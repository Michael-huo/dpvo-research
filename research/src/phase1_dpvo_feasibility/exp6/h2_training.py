"""Training and representation diagnostics for the canonical Exp6 H2 module."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..exp3.runtime import _seed_everything
from .jepa_fmap import coordinate_masks, tokens_to_field
from .jepa_runtime import CompactFeatureStore
from .oracle_packet import FMapZeroContextPacket, _derive_frontend_state
from .predictor import AnchorInterval, RobustTransportBlock5Predictor, prediction_loss
from .protocol import FrameIdentity, canonical_sha256
from .transport import (
    RobustCorrespondence,
    TransportResult,
    estimate_robust_correspondence,
    freeze_train_calibration,
    robust_protocol_metadata,
    robust_transport_interpolation,
    sparse_candidate_statistics,
)

def _field(store: Any, identity: FrameIdentity, transform: Any, device: torch.device) -> torch.Tensor:
    tokens = torch.from_numpy(store.get(identity)).to(device)
    return tokens_to_field(tokens[None], transform)[0]


class RobustCorrespondenceStore:
    def __init__(self, calibration: Mapping[str, Any]) -> None:
        self.calibration = dict(calibration); self.rows: dict[int, RobustCorrespondence] = {}

    def put(self, interval: AnchorInterval, value: RobustCorrespondence) -> None:
        if interval.interval_index in self.rows: raise RuntimeError("duplicate correspondence")
        self.rows[interval.interval_index] = RobustCorrespondence(*[
            getattr(value, name).detach().half().cpu() for name in value.__dataclass_fields__])

    def batch(self, intervals: Sequence[AnchorInterval], device: torch.device, *,
              repeat_queries: bool) -> RobustCorrespondence:
        names = RobustCorrespondence.__dataclass_fields__
        values = []
        for name in names:
            rows = []
            for interval in intervals:
                value = getattr(self.rows[interval.interval_index], name).to(device=device, dtype=torch.float32)
                rows.extend([value] * (len(interval.hidden) if repeat_queries else 1))
            values.append(torch.cat(rows, dim=0))
        return RobustCorrespondence(*values)


@torch.no_grad()
def calibrate_train_only_thresholds(intervals: Sequence[AnchorInterval], store: CompactFeatureStore,
                                    transform: Any, mask: torch.Tensor) -> dict[str, Any]:
    preliminary = []
    for interval in intervals:
        j0 = _field(store, interval.anchor0, transform, mask.device)[None]
        j1 = _field(store, interval.anchor1, transform, mask.device)[None]
        preliminary.append(sparse_candidate_statistics(j0, j1, mask))
    preliminary_values = torch.cat([row[f"chebyshev_displacement_{suffix}"][
        row[f"candidate_mask_{suffix}"]] for row in preliminary for suffix in ("0", "1")])
    radius = int(torch.ceil(torch.quantile(preliminary_values, .99)).clamp(4, 12))
    bounded = []
    for interval in intervals:
        j0 = _field(store, interval.anchor0, transform, mask.device)[None]
        j1 = _field(store, interval.anchor1, transform, mask.device)[None]
        bounded.append(sparse_candidate_statistics(j0, j1, mask, radius))
    identity_sha = canonical_sha256([[row.anchor0.key, row.anchor1.key] for row in intervals])
    result = freeze_train_calibration(preliminary, bounded, train_endpoint_identity_sha256=identity_sha)
    result["calibration_protocol_sha256"] = canonical_sha256(robust_protocol_metadata())
    result["calibration_sha256"] = canonical_sha256(result)
    return result


@torch.no_grad()
def build_robust_correspondence_store(intervals: Sequence[AnchorInterval], store: CompactFeatureStore,
                                      transform: Any, mask: torch.Tensor,
                                      calibration: Mapping[str, Any]) -> tuple[RobustCorrespondenceStore, dict[str, Any]]:
    result = RobustCorrespondenceStore(calibration); started = time.perf_counter()
    for interval in intervals:
        j0 = _field(store, interval.anchor0, transform, mask.device)[None]
        j1 = _field(store, interval.anchor1, transform, mask.device)[None]
        result.put(interval, estimate_robust_correspondence(j0, j1, mask, calibration))
    return result, {"interval_count": len(intervals), "endpoint_only": True,
                    "contains_hidden_target": False, "elapsed_seconds": time.perf_counter()-started,
                    "protocol": robust_protocol_metadata(calibration)}


def _batches(values: Sequence[AnchorInterval], size: int, seed: int | None = None) -> Iterable[tuple[AnchorInterval, ...]]:
    order = np.arange(len(values));
    if seed is not None: np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size): yield tuple(values[index] for index in order[start:start+size])


def _transport_batch(intervals: Sequence[AnchorInterval], store: Any, transform: Any,
                     mask: torch.Tensor, robust: RobustCorrespondenceStore
                     ) -> tuple[TransportResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    j0, j1, target, alpha, delta = [], [], [], [], []
    for interval in intervals:
        left = _field(store, interval.anchor0, transform, mask.device)
        right = _field(store, interval.anchor1, transform, mask.device)
        for query in interval.hidden:
            j0.append(left); j1.append(right); target.append(_field(store, query.identity, transform, mask.device))
            alpha.append(query.alpha); delta.append(query.delta_t_seconds)
    j0t, j1t, targett = torch.stack(j0), torch.stack(j1), torch.stack(target)
    alphat = torch.tensor(alpha, device=mask.device); deltat = torch.tensor(delta, device=mask.device)
    correspondence = robust.batch(intervals, mask.device, repeat_queries=True)
    return robust_transport_interpolation(j0t, j1t, alphat, correspondence, mask), targett, alphat, deltat


def _new_predictor(config: Mapping[str, Any]) -> RobustTransportBlock5Predictor:
    value = config["predictor"]
    return RobustTransportBlock5Predictor(int(value["feature_dim"]), int(value["hidden_dim"]),
        int(value["difference_dim"]), int(value["reliability_dim"]), int(value["residual_blocks"]),
        int(value["group_norm_groups"]), int(value["time_hidden_dim"]))


def _predict(model: torch.nn.Module, transported: TransportResult, alpha: torch.Tensor,
             delta: torch.Tensor) -> torch.Tensor:
    return model(transported.field, transported.warped_difference, transported.warp0.coverage,
                 transported.warp1.coverage, transported.fused_confidence, alpha, delta)


def tiny_overfit(store: CompactFeatureStore, intervals: Sequence[AnchorInterval], transform: Any,
                 mask: torch.Tensor, config: Mapping[str, Any], robust: RobustCorrespondenceStore) -> dict[str, Any]:
    _seed_everything(int(config["experiment"]["seed"])); selected = tuple(intervals[:8])
    model = _new_predictor(config).cuda().train(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    transported, target, alpha, delta = _transport_batch(selected, store, transform, mask, robust)
    with torch.no_grad(): initial = float(prediction_loss(transported.field, target, mask)["total"])
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True): loss = prediction_loss(_predict(model, transported, alpha, delta), target, mask)["total"]
        loss.backward(); optimizer.step()
    with torch.no_grad(): final = float(prediction_loss(_predict(model, transported, alpha, delta), target, mask)["total"])
    required = max(1e-4, 1e-3 * abs(initial)); passed = initial-final > required
    result = {"interval_count": 8, "steps": 200, "initial_total": initial, "final_total": final,
              "improvement": initial-final, "required_improvement": required, "passed": passed,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated())}
    del model, optimizer; torch.cuda.empty_cache()
    if not passed: raise RuntimeError(f"tiny-overfit failed: {result}")
    return result


@torch.no_grad()
def _validation(model: torch.nn.Module, intervals: Sequence[AnchorInterval], store: Any,
                transform: Any, mask: torch.Tensor, robust: RobustCorrespondenceStore) -> dict[str, float]:
    totals = {key: 0. for key in ("total", "cosine", "mse", "smooth_l1", "norm_ratio")}; count = 0; model.eval()
    for batch in _batches(intervals, 2):
        transported, target, alpha, delta = _transport_batch(batch, store, transform, mask, robust)
        metrics = prediction_loss(_predict(model, transported, alpha, delta), target, mask)
        for key in totals: totals[key] += float(metrics[key]) * len(target)
        count += len(target)
    return {key: value/count for key, value in totals.items()} | {"hidden_query_count": count}


def train_predictor(store: CompactFeatureStore, split: Mapping[str, Sequence[AnchorInterval]],
                    transform: Any, mask: torch.Tensor, config: Mapping[str, Any],
                    robust: RobustCorrespondenceStore) -> tuple[torch.nn.Module, dict[str, Any]]:
    _seed_everything(1234); model = _new_predictor(config).cuda().train()
    sample, _, alpha, delta = _transport_batch(split["train"][:1], store, transform, mask, robust)
    if not torch.equal(_predict(model, sample, alpha, delta), sample.field): raise RuntimeError("zero initialization changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True); best, best_value, best_epoch = None, float("inf"), -1
    history = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(30):
        model.train(); total = 0.; count = 0
        for batch in _batches(split["train"], 2, 1234+epoch):
            transported, target, alpha, delta = _transport_batch(batch, store, transform, mask, robust)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True): loss = prediction_loss(_predict(model, transported, alpha, delta), target, mask)["total"]
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(target); count += len(target)
        validation = _validation(model, split["validation"], store, transform, mask, robust)
        history.append({"epoch": epoch+1, "train_total": total/count, "validation_total": validation["total"]})
        if validation["total"] < best_value:
            best_value, best_epoch = validation["total"], epoch+1
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best is None: raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best); model.requires_grad_(False).eval()
    return model, {"best_epoch": best_epoch, "best_validation_total": best_value, "history": history,
        "elapsed_seconds": time.perf_counter()-started, "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "test_was_read_during_training_or_selection": False}


class MetricAccumulator:
    def __init__(self) -> None: self.count = 0; self.sum = {key: 0. for key in ("cosine", "mse", "smooth_l1", "norm_ratio")}; self.errors = []
    def add(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        metrics = prediction_loss(prediction, target, mask); count = len(prediction)
        for key in self.sum: self.sum[key] += float(metrics[key]) * count
        self.errors.append((1-F.cosine_similarity(prediction.float(), target.float(), dim=1))[:, mask.bool()].cpu().numpy().ravel()); self.count += count
    def payload(self) -> dict[str, Any]:
        values = np.concatenate(self.errors)
        return {**{key: value/self.count for key, value in self.sum.items()}, "hidden_query_count": self.count,
                "spatial_cosine_error_quantiles": {name: float(np.quantile(values, q)) for name,q in (("p50",.5),("p90",.9),("p95",.95),("p99",.99))}}


@torch.no_grad()
def held_out_representation(split_test: Sequence[AnchorInterval], store: CompactFeatureStore,
                            transform: Any, mask: torch.Tensor, robust: RobustCorrespondenceStore,
                            predictor: torch.nn.Module, bridge: torch.nn.Module) -> dict[str, Any]:
    names = ("robust_transport", "predicted_jepa", "oracle_jepa"); token = {name: MetricAccumulator() for name in names}; fmap = {name: MetricAccumulator() for name in names}; gmap = {name: 0. for name in names}; count = 0
    fmap_mask = torch.from_numpy(coordinate_masks(transform)["fmap_valid_mask"]).cuda()
    for batch in _batches(split_test, 2):
        transported, target, alpha, delta = _transport_batch(batch, store, transform, mask, robust)
        fields = {names[0]: transported.field, names[1]: _predict(predictor, transported, alpha, delta), names[2]: target}
        for name, value in fields.items(): token[name].add(value, target, mask)
        fmaps = {name: bridge(value.flatten(2).transpose(1,2)) for name,value in fields.items()}; oracle = fmaps[names[2]]
        for name,value in fmaps.items(): fmap[name].add(value, oracle, fmap_mask)
        queries = [query for interval in batch for query in interval.hidden]
        for row, query in enumerate(queries):
            oracle_state,_ = _derive_frontend_state(
                FMapZeroContextPacket(oracle[row:row+1,None]), query.identity, 1234,
                patches_per_image=96, patch_size=3, context_dim=384,
            )
            for name,value in fmaps.items():
                state,_ = _derive_frontend_state(
                    FMapZeroContextPacket(value[row:row+1,None]), query.identity, 1234,
                    patches_per_image=96, patch_size=3, context_dim=384,
                )
                gmap[name] += float(F.cosine_similarity(state.gmap.float().flatten(2), oracle_state.gmap.float().flatten(2), dim=2).mean())
            count += 1
    return {"evaluation_role": "held_out_mh01_representation_test",
            "trajectory_generalization_claim": False,
            "gate1_block5": {name: row.payload() for name,row in token.items()},
            "gate2_frozen_exp6_2": {name: row.payload() | {"derived_gmap_cosine": gmap[name]/count} for name,row in fmap.items()}}


def _shared_pca_rgb(fields: Sequence[np.ndarray], valid_mask: np.ndarray
                    ) -> tuple[list[np.ndarray], dict[str,Any]]:
    """Project comparable CxHxW fields through one deterministic PCA color basis."""
    if not fields or any(value.ndim != 3 for value in fields):
        raise ValueError("PCA visualization requires non-empty CxHxW fields")
    shape=fields[0].shape
    if any(value.shape != shape for value in fields) or valid_mask.shape != shape[1:]:
        raise ValueError("shared PCA fields/mask must have identical spatial shape")
    samples=np.concatenate([value[:,valid_mask].T.astype(np.float32,copy=False) for value in fields])
    mean=samples.mean(axis=0,keepdims=True); centered=samples-mean
    covariance=(centered.T@centered)/max(1,len(centered)-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    basis=eigenvectors[:,-3:].astype(np.float32,copy=False)
    projections=[]; valid_projected=[]
    for value in fields:
        projected=np.moveaxis((np.moveaxis(value,0,-1)-mean.reshape(1,1,-1))@basis,-1,0)
        projected=np.moveaxis(projected,0,-1); projections.append(projected)
        valid_projected.append(projected[valid_mask])
    stacked=np.concatenate(valid_projected)
    low=np.quantile(stacked,.01,axis=0); high=np.quantile(stacked,.99,axis=0)
    rendered=[]
    for projected in projections:
        rgb=np.clip((projected-low)/(high-low+1e-8),0,1)
        rgb[~valid_mask]=0
        rendered.append(rgb)
    return rendered,{"basis":"shared_covariance_eigh_top3","centering":"shared_valid_token_mean",
        "normalization":"shared_valid_p01_p99_per_component","eigenvalues_top3":[float(x) for x in eigenvalues[-3:]],
        "valid_sample_count":int(len(samples))}


def _cosine_error_map(left: np.ndarray, right: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    numerator=(left*right).sum(axis=0)
    denominator=np.linalg.norm(left,axis=0)*np.linalg.norm(right,axis=0)
    error=1-numerator/np.maximum(denominator,1e-8)
    return np.where(valid_mask,error,np.nan)


@torch.no_grad()
def _plot_feature_diagnostics(path: Path, sequence: str, intervals: Sequence[AnchorInterval],
                              token_store: CompactFeatureStore, true_store: CompactFeatureStore,
                              transform: Any, bridge: torch.nn.Module,
                              robust: RobustCorrespondenceStore, predictor: torch.nn.Module,
                              config: Mapping[str,Any]) -> dict[str,Any]:
    """Render offline representation diagnostics; outputs never feed DPVO or training."""
    rows=[(interval,query) for interval in intervals for query in interval.hidden]
    count=min(int(config["diagnostics"]["sampled_hidden_frames_per_sequence"]),len(rows))
    seed=int(config["diagnostics"]["seed"])+int(canonical_sha256(sequence)[:8],16)
    indices=np.sort(np.random.default_rng(seed).choice(len(rows),size=count,replace=False))
    selected=[rows[int(index)] for index in indices]
    token_mask=coordinate_masks(transform)["valid_token_mask"].astype(bool)
    fmap_mask=coordinate_masks(transform)["fmap_valid_mask"].astype(bool)
    oracle_jepa=[]; predicted_jepa=[]; true_fmap=[]; oracle_fmap=[]; predicted_fmap=[]
    sample_metadata=[]; device=torch.device("cuda")
    for interval,query in selected:
        oracle=_field(token_store,query.identity,transform,device)[None]
        j0=_field(token_store,interval.anchor0,transform,device)[None]
        j1=_field(token_store,interval.anchor1,transform,device)[None]
        alpha=torch.tensor([query.alpha],device=device); delta=torch.tensor([query.delta_t_seconds],device=device)
        correspondence=robust.batch((interval,),device,repeat_queries=False)
        transported=robust_transport_interpolation(j0,j1,alpha,correspondence,
            torch.from_numpy(token_mask).to(device))
        predicted=_predict(predictor,transported,alpha,delta)
        oracle_bridge=bridge(oracle.flatten(2).transpose(1,2))
        predicted_bridge=bridge(predicted.flatten(2).transpose(1,2))
        oracle_jepa.append(oracle[0].float().cpu().numpy())
        predicted_jepa.append(predicted[0].float().cpu().numpy())
        true_fmap.append(true_store.get(query.identity))
        oracle_fmap.append(oracle_bridge[0].float().cpu().numpy())
        predicted_fmap.append(predicted_bridge[0].float().cpu().numpy())
        sample_metadata.append({"identity":query.identity.key,"candidate_index":query.identity.candidate_index,
            "timestamp_ns":query.identity.timestamp_ns,"anchor0_identity":interval.anchor0.key,
            "anchor1_identity":interval.anchor1.key,"alpha":query.alpha,
            "delta_t_seconds":query.delta_t_seconds})
    jepa_rgb,jepa_pca=_shared_pca_rgb([*oracle_jepa,*predicted_jepa],token_mask)
    fmap_rgb,fmap_pca=_shared_pca_rgb([*true_fmap,*oracle_fmap,*predicted_fmap],fmap_mask)
    jepa_oracle_rgb,jepa_predicted_rgb=jepa_rgb[:count],jepa_rgb[count:]
    fmap_true_rgb=fmap_rgb[:count]; fmap_oracle_rgb=fmap_rgb[count:2*count]; fmap_predicted_rgb=fmap_rgb[2*count:]
    jepa_error=[_cosine_error_map(a,b,token_mask) for a,b in zip(predicted_jepa,oracle_jepa)]
    h1_error=[_cosine_error_map(a,b,fmap_mask) for a,b in zip(oracle_fmap,true_fmap)]
    h2_error=[_cosine_error_map(a,b,fmap_mask) for a,b in zip(predicted_fmap,oracle_fmap)]
    def vmax(values: Sequence[np.ndarray]) -> float:
        finite=np.concatenate([value[np.isfinite(value)] for value in values])
        return max(1e-6,float(np.quantile(finite,.99)))
    error_limits={"jepa_prediction":vmax(jepa_error),"h1_bridge_to_true_fmap":vmax(h1_error),
                  "h2_prediction_through_bridge":vmax(h2_error)}
    os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-phase1-exp6-3-final")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figure,axes=plt.subplots(count,8,figsize=(22,2.7*count),squeeze=False)
    titles=("Oracle JEPA\n(shared PCA)","Predicted JEPA\n(shared PCA)","JEPA cosine error",
        "True FNet FMap\n(shared PCA)","Oracle JEPA→bridge\n(shared PCA)",
        "Predicted JEPA→bridge\n(shared PCA)","H1 cosine error\nOracle bridge vs True",
        "H2 cosine error\nPred bridge vs Oracle bridge")
    for column,title in enumerate(titles): axes[0,column].set_title(title,fontsize=9)
    for row_index,metadata in enumerate(sample_metadata):
        images=(jepa_oracle_rgb[row_index],jepa_predicted_rgb[row_index],jepa_error[row_index],
            fmap_true_rgb[row_index],fmap_oracle_rgb[row_index],fmap_predicted_rgb[row_index],
            h1_error[row_index],h2_error[row_index])
        for column,value in enumerate(images):
            if column in (2,6,7):
                key=("jepa_prediction","h1_bridge_to_true_fmap","h2_prediction_through_bridge")[(2,6,7).index(column)]
                axes[row_index,column].imshow(value,cmap="magma",vmin=0,vmax=error_limits[key])
            else: axes[row_index,column].imshow(value)
            axes[row_index,column].set_xticks([]); axes[row_index,column].set_yticks([])
        axes[row_index,0].set_ylabel(f"cand {metadata['candidate_index']}\nα={metadata['alpha']:.3f}",fontsize=8)
    figure.suptitle(f"{sequence} — offline hidden representation diagnostics",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.985)); figure.savefig(path,dpi=160); plt.close(figure)
    payload={"role":"offline_diagnostic_only_not_model_or_dpvo_input",
        "sample_selection":"deterministic_random_hidden_queries","seed":seed,"sample_count":count,
        "sample_identity_sha256":canonical_sha256([row["identity"] for row in sample_metadata]),
        "samples":sample_metadata,"jepa_projection":jepa_pca,"fmap_projection":fmap_pca,
        "error_colormap":"magma","error_vmax_p99":error_limits,
        "oracle_hidden_rgb_usage":"offline_reference_extraction_only",
        "hidden_online_rgb_violation_count":0}
    return payload



