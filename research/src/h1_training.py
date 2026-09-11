"""Feature extraction, bridge training, and Oracle controls for Exp6 H1."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dpvo_backend import _seed_everything
from .jepa_fmap import (
    BRIDGE_ARCHITECTURE,
    build_bridge,
    coordinate_masks,
    coordinate_protocol_metadata,
    identity_shuffle,
    masked_reconstruction_loss,
    preprocess_full_fov_rgb,
)
from .jepa_runtime import JepaSidecar, load_dpvo_domain, load_fnet, state_dict_sha256, teacher_fmap
from .oracle_packet import FMapZeroContextPacket, _derive_frontend_state
from .protocol import FrameIdentity, canonical_sha256, repo_path, sha256_file

from .training_runtime import h1_tensor_batch

TRAINING_SEQUENCE = "MH_01_easy"

class FeatureStore:
    """Ephemeral all-frame teacher FMaps and hidden-frame block-5 tokens."""

    def __init__(self, root: Path, records: Sequence[Any], transform: Any) -> None:
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.identities = [record.identity for record in records]
        self.identity_keys = [identity.key for identity in self.identities]
        self.index = {key: row for row, key in enumerate(self.identity_keys)}
        count = len(records)
        token_count = transform.token_grid_height * transform.token_grid_width
        fmap_shape = (128, transform.fmap_height, transform.fmap_width)
        self.block5 = np.memmap(root / "block5.mmap", mode="w+", dtype=np.float16,
                                shape=(count, token_count, 768))
        self.teacher = np.memmap(root / "teacher.mmap", mode="w+", dtype=np.float16,
                                 shape=(count, *fmap_shape))
        self.teacher_written = np.zeros(count, dtype=np.bool_)
        self.token_written = np.zeros(count, dtype=np.bool_)

    def finalize(self, hidden_keys: set[str]) -> dict[str, Any]:
        expected = np.asarray([key in hidden_keys for key in self.identity_keys])
        if not bool(self.teacher_written.all()) or not bool(self.token_written[expected].all()):
            raise RuntimeError("ephemeral feature store is incomplete")
        self.block5.flush(); self.teacher.flush()
        return {
            "frame_count": len(self.identity_keys), "hidden_token_count": int(expected.sum()),
            "identity_list_sha256": canonical_sha256(self.identity_keys),
            "contains_rgb_or_path": False, "teacher_dtype": "float16",
            "token_dtype": "float16",
        }

    def close(self) -> None:
        for value in (self.block5, self.teacher):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def extract_feature_store(
    records: Sequence[Any], hidden_keys: set[str], calibration: np.ndarray,
    config: Mapping[str, Any], temporary: Path, transform: Any,
) -> tuple[FeatureStore, dict[str, Any]]:
    store = FeatureStore(temporary / "feature_store", records, transform)
    fnet = load_fnet(repo_path(config["paths"]["dpvo_checkpoint"]))
    batch_size = int(config["jepa"]["batch_size"])
    with JepaSidecar(config, temporary) as sidecar:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            hidden_tensors, hidden_rows = [], []
            for local, record in enumerate(batch):
                row = start + local
                bgr, _ = load_dpvo_domain(record.rgb_path, calibration)
                if tuple(bgr.shape[:2]) != (transform.source_height, transform.source_width):
                    raise RuntimeError("DPVO-domain geometry changed within a sequence")
                store.teacher[row] = teacher_fmap(
                    fnet, bgr, (transform.fmap_height, transform.fmap_width),
                )
                store.teacher_written[row] = True
                if record.identity.key in hidden_keys:
                    tensor, _ = preprocess_full_fov_rgb(bgr[..., ::-1].copy(), transform)
                    hidden_tensors.append(tensor.numpy()); hidden_rows.append(row)
            if hidden_tensors:
                source = temporary / f"jepa_input_{start}.npy"
                output = temporary / f"block5_{start}.npy"
                np.save(source, np.stack(hidden_tensors), allow_pickle=False)
                sidecar.extract(source, output, str(start))
                values = np.load(output)
                for local, row in enumerate(hidden_rows):
                    store.block5[row] = values[local]; store.token_written[row] = True
                source.unlink(); output.unlink()
    del fnet; torch.cuda.empty_cache()
    return store, {"store": store.finalize(hidden_keys), "jepa": sidecar.provenance}


def _batch_indices(indices: np.ndarray, size: int, seed: int) -> Iterable[np.ndarray]:
    order = indices.copy(); np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield order[start:start + size]


@torch.no_grad()
def _evaluate_bridge(
    model: torch.nn.Module, store: FeatureStore, teacher_rows: np.ndarray,
    token_rows: np.ndarray, mask: torch.Tensor, batch_size: int,
    profiler: Any | None = None,
) -> dict[str, float | int]:
    totals = {"cosine": 0.0, "smooth_l1": 0.0, "mse": 0.0}
    count = 0
    model.eval()
    for start in range(0, len(teacher_rows), batch_size):
        if profiler is not None: profiler.begin_outer()
        target = teacher_rows[start:start + batch_size]
        source = token_rows[start:start + batch_size]
        stage = "resident_batch_gather" if hasattr(store, "tensor_batch") else "batch_data_memmap_h2d"
        scope = contextlib.nullcontext() if profiler is None else profiler.stage(stage)
        with scope:
            tokens = h1_tensor_batch(store, "block5", source)
            teacher = h1_tensor_batch(store, "teacher", target)
        scope = (contextlib.nullcontext() if profiler is None
                 else profiler.stage("validation_forward_loss"))
        with scope:
            prediction = model(tokens)
            losses = masked_reconstruction_loss(prediction, teacher, mask)
        batch = len(target)
        scope = (contextlib.nullcontext() if profiler is None
                 else profiler.stage("synchronization_wait", cuda=False))
        with scope:
            totals["cosine"] += (1.0 - float(losses["cosine"])) * batch
            totals["smooth_l1"] += float(losses["smooth_l1"]) * batch
            totals["mse"] += float(torch.mean((prediction.float() - teacher) ** 2)) * batch
        count += batch
        if profiler is not None: profiler.finish_outer()
    return {key: value / count for key, value in totals.items()} | {"sample_count": count}


@torch.no_grad()
def _derived_gmap_cosine(
    model: torch.nn.Module, store: FeatureStore, teacher_rows: np.ndarray,
    token_rows: np.ndarray, config: Mapping[str, Any],
) -> float:
    total = 0.0
    for teacher_row, token_row in zip(teacher_rows.tolist(), token_rows.tolist()):
        tokens = torch.from_numpy(np.asarray(store.block5[token_row:token_row + 1], np.float32)).cuda()
        prediction = model(tokens)[0]
        teacher = torch.from_numpy(np.asarray(store.teacher[teacher_row], np.float32)).cuda()
        derived = []
        for fmap in (prediction, teacher):
            state, _ = _derive_frontend_state(
                FMapZeroContextPacket(fmap[None, None]), store.identities[teacher_row],
                int(config["experiment"]["seed"]), patches_per_image=96,
                patch_size=3, context_dim=384,
            )
            derived.append(state.gmap.float())
        total += float(F.cosine_similarity(
            derived[0].flatten(2), derived[1].flatten(2), dim=2,
        ).mean())
    return total / len(teacher_rows)


def train_bridge(
    store: FeatureStore, split_keys: Mapping[str, Sequence[str]], transform: Any,
    config: Mapping[str, Any], checkpoint_path: Path,
    training_lineage: Mapping[str, Any] | None = None,
    *, profiler: Any | None = None, validation_profiler: Any | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    bridge = config["bridge"]; seed = int(config["experiment"]["seed"])
    init_seed = int(config["experiment"]["bridge_initialization_seed"])
    batch_size = int(bridge["batch_size"])
    index = store.index
    train_rows = np.asarray([index[key] for key in split_keys["train"]], dtype=np.int64)
    validation_rows = np.asarray([index[key] for key in split_keys["validation"]], dtype=np.int64)
    mask = torch.from_numpy(coordinate_masks(transform)["fmap_valid_mask"]).cuda()
    _seed_everything(init_seed)
    model = build_bridge(transform, channels=int(bridge["hidden_channels"])).cuda().train()
    history: list[dict[str, Any]] = []
    order_hashes: list[list[str]] = []
    best_value, best_epoch = float("inf"), 0
    epochs_per_pass = int(bridge["epochs_per_pass"])
    started = time.perf_counter()
    for pass_index in range(int(bridge["passes"])):
        if pass_index:
            # Reproduce the successful second-stage lineage exactly: the
            # weights continue, while optimizer/scaler and RNG restart.
            _seed_everything(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(bridge["learning_rate"]),
            weight_decay=float(bridge["weight_decay"]),
        )
        if optimizer.state:
            raise AssertionError("each bridge pass must start with empty AdamW state")
        scaler = torch.cuda.amp.GradScaler(enabled=bool(bridge["amp"]))
        pass_hashes = []
        for local_epoch in range(epochs_per_pass):
            epoch_started = time.perf_counter()
            batches = list(_batch_indices(train_rows, batch_size, seed + local_epoch))
            identity_order = [store.identity_keys[row] for rows in batches for row in rows]
            pass_hashes.append(canonical_sha256(identity_order))
            training_total = 0.0
            model.train()
            for rows in batches:
                if profiler is not None: profiler.begin_outer()
                stage = "resident_batch_gather" if hasattr(store, "tensor_batch") else "batch_data_memmap_h2d"
                scope = contextlib.nullcontext() if profiler is None else profiler.stage(stage)
                with scope:
                    tokens = h1_tensor_batch(store, "block5", rows)
                    teacher = h1_tensor_batch(store, "teacher", rows)
                optimizer.zero_grad(set_to_none=True)
                scope = (contextlib.nullcontext() if profiler is None
                         else profiler.stage("forward_loss"))
                with scope:
                    with torch.cuda.amp.autocast(enabled=bool(bridge["amp"])):
                        loss = masked_reconstruction_loss(
                            model(tokens), teacher, mask,
                        )["total"]
                scope = (contextlib.nullcontext() if profiler is None
                         else profiler.stage("backward"))
                with scope: scaler.scale(loss).backward()
                scope = (contextlib.nullcontext() if profiler is None
                         else profiler.stage("optimizer_scaler"))
                with scope: scaler.step(optimizer); scaler.update()
                scope = (contextlib.nullcontext() if profiler is None
                         else profiler.stage("synchronization_wait", cuda=False))
                with scope: training_total += float(loss.detach()) * len(rows)
                if profiler is not None: profiler.finish_outer()
            validation = _evaluate_bridge(
                model, store, validation_rows, validation_rows, mask, batch_size,
                profiler=validation_profiler,
            )
            value = (1.0 - float(validation["cosine"])) + 0.1 * float(validation["smooth_l1"])
            global_epoch = pass_index * epochs_per_pass + local_epoch + 1
            if value < best_value:
                best_value, best_epoch = value, global_epoch
            history.append({
                "global_epoch": global_epoch, "pass": pass_index + 1,
                "epoch_in_pass": local_epoch + 1,
                "training_total": training_total / len(train_rows),
                "validation": validation, "validation_total": value,
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
            })
        order_hashes.append(pass_hashes)
        del optimizer, scaler
    if order_hashes[0] != order_hashes[1]:
        raise AssertionError("the two bridge passes did not repeat the same batch order")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    protocol_metadata = coordinate_protocol_metadata(
        target_height=int(config["jepa"]["target_height"]),
        patch_size=int(config["jepa"]["patch_size"]),
        fmap_scale=int(config["teacher"]["fmap_scale"]),
    )
    torch.save({
        "schema_version": 2, "architecture": BRIDGE_ARCHITECTURE,
        "layer_zero_based": 5, "state_dict": state,
        "state_dict_sha256": state_dict_sha256(state),
        "coordinate_protocol": protocol_metadata,
        "training_recipe": {
            "training_sequence": TRAINING_SEQUENCE, "initialization_seed": init_seed,
            "passes": 2, "epochs_per_pass": 30,
            "optimizer": "AdamW_reset_each_pass", "grad_scaler": "reset_each_pass",
            "learning_rate": float(bridge["learning_rate"]),
            "weight_decay": float(bridge["weight_decay"]), "scheduler": "none",
            "loss": "masked_cosine_plus_0.1_smooth_l1",
            "batch_size": batch_size, "checkpoint_selector": "second_pass_epoch_30_final",
        },
        "training_lineage": dict(training_lineage or {}),
    }, checkpoint_path)
    summary = {
        "lineage": {
            "recipe": (
                "scratch_seed_1236_to_30_epochs_then_optimizer_scaler_rng_reset_"
                "to_same_30_epoch_batch_schedule_then_second_pass_epoch_30_final"
            ),
            "training_sample_population": "MH_01_easy_hidden_contiguous_train",
            "training_sample_count": len(train_rows),
            "loss": "masked_cosine_plus_0.1_smooth_l1",
            "first_pass_epochs": epochs_per_pass,
            "first_pass_checkpoint_selector": "epoch_30_final",
            "second_pass_epochs": epochs_per_pass,
            "optimizer": "AdamW_reset_before_each_pass",
            "grad_scaler": "reset_before_each_pass",
            "second_pass_rng_reset_seed": seed,
            "learning_rate": float(bridge["learning_rate"]),
            "weight_decay": float(bridge["weight_decay"]),
            "scheduler": "none",
            "final_checkpoint_selector": "second_pass_epoch_30_final",
        },
        "initialization_seed": init_seed, "passes": 2, "epochs_per_pass": 30,
        "optimizer_reset_between_passes": True, "grad_scaler_reset_between_passes": True,
        "batch_order_repeated_between_passes": True,
        "batch_order_sha256_by_epoch": order_hashes[0],
        "history": history,
        "first": history[0], "first_pass_final": history[epochs_per_pass - 1],
        "best": history[best_epoch - 1],
        "best_epoch": best_epoch, "final": history[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_sha256": state_dict_sha256(state),
    }
    return model.eval(), summary


def evaluate_representation_control(
    model: torch.nn.Module, store: FeatureStore, test_keys: Sequence[str],
    transform: Any, config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = np.asarray([store.index[key] for key in test_keys], dtype=np.int64)
    shuffle = identity_shuffle(list(test_keys), int(config["experiment"]["seed"]))
    shuffled = np.asarray([store.index[shuffle[key]] for key in test_keys], dtype=np.int64)
    mask = torch.from_numpy(coordinate_masks(transform)["fmap_valid_mask"]).cuda()
    batch_size = int(config["bridge"]["batch_size"])
    correct = _evaluate_bridge(model, store, rows, rows, mask, batch_size)
    incorrect = _evaluate_bridge(model, store, rows, shuffled, mask, batch_size)
    correct["derived_gmap_cosine"] = _derived_gmap_cosine(
        model, store, rows, rows, config,
    )
    incorrect["derived_gmap_cosine"] = _derived_gmap_cosine(
        model, store, rows, shuffled, config,
    )
    return {
        "scope": "held_out_mh01_representation_only",
        "correct": correct, "identity_shuffled": incorrect,
        "identity_shuffle_method": "deterministic_cyclic_no_fixed_point",
        "identity_shuffle_sha256": canonical_sha256(shuffle),
        "population_sha256": canonical_sha256(list(test_keys)),
    }


def _previous_anchor_mapping(records: Sequence[Any], roles: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}; previous: str | None = None
    for record in records:
        key = record.identity.key
        if roles[key] == "anchor":
            previous = key
        elif previous is None:
            raise RuntimeError("hidden frame has no previous anchor")
        else:
            result[key] = previous
    return result


class HiddenFMapProvider:
    """Runtime provider with no hidden RGB/path/GT/graph-state capability."""

    def __init__(self, condition: str, store: FeatureStore,
                 previous_anchor: Mapping[str, str], model: torch.nn.Module | None = None) -> None:
        if condition not in {"true_fmap", "oracle_jepa_bridge"}:
            raise ValueError(condition)
        if condition == "oracle_jepa_bridge" and model is None:
            raise ValueError("Oracle JEPA evaluation requires the fresh MH01 bridge")
        self.condition, self.store = condition, store
        self.previous_anchor, self.model = dict(previous_anchor), model
        self.last_anchor_key: str | None = None
        self.last_anchor_fmap: torch.Tensor | None = None
        self.anchor_count = 0; self.hidden_count = 0

    def observe_anchor(self, identity: FrameIdentity, packet: Any) -> None:
        self.last_anchor_key = identity.key
        self.last_anchor_fmap = packet.fmap.detach()[0, 0].clone()
        self.anchor_count += 1

    @torch.no_grad()
    def get(self, identity: FrameIdentity, device: str = "cuda") -> FMapZeroContextPacket:
        expected = self.previous_anchor[identity.key]
        if self.last_anchor_key != expected or self.last_anchor_fmap is None:
            raise RuntimeError(f"causal anchor mismatch for {identity.key}")
        row = self.store.index[identity.key]
        if self.condition == "true_fmap":
            fmap = torch.from_numpy(np.asarray(self.store.teacher[row], np.float32)).to(device)
        else:
            tokens = torch.from_numpy(
                np.asarray(self.store.block5[row:row + 1], np.float32),
            ).to(device)
            assert self.model is not None
            fmap = self.model(tokens)[0]
        self.hidden_count += 1
        return FMapZeroContextPacket(fmap[None, None])

    def sanitized_descriptor(self) -> dict[str, Any]:
        return {
            "schema": "RuntimeGeneratedFMapZeroContextV1", "condition": self.condition,
            "stored_online_fields": (["oracle_block5_tokens"]
                                     if self.condition == "oracle_jepa_bridge" else []),
            "hidden_rgb_or_path": False, "groundtruth_or_future_state": False,
            "imap": "deterministic_zero", "patch_xy_gmap_fmap2": "derived",
        }

    def usage_payload(self) -> dict[str, Any]:
        return {"condition": self.condition, "anchor_count": self.anchor_count,
                "hidden_count": self.hidden_count}
