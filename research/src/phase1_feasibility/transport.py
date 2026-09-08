"""Canonical robust sparse correspondence and raw-feature transport for Exp6 H2."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


ROBUST_TRANSPORT_PROTOCOL = "exp6_h2_robust_smooth_transport"
CYCLE_SIGMA = 1.5
EPSILON = 1e-6
COARSE_STRIDE = 4
COARSE_RADIUS = 6.0
COARSE_SIGMA = 4.0
HUBER_DELTA = 1.5
AFFINE_RIDGE = 1e-4
MIN_GLOBAL_MATCHES = 12


def _profile_stage(profiler: Any | None, name: str) -> Any:
    """Return an exclusive performance-only scope without changing default calls."""
    return contextlib.nullcontext() if profiler is None else profiler.stage(name)


@dataclass(frozen=True)
class WarpResult:
    feature_sum: torch.Tensor
    coverage: torch.Tensor
    confidence_sum: torch.Tensor
    field: torch.Tensor
    confidence: torch.Tensor
    lost_mass_ratio: torch.Tensor


@dataclass(frozen=True)
class TransportResult:
    field: torch.Tensor
    warp0: WarpResult
    warp1: WarpResult
    warped_difference: torch.Tensor
    fused_confidence: torch.Tensor


@dataclass(frozen=True)
class RobustCorrespondence:
    displacement_0_to_1: torch.Tensor
    displacement_1_to_0: torch.Tensor
    confidence_0: torch.Tensor
    confidence_1: torch.Tensor
    sparse_mask_0: torch.Tensor
    sparse_mask_1: torch.Tensor
    similarity_peak_0: torch.Tensor
    similarity_peak_1: torch.Tensor
    peak_margin_0: torch.Tensor
    peak_margin_1: torch.Tensor
    cycle_error_0: torch.Tensor
    cycle_error_1: torch.Tensor


def robust_protocol_metadata(calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": ROBUST_TRANSPORT_PROTOCOL,
        "coordinate_order": "x_column_y_row_token_index",
        "matching_descriptor": "valid_masked_l2_normalized_raw_block5",
        "transport_feature": "raw_unnormalized_block5",
        "threshold_calibration_population": "train_endpoint_pairs_only",
        "mutual_consistency": "forward_top1_in_reverse_top2",
        "cycle_confidence_sigma_tokens": CYCLE_SIGMA,
        "global_affine": {"target": "q=A[x,y,1]^T",
                          "displacement": "D_affine=q-[x,y]^T",
                          "irls_iterations": 3, "huber_delta_tokens": HUBER_DELTA,
                          "ridge": AFFINE_RIDGE},
        "coarse_residual": {"stride_tokens": COARSE_STRIDE,
                            "radius_chebyshev_tokens": COARSE_RADIUS,
                            "gaussian_sigma_tokens": COARSE_SIGMA,
                            "irls_iterations": 3, "huber_delta_tokens": HUBER_DELTA,
                            "target": "d_sparse-D_affine(c_sparse)",
                            "effective_sample_count": "(sum_w)^2/(sum_w_squared+eps)",
                            "sample_count_factor": "min(1,N_eff/3)"},
        "minimum_global_sparse_matches": MIN_GLOBAL_MATCHES,
        "warp": "bilinear_forward_soft_splat_raw_features",
        "fallback": "raw_same_coordinate_linear_interpolation",
    }
    if calibration is not None:
        result["frozen_train_calibration"] = dict(calibration)
    return result


def token_coordinates(height: int, width: int, *, device: torch.device,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def normalized_descriptors(field: torch.Tensor) -> torch.Tensor:
    if field.ndim != 4:
        raise ValueError("block5 field must be [B,C,H,W]")
    return F.normalize(field.float().permute(0, 2, 3, 1).reshape(
        field.shape[0], -1, field.shape[1]), dim=-1, eps=1e-8)


def _top2_direction(similarity: torch.Tensor, valid: torch.Tensor,
                    coords: torch.Tensor, max_displacement: int | None) -> tuple[torch.Tensor, ...]:
    count = similarity.shape[-1]
    allowed = valid[None, :].expand(count, -1).clone()
    if max_displacement is not None:
        allowed &= (coords[:, None] - coords[None, :]).abs().amax(dim=-1) <= int(max_displacement)
    negative = torch.finfo(similarity.dtype).min
    values, indices = torch.topk(similarity.masked_fill(~allowed[None], negative), k=2, dim=-1)
    return indices, coords[indices[..., 0]], values[..., 0], values[..., 0] - values[..., 1]


def sparse_candidate_statistics(j0: torch.Tensor, j1: torch.Tensor,
                                valid_mask: torch.Tensor,
                                max_displacement: int | None = None, *,
                                profiler: Any | None = None) -> dict[str, torch.Tensor]:
    if j0.shape != j1.shape or j0.ndim != 4:
        raise ValueError("candidate statistics require equal [B,C,H,W] fields")
    batch, _, height, width = j0.shape
    with _profile_stage(profiler, "endpoint_descriptor_matching"):
        valid = valid_mask.to(device=j0.device, dtype=torch.bool).reshape(-1)
        coords = token_coordinates(height, width, device=j0.device)
        similarity = torch.einsum(
            "bnc,bmc->bnm", normalized_descriptors(j0), normalized_descriptors(j1),
        )
    with _profile_stage(profiler, "bidirectional_topk_correspondence"):
        forward = _top2_direction(similarity, valid, coords, max_displacement)
        reverse = _top2_direction(similarity.transpose(1, 2), valid, coords, max_displacement)
        source_indices = torch.arange(height * width, device=j0.device)[None, :, None]

    def directed(primary: tuple[torch.Tensor, ...], opposite: tuple[torch.Tensor, ...]
                 ) -> dict[str, torch.Tensor]:
        indices, destination, peak, margin = primary
        reverse_top2 = torch.gather(opposite[0], 1, indices[..., 0, None].expand(-1, -1, 2))
        mutual = (reverse_top2 == source_indices).any(dim=-1)
        cycle = (coords[reverse_top2[..., 0]] - coords[None]).norm(dim=-1)
        displacement = destination - coords[None]
        return {"candidate_mask": mutual & valid[None].expand(batch, -1) & torch.isfinite(peak),
                "destination": destination, "displacement": displacement,
                "chebyshev_displacement": displacement.abs().amax(dim=-1),
                "similarity_peak": peak, "peak_margin": margin, "cycle_error": cycle}
    with _profile_stage(profiler, "bidirectional_topk_correspondence"):
        rows0, rows1 = directed(forward, reverse), directed(reverse, forward)
    return {f"{name}_0": value for name, value in rows0.items()} | {
        f"{name}_1": value for name, value in rows1.items()}


def freeze_train_calibration(preliminary: Sequence[Mapping[str, torch.Tensor]],
                             bounded: Sequence[Mapping[str, torch.Tensor]], *,
                             train_endpoint_identity_sha256: str) -> dict[str, Any]:
    def collect(rows: Sequence[Mapping[str, torch.Tensor]], field: str) -> torch.Tensor:
        values = [row[f"{field}_{suffix}"].detach().float()[
            row[f"candidate_mask_{suffix}"].detach().bool()].cpu()
            for row in rows for suffix in ("0", "1")]
        result = torch.cat(values) if values else torch.empty(0)
        if not len(result) or not torch.isfinite(result).all():
            raise RuntimeError(f"train calibration has no finite {field} candidates")
        return result
    preliminary_displacement = collect(preliminary, "chebyshev_displacement")
    max_displacement = int(torch.ceil(torch.quantile(preliminary_displacement, .99)).clamp(4, 12))
    similarity = collect(bounded, "similarity_peak"); margin = collect(bounded, "peak_margin")
    cycle = collect(bounded, "cycle_error")
    margin_min, margin_scale = float(torch.quantile(margin, .5)), float(torch.quantile(margin, .9))
    if margin_scale <= margin_min:
        margin_scale = margin_min + EPSILON
    def summary(values: torch.Tensor) -> dict[str, float]:
        return {name: float(torch.quantile(values, quantile)) for name, quantile in (
            ("q10", .1), ("q50", .5), ("q75", .75), ("q90", .9), ("q99", .99))}
    return {"population": "train_endpoint_pairs_only",
            "train_endpoint_identity_sha256": str(train_endpoint_identity_sha256),
            "directed_preliminary_candidate_count": int(len(preliminary_displacement)),
            "directed_bounded_candidate_count": int(len(similarity)),
            "max_displacement_chebyshev_tokens": max_displacement,
            "similarity_min": float(torch.quantile(similarity, .1)),
            "margin_min": margin_min, "margin_scale": margin_scale,
            "cycle_max_tokens": float(torch.quantile(cycle, .75).clamp(1, 3)),
            "quantile_protocol": {"max_displacement": "ceil(clamp(preliminary_q99,4,12))",
                                  "similarity_min": "bounded_q10", "margin_min": "bounded_q50",
                                  "margin_scale": "bounded_q90", "cycle_max": "clamp(bounded_q75,1,3)"},
            "train_statistics": {"preliminary_chebyshev_displacement": summary(preliminary_displacement),
                                 "bounded_similarity_peak": summary(similarity),
                                 "bounded_peak_margin": summary(margin),
                                 "bounded_cycle_error": summary(cycle)}}


def _weighted_affine(source: torch.Tensor, destination: torch.Tensor,
                     confidence: torch.Tensor) -> torch.Tensor | None:
    if len(source) < MIN_GLOBAL_MATCHES:
        return None
    design = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
    base = confidence.clamp_min(0); weights = base
    eye = torch.eye(3, device=source.device, dtype=source.dtype) * AFFINE_RIDGE
    coefficient = None
    for _ in range(3):
        coefficient = torch.linalg.solve(design.T @ (weights[:, None] * design) + eye,
                                         design.T @ (weights[:, None] * destination))
        residual = (design @ coefficient - destination).norm(dim=-1)
        weights = base * torch.clamp(HUBER_DELTA / residual.clamp_min(EPSILON), max=1.)
    return coefficient


def _coarse_axis(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = list(range(0, length, COARSE_STRIDE))
    if values[-1] != length - 1:
        values.append(length - 1)
    return torch.tensor(values, device=device, dtype=dtype)


def _bilinear_irregular(values: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor,
                        height: int, width: int) -> torch.Tensor:
    query_y = torch.arange(height, device=values.device, dtype=values.dtype)
    query_x = torch.arange(width, device=values.device, dtype=values.dtype)
    y1 = torch.searchsorted(ys, query_y).clamp(1, len(ys) - 1); y0 = y1 - 1
    x1 = torch.searchsorted(xs, query_x).clamp(1, len(xs) - 1); x0 = x1 - 1
    fy = ((query_y - ys[y0]) / (ys[y1] - ys[y0]).clamp_min(1))[:, None]
    fx = ((query_x - xs[x0]) / (xs[x1] - xs[x0]).clamp_min(1))[None, :]
    extra = (1,) * (values.ndim - 2)
    fy, fx = fy.reshape(height, 1, *extra), fx.reshape(1, width, *extra)
    top = (1 - fx) * values[y0[:, None], x0[None]] + fx * values[y0[:, None], x1[None]]
    bottom = (1 - fx) * values[y1[:, None], x0[None]] + fx * values[y1[:, None], x1[None]]
    return (1 - fy) * top + fy * bottom


def effective_sample_count(weights: torch.Tensor) -> torch.Tensor:
    return weights.sum().square() / (weights.square().sum() + EPSILON)


def _smooth_direction(statistics: Mapping[str, torch.Tensor], suffix: str,
                      valid_mask: torch.Tensor, calibration: Mapping[str, Any]
                      , *, profiler: Any | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    displacement = statistics[f"displacement_{suffix}"]
    batch, _, _ = displacement.shape; height, width = valid_mask.shape
    coords = token_coordinates(height, width, device=displacement.device)
    with _profile_stage(profiler, "cycle_confidence_filtering"):
        accepted = (statistics[f"candidate_mask_{suffix}"]
                    & (statistics[f"similarity_peak_{suffix}"] >= float(calibration["similarity_min"]))
                    & (statistics[f"peak_margin_{suffix}"] >= float(calibration["margin_min"]))
                    & (statistics[f"cycle_error_{suffix}"] <= float(calibration["cycle_max_tokens"]))
                    & (statistics[f"chebyshev_displacement_{suffix}"]
                       <= int(calibration["max_displacement_chebyshev_tokens"])))
        margin_confidence = ((statistics[f"peak_margin_{suffix}"] - float(calibration["margin_min"]))
                             / (float(calibration["margin_scale"]) - float(calibration["margin_min"]))).clamp(0, 1)
        cycle_confidence = torch.exp(-.5 * (statistics[f"cycle_error_{suffix}"] / CYCLE_SIGMA) ** 2)
        sparse_confidence = torch.where(accepted, margin_confidence * cycle_confidence,
                                        torch.zeros_like(margin_confidence))
    dense_displacements, dense_confidences = [], []
    ys = _coarse_axis(height, displacement.device, displacement.dtype)
    xs = _coarse_axis(width, displacement.device, displacement.dtype)
    for row in range(batch):
        with _profile_stage(profiler, "global_affine_estimation"):
            mask = accepted[row] & (sparse_confidence[row] > 0)
            source, sparse_displacement = coords[mask], displacement[row, mask]
            confidence = sparse_confidence[row, mask]
            coefficient = _weighted_affine(source, source + sparse_displacement, confidence)
            if coefficient is None:
                dense_displacements.append(torch.zeros((height, width, 2), device=displacement.device))
                dense_confidences.append(torch.zeros((height, width), device=displacement.device))
            else:
                affine_displacement = torch.cat(
                    (coords, torch.ones_like(coords[:, :1])), dim=1,
                ) @ coefficient - coords
                sparse_residual = sparse_displacement - affine_displacement[mask]
        if coefficient is None:
            continue
        with _profile_stage(profiler, "coarse_residual_estimation"):
            coarse_residual = torch.zeros((len(ys), len(xs), 2), device=displacement.device)
            coarse_confidence = torch.zeros((len(ys), len(xs)), device=displacement.device)
            for iy, gy in enumerate(ys):
                for ix, gx in enumerate(xs):
                    distance = source - torch.stack((gx, gy)); local = distance.abs().amax(dim=-1) <= COARSE_RADIUS
                    if profiler is None:
                        local_count = int(local.sum())
                    else:
                        wait_started = time.perf_counter()
                        local_count = int(local.sum())
                        profiler.add_nested_wait_diagnostic(
                            "coarse_local_count_gpu_to_cpu_sync",
                            (time.perf_counter() - wait_started) * 1000.0,
                        )
                    if local_count < 3:
                        continue
                    base = confidence[local] * torch.exp(-distance[local].square().sum(dim=-1) / (2 * COARSE_SIGMA ** 2))
                    residual = sparse_residual[local]
                    estimate = (base[:, None] * residual).sum(0) / base.sum().clamp_min(EPSILON)
                    robust = base
                    for _ in range(3):
                        error = (residual - estimate).norm(dim=-1)
                        robust = base * torch.clamp(HUBER_DELTA / error.clamp_min(EPSILON), max=1.)
                        estimate = (robust[:, None] * residual).sum(0) / robust.sum().clamp_min(EPSILON)
                    n_eff = effective_sample_count(robust)
                    mean_confidence = (robust * confidence[local]).sum() / robust.sum().clamp_min(EPSILON)
                    coarse_residual[iy, ix] = estimate
                    coarse_confidence[iy, ix] = mean_confidence * torch.clamp(n_eff / 3., max=1.)
            residual_dense = _bilinear_irregular(coarse_residual, ys, xs, height, width).reshape(-1, 2)
            confidence_dense = _bilinear_irregular(coarse_confidence[..., None], ys, xs, height, width)[..., 0]
            dense = (affine_displacement + residual_dense).reshape(height, width, 2)
            valid = valid_mask.to(device=dense.device, dtype=torch.bool)
            dense_displacements.append(torch.where(valid[..., None], dense, torch.zeros_like(dense)))
            dense_confidences.append(torch.where(valid, confidence_dense.clamp(0, 1), torch.zeros_like(confidence_dense)))
    return torch.stack(dense_displacements), torch.stack(dense_confidences), accepted.reshape(batch, height, width)


def estimate_robust_correspondence(j0: torch.Tensor, j1: torch.Tensor,
                                   valid_mask: torch.Tensor,
                                   calibration: Mapping[str, Any], *,
                                   profiler: Any | None = None) -> RobustCorrespondence:
    statistics = sparse_candidate_statistics(j0, j1, valid_mask,
        int(calibration["max_displacement_chebyshev_tokens"]), profiler=profiler)
    displacement0, confidence0, sparse0 = _smooth_direction(
        statistics, "0", valid_mask, calibration, profiler=profiler,
    )
    displacement1, confidence1, sparse1 = _smooth_direction(
        statistics, "1", valid_mask, calibration, profiler=profiler,
    )
    with _profile_stage(profiler, "correspondence_query_assembly"):
        result = RobustCorrespondence(
            displacement0, displacement1, confidence0, confidence1,
            sparse0, sparse1, statistics["similarity_peak_0"], statistics["similarity_peak_1"],
            statistics["peak_margin_0"], statistics["peak_margin_1"],
            statistics["cycle_error_0"], statistics["cycle_error_1"],
        )
    return result


def forward_soft_splat(raw_field: torch.Tensor, displacement: torch.Tensor,
                       source_confidence: torch.Tensor, scale: torch.Tensor,
                       valid_mask: torch.Tensor) -> WarpResult:
    if raw_field.ndim != 4 or displacement.shape != (*raw_field.shape[:1], *raw_field.shape[-2:], 2):
        raise ValueError("invalid raw field/displacement shape")
    batch, channels, height, width = raw_field.shape
    confidence = source_confidence.to(raw_field).reshape(batch, -1)
    valid = valid_mask.to(device=raw_field.device, dtype=torch.bool).reshape(1, -1).expand(batch, -1)
    coords = token_coordinates(height, width, device=raw_field.device, dtype=raw_field.dtype)
    scale = torch.as_tensor(scale, device=raw_field.device, dtype=raw_field.dtype).reshape(batch, 1, 1)
    destination = coords[None] + scale * displacement.reshape(batch, -1, 2).to(raw_field.dtype)
    x, y = destination[..., 0], destination[..., 1]; x0, y0 = torch.floor(x), torch.floor(y)
    source = raw_field.reshape(batch, channels, -1); feature_sum = torch.zeros_like(source)
    coverage = torch.zeros((batch, 1, height * width), device=raw_field.device, dtype=raw_field.dtype)
    confidence_sum = torch.zeros_like(coverage); retained = torch.zeros((batch, height * width), device=raw_field.device, dtype=raw_field.dtype)
    for nx, ny, weight in ((x0, y0, (1-(x-x0))*(1-(y-y0))),
                           (x0+1, y0, (x-x0)*(1-(y-y0))),
                           (x0, y0+1, (1-(x-x0))*(y-y0)),
                           (x0+1, y0+1, (x-x0)*(y-y0))):
        inside = valid & (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        effective = weight * inside.to(weight.dtype); retained += effective
        index = (ny.clamp(0, height-1) * width + nx.clamp(0, width-1)).long()
        coverage.scatter_add_(2, index[:, None], effective[:, None])
        confidence_weight = effective * confidence
        confidence_sum.scatter_add_(2, index[:, None], confidence_weight[:, None])
        feature_sum.scatter_add_(2, index[:, None].expand(-1, channels, -1), source * confidence_weight[:, None])
    field = feature_sum / confidence_sum.clamp_min(EPSILON)
    query_confidence = confidence_sum / coverage.clamp_min(EPSILON)
    lost = 1 - retained.sum(1) / valid.sum(1).clamp_min(1).to(retained.dtype)
    shape = (batch, 1, height, width)
    return WarpResult(feature_sum.reshape(batch, channels, height, width), coverage.reshape(shape),
                      confidence_sum.reshape(shape), field.reshape(batch, channels, height, width),
                      query_confidence.reshape(shape).clamp(0, 1), lost)


def robust_transport_interpolation(j0: torch.Tensor, j1: torch.Tensor, alpha: torch.Tensor,
                                   correspondence: RobustCorrespondence,
                                   valid_mask: torch.Tensor, *,
                                   profiler: Any | None = None) -> TransportResult:
    with _profile_stage(profiler, "soft_transport_warp"):
        batch = j0.shape[0]
        alpha = torch.as_tensor(alpha, device=j0.device, dtype=j0.dtype).reshape(batch)
        warp0 = forward_soft_splat(j0, correspondence.displacement_0_to_1,
                                   correspondence.confidence_0, alpha, valid_mask)
        warp1 = forward_soft_splat(j1, correspondence.displacement_1_to_0,
                                   correspondence.confidence_1, 1-alpha, valid_mask)
        a = alpha[:, None, None, None]
        weight0 = (1-a) * warp0.coverage * warp0.confidence
        weight1 = a * warp1.coverage * warp1.confidence
        denominator = weight0 + weight1; linear = (1-a) * j0 + a * j1
        transported = (weight0 * warp0.field + weight1 * warp1.field) / denominator.clamp_min(EPSILON)
        valid = valid_mask.to(device=j0.device, dtype=torch.bool)[None, None]
        transported = torch.where((denominator > EPSILON) & valid, transported, linear)
        transported = torch.where((alpha == 0)[:, None, None, None], j0, transported)
        transported = torch.where((alpha == 1)[:, None, None, None], j1, transported)
        base_coverage = (1-a) * warp0.coverage + a * warp1.coverage
        fused_confidence = (weight0 + weight1) / base_coverage.clamp_min(EPSILON)
        result = TransportResult(transported, warp0, warp1, warp1.field-warp0.field,
                                 fused_confidence.clamp(0, 1))
    return result
