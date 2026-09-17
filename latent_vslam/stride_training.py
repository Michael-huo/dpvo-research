"""Fresh predictors per stride with the fixed training recipe."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from latent_vslam.cuda_devices import CudaDevicePool
from latent_vslam.execution_runtime import release_cuda_training_state, require_lifecycle_cleanup
from latent_vslam.predictor_training import (build_robust_correspondence_store, calibrate_train_only_thresholds,
                          held_out_representation, train_predictor)
from latent_vslam.jepa_fmap import coordinate_masks
from prediction.jepa_runtime import (RestrictedFeatureView, extract_block5_store,
                           extract_true_fmap_store, sequence_geometry)
from latent_vslam.parallel_runtime import correspondence_parallel, extract_parallel
from latent_vslam.protocol import atomic_write_json, canonical_sha256, repo_path, sha256_file
from latent_vslam.inference_runtime import _load_bridge, _unique_identities
from latent_vslam.predictor_training import _hidden_identities
from prediction.predictor_checkpoint import build_training_lineage, load_predictor, save_predictor
from latent_vslam.training_runtime import ResidentPredictorView, resident_correspondence
from prediction.transport import robust_protocol_metadata


def training_lineage(budget, protocol, geometry, config):
    train = budget["split"]["train"]
    anchors = {i.key: i for row in train for i in (row.anchor0, row.anchor1)}
    population = [i.public_dict() for i in sorted(anchors.values(), key=lambda i: i.candidate_index)]
    return build_training_lineage(
        config=config, anchor_stride=budget["anchor_stride"], anchor_population=population,
        fixed_split=protocol["fixed_split"], split_sha256=budget["split_payload"]["split_sha256"],
        schedule_sha256=budget["source_schedule"]["schedule_sha256"],
        coordinate_transform_sha256=geometry["transform_sha256"],
    )


def validate_stride_lineage(stored, expected, stride):
    if not stored.get("scientific_training_contract"):
        raise RuntimeError("predictor scientific training lineage required")
    if stored.get("anchor_stride") != stride:
        raise RuntimeError("predictor anchor_stride mismatch")
    if dict(stored) != dict(expected):
        raise RuntimeError("predictor training lineage mismatch")
    body = {k: v for k, v in stored.items() if k != "training_lineage_sha256"}
    if canonical_sha256(body) != stored.get("training_lineage_sha256"):
        raise RuntimeError("predictor training lineage integrity mismatch")


def load_stride_predictor(path, config, expected, stride):
    checkpoint = load_predictor(path, config, expected)
    validate_stride_lineage(checkpoint["training_lineage"], expected, stride)
    return checkpoint


def group_horizon(rows, stride):
    groups = {}
    metrics = ("predicted_jepa_cosine", "transport_baseline_cosine", "bridge_fmap_cosine",
               "transport_bridge_fmap_cosine", "distance_from_previous_anchor_seconds",
               "distance_from_closing_anchor_seconds", "alpha", "delta_t_seconds")
    for row in rows:
        groups.setdefault(row["relative_hidden_index"], []).append(row)
    return [{"anchor_stride": stride, "relative_hidden_index": ordinal,
             "hidden_query_count": len(values), "split": "test", "role": "offline_diagnostics_only",
             **{key: sum(row[key] for row in values) / len(values) for key in metrics}}
            for ordinal, values in sorted(groups.items())]


def train_stride_predictor(records, budget, protocol, config, output, temporary):
    """No predictor resume and no Bridge training; return only CPU model state/metadata.

    train_predictor preserves seed, AMP, 30 epochs, two intervals per batch,
    AdamW, losses, and best-validation selection for every stride. The feasibility
    feasibility-only tiny-overfit gate is not repeated in this sensitivity study.
    """
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]))
    transform, geometry = sequence_geometry(records[0], calibration, config)
    lineage = training_lineage(budget, protocol, geometry, config)
    development = (*budget["split"]["train"], *budget["split"]["validation"])
    stores = []
    resident = None
    robust_resident = None
    predictor = bridge = None
    try:
        bridge, bridge_meta = _load_bridge(config, transform)
        mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
        dev, dev_meta = extract_parallel(
            records, _unique_identities(development), calibration, config,
            temporary / "development", transform, )
        stores.append(dev)
        thresholds = calibrate_train_only_thresholds(
            budget["split"]["train"], dev, transform, mask)
        robust, robust_meta = correspondence_parallel(
            development, dev, transform, mask, thresholds,
            temporary / "correspondence", )
        resident = ResidentPredictorView(dev, development, device=torch.device(CudaDevicePool.discover().primary_device))
        robust_resident = resident_correspondence(robust, torch.device(CudaDevicePool.discover().primary_device))
        predictor, summary = train_predictor(
            resident, budget["split"], transform, mask, config, robust_resident)
        resident.close()
        resident = None
        robust_resident = None
        lineage["train_only_calibration_sha256"] = thresholds["calibration_sha256"]
        lineage["training_lineage_sha256"] = canonical_sha256(lineage)
        checkpoint_path = output / "predictor.pt"
        save_predictor(checkpoint_path, predictor, config, thresholds, lineage,
                       best_epoch=summary["best_epoch"])
        checkpoint = load_stride_predictor(checkpoint_path, config, lineage, budget["anchor_stride"])
        dev.close()
        # Test features and true FMaps are created only after checkpoint freeze.
        test = budget["split"]["test"]
        test_store, test_meta = extract_block5_store(
            records, _unique_identities(test), calibration, config,
            temporary / "test", transform)
        stores.append(test_store)
        teacher_store, teacher_meta = extract_true_fmap_store(
            records, _hidden_identities(test), calibration, config,
            temporary / "test_teacher", transform)
        stores.append(teacher_store)
        teacher = RestrictedFeatureView(teacher_store, {i.key for i in _hidden_identities(test)},
                                        "stride_sweep_held_out_true_fmap")
        test_robust, test_robust_meta = build_robust_correspondence_store(
            test, test_store, transform, mask, thresholds)
        horizon = []
        held_out = held_out_representation(test, test_store, transform, mask, test_robust,
                                           predictor, bridge, teacher, horizon_rows=horizon)
        for row in horizon:
            row.update(anchor_stride=budget["anchor_stride"], split="test", role="offline_diagnostics_only")
        expected_queries = [q.identity.key for interval in test for q in interval.hidden]
        if [row["identity"] for row in horizon] != expected_queries:
            raise RuntimeError("incomplete held-out horizon diagnostics")
        record = {"anchor_stride": budget["anchor_stride"], "lineage": lineage,
                  "split": budget["split_payload"], "summary": summary,
                  "held_out_representation": held_out,
                  "horizon_resolved_quality": group_horizon(horizon, budget["anchor_stride"]),
                  "development_extraction": dev_meta, "development_correspondence": robust_meta,
                  "test_extraction": test_meta, "test_correspondence": test_robust_meta,
                  "test_teacher_extraction": teacher_meta, "test_teacher_usage": teacher.usage_payload(),
                  "test_was_read_during_training_or_selection": False,
                  "bridge": bridge_meta, "scientific_training_contract": lineage["scientific_training_contract"],
                  "training_and_diagnostics_wall_seconds": time.perf_counter()-started}
        bridge_state = {name: value.detach().cpu().clone() for name, value in bridge.state_dict().items()}
        atomic_write_json(output / "horizon_queries.json", horizon)
    finally:
        if resident is not None:
            resident.close()
        robust_resident = None
        for store in stores:
            if not store.closed:
                store.close()
        if predictor is not None:
            predictor.cpu()
        if bridge is not None:
            bridge.cpu()
    # This function's stack (including mask/correspondence/teacher) disappears
    # before the caller checks CUDA cleanup and starts any trajectory worker.
    record["stores_closed_before_deployment"] = all(s.closed for s in stores)
    atomic_write_json(output / "training.json", record)
    return {"checkpoint": checkpoint, "checkpoint_path": checkpoint_path,
            "bridge_state": bridge_state, "transform": transform, "record": record}


def release_training_before_trajectory():
    cleanup = release_cuda_training_state()
    require_lifecycle_cleanup(cleanup)
    return cleanup
