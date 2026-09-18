"""Short real-feature B1 GPU smoke; writes only to a temporary directory."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from latent_vslam.b1_training import prepare_predictor_samples
from latent_vslam.bridge_checkpoint import load_bridge_config
from latent_vslam.inference_runtime import _unique_identities, load_config as load_inference_config
from latent_vslam.jepa_fmap import (
    build_bridge, contiguous_split, coordinate_masks, hidden_split_keys,
    masked_reconstruction_loss,
)
from latent_vslam.parallel_runtime import correspondence_parallel, extract_parallel
from latent_vslam.predictor_training import (
    _new_predictor, _predict, _transport_batch, calibrate_train_only_thresholds,
)
from latent_vslam.protocol import canonical_sha256, load_sequence_records, post_bootstrap_ratio_roles, repo_path
from latent_vslam.runtime import materialize_schedule
from latent_vslam.stride_training import training_lineage
from latent_vslam.training_runtime import bridge_tensor_batch
from prediction.jepa_runtime import sequence_geometry
from prediction.predictor import prediction_loss
from prediction.predictor_checkpoint import load_predictor, save_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=["MH_01_easy", "MH_02_easy"])
    args = parser.parse_args()
    sequences = tuple(args.sequences)
    bridge_config, _ = load_bridge_config()
    predictor_config, _ = load_inference_config()
    predictor_config["experiment"]["training_sequences"] = list(sequences)
    calibration = np.loadtxt(repo_path(bridge_config["paths"]["calibration"]))
    records = {sequence: load_sequence_records(bridge_config, sequence) for sequence in sequences}
    bootstraps = {sequence: materialize_schedule(records[sequence], calibration, bridge_config)[
        "bootstrap_end_candidate_index"] for sequence in sequences}
    transform, geometry = sequence_geometry(records[sequences[0]][0], calibration, bridge_config)
    bridge_counts = {}
    selected_bridge = []
    selected_train_keys = []
    for sequence in sequences:
        identities = [row.identity for row in records[sequence]]
        roles = post_bootstrap_ratio_roles(
            identities, bootstrap_end_candidate_index=bootstraps[sequence], anchor_ratio=.2,
        )
        split = hidden_split_keys(identities, roles, contiguous_split(identities))
        bridge_counts[sequence] = {name: len(keys) for name, keys in split.items()}
        chosen = split["train"][:2]
        selected_train_keys.extend(chosen)
        by_key = {row.identity.key: row for row in records[sequence]}
        selected_bridge.extend(by_key[key] for key in chosen)
    predictor_records, budget, preparation = prepare_predictor_samples(
        records, bootstraps, sequences,
    )
    selected_intervals = tuple(next(row for row in budget["split"]["train"]
                                    if row.anchor0.sequence == sequence)
                               for sequence in sequences)
    with tempfile.TemporaryDirectory(prefix="b1_gpu_smoke_") as name:
        temporary = Path(name)
        bridge_store, _ = extract_parallel(
            selected_bridge, set(selected_train_keys), calibration, bridge_config,
            temporary / "bridge", transform, bridge_hidden_keys=set(selected_train_keys),
        )
        try:
            bridge = build_bridge(transform, channels=int(bridge_config["bridge"]["hidden_channels"])).cuda()
            bridge.train()
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4, weight_decay=1e-4)
            rows = np.asarray([bridge_store.index[key] for key in selected_train_keys], dtype=np.int64)
            mask = torch.from_numpy(coordinate_masks(transform)["fmap_valid_mask"]).cuda()
            tokens = bridge_tensor_batch(bridge_store, "block5", rows)
            teacher = bridge_tensor_batch(bridge_store, "teacher", rows)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                bridge_loss = masked_reconstruction_loss(bridge(tokens), teacher, mask)["total"]
            bridge_loss.backward()
            optimizer.step()
            bridge_loss_value = float(bridge_loss.detach())
            if not np.isfinite(bridge_loss_value):
                raise RuntimeError("non-finite Bridge smoke loss")
            del bridge, optimizer, tokens, teacher, mask, bridge_loss
            torch.cuda.empty_cache()
        finally:
            bridge_store.close()
        selected_identities = _unique_identities(selected_intervals)
        predictor_store, _ = extract_parallel(
            predictor_records, selected_identities, calibration, predictor_config,
            temporary / "predictor", transform,
        )
        try:
            mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
            thresholds = calibrate_train_only_thresholds(
                selected_intervals, predictor_store, transform, mask,
            )
            robust, _ = correspondence_parallel(
                selected_intervals, predictor_store, transform, mask, thresholds,
                temporary / "correspondence",
            )
            if set(robust.rows) != {row.global_interval_index for row in selected_intervals}:
                raise RuntimeError("global interval correspondence keys changed")
            model = _new_predictor(predictor_config).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            transported, target, alpha, delta = _transport_batch(
                selected_intervals, predictor_store, transform, mask, robust,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss = prediction_loss(_predict(model, transported, alpha, delta), target, mask)["total"]
            loss.backward()
            optimizer.step()
            predictor_loss_value = float(loss.detach())
            if not np.isfinite(predictor_loss_value):
                raise RuntimeError("non-finite Predictor smoke loss")
            lineage = training_lineage(budget, preparation["protocol"], geometry, predictor_config)
            lineage["train_only_calibration_sha256"] = thresholds["calibration_sha256"]
            lineage["training_lineage_sha256"] = canonical_sha256(lineage)
            checkpoint_path = temporary / "smoke_predictor.pt"
            save_predictor(checkpoint_path, model, predictor_config, thresholds, lineage,
                           best_epoch=0, sample_counts=budget["sample_counts"])
            checkpoint = load_predictor(checkpoint_path, predictor_config, lineage)
            if checkpoint["sample_counts"] != budget["sample_counts"]:
                raise RuntimeError("smoke checkpoint sample counts changed")
        finally:
            predictor_store.close()
    print(json.dumps({
        "sequences": sequences, "bootstrap_end_candidate_index": bootstraps,
        "bridge_hidden_counts": bridge_counts,
        "predictor_sample_counts": budget["sample_counts"],
        "bridge_smoke_loss": bridge_loss_value,
        "predictor_smoke_loss": predictor_loss_value,
        "predictor_checkpoint_metadata_verified": True,
        "held_out_scope": "training_sequences_internal_only",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
