"""Bridge, Predictor, and B1 multi-sequence training entry point."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from latent_vslam.stride_sweep import (
    TRAINING_ROOT, stride_config, load_protocol, prepare_protocol,
)
from latent_vslam.stride_training import (
    train_stride_predictor, release_training_before_trajectory,
)
from latent_vslam.artifact_runtime import publish_checkpoint_tree, staged_directory
from latent_vslam.canonical import load_yaml, resolve_sequences
from latent_vslam.execution_runtime import initialize_formal_main_process
from latent_vslam.bridge_training_runtime import run as run_bridge_training
from latent_vslam.b1_training import run_predictor as run_b1_predictor
from latent_vslam.protocol import (
    REPO_ROOT, atomic_write_json, load_sequence_records, repo_path, sha256_file,
)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def train_stride_sweep(config_path: str | Path, strides_override=None) -> dict[str, Any]:
    protocol, canonical, resolved = load_protocol(config_path)
    sequence = protocol["sequence"]
    strides = tuple(protocol["anchor_strides"] if strides_override is None else strides_override)
    if (not strides or len(strides) != len(set(strides))
            or any(isinstance(stride, bool) or not isinstance(stride, int) or stride < 2
                   for stride in strides)):
        raise ValueError("stride sweep training requires distinct integer strides >= 2")
    _require_file(repo_path(canonical["paths"]["dpvo_checkpoint"]), "DPVO checkpoint")
    _require_file(repo_path(canonical["paths"]["bridge"]), "Bridge checkpoint")
    _require_file(Path(canonical["jepa"]["checkpoint"]), "V-JEPA checkpoint")
    records = load_sequence_records(canonical, sequence)
    budgets, preparation = prepare_protocol(records, protocol, strides)
    initialize_formal_main_process(require_online=True)
    with staged_directory(TRAINING_ROOT) as staged, \
         tempfile.TemporaryDirectory(prefix="stride_sweep_train_", dir="/tmp") as name:
        work = Path(name)
        manifest: dict[str, Any] = {
            "schema_version": 1, "kind": "stride_predictor_training",
            "sequence": sequence, "anchor_strides": list(strides),
            "config_sha256": sha256_file(resolved),
            "bridge_sha256": sha256_file(repo_path(canonical["paths"]["bridge"])),
            "preparation": preparation, "strides": {},
        }
        for stride in strides:
            config = stride_config(canonical, protocol, stride)
            trained = train_stride_predictor(
                records, budgets[stride], protocol, config,
                staged / f"stride_{stride}", work / f"stride_{stride}",
            )
            cleanup = release_training_before_trajectory()
            manifest["strides"][str(stride)] = {
                "checkpoint_sha256": sha256_file(trained["checkpoint_path"]),
                "training_sha256": sha256_file(staged / f"stride_{stride}" / "training.json"),
                "horizon_sha256": sha256_file(staged / f"stride_{stride}" / "horizon_queries.json"),
                "training_lineage_sha256": trained["record"]["lineage"]["training_lineage_sha256"],
                "cleanup": cleanup,
            }
            del trained
        atomic_write_json(staged / "manifest.json", manifest)
        publish_checkpoint_tree(staged, TRAINING_ROOT)
    return {
        "status": "complete", "training": "predictor_stride_sweep",
        "sequence": sequence, "anchor_strides": list(strides),
        "training_root": str(TRAINING_ROOT.relative_to(REPO_ROOT)),
    }


def run(config_path: str | Path, sequences_override=None) -> dict[str, Any]:
    config, resolved = load_yaml(config_path)
    if "train" in config:
        settings = config["train"]
        configured = settings.get("sequences")
        if not isinstance(configured, list) or not all(isinstance(value, str) for value in configured):
            raise ValueError("train.sequences must be a list of sequence names")
        sequences = resolve_sequences(sequences_override, configured)
        base = repo_path(settings["base_config"])
        if settings["kind"] == "bridge":
            return run_bridge_training(sequences, config_path=base, b1=True)
        if settings["kind"] == "predictor":
            return run_b1_predictor(sequences, base_config_path=base, wrapper_path=resolved)
        raise ValueError(f"unknown B1 training kind: {settings['kind']}")
    if sequences_override is not None:
        raise ValueError("--sequences requires a B1 config with train.sequences")
    if config.get("experiment", {}).get("name") == "bridge_training":
        _require_file(repo_path(config["paths"]["dpvo_checkpoint"]), "DPVO checkpoint")
        _require_file(Path(config["jepa"]["checkpoint"]), "V-JEPA checkpoint")
        return run_bridge_training(
            (config["experiment"]["training_sequence"],),
            config_path=resolved,
        )
    if "anchor_strides" in config:
        return train_stride_sweep(resolved)
    raise ValueError("train config must define Bridge or stride Predictor training")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--sequences", nargs="+", help="override B1 train.sequences")
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(run(args.config, sequences_override=args.sequences), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
