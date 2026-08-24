"""Final Experiment 4 adapter capacity-scaling entry point."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .adapter.train import train_model
from .dataset import prepare_dataset
from .report import write_capacity_report
from .run import _require_dpvo_environment, _result_root, extract_features
from .schema import DEFAULT_CONFIG_PATH, atomic_write_json, load_config


CAPACITY_ORDER = ("small", "medium", "large")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run all three capacities for one epoch on smoke data and retain no artifacts",
    )
    return parser


def _published_capacity_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "model_kind", "capacity", "hidden_dim", "parameter_count",
        "model_config", "model_config_sha256", "epochs", "best_epoch",
        "best_validation_metric", "best_metric_name", "total_training_time",
        "dataset_fingerprint",
    )
    return {key: summary[key] for key in keys}


def run_capacity_benchmark(
    *, smoke: bool, config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config, resolved_config, config_sha256 = load_config(config_path)
    _require_dpvo_environment(config)
    result_root = _result_root(config)
    target = result_root / "capacity_scaling"
    if not smoke and target.exists():
        raise FileExistsError(f"refusing to overwrite formal capacity result: {target}")
    result_root.mkdir(parents=True, exist_ok=True)
    epochs = int(config["training"]["smoke_epochs"] if smoke else config["training"]["capacity_epochs"])
    batch_size = int(
        config["training"]["smoke_batch_size"] if smoke else config["training"]["batch_size"]
    )

    with tempfile.TemporaryDirectory(prefix=".capacity-", dir=result_root) as temporary:
        stage_root = Path(temporary)
        dataset_root = stage_root / "dataset"
        publish = stage_root / "publish"
        publish.mkdir(parents=True, exist_ok=True)
        manifest = prepare_dataset(dataset_root, config_path=resolved_config, smoke=smoke)
        extract_features(config=config, config_path=resolved_config, dataset_root=dataset_root)

        capacity_metrics: dict[str, dict[str, Any]] = {}
        for capacity in CAPACITY_ORDER:
            summary = train_model(
                config=config,
                config_sha256=config_sha256,
                dataset_root=dataset_root,
                output_dir=stage_root / f"training/{capacity}",
                model_kind="adapter",
                capacity=capacity,
                epochs=epochs,
                batch_size=batch_size,
            )
            metrics = _published_capacity_metrics(summary)
            capacity_metrics[capacity] = metrics
            atomic_write_json(publish / capacity / "metrics.json", metrics)

        report = write_capacity_report(
            publish,
            capacity_metrics=capacity_metrics,
            threshold=float(config["capacity_decision"]["relative_improvement_threshold"]),
            eps=float(config["capacity_decision"]["eps"]),
            config_sha256=config_sha256,
            epochs=epochs,
        )
        if smoke:
            return {
                "status": "smoke_complete",
                "artifacts_retained": False,
                "sample_count": manifest["total_indexed_frames"],
                "epochs": epochs,
                "capacities": {
                    name: {
                        "parameter_count": values["parameter_count"],
                        "best_validation_metric": values["best_validation_metric"],
                    }
                    for name, values in capacity_metrics.items()
                },
                "decision": report["decision"],
            }
        os.replace(publish, target)
        return {
            "status": "complete",
            "output_dir": str(target),
            "sample_count": manifest["total_indexed_frames"],
            "decision": report["decision"],
        }


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run_capacity_benchmark(smoke=bool(args.smoke)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
