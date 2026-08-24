"""Final Experiment 4 JEPA-DPVO Bridge Validation entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .adapter.evaluate import evaluate_checkpoint
from .adapter.train import train_model
from .analysis.baseline import evaluate_baseline_ladder
from .analysis.retrieval import analyze_temporal_retrieval
from .dataset import AdapterDataset, prepare_dataset
from .extraction import extract_fmap
from .report import write_bridge_report
from .schema import DEFAULT_CONFIG_PATH, REPO_ROOT, atomic_write_json, load_config, repo_path, select_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the complete pipeline on at most 8 frames per sequence and retain no artifacts",
    )
    return parser


def _result_root(config: dict[str, Any]) -> Path:
    return repo_path(config["experiment"]["result_root"]).resolve()


def _require_dpvo_environment(config: dict[str, Any]) -> None:
    expected = Path(config["runtime"]["dpvo_python"]).resolve()
    observed = Path(sys.executable).resolve()
    if observed != expected:
        raise RuntimeError(f"Exp4 must run in the DPVO environment: expected {expected}, got {observed}")


def _run_jepa_subprocess(config_path: Path, dataset_root: Path, config: dict[str, Any]) -> None:
    code = (
        "from research.src.phase1_dpvo_feasibility.exp4.extraction import jepa_worker; "
        "import sys; jepa_worker(sys.argv[1], sys.argv[2])"
    )
    command = [
        str(Path(config["runtime"]["jepa_python"]).resolve()),
        "-c",
        code,
        str(config_path),
        str(dataset_root),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def extract_features(
    *, config: dict[str, Any], config_path: Path, dataset_root: Path,
) -> dict[str, Any]:
    """Run FMap/Oracle first, then V-JEPA in its fixed environment."""
    samples = select_samples(dataset_root, "all")
    fmap = extract_fmap(
        config_path=config_path,
        samples=samples,
        overwrite=False,
        sanity_only=False,
        dataset_root=dataset_root,
    )
    _run_jepa_subprocess(config_path, dataset_root, config)
    jepa_path = dataset_root / "features/jepa/extraction_manifest.json"
    if not jepa_path.is_file():
        raise RuntimeError("V-JEPA worker completed without an extraction manifest")
    jepa = json.loads(jepa_path.read_text(encoding="utf-8"))
    return {"fmap": fmap, "jepa": jepa}


def _validate_dataset(dataset_root: Path) -> None:
    """Fail before training when any selected feature file is missing."""
    for split in ("train", "val", "test"):
        AdapterDataset(dataset_root / f"{split}.jsonl", dataset_root=dataset_root)


def _compact_extraction(manifests: dict[str, Any]) -> dict[str, Any]:
    fmap, jepa = manifests["fmap"], manifests["jepa"]
    return {
        "fmap": {
            "status": fmap["status"],
            "counts": fmap["counts"],
            "dpvo": fmap["dpvo"],
            "shape_contract": fmap["shape_contract"],
            "sanity": fmap["sanity"],
        },
        "jepa": {
            "status": jepa["status"],
            "counts": jepa["counts"],
            "vjepa": jepa["vjepa"],
        },
    }


def _published_training(summary: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    result = dict(summary)
    result["checkpoint"] = checkpoint
    return result


def run_bridge(*, smoke: bool, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config, resolved_config, config_sha256 = load_config(config_path)
    _require_dpvo_environment(config)
    result_root = _result_root(config)
    target = result_root / "final"
    if not smoke and target.exists():
        raise FileExistsError(f"refusing to overwrite formal Bridge result: {target}")
    result_root.mkdir(parents=True, exist_ok=True)
    epochs = int(config["training"]["smoke_epochs"] if smoke else config["training"]["bridge_epochs"])
    batch_size = int(
        config["training"]["smoke_batch_size"] if smoke else config["training"]["batch_size"]
    )

    with tempfile.TemporaryDirectory(prefix=".bridge-", dir=result_root) as temporary:
        stage_root = Path(temporary)
        dataset_root = stage_root / "dataset"
        publish = stage_root / "publish"
        for directory in (
            publish / "adapter", publish / "lowrank", publish / "baseline",
            publish / "retrieval", publish / "figures",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        manifest = prepare_dataset(dataset_root, config_path=resolved_config, smoke=smoke)
        extraction = extract_features(
            config=config, config_path=resolved_config, dataset_root=dataset_root,
        )
        _validate_dataset(dataset_root)

        adapter_training = train_model(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            output_dir=stage_root / "training/adapter",
            model_kind="adapter",
            capacity="small",
            epochs=epochs,
            batch_size=batch_size,
        )
        lowrank_training = train_model(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            output_dir=stage_root / "training/lowrank",
            model_kind="low_rank_linear",
            capacity="small",
            epochs=epochs,
            batch_size=batch_size,
        )
        adapter_checkpoint = Path(adapter_training["checkpoint"])
        lowrank_checkpoint = Path(lowrank_training["checkpoint"])
        shutil.copy2(adapter_checkpoint, publish / "adapter/best.pt")
        shutil.copy2(lowrank_checkpoint, publish / "lowrank/best.pt")

        baseline_metrics = evaluate_baseline_ladder(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            lowrank_checkpoint=lowrank_checkpoint,
            output_path=publish / "baseline/metrics.json",
            batch_size=batch_size,
        )
        lowrank_metrics = evaluate_checkpoint(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            checkpoint_path=lowrank_checkpoint,
            output_path=publish / "lowrank/metrics.json",
            figures_dir=publish / "figures",
            batch_size=batch_size,
        )
        adapter_metrics = evaluate_checkpoint(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            checkpoint_path=adapter_checkpoint,
            output_path=publish / "adapter/metrics.json",
            figures_dir=publish / "figures",
            batch_size=batch_size,
        )
        retrieval_metrics = analyze_temporal_retrieval(
            config=config,
            config_sha256=config_sha256,
            dataset_root=dataset_root,
            output_path=publish / "retrieval/metrics.json",
        )

        adapter_metrics["mode_metadata"]["checkpoint"] = "adapter/best.pt"
        lowrank_metrics["mode_metadata"]["checkpoint"] = "lowrank/best.pt"
        baseline_metrics["baselines"]["low_rank_linear"]["checkpoint"] = "lowrank/best.pt"
        atomic_write_json(publish / "adapter/metrics.json", adapter_metrics)
        atomic_write_json(publish / "lowrank/metrics.json", lowrank_metrics)
        atomic_write_json(publish / "baseline/metrics.json", baseline_metrics)
        aggregate = write_bridge_report(
            publish,
            protocol=manifest["protocol"],
            adapter_training=_published_training(adapter_training, "adapter/best.pt"),
            lowrank_training=_published_training(lowrank_training, "lowrank/best.pt"),
            adapter_metrics=adapter_metrics,
            lowrank_metrics=lowrank_metrics,
            baseline_metrics=baseline_metrics,
            retrieval_metrics=retrieval_metrics,
            extraction=_compact_extraction(extraction),
            config_sha256=config_sha256,
            eps=float(config["evaluation"]["eps"]),
        )
        if smoke:
            return {
                "status": "smoke_complete",
                "artifacts_retained": False,
                "sample_count": manifest["total_indexed_frames"],
                "adapter_mean_cosine_similarity": adapter_metrics["mean_cosine_similarity"],
                "lowrank_mean_cosine_similarity": lowrank_metrics["mean_cosine_similarity"],
                "interpretation": aggregate["interpretation"],
            }
        os.replace(publish, target)
        return {
            "status": "complete",
            "output_dir": str(target),
            "sample_count": manifest["total_indexed_frames"],
            "interpretation": aggregate["interpretation"],
        }


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run_bridge(smoke=bool(args.smoke)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
