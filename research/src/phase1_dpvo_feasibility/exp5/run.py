"""Single public entry point for Exp5-0 JEPA–DPVO interface validation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .dataset import configured_sequences, load_sequence_records, validate_real_images
from .dpvo_runner import run_mock
from .evaluation import aggregate_metrics, evaluate_sequence, write_outputs
from .jepa_runner import FRAME_PAYLOAD_KEYS


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="validate real MH_01 metadata/images and the pipeline with CPU-only mock models",
    )
    return parser


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Exp5 config must be a mapping: {resolved}")
    config = copy.deepcopy(payload)
    config["repo_root"] = str(REPO_ROOT)
    config["dataset"]["root"] = str(_repo_path(config["dataset"]["root"]))
    for key in ("calibration", "dpvo_checkpoint", "dpvo_config"):
        config["paths"][key] = str(_repo_path(config["paths"][key]))
    config["experiment"]["result_root"] = str(
        _repo_path(config["experiment"]["result_root"])
    )
    config["runtime"]["dpvo_python"] = str(Path(config["runtime"]["dpvo_python"]).resolve())
    config["runtime"]["jepa_python"] = str(Path(config["runtime"]["jepa_python"]).resolve())
    config["jepa"]["repo"] = str(Path(config["jepa"]["repo"]).resolve())
    return config


def _formal_preflight(config: dict[str, Any]) -> None:
    expected_python = Path(config["runtime"]["dpvo_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"formal Exp5-0 requires DPVO Python: {expected_python}")
    for path in (
        Path(config["paths"]["calibration"]),
        Path(config["paths"]["dpvo_checkpoint"]),
        Path(config["paths"]["dpvo_config"]),
        Path(config["runtime"]["jepa_python"]),
        Path(config["jepa"]["repo"]) / str(config["jepa"]["checkpoint"]),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    dpvo_status = subprocess.check_output(
        ["git", "status", "--short", "--", "dpvo"], cwd=REPO_ROOT, text=True
    )
    if dpvo_status.strip():
        raise RuntimeError("formal Exp5-0 requires an unmodified upstream dpvo/ tree")
    import torch
    import cuda_ba  # noqa: F401
    import dpvo.altcorr  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("formal Exp5-0 requires CUDA")


def _invoke_dpvo_worker(
    *, config: dict[str, Any], records: list[dict[str, Any]], mode: str, work_dir: Path,
) -> dict[str, Any]:
    request_path = work_dir / f"{mode}_request.json"
    result_path = work_dir / f"{mode}_result.json"
    request_path.write_text(
        json.dumps({"config": config, "records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        str(config["runtime"]["dpvo_python"]),
        "-c",
        (
            "from research.src.phase1_dpvo_feasibility.exp5.dpvo_runner "
            "import worker_main; raise SystemExit(worker_main())"
        ),
        "--request-json",
        str(request_path),
        "--result-json",
        str(result_path),
        "--mode",
        mode,
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if not result_path.is_file():
        raise RuntimeError(
            f"{mode} DPVO worker produced no result: {completed.stderr[-2000:]}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(f"{mode} DPVO worker failed: {result.get('error', completed.stderr[-2000:])}")
    return result


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    sequence = str(config["dataset"]["smoke_sequence"])
    if sequence != configured_sequences(config)[0]:
        raise ValueError("smoke sequence must be the fixed train sequence")
    records = load_sequence_records(
        config, sequence, limit=int(config["dataset"]["smoke_frames"])
    )
    image_shapes = validate_real_images(records)
    record_dicts = [record.to_dict() for record in records]
    baseline, sidecar, events, jepa_payloads = run_mock(record_dicts)
    expected_events = [
        event
        for record in records
        for event in (
            f"dpvo:{record.frame_id}",
            f"hook_before:{record.frame_id}",
            f"jepa:{record.frame_id}",
            f"hook_after:{record.frame_id}",
        )
    ]
    if events != expected_events:
        raise AssertionError("mock sidecar did not preserve DPVO -> hook -> JEPA ordering")
    if len(jepa_payloads) != len(records) or any(
        set(payload) != FRAME_PAYLOAD_KEYS for payload in jepa_payloads
    ):
        raise AssertionError("mock V-JEPA payload exposed fields outside the dispatch contract")
    sequence_metric = evaluate_sequence(
        expected=records,
        baseline=baseline,
        sidecar=sidecar,
        config=config,
        smoke=True,
    )
    metrics = aggregate_metrics([sequence_metric], config=config, smoke=True)
    with tempfile.TemporaryDirectory(prefix="exp5-smoke-") as temporary:
        output = Path(temporary) / "interface_validation"
        write_outputs(output, metrics)
        if not (output / "metrics.json").is_file() or not (output / "REPORT.md").is_file():
            raise AssertionError("smoke report generation failed")
        persisted = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if persisted.get("status") != "smoke_pass":
            raise AssertionError("smoke metrics did not pass")
    return {
        "status": metrics["status"],
        "sequence": sequence,
        "num_frames": len(records),
        "frame_ids": [record.frame_id for record in records],
        "timestamp_error_max": sequence_metric["timestamp_error"]["max_ns"],
        "existing_pose_max_error": sequence_metric["trajectory_validation"]["existing_pose_max_error"],
        "trajectory_validation": sequence_metric["trajectory_validation"]["passed"],
        "latent_sanity": sequence_metric["latent_sanity"]["passed"],
        "decoded_image_shapes": sorted({tuple(shape) for shape in image_shapes}),
        "models": "mock",
        "artifacts_retained": False,
    }


def run_formal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["experiment"]["result_root"])
    if target.exists():
        raise FileExistsError(f"refusing to overwrite formal Exp5-0 output: {target}")
    _formal_preflight(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    sequence_metrics: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".exp5-", dir=target.parent) as temporary:
        stage = Path(temporary)
        for sequence in configured_sequences(config):
            records = load_sequence_records(config, sequence)
            record_dicts = [record.to_dict() for record in records]
            sequence_dir = stage / sequence
            sequence_dir.mkdir(parents=True)
            baseline = _invoke_dpvo_worker(
                config=config, records=record_dicts, mode="baseline", work_dir=sequence_dir,
            )
            sidecar = _invoke_dpvo_worker(
                config=config, records=record_dicts, mode="sidecar", work_dir=sequence_dir,
            )
            sequence_metrics.append(evaluate_sequence(
                expected=records,
                baseline=baseline,
                sidecar=sidecar,
                config=config,
                smoke=False,
            ))
        metrics = aggregate_metrics(sequence_metrics, config=config, smoke=False)
        publish = stage / "publish"
        write_outputs(publish, metrics)
        os.replace(publish, target)
    failed = [name for name, row in metrics["sequences"].items() if row["status"] != "pass"]
    failed_gates = {
        name: [
            gate
            for gate, passed in (
                ("alignment", row["alignment"]["passed"]),
                ("trajectory_validation", row["trajectory_validation"]["passed"]),
                ("latent_sanity", row["latent_sanity"]["passed"]),
            )
            if not passed
        ]
        for name, row in metrics["sequences"].items()
        if row["status"] != "pass"
    }
    return {
        "status": metrics["status"],
        "output_dir": str(target),
        "sequences": list(metrics["sequences"]),
        "failed_sequences": failed,
        "failed_gates": failed_gates,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    result = run_smoke(config) if args.smoke else run_formal(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
