"""CPU-only contracts for the consolidated Exp4 workflows."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from .. import capacity_benchmark as capacity_module
from .. import run as run_module
from ..capacity_benchmark import CAPACITY_ORDER
from ..dataset import AdapterDataset, prepare_dataset
from ..extraction import compare_fmaps, validate_fmap_shape
from ..report import (
    bridge_interpretation,
    capacity_decision,
    write_bridge_report,
    write_capacity_report,
)
from ..schema import DEFAULT_CONFIG_PATH, PACKAGE_DIR, Exp4Sample, load_config, read_jsonl


def _metric(cosine: float, *, mse: float = 0.1, norm_ratio: float = 1.0) -> dict:
    return {
        "status": "complete",
        "sample_count": 2,
        "spatial_vector_count": 8,
        "mean_cosine_similarity": cosine,
        "median_cosine_similarity": cosine,
        "mse": mse,
        "pred_norm_mean": norm_ratio,
        "teacher_norm_mean": 1.0,
        "norm_ratio": norm_ratio,
        "mode_metadata": {"checkpoint": "temporary", "model_kind": "adapter"},
    }


def _baseline() -> dict:
    return {
        "status": "complete",
        "baselines": {
            "random": _metric(0.0, mse=1.0),
            "mean_fmap": _metric(0.1, mse=0.04),
            "low_rank_linear": _metric(0.2, mse=0.08),
        },
    }


def _retrieval() -> dict:
    modality = {
        "sample_count": 2,
        "self_similarity_mean": 1.0,
        "excluding_self_top1_within_1": 0.8,
        "excluding_self_top1_within_5": 0.9,
        "excluding_self_top5_within_1": 0.95,
        "excluding_self_top5_within_5": 1.0,
    }
    return {"status": "complete", "modalities": {"jepa": modality, "fmap": modality}}


class FinalExp4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _synthetic_config(self) -> tuple[dict, Path]:
        config, _, _ = load_config(DEFAULT_CONFIG_PATH)
        config = copy.deepcopy(config)
        source = self.root / "euroc"
        config["dataset"]["root"] = str(source)
        config["dataset"]["groups"] = ["machine_hall"]
        config["jepa"]["expected_shape"] = [4, 3]
        config["dpvo_fmap"]["reference_shape"] = [2, 2, 2]
        for sequence in ("MH_01_easy", "MH_03_medium", "MH_05_difficult"):
            camera = source / "machine_hall" / sequence / "mav0/cam0"
            image_dir = camera / "data"
            image_dir.mkdir(parents=True)
            rows = ["#timestamp [ns],filename"]
            for index in range(20):
                timestamp = 1_000_000 + index
                filename = f"{timestamp}.png"
                (image_dir / filename).touch()
                rows.append(f"{timestamp},{filename}")
            (camera / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return config, path

    def test_prepare_schema_smoke_stride_and_loading_identity(self) -> None:
        _, config_path = self._synthetic_config()
        dataset_root = self.root / "dataset"
        manifest = prepare_dataset(dataset_root, config_path=config_path, smoke=True)
        self.assertEqual(manifest["total_indexed_frames"], 24)
        self.assertEqual(manifest["indices"]["train"]["frames"], 8)
        train = read_jsonl(dataset_root / "train.jsonl")
        self.assertEqual([sample.frame_id for sample in train], list(range(0, 16, 2)))
        sample = train[0]
        self.assertEqual(sample.identity[:3], ("euroc", "machine_hall", "MH_01_easy"))

        identity = {
            "dataset_type": sample.dataset_type,
            "dataset_group": sample.dataset_group,
            "sequence_category": sample.sequence_category,
            "sequence": sample.sequence,
            "frame_id": sample.frame_id,
            "timestamp_ns": sample.timestamp_ns,
            "image_path": sample.image_path,
            "split": sample.split,
        }
        jepa_path = sample.feature_path(dataset_root, "jepa")
        fmap_path = sample.feature_path(dataset_root, "dpvo_fmap")
        jepa_path.parent.mkdir(parents=True)
        fmap_path.parent.mkdir(parents=True)
        torch.save({
            "tokens": torch.ones(sample.jepa_shape, dtype=torch.float16),
            "metadata": {
                **identity, "feature_kind": "jepa_tokens",
                "shape": list(sample.jepa_shape), "config_sha256": manifest["config_sha256"],
            },
        }, jepa_path)
        torch.save({
            "fmap": torch.ones(sample.fmap_shape, dtype=torch.float16),
            "metadata": {
                **identity, "feature_kind": "dpvo_fmap",
                "shape": list(sample.fmap_shape), "config_sha256": manifest["config_sha256"],
            },
        }, fmap_path)
        one_index = dataset_root / "one.jsonl"
        one_index.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
        item = AdapterDataset(one_index, dataset_root=dataset_root)[0]
        self.assertEqual(item["jepa_tokens"].dtype, torch.float32)
        self.assertEqual(item["fmap_teacher"].dtype, torch.float32)

    def test_schema_rejects_invalid_category(self) -> None:
        with self.assertRaises(ValueError):
            Exp4Sample(
                dataset_type="euroc", dataset_group="machine_hall",
                sequence_category="unknown", sequence="MH_01_easy",
                frame_id=0, timestamp_ns=1, image_path="/tmp/1.png", split="train",
                jepa_feature_path="features/jepa/euroc/machine_hall/MH_01_easy/1.pt",
                dpvo_feature_path="features/dpvo_fmap/euroc/machine_hall/MH_01_easy/1.pt",
                jepa_shape=(4, 3), fmap_shape=(2, 2, 2),
            )

    def test_fmap_shape_and_oracle_metric_helpers(self) -> None:
        config = {
            "fmap_channels": 2, "spatial_scale": 2,
            "reference_shape": [2, 2, 3], "shape_policy": "strict",
        }
        self.assertEqual(validate_fmap_shape((2, 2, 3), (4, 6), config), [])
        with self.assertRaises(ValueError):
            validate_fmap_shape((2, 3, 3), (4, 6), config)
        metrics = compare_fmaps(torch.ones(1, 1, 2), torch.ones(1, 1, 2))
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)

    def test_bridge_interpretation_ladder_and_gaps(self) -> None:
        result = bridge_interpretation(_baseline(), _metric(0.3))
        self.assertTrue(result["bridge_evidence"])
        self.assertEqual(result["mapping_interpretation"], "linear_signal_with_nonlinear_gain")
        self.assertAlmostEqual(result["lowrank_adapter_absolute_gap"], 0.1)
        self.assertAlmostEqual(result["lowrank_adapter_relative_gap"], 1.0 / 3.0)
        weak = _baseline()
        weak["baselines"]["low_rank_linear"]["mean_cosine_similarity"] = 0.05
        self.assertEqual(
            bridge_interpretation(weak, _metric(0.3))["mapping_interpretation"],
            "direct_linear_relation_is_weak",
        )

    def test_capacity_one_percent_decision(self) -> None:
        values = {
            "small": {"best_validation_metric": 1.0},
            "medium": {"best_validation_metric": 0.995},
            "large": {"best_validation_metric": 0.991},
        }
        self.assertFalse(capacity_decision(values, threshold=0.01, eps=1e-8)["capacity_effective"])
        values["large"]["best_validation_metric"] = 0.99
        self.assertTrue(capacity_decision(values, threshold=0.01, eps=1e-8)["capacity_effective"])

    def test_report_rendering(self) -> None:
        protocol = {
            "splits": {"train": ["MH_01_easy"], "val": ["MH_03_medium"], "test": ["MH_05_difficult"]},
            "camera": "cam0", "stride": 2,
        }
        training = {"epochs": 50}
        bridge = write_bridge_report(
            self.root / "bridge",
            protocol=protocol,
            adapter_training=training,
            lowrank_training=training,
            adapter_metrics=_metric(0.3),
            lowrank_metrics=_metric(0.2),
            baseline_metrics=_baseline(),
            retrieval_metrics=_retrieval(),
            extraction={"status": "complete"},
            config_sha256="config",
            eps=1e-8,
        )
        self.assertTrue((self.root / "bridge/REPORT.md").is_file())
        self.assertIn("lowrank_adapter_relative_gap", bridge["interpretation"])

        capacities = {}
        for index, name in enumerate(CAPACITY_ORDER):
            capacities[name] = {
                "hidden_dim": (256, 512, 768)[index],
                "parameter_count": index + 1,
                "best_epoch": 10,
                "best_validation_metric": (1.0, 0.999, 0.995)[index],
                "total_training_time": 1.0,
                "model_config_sha256": name,
            }
        summary = write_capacity_report(
            self.root / "capacity",
            capacity_metrics=capacities,
            threshold=0.01,
            eps=1e-8,
            config_sha256="config",
            epochs=100,
        )
        self.assertFalse(summary["decision"]["capacity_effective"])
        self.assertTrue((self.root / "capacity/REPORT.md").is_file())

    def test_only_two_public_parsers(self) -> None:
        self.assertFalse(run_module.build_parser().parse_args([]).smoke)
        self.assertTrue(run_module.build_parser().parse_args(["--smoke"]).smoke)
        self.assertFalse(capacity_module.build_parser().parse_args([]).smoke)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_module.build_parser().parse_args(["--profile", "alignment"])
        executable = []
        for path in PACKAGE_DIR.rglob("*.py"):
            if "tests" not in path.parts and 'if __name__ == "__main__"' in path.read_text(encoding="utf-8"):
                executable.append(path.relative_to(PACKAGE_DIR).as_posix())
        self.assertEqual(sorted(executable), ["capacity_benchmark.py", "run.py"])

    def _workflow_config(self) -> tuple[dict, Path, str]:
        config, resolved, config_hash = load_config(DEFAULT_CONFIG_PATH)
        config = copy.deepcopy(config)
        config["experiment"]["result_root"] = str(self.root / "results")
        return config, resolved, config_hash

    def test_bridge_smoke_order_and_no_published_artifacts(self) -> None:
        config, resolved, config_hash = self._workflow_config()
        calls: list[str] = []

        def prepare(root: Path, **_: object) -> dict:
            calls.append("prepare")
            root.mkdir(parents=True)
            return {
                "total_indexed_frames": 24,
                "protocol": {
                    "splits": config["dataset"]["splits"], "camera": "cam0", "stride": 2,
                },
            }

        extraction = {
            "fmap": {
                "status": "complete", "counts": {"total": 24}, "dpvo": {},
                "shape_contract": {}, "sanity": {"passed": True},
            },
            "jepa": {"status": "complete", "counts": {"total": 24}, "vjepa": {}},
        }

        def train(**kwargs: object) -> dict:
            kind = str(kwargs["model_kind"])
            calls.append(f"train:{kind}")
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True)
            checkpoint = output / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            return {
                "status": "complete", "model_kind": kind,
                "capacity": "small" if kind == "adapter" else None,
                "hidden_dim": 256 if kind == "adapter" else None,
                "rank": 16 if kind != "adapter" else None,
                "parameter_count": 1, "model_config": {}, "model_config_sha256": kind,
                "epochs": 1, "best_epoch": 1, "best_validation_metric": 0.5,
                "best_metric_name": "val_total_loss", "total_training_time": 1.0,
                "dataset_fingerprint": {}, "checkpoint": str(checkpoint),
            }

        def baseline(**kwargs: object) -> dict:
            calls.append("baseline")
            return _baseline()

        def evaluate(**kwargs: object) -> dict:
            checkpoint = str(kwargs["checkpoint_path"])
            kind = "lowrank" if "lowrank" in checkpoint else "adapter"
            calls.append(f"evaluate:{kind}")
            return _metric(0.2 if kind == "lowrank" else 0.3)

        def retrieval(**_: object) -> dict:
            calls.append("retrieval")
            return _retrieval()

        def report(output: Path, **_: object) -> dict:
            calls.append("report")
            (output / "REPORT.md").write_text("report", encoding="utf-8")
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            return {"interpretation": {"bridge_evidence": True}}

        with (
            mock.patch.object(run_module, "load_config", return_value=(config, resolved, config_hash)),
            mock.patch.object(run_module, "_require_dpvo_environment"),
            mock.patch.object(run_module, "prepare_dataset", side_effect=prepare),
            mock.patch.object(run_module, "extract_features", side_effect=lambda **_: calls.append("extract") or extraction),
            mock.patch.object(run_module, "_validate_dataset", side_effect=lambda *_: calls.append("validate")),
            mock.patch.object(run_module, "train_model", side_effect=train),
            mock.patch.object(run_module, "evaluate_baseline_ladder", side_effect=baseline),
            mock.patch.object(run_module, "evaluate_checkpoint", side_effect=evaluate),
            mock.patch.object(run_module, "analyze_temporal_retrieval", side_effect=retrieval),
            mock.patch.object(run_module, "write_bridge_report", side_effect=report),
        ):
            result = run_module.run_bridge(smoke=True)
        self.assertEqual(result["status"], "smoke_complete")
        self.assertEqual(calls, [
            "prepare", "extract", "validate", "train:adapter", "train:low_rank_linear",
            "baseline", "evaluate:lowrank", "evaluate:adapter", "retrieval", "report",
        ])
        self.assertFalse((self.root / "results/final").exists())
        self.assertEqual(list((self.root / "results").glob(".bridge-*")), [])

    def test_capacity_smoke_has_three_adapters_and_no_lowrank(self) -> None:
        config, resolved, config_hash = self._workflow_config()
        observed: list[tuple[str, str, int]] = []

        def train(**kwargs: object) -> dict:
            observed.append((str(kwargs["model_kind"]), str(kwargs["capacity"]), int(kwargs["epochs"])))
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True)
            checkpoint = output / "best.pt"
            checkpoint.touch()
            capacity = str(kwargs["capacity"])
            return {
                "status": "complete", "model_kind": "adapter", "capacity": capacity,
                "hidden_dim": {"small": 256, "medium": 512, "large": 768}[capacity],
                "parameter_count": 1, "model_config": {}, "model_config_sha256": capacity,
                "epochs": 1, "best_epoch": 1, "best_validation_metric": 1.0,
                "best_metric_name": "val_total_loss", "total_training_time": 1.0,
                "dataset_fingerprint": {}, "checkpoint": str(checkpoint),
            }

        with (
            mock.patch.object(capacity_module, "load_config", return_value=(config, resolved, config_hash)),
            mock.patch.object(capacity_module, "_require_dpvo_environment"),
            mock.patch.object(capacity_module, "prepare_dataset", return_value={"total_indexed_frames": 24}),
            mock.patch.object(capacity_module, "extract_features"),
            mock.patch.object(capacity_module, "train_model", side_effect=train),
        ):
            result = capacity_module.run_capacity_benchmark(smoke=True)
        self.assertEqual(observed, [
            ("adapter", "small", 1), ("adapter", "medium", 1), ("adapter", "large", 1),
        ])
        self.assertEqual(result["status"], "smoke_complete")
        self.assertFalse((self.root / "results/capacity_scaling").exists())

    def test_formal_targets_refuse_overwrite_before_work(self) -> None:
        config, resolved, config_hash = self._workflow_config()
        (self.root / "results/final").mkdir(parents=True)
        with (
            mock.patch.object(run_module, "load_config", return_value=(config, resolved, config_hash)),
            mock.patch.object(run_module, "_require_dpvo_environment"),
            mock.patch.object(run_module, "prepare_dataset") as prepare,
        ):
            with self.assertRaises(FileExistsError):
                run_module.run_bridge(smoke=False)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
