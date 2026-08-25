"""CPU-only regression tests for Exp5-0, Exp5-2 and Exp5-Oracle contracts."""

from __future__ import annotations

import copy
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from .dpvo_runner import run_mock
from .evaluation import evaluate_sequence
from .fusion.feature_hook import (
    FNetInjectionHook,
    InjectionContractError,
    OracleFNetReplacementHook,
)
from .fusion.projection import (
    DenseJepaProjection,
    build_random_dense_projection,
    dense_spatial_cosine_loss,
    estimate_pair_bytes,
    load_dense_projection_checkpoint,
    state_dict_sha256,
    train_dense_projection,
    validate_memory_capacity,
)
from .fusion.run import (
    _figure_relative_path,
    _mock_global_reference,
    _trajectory_relative_path,
    _write_outputs,
    build_metrics,
    build_oracle_metrics,
    decide_dense_fusion,
    decide_oracle_replacement,
    load_config,
    mock_evaluations,
    mock_oracle_evaluations,
    publish_replacing,
    render_report,
    render_oracle_report,
    run_formal,
    run_oracle_formal,
    validate_final_inventory,
    validate_oracle_inventory,
)
from .fusion.runtime import (
    DENSE_FRAME_REQUEST_KEYS,
    RUNTIME_KEYS,
    MockDenseJepaWorkerClient,
    DenseJepaWorkerClient,
    decode_dense_token_response,
    dense_frame_request,
    encode_dense_token_response,
    frame_availability,
    separate_environment_preflight,
)
from .fusion.trajectory_eval import (
    METHOD_ALIASES,
    POSE_FORMAT,
    SEQUENCE_ROLES,
    TIMESTAMP_UNIT,
    build_estimated_trajectory,
    build_ground_truth_subset,
    evaluate_trajectory,
    load_trajectory,
    validate_estimated_trajectory,
    write_trajectory,
)
from .fusion.trajectory_plot import plot_xy_trajectories
from .jepa_runner import FRAME_PAYLOAD_KEYS, sidecar_frame_payload


def _records(count: int = 3) -> list[dict]:
    return [
        {
            "dataset": "euroc",
            "dataset_group": "machine_hall",
            "split": "train",
            "sequence": "MH_01_easy",
            "frame_id": index * 2,
            "stream_index": index,
            "timestamp": 1_403_636_579_763_555_584 + index * 100,
            "image_path": f"/tmp/{1_403_636_579_763_555_584 + index * 100}.png",
        }
        for index in range(count)
    ]


def _exp50_config() -> dict:
    return {
        "schema_version": 2,
        "experiment": {"name": "phase1_exp5_0_jepa_dpvo_interface_validation"},
        "dataset": {"dataset_type": "euroc", "camera": "cam0", "stride": 2, "skip": 0},
        "jepa": {"expected_global_shape": [768]},
        "validation": {
            "timestamp_error_max_ns": 0,
            "max_pose_error": 1.0e-5,
            "latent_epsilon": 1.0e-8,
        },
    }


def _exp50_results(records: list[dict]) -> tuple[dict, dict]:
    frames = [
        {"frame_id": row["frame_id"], "timestamp": row["timestamp"], "runtime": 0.01}
        for row in records
    ]
    baseline = {"status": "ok", "runtime": 1.0, "frames": frames, "trajectory": []}
    sidecar = {
        "status": "ok",
        "runtime": 2.0,
        "frames": copy.deepcopy(frames),
        "trajectory": [[99.0] * 7 for _ in records],
        "jepa_frames": [
            {
                "frame_id": row["frame_id"], "timestamp": row["timestamp"],
                "jepa_shape": [768], "extraction_time": 0.02,
                "latent_norm": 2.0, "latent_mean": 0.1, "latent_std": 0.2,
                "finite_check": True,
            }
            for row in records
        ],
        "trajectory_validation": {
            "mode": "single_run_hook_validation",
            "existing_pose_max_error": 0.0,
            "existing_pose_count": 6,
            "compared_frame_count": len(records),
            "frame_index_unchanged": True,
            "pose_shape_unchanged": True,
            "finite_check": True,
        },
    }
    return baseline, sidecar


class Exp50RegressionTests(unittest.TestCase):
    def test_independent_trajectory_is_not_gate(self) -> None:
        records = _records()
        baseline, sidecar = _exp50_results(records)
        metric = evaluate_sequence(
            expected=records, baseline=baseline, sidecar=sidecar,
            config=_exp50_config(), smoke=False,
        )
        self.assertEqual(metric["status"], "pass")
        self.assertNotIn("trajectory", metric)

    def test_payload_and_dispatch_order_remain_scoped(self) -> None:
        records = _records(2)
        payload = sidecar_frame_payload(records[0] | {"slam": object(), "tensor": object()})
        self.assertEqual(set(payload), FRAME_PAYLOAD_KEYS)
        _, sidecar, events, payloads = run_mock(records)
        self.assertEqual(events[0:4], ["dpvo:0", "hook_before:0", "jepa:0", "hook_after:0"])
        self.assertEqual(sidecar["trajectory_validation"]["compared_frame_count"], 2)
        self.assertTrue(all(set(row) == FRAME_PAYLOAD_KEYS for row in payloads))


class DenseProjectionTests(unittest.TestCase):
    def test_token_reshape_preserves_spatial_order(self) -> None:
        model = DenseJepaProjection(input_dim=2, output_dim=1, token_grid=(2, 2))
        with torch.no_grad():
            model.channel_projection.weight.zero_()
            model.channel_projection.weight[0, 0, 0, 0] = 1.0
            model.channel_projection.bias.zero_()
        tokens = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]])
        output = model(tokens, target_size=(2, 2))
        torch.testing.assert_close(output[0, 0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_real_projection_shape_parameters_and_interpolation(self) -> None:
        model = DenseJepaProjection()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 98_432)
        output = model(torch.zeros(1, 576, 768), target_size=(120, 188))
        self.assertEqual(tuple(output.shape), (1, 128, 120, 188))
        self.assertTrue(torch.isfinite(output).all())

    def test_dense_spatial_cosine_loss(self) -> None:
        teacher = torch.randn(2, 4, 3, 5)
        self.assertAlmostEqual(
            float(dense_spatial_cosine_loss(teacher.clone(), teacher).item()), 0.0, places=6
        )
        with self.assertRaises(ValueError):
            dense_spatial_cosine_loss(teacher, teacher[:, :, :, :-1])

    def test_ram_estimate_and_capacity_gate(self) -> None:
        expected = (576 * 768 + 128 * 120 * 188) * 4 * 1841
        self.assertEqual(estimate_pair_bytes(1841), expected)
        gate = validate_memory_capacity(
            required_bytes=100, available_bytes=125, safety_factor=1.25
        )
        self.assertTrue(gate["passed"])
        with self.assertRaises(MemoryError):
            validate_memory_capacity(
                required_bytes=100, available_bytes=124, safety_factor=1.25
            )

    def test_random_dense_projection_is_deterministic(self) -> None:
        config = load_config()
        left = build_random_dense_projection(config, device=torch.device("cpu"))
        right = build_random_dense_projection(config, device=torch.device("cpu"))
        self.assertEqual(
            state_dict_sha256(left.state_dict()), state_dict_sha256(right.state_dict())
        )


class DenseEnvironmentBridgeTests(unittest.TestCase):
    def test_environment_config_is_separate_and_invalid_path_fails(self) -> None:
        config = load_config()
        self.assertNotIn("jepa_python", config["runtime"])
        self.assertNotEqual(
            Path(config["runtime"]["dpvo_python"]).resolve(),
            Path(config["environment"]["vjepa_python"]).resolve(),
        )
        invalid = copy.deepcopy(config)
        invalid["environment"]["vjepa_python"] = "/nonexistent/exp5-vjepa-python"
        with self.assertRaises(FileNotFoundError):
            separate_environment_preflight(invalid)

    def test_frame_request_and_base64_token_roundtrip(self) -> None:
        record = _records(1)[0]
        request = dense_frame_request(record, 0)
        self.assertEqual(set(request), DENSE_FRAME_REQUEST_KEYS)
        tokens = np.arange(576 * 768, dtype=np.float32).reshape(576, 768)
        response = encode_dense_token_response(
            request=request, tokens=tokens, extraction_time=0.25
        )
        decoded, runtime = decode_dense_token_response(
            response, expected_request=request
        )
        np.testing.assert_array_equal(decoded, tokens)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(runtime, 0.25)
        self.assertEqual(response["token_encoding"], "base64_raw_little_endian")

    def test_protocol_rejects_identity_nonfinite_and_invalid_base64(self) -> None:
        request = dense_frame_request(_records(1)[0], 0)
        tokens = np.zeros((576, 768), dtype=np.float32)
        response = encode_dense_token_response(
            request=request, tokens=tokens, extraction_time=0.1
        )
        mismatch = copy.deepcopy(response)
        mismatch["request_id"] = 1
        with self.assertRaises(RuntimeError):
            decode_dense_token_response(mismatch, expected_request=request)
        invalid = copy.deepcopy(response)
        invalid["token_data"] = "not-base64!"
        with self.assertRaises(RuntimeError):
            decode_dense_token_response(invalid, expected_request=request)
        tokens[0, 0] = np.nan
        with self.assertRaises(ValueError):
            encode_dense_token_response(
                request=request, tokens=tokens, extraction_time=0.1
            )

    def test_mock_worker_is_persistent_for_complete_sequence(self) -> None:
        records = _records(8)
        token = np.ones((576, 768), dtype=np.float32)
        worker = MockDenseJepaWorkerClient({row["frame_id"]: token for row in records})
        for row in records:
            dense, extraction, bridge = worker.extract(row)
            self.assertEqual(dense.shape, (576, 768))
            self.assertGreater(extraction, 0.0)
            self.assertGreater(bridge, 0.0)
        worker.close()
        lifecycle = worker.lifecycle()
        self.assertTrue(lifecycle["initialized"])
        self.assertEqual(lifecycle["request_count"], 8)
        self.assertTrue(lifecycle["stopped"])
        self.assertTrue(lifecycle["cleanup_passed"])

    def test_client_reports_worker_early_exit(self) -> None:
        class ExitedProcess:
            def __init__(self) -> None:
                self.stdout = io.StringIO("")
                self.stdin = io.StringIO()
                self.returncode = 7

            def poll(self) -> int:
                return 7

        config = load_config()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "research.src.phase1_dpvo_feasibility.exp5.fusion.runtime.subprocess.Popen",
            return_value=ExitedProcess(),
        ):
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                DenseJepaWorkerClient(config, Path(temporary) / "worker.json")

    def test_mock_evaluations_scope_workers_per_fused_run(self) -> None:
        evaluations = mock_evaluations(8)
        for methods in evaluations.values():
            self.assertFalse(methods["dpvo_baseline"]["worker_lifecycle"]["used"])
            for method in (
                "random_dense_projection_fusion",
                "jepa_dense_projection_fusion",
            ):
                lifecycle = methods[method]["worker_lifecycle"]
                self.assertTrue(lifecycle["used"])
                self.assertEqual(lifecycle["request_count"], 8)
                self.assertTrue(lifecycle["cleanup_passed"])

    def test_runtime_has_no_distributed_or_dpvo_worker_bridge(self) -> None:
        source = (Path(__file__).parent / "fusion" / "runtime.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "unified_dependency_preflight",
            "evaluation_worker_main",
            "invoke_evaluation_worker",
            "SharedMemory",
            "multiprocessing.shared_memory",
            "from multiprocessing import shared_memory",
            "AF_UNIX",
        ):
            self.assertNotIn(forbidden, source)


class DenseTrainingTests(unittest.TestCase):
    def test_validation_schedule_and_checkpoint(self) -> None:
        config = load_config()
        config["projection"].update({
            "input_dim": 2,
            "output_dim": 1,
            "token_count": 4,
            "token_grid": [2, 2],
            "epochs": 10,
            "batch_size": 2,
            "validation_interval": 5,
        })
        rng = np.random.default_rng(4)
        tokens = rng.standard_normal((4, 4, 2), dtype=np.float32)
        teacher = tokens[..., :1].reshape(4, 2, 2, 1).transpose(0, 3, 1, 2).copy()
        pairs = {"tokens": tokens, "teacher": teacher}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "projection.pt"
            model, metadata = train_dense_projection(
                train_pairs=pairs,
                val_pairs={key: value.copy() for key, value in pairs.items()},
                config=config,
                checkpoint_path=checkpoint,
                device=torch.device("cpu"),
            )
            loaded, loaded_metadata = load_dense_projection_checkpoint(
                checkpoint, device=torch.device("cpu")
            )
            self.assertEqual(metadata["validation_epochs"], [5, 10])
            self.assertIn(metadata["best_epoch"], (5, 10))
            self.assertEqual(metadata["checkpoint_schema_version"], 2)
            self.assertEqual(loaded_metadata["projection_kind"], "dense_tokens")
            self.assertEqual(type(model), type(loaded))


class DenseInjectionTests(unittest.TestCase):
    class FNet(nn.Module):
        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            return torch.ones(1, 1, 128, 2, 3)

    def test_raw_fnet_scale_contract_and_cleanup(self) -> None:
        record = _records(1)[0]
        projected = torch.randn(1, 128, 2, 3)
        fnet = self.FNet()
        hook = FNetInjectionHook(
            alpha=0.1, channels=128, projection_to_raw_scale=4.0
        )
        hook.install(fnet)
        hook.bind(record, projected)
        raw_fused = fnet(torch.zeros(1))
        hook.finish_frame(record)
        hook.close()
        expected_fmap = torch.ones_like(raw_fused) / 4.0 + 0.1 * projected[:, None]
        torch.testing.assert_close(raw_fused / 4.0, expected_fmap)
        self.assertTrue(hook.cleanup_passed)
        self.assertEqual(hook.lifecycle()["consumed_frames"], 1)

    def test_dense_hook_rejects_wrong_spatial_shape_and_nonfinite(self) -> None:
        record = _records(1)[0]
        hook = FNetInjectionHook()
        fnet = self.FNet()
        hook.install(fnet)
        hook.bind(record, torch.zeros(1, 128, 1, 1))
        with self.assertRaises(InjectionContractError):
            fnet(torch.zeros(1))
        hook.close()
        hook = FNetInjectionHook()
        hook.install(self.FNet())
        with self.assertRaises(InjectionContractError):
            hook.bind(record, torch.full((1, 128, 2, 3), float("nan")))
        hook.close()


def _projection_metadata() -> dict:
    alignment = {
        "sample_count": 8,
        "spatial_vector_count": 768,
        "mean_spatial_cosine_similarity": 0.8,
        "spatial_cosine_loss": 0.2,
    }
    return {
        "checkpoint_schema_version": 2,
        "projection_kind": "dense_tokens",
        "model": "Conv2d(768,128,kernel_size=1)",
        "input_shape": [576, 768],
        "token_grid": [24, 24],
        "output_channels": 128,
        "teacher_contract": "raw Patchifier.fnet output / 4.0",
        "interpolation": {"mode": "bilinear", "align_corners": False},
        "parameter_count": 98_432,
        "seed": 1234,
        "best_epoch": 50,
        "best_validation_spatial_cosine": 0.8,
        "validation_epochs": list(range(5, 51, 5)),
        "requested_epochs": 50,
        "batch_size": 8,
        "training_seconds": 10.0,
        "split_counts": {"train": 8, "val": 8},
        "learned_validation_alignment": alignment,
        "random_validation_alignment": alignment | {"mean_spatial_cosine_similarity": 0.0},
        "random_seed": 4321,
        "random_weight_hash": "a" * 64,
        "uses_groundtruth": False,
        "uses_trajectory": False,
        "uses_future_frames": False,
        "pair_storage": "CPU_RAM_float32_only",
        "config_sha256": "b" * 64,
    }


def _extraction_metadata() -> dict:
    return {
        "pair_storage": "preallocated_CPU_RAM_float32_only",
        "persistent_cache": False,
        "split_counts": {"train": 8, "val": 8},
        "token_shape": [576, 768],
        "teacher_shape": [128, 120, 188],
        "teacher_contract": "raw Patchifier.fnet output / 4.0",
        "pair_bytes": 100,
        "memory_gate": {"passed": True},
        "extraction_seconds": 1.0,
        "jepa_extraction_seconds": 0.5,
        "uses_groundtruth": False,
        "uses_trajectory": False,
        "uses_future_frames": False,
    }


def _test_alignment() -> dict:
    return {
        "sample_count": 8,
        "spatial_vector_count": 8 * 120 * 188,
        "learned_mean_spatial_cosine_similarity": 0.8,
        "random_mean_spatial_cosine_similarity": 0.0,
        "evaluation_seconds": 1.0,
        "pair_retained": False,
    }


def _trajectory_visualization_metadata() -> dict:
    result = {}
    for sequence, role in SEQUENCE_ROLES.items():
        result[sequence] = {
            "role": role,
            "trajectory_artifacts": {
                "baseline": f"trajectories/{sequence}_baseline.json",
                "random": f"trajectories/{sequence}_random.json",
                "learned": f"trajectories/{sequence}_learned.json",
                "ground_truth": f"trajectories/{sequence}_gt.json",
            },
            "figure_path": f"figures/{sequence.rsplit('_', 1)[0]}_xy.png",
            "completeness": True,
            "ate_verification": {
                alias: {
                    "runtime_ate_rmse": 0.1,
                    "exported_ate_rmse": 0.1,
                    "associated_count": 8,
                    "alignment": "Sim(3)",
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "diagnostic_only": True,
                }
                for alias in ("baseline", "random", "learned")
            },
            "decision_input": sequence == "MH_05_difficult",
        }
    return result


class DenseTrajectoryVisualizationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_payloads(source: Path) -> tuple[dict, dict]:
        source.write_text("synthetic ground truth\n", encoding="utf-8")
        count = 8
        timestamps = np.arange(count, dtype=np.int64) * 50_000_000 + 1_000_000_000
        index = np.arange(count, dtype=np.float64)
        gt_positions = np.stack(
            (index, np.sin(index * 0.5), np.cos(index * 0.3)), axis=1
        )
        estimate_positions = 2.5 * gt_positions + np.asarray([3.0, -2.0, 1.0])
        quaternion = np.tile([[0.0, 0.0, 0.0, 1.0]], (count, 1))
        records = [
            {"timestamp": int(timestamp), "frame_id": ordinal * 2, "stream_index": ordinal}
            for ordinal, timestamp in enumerate(timestamps)
        ]
        estimate = build_estimated_trajectory(
            sequence="MH_05_difficult",
            method="jepa_dense_projection_fusion",
            records=records,
            ordinals=np.arange(count, dtype=np.float64),
            poses=np.concatenate((estimate_positions, quaternion), axis=1),
        )
        ground_truth = {
            "schema_version": 1,
            "kind": "ground_truth",
            "sequence": "MH_05_difficult",
            "method": "ground_truth",
            "timestamp_unit": TIMESTAMP_UNIT,
            "pose_format": POSE_FORMAT,
            "timestamps": timestamps.tolist(),
            "poses": np.concatenate((gt_positions, quaternion), axis=1).tolist(),
            "num_poses": count,
            "source_path": str(source.resolve()),
            "source_sha256": "0" * 64,
        }
        return estimate, ground_truth

    def test_schema_roundtrip_and_sim3_ate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            estimate, ground_truth = self._synthetic_payloads(root / "gt.txt")
            estimate_path = root / "estimate.json"
            gt_path = root / "gt.json"
            write_trajectory(estimate_path, estimate)
            write_trajectory(gt_path, ground_truth)
            loaded_estimate = load_trajectory(estimate_path)
            loaded_gt = load_trajectory(gt_path)
            metric = evaluate_trajectory(loaded_estimate, loaded_gt)
            self.assertEqual(metric["associated_count"], 8)
            self.assertEqual(metric["alignment"], "Sim(3)")
            self.assertLess(metric["ate_rmse"], 1.0e-10)
            self.assertEqual(loaded_estimate["num_poses"], len(loaded_estimate["poses"]))

    def test_empty_duplicate_bad_ordinal_and_nonfinite_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            estimate, _ = self._synthetic_payloads(Path(temporary) / "gt.txt")
            empty = copy.deepcopy(estimate)
            empty.update({"timestamps": [], "frame_ids": [], "ordinals": [], "poses": [],
                          "num_poses": 0})
            with self.assertRaises(ValueError):
                validate_estimated_trajectory(empty)
            duplicate = copy.deepcopy(estimate)
            duplicate["timestamps"][1] = duplicate["timestamps"][0]
            with self.assertRaises(ValueError):
                validate_estimated_trajectory(duplicate)
            bad_ordinal = copy.deepcopy(estimate)
            bad_ordinal["ordinals"][1] = 9
            with self.assertRaises(ValueError):
                validate_estimated_trajectory(bad_ordinal)
            nonfinite = copy.deepcopy(estimate)
            nonfinite["poses"][0][0] = float("nan")
            with self.assertRaises(ValueError):
                validate_estimated_trajectory(nonfinite)

    def test_real_ground_truth_subset(self) -> None:
        from evo.tools import file_interface

        config = load_config()
        source = Path(config["repo_root"]) / str(
            config["evaluation"]["groundtruth_pattern"]
        ).format(sequence="MH_03_medium")
        reference = file_interface.read_tum_trajectory_file(str(source))
        timestamps = np.rint(reference.timestamps).astype(np.int64)
        payload = build_ground_truth_subset(
            sequence="MH_03_medium",
            source_path=source,
            timestamp_min=int(timestamps[10]),
            timestamp_max=int(timestamps[20]),
        )
        self.assertGreater(payload["num_poses"], 0)
        self.assertTrue(all(
            int(timestamps[10]) <= value <= int(timestamps[20])
            for value in payload["timestamps"]
        ))

    def test_png_plot_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            estimate, ground_truth = self._synthetic_payloads(root / "gt.txt")
            estimates = {}
            for alias, method in (
                ("baseline", "dpvo_baseline"),
                ("random", "random_dense_projection_fusion"),
                ("learned", "jepa_dense_projection_fusion"),
            ):
                payload = copy.deepcopy(estimate)
                payload["method"] = method
                estimates[alias] = payload
            output = root / "MH_05_xy.png"
            metadata = plot_xy_trajectories(
                sequence="MH_05_difficult",
                estimates=estimates,
                ground_truth=ground_truth,
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(metadata["format"], "png")
            self.assertFalse(any(root.glob("*.pdf")))

    def test_replacing_publish_and_failure_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            target = root / "result"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            staging = work / "staging"
            staging.mkdir()
            (staging / "new.txt").write_text("new", encoding="utf-8")
            publish_replacing(staging, target, work)
            self.assertTrue((target / "new.txt").is_file())
            self.assertFalse((target / "old.txt").exists())

            failed_staging = work / "failed_staging"
            failed_staging.mkdir()
            (failed_staging / "bad.txt").write_text("bad", encoding="utf-8")
            real_replace = __import__("os").replace
            calls = 0

            def fail_second(source: str | Path, destination: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic publish failure")
                real_replace(source, destination)

            with mock.patch(
                "research.src.phase1_dpvo_feasibility.exp5.fusion.run.os.replace",
                side_effect=fail_second,
            ):
                with self.assertRaises(OSError):
                    publish_replacing(failed_staging, target, work)
            self.assertTrue((target / "new.txt").is_file())


class DenseDecisionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_strong_feasible_inconclusive_and_failure_boundaries(self) -> None:
        evaluations = mock_evaluations(8)
        strong = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual((strong["outcome"], strong["evidence"]), (
            "success_dense_fusion", "strong"
        ))

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["ate"][
            "translation_rmse"
        ] = 0.315
        exact_strong = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(exact_strong["evidence"], "strong")

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["ate"]["translation_rmse"] = 0.324
        feasible = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(feasible["evidence"], "feasible")

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["random_dense_projection_fusion"]["ate"][
            "translation_rmse"
        ] = 0.34
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["ate"][
            "translation_rmse"
        ] = 0.33
        exact_feasible = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(exact_feasible["evidence"], "feasible")

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["random_dense_projection_fusion"]["ate"]["translation_rmse"] = 0.32
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["ate"]["translation_rmse"] = 0.32
        inconclusive = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(inconclusive["outcome"], "inconclusive_dense_fusion")

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["ate"]["translation_rmse"] = 0.331
        failure = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(failure["outcome"], "failure_dense_fusion")

        evaluations = mock_evaluations(8)
        evaluations["MH_05_difficult"]["jepa_dense_projection_fusion"]["tracking"]["success"] = False
        engineering = decide_dense_fusion(evaluations=evaluations, config=self.config)
        self.assertEqual(engineering["evidence"], "engineering_failure")

    def test_metrics_report_and_output_inventory(self) -> None:
        evaluations = mock_evaluations(8)
        decision_before = decide_dense_fusion(
            evaluations=copy.deepcopy(evaluations), config=self.config
        )
        metrics = build_metrics(
            config=self.config,
            projection_metadata=_projection_metadata(),
            extraction_metadata=_extraction_metadata(),
            test_alignment=_test_alignment(),
            evaluations=evaluations,
            global_reference=_mock_global_reference(),
            smoke=False,
            trajectory_visualization=_trajectory_visualization_metadata(),
        )
        report = render_report(metrics)
        self.assertEqual(metrics["decision"], decision_before)
        self.assertFalse(metrics["representation_alignment"]["direct_numeric_comparison_valid"])
        self.assertIn("must not be compared numerically", report)
        self.assertIn("384×384 center crop", report)
        self.assertIn("Learned global cosine", report)
        self.assertIn("Learned dense cosine", report)
        self.assertIn("Projection s", report)
        self.assertIn("Training-sequence diagnostic", report)
        self.assertIn("Transfer diagnostic", report)
        self.assertIn("Decision-sequence visualization", report)
        self.assertIn("continues to use MH_05 alone", report)
        self.assertEqual(
            {sequence: row["role"] for sequence, row in metrics["trajectory_visualization"].items()},
            SEQUENCE_ROLES,
        )
        for methods in metrics["evaluations"].values():
            for row in methods.values():
                self.assertEqual(set(row["runtime"]), RUNTIME_KEYS)
                self.assertTrue(row["runtime_overhead_vs_baseline"]["diagnostic_only"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            torch.save({"mock": True}, output / "projection.pt")
            _write_outputs(output, metrics)
            (output / "trajectories").mkdir()
            (output / "figures").mkdir()
            for sequence in SEQUENCE_ROLES:
                for method in METHOD_ALIASES:
                    (output / _trajectory_relative_path(sequence, method)).touch()
                (output / "trajectories" / f"{sequence}_gt.json").touch()
                (output / _figure_relative_path(sequence)).touch()
            validate_final_inventory(output)
            self.assertFalse(any(output.rglob("*.pdf")))
            json.loads((output / "metrics.json").read_text(encoding="utf-8"))


class OracleFNetReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_stride_availability_and_replacement_boundary(self) -> None:
        records = _records(8)
        frames, availability = frame_availability(records, keyframe_stride=5)
        self.assertEqual(
            [index for index, frame in enumerate(frames) if frame["is_keyframe"]],
            [0, 5],
        )
        self.assertEqual(availability["keyframe_count"], 2)
        self.assertEqual(availability["non_keyframe_count"], 6)
        self.assertEqual(availability["matching_rgb_ratio"], 0.25)
        self.assertEqual(availability["jepa_replaced_ratio"], 0.75)
        _, repeated = frame_availability(records, keyframe_stride=5)
        self.assertEqual(
            availability["canonical_availability_sha256"],
            repeated["canonical_availability_sha256"],
        )

        class CountingFNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, _image: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return torch.ones(1, 1, 128, 4, 6)

        fnet = CountingFNet()
        hook = OracleFNetReplacementHook(
            alpha=0.1, channels=128, projection_to_raw_scale=4.0,
            replacement_dtype=torch.float32,
        )
        hook.install(fnet)
        outputs = []
        projections = []
        for index, record in enumerate(records):
            projection = torch.full((1, 128, 4, 6), float(index + 1))
            projections.append(projection)
            is_keyframe = frames[index]["is_keyframe"]
            hook.bind(record, projection, is_keyframe=is_keyframe)
            output = fnet(torch.randn(1, 1, 3, 16, 24))
            outputs.append(output)
            hook.finish_frame(record, is_keyframe=is_keyframe)
        hook.close()
        lifecycle = hook.lifecycle()
        self.assertEqual(fnet.calls, 2)
        self.assertEqual(lifecycle["fnet_executed_frames"], 2)
        self.assertEqual(lifecycle["jepa_replaced_frames"], 6)
        self.assertEqual(lifecycle["output_contract"], [1, 1, 128, 4, 6])
        self.assertTrue(lifecycle["cleanup_passed"])
        self.assertEqual(outputs[0].shape, outputs[1].shape)
        self.assertEqual(outputs[0].dtype, outputs[1].dtype)
        self.assertTrue(torch.allclose(
            outputs[0] / 4.0,
            torch.ones_like(outputs[0]) / 4.0 + 0.1 * projections[0][:, None],
        ))
        self.assertTrue(torch.allclose(outputs[1] / 4.0, projections[1][:, None]))

    def test_oracle_hook_rejects_nonfinite_and_duplicate_consumption(self) -> None:
        fnet = nn.Identity()
        hook = OracleFNetReplacementHook(replacement_dtype=torch.float32)
        hook.install(fnet)
        record = _records(1)[0]
        with self.assertRaises(InjectionContractError):
            hook.bind(
                record, torch.full((1, 128, 4, 6), float("nan")),
                is_keyframe=False,
            )
        projection = torch.ones(1, 128, 4, 6)
        hook.bind(record, projection, is_keyframe=False)
        fnet(torch.zeros(1))
        with self.assertRaises(InjectionContractError):
            fnet(torch.zeros(1))
        hook.close()
        self.assertFalse(hook.cleanup_passed)

    def test_oracle_decision_boundaries_and_coverage_not_gate(self) -> None:
        evaluations = mock_oracle_evaluations(8)
        strong = decide_oracle_replacement(
            evaluations=evaluations, config=self.config
        )
        self.assertEqual((strong["outcome"], strong["evidence"]), (
            "success_oracle_fmap_replacement", "strong"
        ))

        evaluations = mock_oracle_evaluations(8)
        oracle = evaluations["MH_05_difficult"]["oracle_jepa_missing_rgb"]
        oracle["ate"]["translation_rmse"] = 0.315
        oracle["tracking"]["pose_count"] = 7
        exact_strong = decide_oracle_replacement(
            evaluations=evaluations, config=self.config
        )
        self.assertEqual(exact_strong["evidence"], "strong")
        self.assertFalse(exact_strong["coverage_is_independent_gate"])

        evaluations = mock_oracle_evaluations(8)
        evaluations["MH_05_difficult"]["oracle_jepa_missing_rgb"]["ate"][
            "translation_rmse"
        ] = 0.324
        feasible = decide_oracle_replacement(
            evaluations=evaluations, config=self.config
        )
        self.assertEqual(feasible["evidence"], "feasible")

        evaluations = mock_oracle_evaluations(8)
        evaluations["MH_05_difficult"]["oracle_jepa_missing_rgb"]["ate"][
            "translation_rmse"
        ] = 0.331
        failure = decide_oracle_replacement(
            evaluations=evaluations, config=self.config
        )
        self.assertEqual(failure["outcome"], "failure_oracle_fmap_replacement")

        evaluations = mock_oracle_evaluations(8)
        evaluations["MH_05_difficult"]["oracle_jepa_missing_rgb"][
            "reproducibility_contract_sha256"
        ] = "different"
        engineering = decide_oracle_replacement(
            evaluations=evaluations, config=self.config
        )
        self.assertEqual(engineering["evidence"], "engineering_failure")

    def test_oracle_report_inventory_and_no_training(self) -> None:
        evaluations = mock_oracle_evaluations(8)
        reproducibility = {
            "contract_sha256": "c" * 64,
            "git_head": "mock",
        }
        evidence = {
            "sequence": "MH_05_difficult",
            "role": "decision",
            "trajectory_artifacts": {
                "baseline": "trajectories/MH_05_baseline.json",
                "oracle": "trajectories/MH_05_oracle.json",
                "ground_truth": "trajectories/MH_05_gt.json",
            },
            "figure_path": "figures/MH_05_oracle_missing_rgb.png",
            "ate_verification": {},
            "diagnostic_only": True,
            "completeness": True,
        }
        metrics = build_oracle_metrics(
            config=self.config,
            evaluations=evaluations,
            reproducibility=reproducibility,
            source_reference={
                "projection_retrained": False,
                "projection_sha256": "1" * 64,
                "metrics_sha256": "2" * 64,
            },
            trajectory_evidence=evidence,
            smoke=False,
        )
        report = render_oracle_report(metrics)
        self.assertIn(
            "replacement for missing RGB-derived matching features, not a complete replacement",
            report,
        )
        self.assertIn("INet/context", report)
        self.assertNotIn("train_dense_projection", inspect.getsource(run_oracle_formal))
        self.assertIn("run_dense_formal", inspect.getsource(run_formal))
        self.assertIn("run_oracle_formal", inspect.getsource(run_formal))
        baseline = evaluations["MH_05_difficult"]["dpvo_baseline"]
        self.assertFalse(baseline["worker_lifecycle"]["used"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "trajectories").mkdir()
            (output / "figures").mkdir()
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            (output / "REPORT.md").write_text(report, encoding="utf-8")
            for filename in (
                "MH_05_baseline.json", "MH_05_oracle.json", "MH_05_gt.json"
            ):
                (output / "trajectories" / filename).touch()
            (output / "figures" / "MH_05_oracle_missing_rgb.png").touch()
            validate_oracle_inventory(output)

    def test_formal_orchestrator_always_rebuilds_dense_stage_first(self) -> None:
        self.assertNotIn("global_reference_metrics", self.config["evaluation"])
        with mock.patch(
            "research.src.phase1_dpvo_feasibility.exp5.fusion.run.run_dense_formal",
            return_value={"mode": "formal", "status": "inconclusive_dense_fusion"},
        ) as dense, mock.patch(
            "research.src.phase1_dpvo_feasibility.exp5.fusion.run.run_oracle_formal",
            return_value={"mode": "formal", "status": "success_oracle_fmap_replacement"},
        ) as oracle:
            result = run_formal(self.config)
        dense.assert_called_once_with(self.config)
        oracle.assert_called_once_with(self.config)
        self.assertEqual(result["pipeline"], "from_scratch_exp5_2_then_exp5_oracle")
        self.assertFalse(result["old_results_reused"])
        self.assertTrue(result["projection_retrained_as_exp5_2_prerequisite"])


if __name__ == "__main__":
    unittest.main()
