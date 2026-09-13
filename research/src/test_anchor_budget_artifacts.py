"""CPU-only publication tests; fixtures contain synthetic stored results, not SLAM runs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import anchor_budget_artifacts as artifacts
from . import run_anchor_budget as runner
from .anchor_budget_figures import aligned_translation, render_figures
from .predictor import build_anchor_intervals, predictor_state_sha256
from .protocol import FrameIdentity, atomic_write_json, canonical_sha256, sha256_file


def staging_fixture(parent, strides=(3, 5, 10)):
    parent.mkdir(parents=True, exist_ok=True)
    worker_log = parent / ".research_anchor_budget_run/workers/job_000/worker.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("runtime diagnostics")
    root = parent / "staging"
    sequence = root / "sequences/MH_01_easy"
    sequence.mkdir(parents=True)
    ts = np.asarray([1403636579763555584 + i*100_000_000 for i in range(31)], dtype=np.uint64)
    xyz = np.asarray([[np.cos(i/5), np.sin(i/5), i/20] for i in range(31)])
    poses = np.column_stack((xyz, np.zeros((31,3)), np.ones(31)))
    gt = parent / "groundtruth.txt"
    np.savetxt(gt, np.column_stack((ts, xyz, np.ones(31), np.zeros((31,3)))))
    atomic_write_json(sequence / "groundtruth.json", {"path": str(gt), "sha256": sha256_file(gt), "role": "reference_not_slam_run"})
    identities = [FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy", i*2, i, int(t)) for i,t in enumerate(ts)]
    fixed = {"regions": {name: {"candidate_start": a, "candidate_end": b}
                         for name,a,b in (("train",0,9),("validation",11,19),("test",21,30))}}
    comparisons, horizons, references = [], [], {}
    populations = {s: {"population_sha256": str(s), "common_anchor_timestamps_ns": [int(t) for t in ts[::s]],
                       "rpe_pairs_ns": [] if s == 3 else [[int(ts[0]),int(ts[10])]]} for s in set((*strides, 5))}
    profile = {"context_wait_ms": {"mean_ms": 150, "p95_ms": 200},
               "effective_hidden_delay_ms": {"mean_ms": 900}, "stages": {"jepa_encoder": {"total_ms": 100}},
               "stage_c_runtime": {"pid": 123}, "stage_c_timing": {"total_ms": 30}}
    def condition(folder, selected, stride, ours=False):
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "trajectory.npz", timestamps_ns=ts[selected], poses=poses[selected])
        row = {"condition": folder.name, "status": "complete", "runtime": {"elapsed_seconds": 3, "stage_c_timing": {"total_ms": 30}},
               "canonical_evaluation": {"ate_rmse_m": .12345678901234568 + stride/100,
                    "translation_rpe_rmse_m": None if stride == 3 else .01, "rotation_rpe_rmse_deg": None if stride == 3 else .2,
                    "rpe_pair_count": 0 if stride == 3 else 1,
                    "sim3": {"scale": 1.25, "rotation": np.eye(3).tolist(), "translation": [1,2,3]}},
               "canonical_population": populations[stride], "canonical_coverage": {"canonical_pose_coverage": 1.0},
               "graph_workload": {"final_node_count": 12, "cumulative_factor_count": 200},
               "matched_trajectory_wall_seconds": 3.456789,
               "sequential_execution": {"logical_device": 0, "elapsed_seconds": 5, "worker_log": str(worker_log)},
               "performance_diagnostics": {"jobs": [{"worker_log": str(worker_log), "elapsed_seconds": 5}]},
               "trajectory_sha256": sha256_file(folder / "trajectory.npz")}
        if ours:
            row["h2_stage_profile"] = copy.deepcopy(profile)
            row["provider_usage"] = {"timeline": [{"anchor": "debug", "admitted_ns": 1}], "waits": [{"ms": .1}],
                                     "hidden_consumed_exactly_once": True}
        atomic_write_json(folder / "results.json", row)
        return row
    condition(sequence / "full_rgb", slice(None), 5)
    for stride in strides:
        folder = sequence / f"stride_{stride}"
        sparse = condition(folder / "sparse_rgb", slice(None,None,stride), stride)
        ours = condition(folder / "predicted_jepa", slice(None), stride, ours=True)
        roles = {i.key: "anchor" if i.candidate_index % stride == 0 else "hidden" for i in identities}
        intervals = build_anchor_intervals(identities, roles)
        population = {"anchor_stride": stride, "theoretical_anchor_ratio": 1/stride, "complete_intervals": len(intervals),
                      "effective_population": {"effective_candidate_count":31}, "split": fixed, "schedule": {"anchor_count": len(ts[::stride])},
                      "communication": {"actual_anchor_ratio":len(ts[::stride])/31, "encoded_byte_reduction": .7,
                                        "encoded_anchor_bytes": 30, "encoded_full_bytes":100}}
        atomic_write_json(folder / "schedule.json", population | {"roles":roles,"intervals":[r.payload() for r in intervals]})
        lineage = {"protocol":"anchor_budget_fresh_predictor_v1", "anchor_stride":stride, "seed":1234,
                   "h1_bridge_sha256":"h1", "scientific_config":{"config_protocol_sha256":"h2"}}
        lineage["training_lineage_sha256"] = canonical_sha256(lineage)
        modelroot = root / "models" / f"stride_{stride}"
        modelroot.mkdir(parents=True)
        state = {"weight": torch.arange(4,dtype=torch.float32)}
        torch.save({"training_lineage":lineage, "state_dict":state, "state_dict_sha256":predictor_state_sha256(state)}, modelroot / "predictor.pt")
        horizon = [{"anchor_stride":stride, "relative_hidden_index":q, "distance_from_previous_anchor_seconds":q/10,
                    "distance_from_closing_anchor_seconds":(stride-q)/10,
                    "predicted_jepa_cosine":.9-q/100, "transport_baseline_cosine":.8-q/100,
                    "bridge_fmap_cosine":.7-q/100, "transport_bridge_fmap_cosine":.6-q/100}
                   for q in range(1,stride)]
        training = {"anchor_stride":stride, "lineage":lineage, "split":fixed, "horizon_resolved_quality":horizon,
                    "summary":{"best_epoch":2, "best_validation_total":.2, "elapsed_seconds":1.2345,
                               "history":[{"epoch":1,"train_total":.5,"validation_total":.3},
                                          {"epoch":2,"train_total":.4,"validation_total":.2}]},
                    "training_and_diagnostics_wall_seconds":3.14159,
                    "development_extraction":{"elapsed_seconds":.123,"workers":[{"pid":123,"worker_log":str(worker_log)}]},
                    "test_extraction":{"encoder_inference_ms":[1.0,2.0],"jepa":{"worker_pid":5,"runtime":{}}},
                    "held_out_representation":{"gate1_block5":{"predicted_jepa":{"cosine":.9}},
                        "gate2_frozen_h1_bridge_vs_true_fmap":{"predicted_jepa":{"cosine":.7}}}}
        atomic_write_json(modelroot / "training.json", training)
        atomic_write_json(modelroot / "horizon_queries.json", horizon)
        comparison = {"anchor_stride":stride, "actual_anchor_ratio":population["communication"]["actual_anchor_ratio"],
                      "encoded_byte_reduction":.7,"sparse_ate_rmse_m":sparse["canonical_evaluation"]["ate_rmse_m"],
                      "ours_ate_rmse_m":ours["canonical_evaluation"]["ate_rmse_m"],"ours_minus_sparse_ate_m":0.,
                      "ours_wall_seconds":ours["matched_trajectory_wall_seconds"],"context_wait_mean_ms":150.}
        legacy = {"anchor_stride":stride,"population":population,"comparison":comparison,
                  "predictor_ref":str((modelroot / "predictor.pt").relative_to(root)),
                  "predictor_sha256":sha256_file(modelroot / "predictor.pt"),
                  "training_ref":str((modelroot / "training.json").relative_to(root)),"training_cleanup":{"success":True}}
        atomic_write_json(folder / "results.json",legacy)
        references[str(stride)] = str((folder / "results.json").relative_to(root))
        comparisons.append(comparison); horizons.extend(horizon)
    index = {"status":"complete","anchor_strides":list(strides),"sequence":"MH_01_easy","protocol":"test",
             "trajectory_comparison":["GT","Full RGB","Sparse RGB","Ours"], "config":"/tmp/config.yaml","config_sha256":"config",
             "preparation":{"fixed_split":fixed,"canonical_artifacts":{},"stride5_equivalence":{"all_exact":True}},
             "repository":{"git_commit":"fixture"}, "execution":{"maximum_concurrent_dpvo_instances":1,
                 "sequential_evaluation":{"jobs":[{"worker_log":str(worker_log),"elapsed_seconds":5}]}},"strides":references,
             "accuracy_vs_communication":comparisons,"prediction_quality_vs_horizon":horizons}
    atomic_write_json(root / "INDEX.json",index)
    atomic_write_json(root / "accuracy_vs_communication.json",comparisons)
    atomic_write_json(root / "prediction_quality_vs_horizon.json",horizons)
    return root


class ArtifactAggregationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="anchor_budget_test_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.source = staging_fixture(self.parent)
        self.destination = self.parent / "results"
        self.checkpoints = self.parent / "checkpoints"

    def test_scientific_values_and_all_trajectories_are_exact(self):
        data,bundle,checkpoints,proof = artifacts.build_compact(self.source,self.checkpoints)
        self.assertEqual(len(data["metadata"]["trajectories"]),8)
        for name,old in proof["trajectories"].items():
            restored = artifacts.reconstruct_trajectory(bundle,name)
            for key in old:
                self.assertEqual(restored[key].dtype,old[key].dtype)
                self.assertTrue(np.array_equal(restored[key],old[key]))
        for stride in (3,5,10):
            row=data["strides"][str(stride)]
            base=self.source/f"sequences/MH_01_easy/stride_{stride}"
            for kind,folder in (("sparse","sparse_rgb"),("ours","predicted_jepa")):
                old=json.loads((base/folder/"results.json").read_text())
                for key in ("canonical_evaluation","canonical_coverage","graph_workload","matched_trajectory_wall_seconds"):
                    self.assertEqual(row[kind][key],old[key])
            old_train=json.loads((self.source/f"models/stride_{stride}/training.json").read_text())
            self.assertEqual(row["training"]["held_out_representation"],old_train["held_out_representation"])
            self.assertEqual(artifacts.expand_columns(row["training"]["epoch_metrics"]),old_train["summary"]["history"])
            self.assertEqual(artifacts.reconstruct_schedule(bundle,stride),proof["schedules"][stride])
            self.assertEqual(row["provenance"]["checkpoint"]["scientific_lineage"],old_train["lineage"])
        self.assertEqual(data["metadata"]["repository"]["git_commit"], "fixture")
        self.assertNotIn('"timeline"',json.dumps(data))
        self.assertNotIn('"waits"',json.dumps(data))

    def test_publish_separates_checkpoints_and_preserves_fresh_source(self):
        before=artifacts.inventory(self.source)
        data=artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        self.assertEqual(set(artifacts.inventory(self.destination)),artifacts.FORMAL_FILES)
        for stride in (3,5,10):
            old=f"models/stride_{stride}/predictor.pt"
            new=self.checkpoints/f"predictor_stride_{stride}.pt"
            self.assertEqual(sha256_file(new),before[old]["sha256"])
        self.assertEqual(len(list(self.checkpoints.iterdir())),3)
        for path in self.destination.rglob("*.json"):
            self.assertNotIn(".research_", path.read_text(), str(path))
            self.assertNotIn('"worker_log"', path.read_text(), str(path))
        self.assertEqual(data["metadata"]["execution"]["sequential_evaluation"]["jobs"],
                         [{"elapsed_seconds": 5}])
        for condition in [data["full_rgb"], *[row[kind] for row in data["strides"].values()
                                             for kind in ("sparse", "ours")]]:
            self.assertEqual(condition["performance_diagnostics"]["jobs"], [{"elapsed_seconds": 5}])
            self.assertEqual(condition["execution"]["elapsed_seconds"], 5)
        for row in data["strides"].values():
            self.assertEqual(row["training"]["development_extraction"],
                             {"elapsed_seconds": .123, "workers": [{"pid": 123}]})
        self.assertEqual((self.parent / ".research_anchor_budget_run/workers/job_000/worker.log").read_text(),
                         "runtime diagnostics")
        artifacts.validate_compact(self.destination)
        self.assertLess(len((self.destination/"SUMMARY.md").read_text().splitlines()),40)
        self.assertEqual(artifacts.inventory(self.source), before)
        self.assertEqual(artifacts.publish_compact(self.source,self.destination,self.checkpoints)["strides"],data["strides"])

    def test_figures_need_only_json_and_npz_and_do_not_refit(self):
        artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        package=self.parent/"standalone"
        package.mkdir()
        import shutil
        for filename in ("results.json","trajectories.npz"):
            shutil.copy2(self.destination/filename,package/filename)
        before={name:sha256_file(package/name) for name in ("results.json","trajectories.npz")}
        with patch("research.src.evaluation.fit_sim3_trajectory",side_effect=AssertionError("no refit")), \
             patch("research.src.evaluation.evaluate_paired_trajectory",side_effect=AssertionError("no evaluation")), \
             patch.object(torch,"load",side_effect=AssertionError("no checkpoint read")):
            render_figures(package)
        self.assertEqual(before,{name:sha256_file(package/name) for name in before})
        self.assertEqual({p.name for p in (package/"figures").iterdir()},{"trajectories.png","tradeoffs.png"})
        for p in (package/"figures").iterdir():
            self.assertTrue(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_unmapped_scientific_file_prevents_cleanup(self):
        (self.source/"unmapped_science.json").write_text('{"measurement": 17}')
        before=artifacts.inventory(self.source)
        with self.assertRaisesRegex(RuntimeError,"unmapped"):
            artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        self.assertEqual(before,artifacts.inventory(self.source))

    def test_duplicate_scientific_disagreement_prevents_cleanup(self):
        (self.source/"accuracy_vs_communication.json").write_text('[]')
        before=artifacts.inventory(self.source)
        with self.assertRaisesRegex(RuntimeError,"disagrees"):
            artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        self.assertEqual(before,artifacts.inventory(self.source))

    def test_existing_checkpoint_is_replaced_after_success(self):
        self.checkpoints.mkdir()
        old = self.checkpoints/"predictor_stride_3.pt"
        old.write_bytes(b"different model")
        artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        self.assertEqual(sha256_file(old), sha256_file(self.source/"models/stride_3/predictor.pt"))
        artifacts.validate_compact(self.destination)

    def test_new_stride_request_replaces_both_canonical_sets(self):
        artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        source = staging_fixture(self.parent / "new_request", (5, 7, 8))
        artifacts.publish_compact(source,self.destination,self.checkpoints)
        result = artifacts.validate_compact(self.destination)
        self.assertEqual(set(result["strides"]), {"5", "7", "8"})
        self.assertEqual({p.name for p in self.checkpoints.iterdir()}, {f"predictor_stride_{s}.pt" for s in (5,7,8)})
        self.assertEqual(set(result["metadata"]["trajectories"]), {"GT", "Full_RGB", *[f"stride_{s}_{c}" for s in (5,7,8) for c in ("sparse", "ours")]})
        source = staging_fixture(self.parent / "subset", (7,))
        artifacts.publish_compact(source,self.destination,self.checkpoints)
        self.assertEqual(set(artifacts.validate_compact(self.destination)["strides"]), {"7"})
        self.assertEqual({p.name for p in self.checkpoints.iterdir()}, {"predictor_stride_7.pt"})
        self.assertEqual(set(artifacts.inventory(self.destination)), artifacts.FORMAL_FILES)

    def test_render_failure_preserves_both_canonical_trees_and_cleans_staging(self):
        artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        before = (artifacts.inventory(self.destination), artifacts.inventory(self.checkpoints))
        with patch("research.src.anchor_budget_figures.render_figures", side_effect=RuntimeError("plot failed")):
            with self.assertRaisesRegex(RuntimeError, "plot failed"):
                artifacts.publish_compact(self.source,self.destination,self.checkpoints)
        self.assertEqual(before, (artifacts.inventory(self.destination), artifacts.inventory(self.checkpoints)))
        self.assertFalse(any("staging-" in p.name or "backup-" in p.name for p in self.parent.iterdir()))

    def test_stored_sim3_is_applied_without_changing_raw_arrays(self):
        value=np.arange(9,dtype=np.float64).reshape(3,3)
        before=value.copy()
        output=aligned_translation({"x__translation":value},"x",{"canonical_evaluation":{"sim3":{
            "scale":2.,"rotation":np.eye(3),"translation":[1,2,3]}}})
        self.assertTrue(np.array_equal(value,before))
        self.assertTrue(np.array_equal(output,value*2+np.asarray([1,2,3])))

    def test_preparation_does_not_read_existing_canonical_results(self):
        self.destination.mkdir()
        (self.destination / "results.json").write_text("invalid old result")
        protocol,canonical,path=runner.load_protocol()
        protocol=protocol | {"output_root":str(self.destination)}
        with patch.object(runner,"load_protocol",return_value=(protocol,canonical,path)), \
             patch.object(runner,"load_sequence_records",return_value=()), \
             patch.object(runner,"prepare_protocol",return_value=({}, {})), \
             patch.object(runner,"execute_sequence",side_effect=AssertionError("no experiment")):
            self.assertEqual(runner.run()["status"],"prepared_no_experiment_run")
        self.assertEqual((self.destination / "results.json").read_text(), "invalid old result")

    def test_offline_cli_actions_are_mutually_exclusive_with_execution(self):
        import contextlib,io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--execute","--consolidate-existing"])


if __name__ == "__main__":
    unittest.main()
