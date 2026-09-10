"""Three-GPU preparation and canonical sequential GPU0 trajectory runtime."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .execution_runtime import (
    apply_cpu_profile, apply_runtime, capture_runtime,
    release_cuda_training_state, require_lifecycle_cleanup, runtime_provenance,
)
from .protocol import REPO_ROOT, canonical_sha256, atomic_write_json


FORMAL_DEVICES = (0, 1, 2)


def shard_batches(values, batch_size, devices):
    if batch_size<=0 or not devices or len(set(devices))!=len(devices):
        raise ValueError("nonempty distinct devices and positive batch size required")
    shards=[[] for _ in devices]
    for ordinal,start in enumerate(range(0,len(values),batch_size)):
        shards[ordinal%len(shards)].extend(values[start:start+batch_size])
    return shards


def merge_identity_rows(expected_keys, shards):
    if len(set(expected_keys))!=len(expected_keys): raise ValueError("duplicate expected identity")
    merged={}
    for rows in shards:
        for key,value in rows:
            if key in merged: raise RuntimeError(f"duplicate shard identity: {key}")
            if key not in expected_keys: raise RuntimeError(f"unknown shard identity: {key}")
            merged[key]=value
    if set(merged)!=set(expected_keys): raise RuntimeError("incomplete shard identity population")
    values=[merged[key] for key in expected_keys]
    hashes=[hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in values]
    return values,{"identity_sha256":canonical_sha256(expected_keys),
                   "ordered_content_sha256":canonical_sha256(list(zip(expected_keys,hashes))),
                   "content_sha256_by_identity":dict(zip(expected_keys,hashes))}


def _launch(task,path,device):
    torch.save(task,path/"task.pt")
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(device)
    command=[sys.executable,"-m","research.src.phase1_feasibility.parallel_runtime",
             str(path/"task.pt")]
    cpu_profile=task.get("cpu_profile")
    if cpu_profile is not None:
        cpus=",".join(str(value) for value in cpu_profile["cpus"])
        env["OMP_NUM_THREADS"]=str(cpu_profile["omp_num_threads"])
        env["MKL_NUM_THREADS"]=str(cpu_profile["mkl_num_threads"])
        env["PHASE1_CPU_PROFILE"]=json.dumps(cpu_profile,sort_keys=True)
        if shutil.which("numactl") and cpu_profile.get("numa_node") is not None:
            command=["numactl",f"--physcpubind={cpus}",
                     f"--membind={int(cpu_profile['numa_node'])}",*command]
        elif shutil.which("taskset"):
            command=["taskset","--cpu-list",cpus,*command]
        else:
            raise RuntimeError("NUMA/CPU binding requires numactl or taskset")
    with (path/"worker.log").open("w") as log:
        subprocess.run(command,cwd=REPO_ROOT,env=env,stdout=log,
                       stderr=subprocess.STDOUT,check=True)
    return path


def _gpu_process_telemetry():
    try:
        from .efficiency_profiling import gpu_process_snapshot
        return gpu_process_snapshot()
    except Exception as error:
        return {
            "rows": [],
            "error": f"{type(error).__name__}: {error}",
            "telemetry_only": True,
        }


def _validate_isolated_worker_binding(runtime, requested_device):
    if (
        not isinstance(runtime, dict)
        or runtime.get("cuda_visible_devices") != str(requested_device)
        or runtime.get("logical_cuda_device_count") != 1
        or runtime.get("current_logical_cuda_device") != 0
    ):
        raise RuntimeError(
            "formal worker device binding mismatch: "
            f"requested={requested_device}, runtime={runtime}"
        )


def run_sequential_trajectory_jobs(tasks, temporary, *, cpu_profile=None, hardware):
    """Run formal trajectories one at a time in fresh GPU0 processes."""
    temporary = Path(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    timeline = []
    for ordinal, task in enumerate(tasks):
        required_devices = (
            FORMAL_DEVICES if task["kind"] == "formal_h2_predicted" else (0,)
        )
        before = _gpu_process_telemetry()
        path = temporary / f"job_{ordinal:03d}"
        path.mkdir()
        value = dict(task) | {
            "settings": task.get("settings", capture_runtime()),
            "physical_device": 0,
            "submitted_ns": time.monotonic_ns(),
            "cpu_profile": cpu_profile,
        }
        _launch(value, path, 0)
        payload = torch.load(path / "result.pt", map_location="cpu", weights_only=False)
        _validate_isolated_worker_binding(payload.get("worker_runtime"), 0)
        cleanup = payload.get("cleanup", {})
        require_lifecycle_cleanup(cleanup)
        started_ns = payload.pop("started_ns")
        completed_ns = payload.pop("completed_ns")
        record = {
            "kind": value["kind"],
            "sequence": value.get("sequence", payload.get("sequence")),
            "condition": value.get("condition", payload.get("condition")),
            "physical_device": 0,
            "submitted_ns": value["submitted_ns"],
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "elapsed_seconds": (completed_ns - started_ns) / 1e9,
            "worker_log": str(path / "worker.log"),
            "runtime": payload.get("worker_runtime"),
            "processes_before": before,
            "required_physical_devices": list(required_devices),
            "gpu_process_telemetry_only": True,
            "cleanup": cleanup,
        }
        payload["sequential_execution"] = record
        rows.append(payload)
        timeline.append(record)
        after = _gpu_process_telemetry()
        record["processes_after"] = after
    return rows, {
        "schema": "phase1_sequential_gpu0_trajectory_execution_v1",
        "device": 0,
        "job_count": len(tasks),
        "makespan_seconds": time.perf_counter() - started,
        "jobs": timeline,
        "maximum_concurrent_dpvo_instances": 1,
        "condition_concurrency": False,
        "sequence_concurrency": False,
    }


def extract_parallel(records, identities, calibration, config, temporary, transform, *, devices,
                     h1_hidden_keys=None):
    """Whole original JEPA batches are indivisible sharding units."""
    from .jepa_runtime import CompactFeatureStore
    from .h1_training import FeatureStore
    temporary=Path(temporary);temporary.mkdir(parents=True,exist_ok=True)
    values=records if h1_hidden_keys is not None else identities
    shards=shard_batches(values,int(config["jepa"]["batch_size"]),devices)
    settings=capture_runtime(); started=time.perf_counter(); jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for ordinal,(device,shard) in enumerate(zip(devices,shards)):
            if not shard: continue
            path=temporary/f"shard_{ordinal}";path.mkdir()
            keys={v.identity.key if hasattr(v,"identity") else v.key for v in shard}
            task={"kind":"h1" if h1_hidden_keys is not None else "h2","settings":settings,
                  "physical_device":device,
                  "records":shard if h1_hidden_keys is not None else [r for r in records if r.identity.key in keys],
                  "identities":shard,"hidden_keys":h1_hidden_keys,"calibration":calibration,
                  "config":config,"transform":transform}
            jobs.append(pool.submit(_launch,task,path,device))
        paths=[future.result() for future in jobs]
    h1=h1_hidden_keys is not None
    store=(FeatureStore(temporary/"merged",records,transform) if h1 else
           CompactFeatureStore(temporary/"merged","block5",identities,(transform.token_grid_height*transform.token_grid_width,768)))
    seen=set(); hashes={}; worker_provenance=[]
    for path in paths:
        meta=json.loads((path/"manifest.json").read_text());worker_provenance.append(meta["runtime"])
        tokens=np.load(path/"block5.npy",mmap_mode="r")
        teachers=np.load(path/"teacher.npy",mmap_mode="r") if h1 else None
        token_index={key:index for index,key in enumerate(meta.get("token_keys",meta["keys"]))}
        for i,key in enumerate(meta["keys"]):
            if key in seen or key not in store.index: raise RuntimeError("duplicate/unknown extraction shard identity")
            seen.add(key); row=store.index[key]
            names={"teacher":teachers[i]} if h1 else {"block5":tokens[token_index[key]]}
            if h1 and key in token_index: names["block5"]=tokens[token_index[key]]
            for name,value in names.items():
                if value.dtype!=np.float16: raise RuntimeError("shard native dtype changed")
                digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                if digest!=meta["hashes"][key][name]: raise RuntimeError("shard payload hash mismatch")
                if h1:
                    getattr(store,name)[row]=value
                else: store.put(store.identities[row],value)
            if h1:
                store.teacher_written[row]=True;store.token_written[row]=key in h1_hidden_keys
            hashes[key]=meta["hashes"][key]
    keys=[r.identity.key for r in records] if h1 else [i.key for i in identities]
    if seen!=set(keys): raise RuntimeError("incomplete extraction shard merge")
    meta=store.finalize(set(h1_hidden_keys)) if h1 else store.finalize()
    return store,{"store":meta,"elapsed_seconds":time.perf_counter()-started,
        "identity_sha256":canonical_sha256(keys),"ordered_content_sha256":canonical_sha256([(k,hashes[k]) for k in keys]),
        "workers":worker_provenance,"batch_membership_preserved":True,"domain":"research_throughput"}


class ReadonlyFeatureRows:
    def __init__(self, descriptor):
        self.values=np.memmap(descriptor["path"],mode="r",dtype=np.float16,shape=tuple(descriptor["shape"]))
        self.index=descriptor["index"]
    def get(self,identity): return np.asarray(self.values[self.index[identity.key]],np.float32)


class ReadonlyH1FeatureStore:
    """Open an H1 temporary store read-only in independent condition workers."""
    def __init__(self, descriptor):
        self.block5 = np.memmap(
            descriptor["block5_path"], mode="r", dtype=np.float16,
            shape=tuple(descriptor["block5_shape"]),
        )
        self.teacher = np.memmap(
            descriptor["teacher_path"], mode="r", dtype=np.float16,
            shape=tuple(descriptor["teacher_shape"]),
        )
        self.index = dict(descriptor["index"])
        self.identity_keys = list(descriptor["identity_keys"])

    def close(self):
        for value in (self.block5, self.teacher):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def correspondence_parallel(intervals,store,transform,mask,calibration,temporary,*,devices):
    from .h2_training import RobustCorrespondenceStore
    from .transport import RobustCorrespondence
    path=Path(temporary);path.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    tasks=shard_batches(intervals,1,devices)
    descriptor={"path":str(store.values.filename),"shape":list(store.values.shape),"index":store.index}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[]
        for i,(device,rows) in enumerate(zip(devices,tasks)):
            if not rows:continue
            directory=path/f"correspondence_{i}";directory.mkdir()
            task={"kind":"correspondence","settings":capture_runtime(),"intervals":rows,
                  "store":descriptor,"transform":transform,"calibration":calibration,
                  "mask":mask.cpu()}
            futures.append(pool.submit(_launch,task,directory,device))
        paths=[future.result() for future in futures]
    result=RobustCorrespondenceStore(calibration);expected=[i.interval_index for i in intervals]
    by_key={}
    for directory in paths:
        data=torch.load(directory/"correspondence.pt",map_location="cpu",weights_only=False)
        for key,value in data.items():
            if key in by_key or key not in expected:raise RuntimeError("invalid correspondence shard identity")
            by_key[key]=value
    if set(by_key)!=set(expected):raise RuntimeError("missing correspondence shard")
    result.rows={key:by_key[key] for key in expected}
    hashes={str(key):{name:hashlib.sha256(getattr(by_key[key],name).numpy().tobytes()).hexdigest()
                      for name in RobustCorrespondence.__dataclass_fields__} for key in expected}
    return result,{"interval_count":len(intervals),"endpoint_only":True,"contains_hidden_target":False,
                   "calibration_sha256":canonical_sha256(calibration),"elapsed_seconds":time.perf_counter()-started,
                   "ordered_content_sha256":canonical_sha256([(key,hashes[str(key)]) for key in expected]),
                   "domain":"research_throughput"}


def worker(task_path):
    task=torch.load(task_path,map_location="cpu",weights_only=False);path=Path(task_path).parent
    if task.get("cpu_profile") is not None:
        apply_cpu_profile(task["cpu_profile"])
    apply_runtime(task["settings"])
    started_ns = time.monotonic_ns()
    if task["kind"] == "materialize_schedule":
        from .runtime import materialize_schedule
        schedule = materialize_schedule(
            task["records"], task["calibration"], task["config"],
        )
        cleanup = release_cuda_training_state()
        torch.save({
            "sequence": task["records"][0].identity.sequence,
            "condition": "bootstrap_schedule", "schedule": schedule,
            "worker_runtime": runtime_provenance(
                task["settings"], component="gpu0_bootstrap_schedule",
            ),
            "cleanup": cleanup, "started_ns": started_ns,
            "completed_ns": time.monotonic_ns(),
        }, path / "result.pt")
        return
    if task["kind"] == "formal_h0_condition":
        from .efficiency_profiling import (
            PersistentPerformanceAudit, performance_diagnosis,
        )
        from .run_h0 import _run_condition
        with PersistentPerformanceAudit(
            "h0_state", f"{task['sequence']}:{task['condition']}",
            components=("dpvo_condition_worker",),
        ) as performance:
            with performance.phase("condition_execution"):
                result = _run_condition(
                    task["condition"], task["records"], task["calibration"],
                    task["config"], task["roles"], path / "condition_work",
                )
        perf = performance.payload()
        perf["diagnosis"] = performance_diagnosis(perf)
        cleanup = release_cuda_training_state()
        torch.save({"sequence": task["sequence"], **result,
                    "worker_runtime": runtime_provenance(
                        task["settings"], component="h0_formal_evaluation",
                    ),
                    "performance": perf, "cleanup": cleanup, "started_ns": started_ns,
                    "completed_ns": time.monotonic_ns()},
                   path / "result.pt")
        return
    if task["kind"] == "formal_h1_condition":
        from .efficiency_profiling import (
            PersistentPerformanceAudit, performance_diagnosis,
        )
        from .jepa_fmap import build_bridge
        from .run_h1 import _run_condition
        condition = task["condition"]
        store = None
        if condition in {"true_fmap", "oracle_jepa_bridge"}:
            store = ReadonlyH1FeatureStore(task["store"])
        model = None
        if condition == "oracle_jepa_bridge":
            checkpoint = torch.load(
                task["checkpoint"], map_location="cpu", weights_only=False,
            )
            model = build_bridge(
                task["transform"],
                channels=int(task["config"]["bridge"]["hidden_channels"]),
            ).cuda().eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.requires_grad_(False)
        with PersistentPerformanceAudit(
            "h1_interface", f"{task['sequence']}:{condition}",
            components=("dpvo_bridge_condition_worker",),
        ) as performance:
            with performance.phase("condition_execution"):
                result = _run_condition(
                    condition, task["records"], task["calibration"], task["roles"],
                    store, model, task["config"], path,
                )
        perf = performance.payload()
        perf["diagnosis"] = performance_diagnosis(perf)
        if store is not None:
            store.close()
        worker_runtime = runtime_provenance(
            task["settings"], component=f"h1_{condition}", model=model,
        )
        if model is not None:
            model.cpu()
        cleanup = release_cuda_training_state(store)
        torch.save({
            "sequence": task["sequence"], **result, "performance": perf,
            "worker_runtime": worker_runtime,
            "cleanup": cleanup, "started_ns": started_ns,
            "completed_ns": time.monotonic_ns(),
        }, path / "result.pt")
        return
    if task["kind"] in {"formal_h2_full", "formal_h2_sparse"}:
        from .runtime import (
            OnlineFrame, run_formal_mode, sanitize_full_oracle_frames,
            warmup_dpvo_frontend,
        )
        records = task["records"]
        calibration = task["calibration"]
        config = task["config"]
        if task["kind"] == "formal_h2_full":
            observations = [OnlineFrame(row.identity, row.rgb_path) for row in records]
            roles = {row.identity.key: "anchor" for row in records}
            condition = "full_rgb_reference"
            mode = "matched_full_rgb"
        else:
            observations = sanitize_full_oracle_frames(records, task["roles"])
            roles = task["roles"]
            condition = "sparse_rgb_reference"
            mode = "sparse_rgb"
        warmup = warmup_dpvo_frontend(
            records[0], calibration, config, packet_runtime=False,
        )
        runtime, arrays = run_formal_mode(
            mode, observations, calibration, config, roles=roles,
            condition_name=condition, matched_timing=True,
            profile_graph_runtime=True, collect_graph_trace=False,
        )
        cleanup = release_cuda_training_state()
        torch.save({
            "sequence": records[0].identity.sequence,
            "condition": condition, "runtime": runtime, "arrays": arrays,
            "worker_runtime": runtime_provenance(
                task["settings"], component=f"h2_{condition}",
            ),
            "warmup": warmup, "cleanup": cleanup, "started_ns": started_ns,
            "completed_ns": time.monotonic_ns(),
        }, path / "result.pt")
        return
    if task["kind"] == "formal_h2_representation_control":
        from .h2_training import (
            _field, _plot_feature_diagnostics, build_robust_correspondence_store,
        )
        from .jepa_fmap import build_bridge, coordinate_masks
        from .jepa_runtime import extract_block5_store, extract_true_fmap_store
        from .runtime import run_packet_observations
        from .run_h2 import _store_observations
        from .h2_training import _new_predictor
        records = task["records"]
        identities = [row.identity for row in records]
        roles = task["roles"]
        transform = task["transform"]
        config = task["config"]
        calibration = task["calibration"]
        setup_started = time.perf_counter()
        bridge = build_bridge(
            transform, channels=int(config["bridge"]["hidden_channels"]),
        ).cuda().eval().requires_grad_(False)
        bridge.load_state_dict(task["bridge_state"], strict=True)
        model_setup_seconds = time.perf_counter() - setup_started
        predictor = None
        work = path / "references"
        preparation_started = time.perf_counter()
        all_store, all_extraction = extract_block5_store(
            records, identities, calibration, config, work, transform,
        )
        block5_preparation_seconds = time.perf_counter() - preparation_started
        condition = task["condition"]
        if condition not in {"anchor_jepa_only", "oracle_jepa_hidden_reference"}:
            raise ValueError(condition)
        include_hidden = condition == "oracle_jepa_hidden_reference"
        height, width = transform.source_height, transform.source_width
        trajectory_started = time.perf_counter()
        runtime, arrays = run_packet_observations(
            _store_observations(
                identities, roles, all_store, transform, bridge,
                include_hidden=include_hidden,
            ),
            calibration, config, image_height=height, image_width=width,
            condition_name=condition,
        )
        trajectory_call_seconds = time.perf_counter() - trajectory_started
        diagnostics = diagnostic_path = true_extraction = robust_meta = None
        diagnostics_started = time.perf_counter()
        if include_hidden:
            predictor = _new_predictor(config).cuda().eval().requires_grad_(False)
            predictor.load_state_dict(task["predictor_state"], strict=True)
            hidden = tuple(row.identity for row in records if roles[row.identity.key] == "hidden")
            true_store, true_extraction = extract_true_fmap_store(
                records, hidden, calibration, config, work / "true", transform,
            )
            mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
            robust, robust_meta = build_robust_correspondence_store(
                task["intervals"], all_store, transform, mask, task["thresholds"],
            )
            diagnostic_path = path / "feature_diagnostics.png"
            diagnostics = _plot_feature_diagnostics(
                diagnostic_path, records[0].identity.sequence, task["intervals"],
                all_store, true_store, transform, bridge, robust, predictor, config,
            )
            true_store.close()
            del mask, robust, true_store
        offline_diagnostics_seconds = time.perf_counter() - diagnostics_started
        all_store.close()
        worker_runtime = runtime_provenance(
            task["settings"], component=f"h2_{condition}", model=bridge,
        )
        bridge.cpu()
        if predictor is not None:
            predictor.cpu()
        cleanup_started = time.perf_counter()
        cleanup = release_cuda_training_state()
        cleanup_seconds = time.perf_counter() - cleanup_started
        completed_ns = time.monotonic_ns()
        complete_worker_seconds = (completed_ns - started_ns) / 1e9
        named_seconds = (
            model_setup_seconds + block5_preparation_seconds
            + trajectory_call_seconds + offline_diagnostics_seconds
            + cleanup_seconds
        )
        timing_scopes = {
            "schema": "phase1_h2_condition_timing_scopes_v1",
            "model_and_checkpoint_setup_seconds": model_setup_seconds,
            "offline_block5_preparation_seconds": block5_preparation_seconds,
            "offline_preparation_and_diagnostics_seconds": (
                block5_preparation_seconds + offline_diagnostics_seconds
            ),
            "matched_trajectory_seconds": float(runtime["elapsed_seconds"]),
            "matched_trajectory_call_seconds": trajectory_call_seconds,
            "offline_diagnostics_seconds": offline_diagnostics_seconds,
            "cleanup_seconds": cleanup_seconds,
            "complete_condition_worker_seconds": complete_worker_seconds,
            "unattributed_worker_overhead_seconds": max(
                0.0, complete_worker_seconds - named_seconds,
            ),
            "matched_timing_source": "runtime.elapsed_seconds",
            "complete_worker_excludes_result_serialization": True,
        }
        torch.save({
            "sequence": records[0].identity.sequence,
            "condition": condition,
            "worker_runtime": worker_runtime,
            "runtime": runtime, "arrays": arrays,
            "diagnostics": diagnostics,
            "diagnostic_path": str(diagnostic_path) if diagnostic_path else None,
            "offline_reference": {
                "block5_extraction": all_extraction,
                "true_fmap_extraction": true_extraction,
                "robust_correspondence": robust_meta,
            },
            "condition_timing_scopes": timing_scopes,
            "cleanup": cleanup,
            "started_ns": started_ns, "completed_ns": completed_ns,
        }, path / "result.pt")
        return
    if task["kind"] == "formal_h2_predicted":
        from .efficiency_profiling import PerformanceRecorder, TransferLedger
        from .h2_training import _new_predictor
        from .jepa_fmap import build_bridge
        from .run_h2 import _run_strict_replay
        from .execution_runtime import FormalExecution
        bridge = build_bridge(
            task["transform"], channels=int(task["config"]["bridge"]["hidden_channels"]),
        ).cuda().eval().requires_grad_(False)
        bridge.load_state_dict(task["bridge_state"], strict=True)
        predictor = _new_predictor(task["config"]).cuda().eval().requires_grad_(False)
        predictor.load_state_dict(task["predictor_state"], strict=True)
        recorder = PerformanceRecorder(enable_cuda=True)
        ledger = TransferLedger()
        runtime, arrays, online_profile = _run_strict_replay(
            task["records"], task["roles"], task["intervals"], task["calibration"],
            task["transform"], bridge, predictor, task["thresholds"], task["config"],
            path / "strict_online", performance=recorder, transfer_ledger=ledger,
            predictor_checkpoint=Path(task["predictor_checkpoint"]),
            predictor_state_hash=task["predictor_state_hash"],
            execution=FormalExecution(**task.get("execution", {})),
            cpu_profile=task.get("pipeline_cpu_profile"),
        )
        worker_runtime = runtime_provenance(
            task["settings"], component="h2_stage_c_native_frontend_bridge_dpvo",
            model=bridge,
        )
        bridge.cpu(); predictor.cpu()
        cleanup = release_cuda_training_state()
        torch.save({
            "sequence": task["records"][0].identity.sequence,
            "condition": "predicted_jepa_hidden", "runtime": runtime,
            "arrays": arrays, "online_profile": online_profile,
            "worker_runtime": worker_runtime, "cleanup": cleanup,
            "started_ns": started_ns, "completed_ns": time.monotonic_ns(),
        }, path / "result.pt")
        return
    if task["kind"]=="correspondence":
        from .h2_training import build_robust_correspondence_store
        store,meta=build_robust_correspondence_store(task["intervals"],ReadonlyFeatureRows(task["store"]),
                            task["transform"],task["mask"].cuda(),task["calibration"])
        torch.save(store.rows,path/"correspondence.pt");atomic_write_json(path/"manifest.json",meta);return
    if task["kind"]=="h1":
        from .h1_training import extract_feature_store
        store,_=extract_feature_store(task["records"],set(task["hidden_keys"]),task["calibration"],
                                      task["config"],path,task["transform"])
        keys=store.identity_keys
        token_keys=[key for key in keys if key in task["hidden_keys"]]
        tokens=np.stack([store.block5[store.index[key]] for key in token_keys])
        np.save(path/"teacher.npy",store.teacher)
    else:
        from .jepa_runtime import extract_block5_store
        store,_=extract_block5_store(task["records"],task["identities"],task["calibration"],
                                     task["config"],path,task["transform"])
        keys=[i.key for i in store.identities];token_keys=keys;tokens=store.values
    np.save(path/"block5.npy",tokens)
    hashes={key:{} for key in keys}
    for i,key in enumerate(token_keys):hashes[key]["block5"]=hashlib.sha256(tokens[i].tobytes()).hexdigest()
    if task["kind"]=="h1":
        for i,key in enumerate(keys):hashes[key]["teacher"]=hashlib.sha256(store.teacher[i].tobytes()).hexdigest()
    worker_runtime=runtime_provenance(task["settings"],component="offline_extraction",amp=True)
    worker_runtime["physical_device"]=task.get("physical_device")
    worker_runtime["logical_cuda_ordinal"]=0
    atomic_write_json(path/"manifest.json",{"keys":keys,"token_keys":token_keys,"hashes":hashes,
        "runtime":worker_runtime})
    store.close()


if __name__=="__main__":worker(sys.argv[1])
