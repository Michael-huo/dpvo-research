"""Separate-environment JEPA and predictor CUDA workers; CPU shared-memory IPC only."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

from .execution_runtime import apply_cpu_profile, apply_runtime, runtime_provenance
from .staged_transfer import PinnedTransfer, shared_array


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def run(config, component):
    encoded_profile = os.environ.get("PHASE1_CPU_PROFILE")
    if encoded_profile:
        apply_cpu_profile(json.loads(encoded_profile))
    settings = config["worker_settings"]
    apply_runtime(settings)
    if component == "encoder":
        from .jepa_worker import _load
        _, model, original = _load(config)
        provenance = runtime_provenance(settings, component=component, model=model,
                                        amp=True, autocast_dtype="bfloat16") | {"jepa": original}
    else:
        from .h2_training import _new_predictor
        model = _new_predictor(config).cuda().eval().requires_grad_(False)
        checkpoint = torch.load(config["predictor_checkpoint"], map_location="cpu", weights_only=False)
        from .predictor import predictor_state_sha256
        if predictor_state_sha256(checkpoint["state_dict"]) != config["predictor_state_sha256"]:
            raise RuntimeError("predictor worker frozen state mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        provenance = runtime_provenance(settings, component=component, model=model, amp=False)
    transfers = PinnedTransfer(verify=config.get("verify_transfers", False))
    emit({"status": "ready", "provenance": provenance})
    for line in sys.stdin:
        request = json.loads(line); memories = []
        try:
            if request["action"] == "close":
                transfers.close()
                del model
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                emit({"status": "closed", "request_id": request["request_id"]})
                return
            if request["action"] == "barrier":
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                emit({"status": "ok", "request_id": request["request_id"],
                      "peak_vram_bytes": peak, "transfer": transfers.payload()})
                if request["request_id"] == "ready":
                    transfers.close()
                    transfers = PinnedTransfer(verify=config.get("verify_transfers", False))
                continue
            arrays = {}
            for name, descriptor in request["slots"].items():
                memory, array = shared_array(descriptor); memories.append(memory); arrays[name] = array
            with torch.inference_mode():
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                if component == "encoder":
                    batch = transfers.h2d(arrays["input"], "jepa_input")
                    start.record()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(batch)
                    # Original sidecar exports float16 and the consumer reads float32.
                    result = outputs[0].float().half().float()
                    end.record(); end.synchronize()
                else:
                    from .transport import (
                        RobustCorrespondence, decision_trace_payload,
                        estimate_robust_correspondence, robust_transport_interpolation,
                    )
                    from .jepa_fmap import coordinate_masks
                    from .jepa_fmap import FullFOVTransform
                    # Geometry travels as dataclass fields, not any dataset capability.
                    transform = FullFOVTransform(**config["transform"])
                    left = transfers.h2d(arrays["left"], "endpoint_left")
                    right = transfers.h2d(arrays["right"], "endpoint_right")
                    mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
                    start.record()
                    trace = {} if config.get("decision_trace", False) else None
                    corr = estimate_robust_correspondence(
                        left, right, mask, config["transport_calibration"],
                        decision_trace=trace,
                    )
                    count = len(request["alpha"])
                    repeated = RobustCorrespondence(*[getattr(corr, n).repeat(count, *([1]*(getattr(corr,n).ndim-1)))
                                                      for n in corr.__dataclass_fields__])
                    alpha = torch.tensor(request["alpha"], device="cuda")
                    delta = torch.tensor(request["delta"], device="cuda")
                    transported = robust_transport_interpolation(
                        left.repeat(count,1,1,1), right.repeat(count,1,1,1),
                        alpha, repeated, mask, decision_trace=trace,
                    )
                    result = model(transported.field, transported.warped_difference,
                                   transported.warp0.coverage, transported.warp1.coverage,
                                   transported.fused_confidence, alpha, delta)
                    end.record(); end.synchronize()
                destination = arrays["output"][:len(result)]
                transfers.d2h(result, destination, component+"_output")
            emit({"status": "ok", "request_id": request["request_id"],
                  "generations": {name: slot["generation"] for name, slot in request["slots"].items()},
                  "shape": list(result.shape), "dtype": str(result.dtype),
                  "device": str(result.device),
                  "compute_ms": float(start.elapsed_time(end)),
                  "decision_trace": (
                      decision_trace_payload(trace)
                      if component == "predictor" and trace is not None else None
                  )})
        finally:
            for memory in memories: memory.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--component", choices=["encoder", "predictor"], required=True)
    args = parser.parse_args()
    try:
        with open(args.config) as source: config = json.load(source)
        run(config, args.component)
    except BaseException as error:
        import traceback
        emit({"status": "error", "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
