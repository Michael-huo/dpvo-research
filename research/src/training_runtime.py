"""Native-FP16 resident training data for the formal H1/H2 paths."""
from __future__ import annotations

import gc
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def tensor_footprint(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    seen = set(); total = 0; rows = {}
    for name, value in values.items():
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes())
        unique = key not in seen
        if unique: total += storage.nbytes(); seen.add(key)
        rows[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                      "device": str(value.device), "bytes": value.numel()*value.element_size(),
                      "storage_bytes": storage.nbytes(), "unique_storage": unique}
    return {"tensors": rows, "unique_storage_bytes": total}


class ResidentRows:
    """Upload every explicitly allowed row once in canonical order."""
    def __init__(self, arrays: Mapping[str, np.ndarray], allowed_rows: Sequence[int], *,
                 device: torch.device):
        self.arrays = dict(arrays); self.device = torch.device(device)
        self.allowed = tuple(dict.fromkeys(int(row) for row in allowed_rows))
        self.allowed_set = set(self.allowed); self.closed = False
        self.resident: dict[str, torch.Tensor] = {}; self.index = {}
        allocated_before = self._memory_value(torch.cuda.memory_allocated)
        reserved_before = self._memory_value(torch.cuda.memory_reserved)
        per_row = sum(int(np.prod(a.shape[1:]))*a.dtype.itemsize for a in arrays.values())
        started = time.perf_counter()
        # A single native-dtype upload per array; ordering is train then validation.
        selected = self.allowed
        try:
            for name, array in arrays.items():
                host = torch.from_numpy(
                    np.array(array[list(selected)], copy=True),
                )
                try:
                    self.resident[name] = host.to(self.device)
                finally:
                    del host
            self.index = {row: i for i, row in enumerate(selected)}
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as error:
            try:
                cleanup = self._release_owned(synchronize=True)
            except BaseException as cleanup_error:
                cleanup = {
                    "resident_tensors_cleared": False,
                    "errors": [
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    ],
                }
            setattr(error, "phase1_resident_cleanup", cleanup)
            raise
        allocated_after = self._memory_value(torch.cuda.memory_allocated)
        reserved_after = self._memory_value(torch.cuda.memory_reserved)
        self.diagnostics = {"allowed_rows": len(self.allowed), "resident_rows": len(selected),
                            "native_bytes": per_row*len(self.allowed),
                            "resident_bytes": per_row*len(selected),
                            "initialization_seconds": time.perf_counter()-started,
                            "batch_h2d_bytes": 0, "batch_h2d_seconds": 0.,
                            "allocated_before_bytes": allocated_before,
                            "allocated_after_bytes": allocated_after,
                            "allocated_delta_bytes": self._delta(allocated_before, allocated_after),
                            "reserved_before_bytes": reserved_before,
                            "reserved_after_bytes": reserved_after,
                            "reserved_delta_bytes": self._delta(reserved_before, reserved_after),
                            "memory_telemetry_only": True}

    def _memory_value(self, function):
        if self.device.type != "cuda":
            return 0
        try:
            return int(function(self.device))
        except Exception:
            return 0

    @staticmethod
    def _delta(before, after):
        return None if before is None or after is None else after - before

    def _release_owned(self, *, synchronize: bool) -> dict[str, Any]:
        self.closed = True
        self.index.clear()
        self.resident.clear()
        errors = []
        cuda_initialized = False
        if self.device.type == "cuda":
            try:
                cuda_initialized = bool(torch.cuda.is_initialized())
            except Exception as error:
                errors.append(
                    f"CUDA initialization query: {type(error).__name__}: {error}"
                )
        if synchronize and cuda_initialized:
            try:
                torch.cuda.synchronize(self.device)
            except Exception as error:
                errors.append(f"synchronize: {type(error).__name__}: {error}")
        try:
            gc.collect()
        except Exception as error:
            errors.append(f"gc: {type(error).__name__}: {error}")
        if cuda_initialized:
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                errors.append(f"empty_cache: {type(error).__name__}: {error}")
        return {"resident_tensors_cleared": True, "errors": errors}

    def batch(self, name: str, rows: Sequence[int]) -> torch.Tensor:
        if self.closed: raise RuntimeError("resident store closed")
        rows = [int(r) for r in rows]
        if not set(rows) <= self.allowed_set: raise PermissionError("row outside train/validation capability")
        if not all(row in self.index for row in rows):
            raise RuntimeError("formal resident store is missing an allowed row")
        indices = torch.tensor([self.index[row] for row in rows], device=self.device)
        return self.resident[name].index_select(0, indices).float()

    def close(self):
        if not self.closed:
            self._release_owned(synchronize=False)


class ResidentH1View:
    def __init__(self, store, split_keys, *, device):
        self.store = store
        keys = [*split_keys["train"], *split_keys["validation"]]
        self.rows = ResidentRows({"block5": store.block5, "teacher": store.teacher},
                                  [store.index[k] for k in keys], device=device)

    def __getattr__(self, name): return getattr(self.store, name)
    def tensor_batch(self, name, rows): return self.rows.batch(name, rows)
    def close(self): self.rows.close()


class ResidentH2View:
    def __init__(self, store, intervals, *, device):
        self.store = store
        keys = list(dict.fromkeys(identity.key for interval in intervals for identity in
                                 (interval.anchor0, *[q.identity for q in interval.hidden], interval.anchor1)))
        self.rows = ResidentRows({"block5": store.values}, [store.index[k] for k in keys],
                                  device=device)
        self.allowed = set(keys)

    def __getattr__(self, name): return getattr(self.store, name)
    def get_tensor(self, identity):
        if self.store.closed: raise RuntimeError("underlying feature store closed")
        key = identity.key if hasattr(identity, "key") else identity
        if key not in self.allowed: raise PermissionError("feature outside development capability")
        return self.rows.batch("block5", [self.store.index[key]])[0]
    def close(self): self.rows.close()


def h1_tensor_batch(store, name, rows):
    if hasattr(store, "tensor_batch"): return store.tensor_batch(name, rows)
    return torch.from_numpy(np.asarray(getattr(store, name)[rows], np.float32)).cuda()


def resident_correspondence(store, device):
    """Preserve the original half quantization, including boolean fields stored as half."""
    from .h2_training import RobustCorrespondenceStore
    from .transport import RobustCorrespondence
    result = RobustCorrespondenceStore(store.calibration)
    partial = []
    try:
        for key, row in store.rows.items():
            partial = []
            for name in row.__dataclass_fields__:
                partial.append(getattr(row, name).to(device))
            result.rows[key] = RobustCorrespondence(*partial)
            partial = []
    except torch.cuda.OutOfMemoryError as error:
        partial.clear()
        result.rows.clear()
        cleanup_errors = []
        cuda_initialized = False
        if torch.device(device).type == "cuda":
            try:
                cuda_initialized = bool(torch.cuda.is_initialized())
            except Exception as cleanup_error:
                cleanup_errors.append(
                    "CUDA initialization query: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if cuda_initialized:
            try:
                torch.cuda.synchronize(device)
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"synchronize: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        try:
            gc.collect()
        except Exception as cleanup_error:
            cleanup_errors.append(
                f"gc: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        if cuda_initialized:
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"empty_cache: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        setattr(error, "phase1_correspondence_resident_cleanup", {
            "resident_tensors_cleared": True,
            "errors": cleanup_errors,
        })
        raise
    return result
