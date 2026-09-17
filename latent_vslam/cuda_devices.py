"""Logical CUDA resources. Physical identities are optional telemetry only."""
from __future__ import annotations

import ctypes
import dataclasses
import os
import uuid

import torch


ONLINE_DEVICE_ERROR = (
    "current online runtime requires 3 visible CUDA devices; "
    "single-GPU runtime will be evaluated separately"
)


@dataclasses.dataclass(frozen=True)
class CudaDevicePool:
    device_count: int

    def __post_init__(self):
        if self.device_count < 0:
            raise ValueError("visible CUDA count must be nonnegative")

    @classmethod
    def discover(cls):
        return cls(int(torch.cuda.device_count()))

    @property
    def visible_devices(self):
        return tuple(f"cuda:{i}" for i in range(self.device_count))

    @property
    def logical_ids(self):
        return tuple(range(self.device_count))

    @property
    def primary_device(self):
        self.require(0)
        return self.visible_devices[0]

    @property
    def worker_devices(self):
        return self.visible_devices[1:]

    @property
    def preparation_devices(self):
        self.require(0)
        return self.logical_ids

    def require(self, logical_id):
        if int(logical_id) not in self.logical_ids:
            raise RuntimeError(f"logical CUDA device {logical_id} unavailable; visible_count={self.device_count}")
        return int(logical_id)

    def online_mapping(self):
        if self.device_count < 3:
            raise RuntimeError(ONLINE_DEVICE_ERROR)
        return {"stage_c": self.visible_devices[0],
                "predictor": self.visible_devices[1], "encoder": self.visible_devices[2]}

    def provenance(self):
        return {"visible_cuda_count": self.device_count,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "visible_devices": list(self.visible_devices),
                "primary_device": self.primary_device if self.device_count else None,
                "worker_devices": list(self.worker_devices),
                "preparation_devices": list(self.visible_devices),
                "preparation_mode": "serial" if self.device_count == 1 else "sharded",
                "online_runtime_mapping": self.online_mapping() if self.device_count >= 3 else None,
                "online_runtime_requires_visible_devices": 3}


def select_worker_device(logical_id):
    """Called in a fresh process before model construction or CUDA allocation."""
    pool = CudaDevicePool.discover()
    device = pool.require(logical_id)
    torch.cuda.set_device(device)
    return device


def validate_worker_binding(runtime, logical_id, pool=None):
    pool = pool or CudaDevicePool.discover()
    if (not isinstance(runtime, dict)
            or runtime.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")
            or runtime.get("logical_cuda_device_count") != pool.device_count
            or runtime.get("current_logical_cuda_device") != int(logical_id)):
        raise RuntimeError(f"worker device binding mismatch: logical={logical_id}, runtime={runtime}")


def logical_device_telemetry(count, physical_devices=()):
    """Query driver identities without retaining/creating any CUDA context.

    cuDevice ordinals respect the inherited visibility mask and CUDA ordering.
    Never guess a physical ordinal from a numeric visibility selector.
    """
    rows = [{"logical_index": i, "logical_device": f"cuda:{i}",
             "physical_index": None, "uuid": None, "pci_bus_id": None,
             "name": None, "mapping_basis": "unavailable", "mapping_error": None}
            for i in range(count)]
    if not rows:
        return rows
    try:
        driver = ctypes.CDLL("libcuda.so.1")

        def checked(function, *args):
            status = function(*args)
            if status:
                raise RuntimeError(f"CUDA driver telemetry status {status}")

        checked(driver.cuInit, 0)
        driver_count = ctypes.c_int()
        checked(driver.cuDeviceGetCount, ctypes.byref(driver_count))
        if driver_count.value != count:
            raise RuntimeError("driver and torch visible counts differ")
        for row in rows:
            device = ctypes.c_int()
            checked(driver.cuDeviceGet, ctypes.byref(device), row["logical_index"])
            raw_uuid = (ctypes.c_ubyte * 16)()
            checked(driver.cuDeviceGetUuid, ctypes.byref(raw_uuid), device)
            bus, name = ctypes.create_string_buffer(32), ctypes.create_string_buffer(256)
            checked(driver.cuDeviceGetPCIBusId, bus, len(bus), device)
            checked(driver.cuDeviceGetName, name, len(name), device)
            identity = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
            physical = next((p for p in physical_devices if p.get("uuid") == identity), {})
            row.update(physical)
            row.update(uuid=identity, pci_bus_id=bus.value.decode(), name=name.value.decode(),
                       mapping_basis="CUDA_driver_UUID_PCI_no_context")
    except Exception as error:
        for row in rows:
            row["mapping_error"] = f"{type(error).__name__}: {error}"
    return rows
