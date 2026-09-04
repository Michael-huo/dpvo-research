"""Full RGB-derived DPVO frontend packets for the Exp6 upper-bound control.

The packet deliberately stops at the Patchifier boundary.  In particular it
does not contain pose, inverse depth, recurrent, factor, BA, trajectory, or
ground-truth state.  Patch depth is reconstructed by the online DPVO graph.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .protocol import FrameIdentity, canonical_sha256


PACKET_NAME = "FullOracleHiddenFrontendPacket"
PACKET_SCHEMA = "FullOracleHiddenFrontendPacketV1"
PACKET_FIELDS = ("fmap", "gmap", "imap", "patch_xy", "colors")
ZERO_PACKET_NAME = "FMapZeroContextPacket"
ZERO_PACKET_SCHEMA = "FMapZeroContextPacketV2"
ZERO_PACKET_FIELDS = ("fmap",)
PACKET_SCHEMAS = {
    PACKET_SCHEMA: (PACKET_NAME, PACKET_FIELDS),
    ZERO_PACKET_SCHEMA: (ZERO_PACKET_NAME, ZERO_PACKET_FIELDS),
}
FORBIDDEN_PACKET_FIELDS = frozenset({
    "depth", "inverse_depth", "pose", "pose_seed", "net", "recurrent_state",
    "factor", "target", "weight", "ba", "trajectory", "groundtruth", "gt",
    "rgb", "rgb_path", "image_path",
})


def frontend_seed(identity: FrameIdentity, experiment_seed: int) -> int:
    payload = f"exp6-full-oracle-v1{int(experiment_seed)}{identity.key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@contextlib.contextmanager
def frontend_rng_scope(seed: int) -> Iterator[None]:
    """Temporarily install a frame-local frontend RNG without advancing globals."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


@dataclass(frozen=True)
class FullOracleHiddenFrontendPacket:
    fmap: Any
    gmap: Any
    imap: Any
    patch_xy: Any
    colors: Any

    def __post_init__(self) -> None:
        names = set(vars(self))
        if names != set(PACKET_FIELDS):
            raise ValueError("frontend packet schema changed")
        if names & FORBIDDEN_PACKET_FIELDS:
            raise ValueError("frontend packet contains forbidden graph/future state")

    def tensors(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in PACKET_FIELDS}




@dataclass(frozen=True)
class FMapZeroContextPacket:
    fmap: Any

    def tensors(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ZERO_PACKET_FIELDS}


def representation_contract(schema: str) -> dict[str, list[str]]:
    if schema == PACKET_SCHEMA:
        return {
            "stored": list(PACKET_FIELDS), "derived": ["fmap2"],
            "synthetic": [], "removed": [],
        }
    if schema == ZERO_PACKET_SCHEMA:
        return {
            "stored": list(ZERO_PACKET_FIELDS),
            "derived": ["patch_xy", "gmap", "fmap2"],
            "synthetic": ["imap_zero", "colors_zero_buffer_compatibility"],
            "removed": ["imap", "colors"],
        }
    raise ValueError(f"unsupported frontend packet schema: {schema}")


def extract_frontend_packet(slam: Any, image: Any, identity: FrameIdentity, experiment_seed: int) -> FullOracleHiddenFrontendPacket:
    """Run exactly the online Patchifier preprocessing under identity RNG."""
    import torch

    from ..exp3.oracle import normalize_rgb

    normalized = normalize_rgb(image)
    seed = frontend_seed(identity, experiment_seed)
    with torch.no_grad(), frontend_rng_scope(seed), torch.cuda.amp.autocast(enabled=bool(slam.cfg.MIXED_PRECISION)):
        fmap, gmap, imap, patches, _, colors = slam.network.patchify(
            normalized,
            patches_per_image=int(slam.M),
            centroid_sel_strat=slam.cfg.CENTROID_SEL_STRAT,
            return_color=True,
        )
    return FullOracleHiddenFrontendPacket(
        fmap=fmap.detach(),
        gmap=gmap.detach(),
        imap=imap.detach(),
        patch_xy=patches[:, :, :2].detach(),
        colors=colors.detach(),
    )


def _derive_frontend_state(
    packet: FMapZeroContextPacket,
    identity: FrameIdentity, experiment_seed: int, *, patches_per_image: int,
    patch_size: int, context_dim: int,
) -> tuple[Any, dict[str, Any]]:
    """Recreate the Patchifier sampling path from fmap under identity RNG."""
    import torch
    from dpvo import altcorr
    from dpvo.utils import coords_grid_with_index

    fmap = packet.fmap
    if fmap.ndim != 5 or fmap.shape[:2] != (1, 1):
        raise ValueError(f"unsupported fmap layout for derivation: {tuple(fmap.shape)}")
    _, n, _, height, width = fmap.shape
    seed = frontend_seed(identity, experiment_seed)
    with frontend_rng_scope(seed):
        x = torch.randint(1, width - 1, size=[n, int(patches_per_image)], device=fmap.device)
        y = torch.randint(1, height - 1, size=[n, int(patches_per_image)], device=fmap.device)
    centroids = torch.stack([x, y], dim=-1).float()
    disps = torch.ones((1, n, height, width), dtype=torch.float32, device=fmap.device)
    grid, _ = coords_grid_with_index(disps, device=fmap.device)
    radius = int(patch_size) // 2
    patch_xy = altcorr.patchify(grid[0], centroids, radius).view(
        1, -1, 3, int(patch_size), int(patch_size),
    )[:, :, :2]
    gmap = altcorr.patchify(fmap[0], centroids, radius).view(
        1, -1, fmap.shape[2], int(patch_size), int(patch_size),
    )
    imap = torch.zeros(
        (1, int(patches_per_image), int(context_dim), 1, 1),
        dtype=torch.float32, device=fmap.device,
    )
    colors = torch.full(
        (1, int(patches_per_image), 3), -0.5,
        dtype=torch.float32, device=fmap.device,
    )
    full = FullOracleHiddenFrontendPacket(
        fmap=fmap, gmap=gmap, imap=imap, patch_xy=patch_xy, colors=colors,
    )
    return full, {
        "frontend_seed": seed, "centroids": centroids,
        "patch_xy": patch_xy, "gmap": gmap, "imap": imap,
    }


def packet_to_native(
    packet: FullOracleHiddenFrontendPacket | FMapZeroContextPacket,
    *, identity: FrameIdentity | None = None, experiment_seed: int | None = None,
    patches_per_image: int | None = None, patch_size: int | None = None,
    context_dim: int | None = None,
) -> Any:
    """Build the existing NativeFeaturePacket with non-oracle placeholder depth."""
    import torch

    from ..exp3.oracle import NativeFeaturePacket

    if not isinstance(packet, FullOracleHiddenFrontendPacket):
        if any(value is None for value in (
            identity, experiment_seed, patches_per_image, patch_size, context_dim,
        )):
            raise ValueError("reduced packet derivation requires identity and DPVO dimensions")
        packet, _ = _derive_frontend_state(
            packet, identity, int(experiment_seed),
            patches_per_image=int(patches_per_image), patch_size=int(patch_size),
            context_dim=int(context_dim),
        )

    xy = packet.patch_xy
    depth = torch.ones(
        (*xy.shape[:2], 1, *xy.shape[-2:]), dtype=xy.dtype, device=xy.device,
    )
    patches = torch.cat((xy, depth), dim=2)
    return NativeFeaturePacket(
        fmap=packet.fmap, gmap=packet.gmap, imap=packet.imap,
        patches=patches, colors=packet.colors,
    )


def packet_arrays(packet: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, tensor in packet.tensors().items():
        result[name] = tensor.detach().cpu().contiguous().numpy()
    return result


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "shape_exact": left.shape == right.shape,
            "dtype_exact": left.dtype == right.dtype,
            "bitwise_exact": False,
            "max_abs": None, "relative_l2": None, "cosine_error": None,
        }
    bitwise = bool(np.array_equal(left.view(np.uint8), right.view(np.uint8)))
    l64, r64 = left.astype(np.float64, copy=False), right.astype(np.float64, copy=False)
    difference = l64 - r64
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = float(np.linalg.norm(l64.reshape(-1)))
    relative = float(np.linalg.norm(difference.reshape(-1)) / max(denominator, 1e-30))
    lflat, rflat = l64.reshape(-1), r64.reshape(-1)
    norm_product = float(np.linalg.norm(lflat) * np.linalg.norm(rflat))
    cosine_error = float(1.0 - np.dot(lflat, rflat) / norm_product) if norm_product else 0.0
    return {
        "shape_exact": True, "dtype_exact": True, "bitwise_exact": bitwise,
        "max_abs": maximum, "relative_l2": relative, "cosine_error": cosine_error,
    }






class FrontendPacketWriter:
    """Preallocated streaming writer; at most one packet is resident at a time."""

    def __init__(
        self, root: str | Path, identities: Sequence[FrameIdentity], *,
        provenance: Mapping[str, Any], schema: str = PACKET_SCHEMA,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.identities = tuple(identities)
        if not self.identities or len({item.key for item in self.identities}) != len(self.identities):
            raise ValueError("streaming packet identities must be non-empty and unique")
        self.provenance = dict(provenance)
        if schema not in PACKET_SCHEMAS:
            raise ValueError(f"unsupported frontend packet schema: {schema}")
        self.schema = schema
        self.name, self.fields = PACKET_SCHEMAS[schema]
        self.cursor = 0
        self.maps: dict[str, np.memmap] = {}
        self.field_templates: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        self.packet_hashes: list[dict[str, Any]] = []

    def append(self, identity: FrameIdentity, packet: Any) -> None:
        if self.cursor >= len(self.identities) or identity.key != self.identities[self.cursor].key:
            raise ValueError("streaming packet identity order mismatch")
        arrays = packet_arrays(packet)
        if not self.maps:
            if tuple(arrays) != tuple(self.fields):
                raise ValueError("packet fields do not match writer schema")
            for name in self.fields:
                array = arrays[name]
                self.field_templates[name] = (array.shape, array.dtype)
                self.maps[name] = np.memmap(
                    self.root / f"{name}.mmap", mode="w+", dtype=array.dtype,
                    shape=(len(self.identities), *array.shape),
                )
        field_hashes: dict[str, str] = {}
        for name in self.fields:
            array = arrays[name]
            expected_shape, expected_dtype = self.field_templates[name]
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(f"inhomogeneous streaming packet field: {name}")
            self.maps[name][self.cursor] = array
            field_hashes[name] = _tensor_sha256(array)
        row = {"identity_key": identity.key, "field_sha256": field_hashes}
        row["packet_sha256"] = canonical_sha256(row)
        self.packet_hashes.append(row)
        self.cursor += 1

    def finalize(self) -> "FrontendPacketStore":
        if self.cursor != len(self.identities):
            raise ValueError("streaming packet store is incomplete")
        fields: dict[str, Any] = {}
        for name, mmap in self.maps.items():
            mmap.flush()
            item_shape, dtype = self.field_templates[name]
            shape = (len(self.identities), *item_shape)
            readonly = np.memmap(self.root / f"{name}.mmap", mode="r", dtype=dtype, shape=shape)
            fields[name] = {
                "file": f"{name}.mmap", "shape": list(shape),
                "item_shape": list(item_shape), "dtype": dtype.str,
                "byte_length": int(readonly.nbytes),
                "sha256": _tensor_sha256(np.asarray(readonly)),
            }
            del readonly
        self.maps.clear()
        keys = [identity.key for identity in self.identities]
        descriptor = {
            "name": self.name, "schema": self.schema,
            "storage": "streaming_preallocated_memmap_v1",
            "identity_keys": keys, "identity_list_sha256": canonical_sha256(keys),
            "fields": fields, "packet_hashes": self.packet_hashes,
            "provenance": self.provenance,
            "forbidden_fields": sorted(FORBIDDEN_PACKET_FIELDS),
            "contains_graph_state": False, "contains_hidden_rgb_or_path": False,
        }
        descriptor["packet_sha256"] = canonical_sha256(descriptor)
        (self.root / "descriptor.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return FrontendPacketStore(self.root, descriptor)




class FrontendPacketStore:
    """Identity-keyed temporary memmap store with whole-file and row hashes."""

    def __init__(self, root: str | Path, descriptor: Mapping[str, Any]):
        self.root = Path(root)
        self.descriptor = dict(descriptor)
        schema = str(self.descriptor.get("schema"))
        if schema not in PACKET_SCHEMAS:
            raise ValueError("unsupported frontend packet schema")
        expected_name, self.fields = PACKET_SCHEMAS[schema]
        if self.descriptor.get("name") != expected_name or tuple(self.descriptor.get("fields", {})) != tuple(self.fields):
            raise ValueError("frontend packet descriptor field allowlist mismatch")
        self._keys = tuple(str(value) for value in self.descriptor["identity_keys"])
        self._index = {key: index for index, key in enumerate(self._keys)}
        if len(self._index) != len(self._keys):
            raise ValueError("duplicate packet identity")
        hashes = self.descriptor.get("packet_hashes", [])
        if hashes and [row["identity_key"] for row in hashes] != list(self._keys):
            raise ValueError("packet hash identity order mismatch")
        self._maps: dict[str, np.memmap] = {}
        for name in self.fields:
            spec = self.descriptor["fields"][name]
            path = self.root / spec["file"]
            if path.stat().st_size != int(spec["byte_length"]):
                raise ValueError(f"packet field length mismatch: {name}")
            mmap = np.memmap(path, mode="r", dtype=np.dtype(spec["dtype"]), shape=tuple(spec["shape"]))
            if _tensor_sha256(np.asarray(mmap)) != spec["sha256"]:
                raise ValueError(f"packet field hash mismatch: {name}")
            self._maps[name] = mmap


    def get(self, identity: FrameIdentity, device: str = "cuda") -> Any:
        import torch

        try:
            index = self._index[identity.key]
        except KeyError as error:
            raise KeyError(f"no packet for hidden identity {identity.key}") from error
        arrays = {name: np.array(self._maps[name][index], copy=True) for name in self.fields}
        if self.descriptor.get("packet_hashes"):
            expected = self.descriptor["packet_hashes"][index]
            actual = {
                "identity_key": identity.key,
                "field_sha256": {name: _tensor_sha256(array) for name, array in arrays.items()},
            }
            actual["packet_sha256"] = canonical_sha256(actual)
            if actual != expected:
                raise ValueError(f"packet row hash mismatch: {identity.key}")
        tensors = {name: torch.from_numpy(array).to(device=device) for name, array in arrays.items()}
        schema = self.descriptor["schema"]
        packet_class = {
            PACKET_SCHEMA: FullOracleHiddenFrontendPacket,
            ZERO_PACKET_SCHEMA: FMapZeroContextPacket,
        }[schema]
        return packet_class(**tensors)


    def sanitized_descriptor(self) -> dict[str, Any]:
        return dict(self.descriptor)

    def close(self) -> None:
        self._maps.clear()
