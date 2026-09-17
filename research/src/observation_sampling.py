"""Explicit scientific RNG isolation; no deterministic CUDA algorithm changes."""
from __future__ import annotations

import hashlib
import json

from .oracle_packet import frontend_rng_scope

SAMPLING_PROTOCOL = {
    "identity_seed_derivation": "SHA256(namespace, scientific_seed, identity.key, timestamp_ns, role)",
    "frontend": "unchanged frontend_seed(identity, scientific_seed), saved/restored RNG scope",
    "hidden_patch_xy": "unchanged uniform randint x then y; native bounds and M",
    "native_patch_xy": "unchanged native Patchifier distribution and M",
    "initial_depth": "SHA256(scientific_seed, identity.key, timestamp_ns, dpvo_track_packet); native rand_like",
    "process_rng": "Python, NumPy, torch CPU and CUDA seeded with scientific_seed in all three workers",
    "cuda_bit_exact_required": False,
    "global_deterministic_algorithms_enabled_by_experiment": False,
    "geometry_limit": "reprojected coordinates and gmap values remain state/feature dependent",
}


def validate_scientific_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("scientific seed must be an integer in [0, 2**32)")
    return seed


def observation_seed(identity, scientific_seed, role):
    validate_scientific_seed(scientific_seed)
    payload = json.dumps(["reliability_identity_rng_v1", scientific_seed, identity.key,
                          int(identity.timestamp_ns), role], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def observation_rng_scope(identity, scientific_seed, role="dpvo_track_packet"):
    # Reuse the existing save/restore context: native sampling code/distribution
    # stays untouched, and draws cannot advance another observation's stream.
    return frontend_rng_scope(observation_seed(identity, scientific_seed, role))
