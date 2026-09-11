"""Canonical Phase 1 condition metadata and visual-state contracts."""

from __future__ import annotations

from typing import Any

from .protocol import canonical_sha256


VISUAL_STATE_CONTRACT = {
    "schema": "exp6_fmap_only_visual_state_v1",
    "stored_hidden_visual_state": ["fmap"],
    "derived": {
        "patch_xy": "frame_identity_seed_dpvo_coordinate_rng",
        "gmap": "dpvo_altcorr_patchify_from_fmap",
        "fmap2": "average_pool_from_fmap",
    },
    "imap": "deterministic_zero",
    "colors": "removed_zero_buffer_compatibility",
    "pose_depth": "online_dpvo_initialization_update_and_ba",
    "graph_semantics": "normal_source_target_update_ba_and_upstream_culling",
}
VISUAL_STATE_CONTRACT_SHA256 = canonical_sha256(VISUAL_STATE_CONTRACT)


def condition_metadata(
    *, experiment_mode: str, input_source: str,
    online_allowed_fields: list[str] | tuple[str, ...],
    offline_reference_only: bool, strict_deployment: bool,
    timestamp_causal: bool, closing_anchor_online_available: bool,
) -> dict[str, Any]:
    payload = {
        "experiment_mode": str(experiment_mode),
        "input_source": str(input_source),
        "online_allowed_fields": list(online_allowed_fields),
        "offline_reference_only": bool(offline_reference_only),
        "strict_deployment": bool(strict_deployment),
        "timestamp_causal": bool(timestamp_causal),
        "closing_anchor_online_available": bool(closing_anchor_online_available),
    }
    payload["metadata_sha256"] = canonical_sha256(payload)
    return payload
