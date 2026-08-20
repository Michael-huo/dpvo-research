"""DPVO-native packet extraction for Experiment 3.

The Oracle path intentionally invokes only Patchifier.fnet.  It never computes
or returns hidden-frame gmap, imap, patch locations, or colors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NativeFeaturePacket:
    fmap: torch.Tensor
    gmap: torch.Tensor
    imap: torch.Tensor
    patches: torch.Tensor
    colors: torch.Tensor


@dataclass(frozen=True)
class OracleFMap:
    fmap: torch.Tensor


def normalize_rgb(image: torch.Tensor) -> torch.Tensor:
    """Apply exactly the normalization used by DPVO.__call__."""
    return 2.0 * (image[None, None] / 255.0) - 0.5


@torch.no_grad()
def extract_native_packet(slam: Any, image: torch.Tensor) -> NativeFeaturePacket:
    normalized = normalize_rgb(image)
    with torch.cuda.amp.autocast(enabled=bool(slam.cfg.MIXED_PRECISION)):
        fmap, gmap, imap, patches, _, colors = slam.network.patchify(
            normalized,
            patches_per_image=int(slam.M),
            centroid_sel_strat=slam.cfg.CENTROID_SEL_STRAT,
            return_color=True,
        )
    return NativeFeaturePacket(
        fmap=fmap.detach(),
        gmap=gmap.detach(),
        imap=imap.detach(),
        patches=patches.detach(),
        colors=colors.detach(),
    )


@torch.no_grad()
def extract_oracle_fmap(slam: Any, image: torch.Tensor) -> OracleFMap:
    """Extract target fmap only; no hidden-frame patch sampling is performed."""
    normalized = normalize_rgb(image)
    with torch.cuda.amp.autocast(enabled=bool(slam.cfg.MIXED_PRECISION)):
        fmap = slam.network.patchify.fnet(normalized) / 4.0
    return OracleFMap(fmap=fmap.detach())
