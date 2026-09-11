"""Shared research components organized by function."""

from .protocol import FrameIdentity, FrameRole, RepresentationOrigin
from .schema import VISUAL_STATE_CONTRACT, VISUAL_STATE_CONTRACT_SHA256

__all__ = [
    "FrameIdentity", "FrameRole", "RepresentationOrigin",
    "VISUAL_STATE_CONTRACT", "VISUAL_STATE_CONTRACT_SHA256",
]
