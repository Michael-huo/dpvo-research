"""Local V-JEPA 2.1 ViT-B factory used by the block5 worker."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from prediction.vjepa2 import native_predictor, vision_transformer


SOURCE_COMMIT = "be4a39cf6252ee003dbe96f22176529c5c0a39c9"
MODEL_ALIAS = "vjepa2_1_vit_base_384"


def _clean_backbone_key(state_dict: dict) -> dict:
    # Same key normalization as src/hub/backbones.py at SOURCE_COMMIT.
    for key, value in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = value
    return state_dict


def build_vitb_encoder_predictor() -> tuple[torch.nn.Module, torch.nn.Module]:
    """Construct both native modules with the original 2.1 ViT-B hub defaults."""
    encoder = vision_transformer.vit_base(
        patch_size=16, img_size=(384, 384), num_frames=64,
        tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True, img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    predictor = native_predictor.vit_predictor(
        img_size=(384, 384), patch_size=16, use_mask_tokens=True,
        embed_dim=encoder.embed_dim, predictor_embed_dim=384,
        teacher_embed_dim=1664, num_frames=64, tubelet_size=2,
        depth=12, num_heads=12, num_mask_tokens=8, use_rope=True,
        uniform_power=False, use_sdpa=True, use_silu=False,
        wide_silu=True, n_output_distillation=1, return_all_tokens=True,
        img_temporal_dim_size=1,
    )
    return encoder, predictor


def load_vitb_encoder_predictor(
    checkpoint: Path, device: torch.device, *, dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load both native checkpoint branches before moving them to the device."""
    encoder, predictor = build_vitb_encoder_predictor()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(_clean_backbone_key(state_dict["ema_encoder"]), strict=True)
    predictor.load_state_dict(_clean_backbone_key(state_dict["predictor"]), strict=True)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()
    return encoder, predictor


def load_vitb_encoder(
    checkpoint: Path, device: torch.device, *, dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    encoder, predictor = load_vitb_encoder_predictor(checkpoint, device, dtype=dtype)
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoder
