"""CPU contracts for the vendored formal ViT-B block5 loader."""

from __future__ import annotations

from copy import deepcopy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from prediction.vjepa2 import runtime
from latent_vslam.bridge_checkpoint import (
    PATH_INDEPENDENT_BRIDGE_CONFIG_SHA256, bridge_training_context, load_bridge_config,
)
from latent_vslam.manifests import config_protocol_fingerprint
from latent_vslam.inference_runtime import load_config


class Vjepa2ContractsTest(unittest.TestCase):
    def test_checkpoint_location_is_runtime_only_and_bridge_identity_is_stable(self):
        h1, _ = load_bridge_config()
        self.assertEqual(
            bridge_training_context(h1)[1]["config_protocol_sha256"],
            PATH_INDEPENDENT_BRIDGE_CONFIG_SHA256,
        )
        for config in (h1, load_config()[0]):
            self.assertNotIn("repo", config["jepa"])
            other = deepcopy(config)
            other["jepa"]["checkpoint"] = "/different/mount/same_checkpoint.pt"
            self.assertEqual(
                config_protocol_fingerprint(config)["config_protocol_sha256"],
                config_protocol_fingerprint(other)["config_protocol_sha256"],
            )

    def test_vitb_factory_builds_both_native_modules_with_frozen_defaults(self):
        encoder = SimpleNamespace(embed_dim=768)
        predictor = object()
        with patch.object(runtime.vision_transformer, "vit_base", return_value=encoder) as build_encoder, \
             patch.object(runtime.native_predictor, "vit_predictor", return_value=predictor) as build_predictor:
            self.assertEqual(runtime.build_vitb_encoder_predictor(), (encoder, predictor))
        e = build_encoder.call_args.kwargs
        p = build_predictor.call_args.kwargs
        self.assertEqual((e["img_size"], e["patch_size"], e["num_frames"],
                          e["tubelet_size"], e["img_temporal_dim_size"]),
                         ((384, 384), 16, 64, 2, 1))
        self.assertEqual((p["depth"], p["num_heads"], p["num_mask_tokens"],
                          p["teacher_embed_dim"], p["n_output_distillation"]),
                         (12, 12, 8, 1664, 1))
        self.assertTrue(e["use_rope"] and e["interpolate_rope"] and e["use_sdpa"])
        self.assertTrue(p["use_rope"] and p["use_sdpa"] and p["return_all_tokens"])

    def test_checkpoint_loads_encoder_and_native_predictor_strictly(self):
        class Model:
            def __init__(self):
                self.loaded = None
                self.device = None
                self.dtype = None
                self.training = True

            def load_state_dict(self, state, strict):
                self.loaded = (state, strict)

            def to(self, *, device, dtype):
                self.device, self.dtype = device, dtype
                return self

            def eval(self):
                self.training = False
                return self

        encoder, predictor = Model(), Model()
        checkpoint = Path("/checkpoint/unchanged.pt")
        with patch.object(runtime, "build_vitb_encoder_predictor", return_value=(encoder, predictor)), \
             patch.object(runtime.torch, "load", return_value={
                 "ema_encoder": {"module.backbone.encoder": torch.tensor(1)},
                 "predictor": {"module.backbone.predictor": torch.tensor(2)},
             }) as load:
            result = runtime.load_vitb_encoder_predictor(checkpoint, torch.device("cpu"))
        self.assertEqual(result, (encoder, predictor))
        self.assertEqual(list(encoder.loaded[0]), ["encoder"])
        self.assertEqual(list(predictor.loaded[0]), ["predictor"])
        self.assertTrue(encoder.loaded[1] and predictor.loaded[1])
        self.assertEqual(encoder.dtype, torch.bfloat16)
        self.assertFalse(encoder.training or predictor.training)
        load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=False)

    def test_native_vitb_block5_cpu_shape(self):
        torch.set_num_threads(2)
        encoder, predictor = runtime.build_vitb_encoder_predictor()
        self.assertEqual(encoder.embed_dim, 768)
        self.assertEqual(len(encoder.blocks), 12)
        self.assertEqual(len(predictor.predictor_blocks), 12)
        del predictor
        encoder.out_layers = [5]
        with torch.inference_mode():
            output = encoder(torch.zeros(1, 3, 1, 32, 32))
        self.assertEqual(len(output), 1)
        self.assertEqual(tuple(output[0].shape), (1, 4, 768))


if __name__ == "__main__":
    unittest.main()
