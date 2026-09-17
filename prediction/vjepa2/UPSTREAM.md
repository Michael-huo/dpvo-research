# V-JEPA 2.1 ViT-B runtime source

This directory contains only the source needed to construct and load the formal
ViT-B/16 block5 encoder **and its native predictor**. The predictor is built and
loaded before it is released, matching the existing worker's `torch.hub` path.

- Source repository: `https://github.com/Michael-huo/vjepa2-research`
- Source commit: `be4a39cf6252ee003dbe96f22176529c5c0a39c9`
- Original project: `https://github.com/facebookresearch/vjepa2`
- Original upstream baseline noted by the source repository: `20e498b`
- License: MIT; see [LICENSE](LICENSE). The original file copyright headers
  remain in the copied model code.

| Here | Source at recorded commit |
| --- | --- |
| `vision_transformer.py` | `app/vjepa_2_1/models/vision_transformer.py` |
| `native_predictor.py` | `app/vjepa_2_1/models/predictor.py` |
| `modules.py` | `app/vjepa_2_1/models/utils/modules.py` |
| `patch_embed.py` | `app/vjepa_2_1/models/utils/patch_embed.py` |
| `masks.py` | `src/masks/utils.py` |
| `tensors.py` | `src/utils/tensors.py` |
| `runtime.py` | ViT-B branch of `src/hub/backbones.py`, plus the formal
  `research/scripts/common/video_models.py` load sequence |

Changes to copied model code are limited to import paths and removing unused
model variants, audio patch embedding, cross attention, and scheduling helpers.
The ViT-B class and factory, predictor class and factory, and their required
attention, embedding, masking, and tensor helpers retain their source bodies.
The source repository's phase scripts, probes, video decoding, DAVIS tools,
other model sizes, training, demos, results, and test suite are excluded.

The checkpoint remains at its existing absolute path outside this package.
