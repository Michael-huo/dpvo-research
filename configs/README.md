# Train and Infer configurations

Run from the repository root with the `latent-vslam` environment.

| Task | Config | Output |
| --- | --- | --- |
| Bridge training | `configs/train/bridge_mh01.yaml` | `checkpoints/bridge/` |
| Fresh Predictor training for strides 3/5/10 | `configs/train/predictor_stride_sweep_mh01.yaml` | `checkpoints/predictor/stride_sweep_training/` |
| Four-mode inference | `configs/infer/mh01_four_modes.yaml` | `research/results/inference/mh01_four_modes/` |
| Stride sweep inference | `configs/infer/mh01_stride_sweep.yaml` | `research/results/inference/mh01_stride_sweep/` |

The two stride sweep files have identical content so the training manifest binds the same configuration that Infer validates. Infer requires existing checkpoints and never trains. Local EuRoC and ground-truth roots are resolved from ignored `configs/paths.local.yaml`; `configs/paths.example.yaml` documents its format. The V-JEPA checkpoint lives under `checkpoints/vjepa/` as a local symlink until manually copied, as described there.
