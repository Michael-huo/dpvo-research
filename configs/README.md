# Train and Infer configurations

Run from the repository root with the `latent-vslam` environment.

| Task | Config | Output |
| --- | --- | --- |
| Bridge training | `configs/train/bridge_mh01.yaml` | `checkpoints/bridge/` |
| Fresh Predictor training for strides 3/5/10 | `configs/train/predictor_stride_sweep_mh01.yaml` | `checkpoints/predictor/stride_sweep_training/` |
| B1 multi-sequence Bridge training | `configs/train/b1_bridge.yaml` | `checkpoints/b1/bridge/<sequences>/` |
| B1 multi-sequence Predictor training | `configs/train/b1_predictor.yaml` | `checkpoints/b1/predictor/<sequences>/` |
| Four-mode inference | `configs/infer/mh01_four_modes.yaml` | `research/results/inference/mh01_four_modes/` |
| Stride sweep inference | `configs/infer/mh01_stride_sweep.yaml` | `research/results/inference/mh01_stride_sweep/` |

The two stride sweep files have identical content so the training manifest binds the same configuration that Infer validates. Infer requires existing checkpoints and never trains. Local EuRoC and ground-truth roots are resolved from ignored `configs/paths.local.yaml`; `configs/paths.example.yaml` documents its format. The V-JEPA checkpoint lives under `checkpoints/vjepa/` as a local symlink until manually copied, as described there.

B1 reads `train.sequences` in the two B1 configs. Run Bridge before Predictor with the same sequence list. `--sequences MH_01_easy` overrides the list for a single-sequence regression. B1 test splits are held-out portions of the training sequences, not an unseen-sequence benchmark.
