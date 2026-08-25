# Phase 1 / Experiment 5-0 — JEPA–DPVO Interface Validation

## Scope

Exp5-0 只验证 frozen V-JEPA sidecar 能否稳定接入 DPVO inference runtime：同一 EuRoC frame 的身份是否严格同步、JEPA latent 是否有效、sidecar 执行后 DPVO 在 frame-dispatch boundary 已暴露的 pose state 是否保持不变，以及逐帧 extraction 带来多少额外运行时间。

本实验不评价 SLAM 精度提升，也不评价 JEPA representation quality。它不训练模型，不实现 adapter/fusion，不保存 latent 或 dense-token cache，不生成轨迹 benchmark、轨迹图或可视化。

## DPVO interface analysis

### Image loading and timestamp flow

Upstream `dpvo.stream.image_stream` 按文件名排序图片，应用 `skip/stride`，通过 OpenCV 读取 BGR 图像，按 EuRoC calibration 去畸变，并把高宽裁剪为 16 的倍数。它输出的 `t` 是采样后从零开始的 stream ordinal，并不是 EuRoC sensor timestamp。

Exp5 在内存中复用 Exp4 的 EuRoC `data.csv` discovery/identity 协议，将以下三种身份显式分开：

- `stream_index`：传给原始 `DPVO.__call__()` 的连续序号，保持 upstream inference 语义；
- `frame_id`：stride 前的零基 cam0 CSV 行号；
- `timestamp`：CSV 和图片文件名中的 EuRoC 纳秒时间戳。

`DPVO.terminate()` 返回的 ordinal timestamp 会被验证。JEPA sidecar 每帧只接收 `image_path`、`frame_id` 和 `timestamp`，因此两条路径共享完全相同的 frame identity，但 JEPA 不能访问 DPVO graph 或 tensor。

### DPVO feature flow

```text
EuRoC cam0 image
  -> dpvo.stream.image_stream
     (sort, stride, OpenCV decode, undistort, crop to multiple of 16)
  -> DPVO.__call__(stream_index, image, intrinsics)
  -> input normalization: 2 * (image / 255) - 0.5
  -> Patchifier
       -> FNet / 4.0 -> full fmap
       -> INet / 4.0 -> context imap
       -> patch sampling from fmap -> gmap
  -> current-frame fmap pyramid + source gmap
  -> local correlation volumes
  -> recurrent update operator (delta, weight)
  -> bundle adjustment and keyframe/graph maintenance
  -> DPVO.terminate()
       -> final updates
       -> interpolation of motion-rejected/keyframe-removed frames
       -> inverse camera poses, one pose per submitted stream ordinal
```

FNet produces the dense 128-channel feature map. Patchifier samples source patches from it into `gmap`, while DPVO stores the full-resolution and pooled maps in its target-frame pyramid. Correlation feeds the recurrent update operator; predicted deltas/weights then drive BA. The final trajectory is produced only by `terminate()` and is generated once per run; it is not used for an independent-run correctness comparison.

### Exp5-0 hook point

The hook is the research-only frame-dispatch boundary outside DPVO:

```text
shared FrameRecord
  -> upstream DPVO inference for this frame
  -> snapshot current graph index and existing pg.poses_[:slam.n]
  -> frozen V-JEPA extraction for this same frame
  -> compare the same existing pose prefix
  -> discard the [768] pooled latent after numeric/shape checks
```

No module hook is installed inside DPVO. JEPA receives no DPVO object, feature, pose, or tensor. The single-run check verifies only that the current graph index, pose-prefix shape, and existing `pg.poses_[:slam.n]` entries remain unchanged while JEPA executes. It does not inspect candidate/unused slots and does not claim to validate DPVO's complete internal computation graph. Baseline and sidecar remain separate processes, but baseline is used only for runtime measurement.

## Protocol and outputs

- Dataset: EuRoC cam0, stride 2, seed 1234.
- Train metadata sequence: `MH_01_easy`.
- Validation metadata sequence: `MH_03_medium`.
- Test metadata sequence: `MH_05_difficult`.
- JEPA output: final EMA V-JEPA 2.1 ViT-B/16 dense tokens, immediately mean-pooled to `[768]` and discarded.
- Hook validation: existing pose entries at the dispatch boundary must have `existing_pose_max_error < 1e-5`; independent-run trajectories are not compared.
- Runtime is descriptive only. Exp5-0 does not impose a runtime acceptance threshold.

Formal outputs contain only:

```text
research/results/phase1-dpvo-feasibility/exp5/interface_validation/
├── metrics.json
└── REPORT.md
```

Per-frame identity/runtime records and JEPA statistics live only in a temporary staging directory and are removed after evaluation. Pose snapshots are never saved. No ATE or other trajectory benchmark is computed.

`pass` and `fail` are both valid completed experiment outcomes. Once both workers complete, Exp5 publishes `metrics.json` and `REPORT.md`; synchronization, latent, and dispatch-boundary validation determine status. Runtime is diagnostic only. Worker/preflight/I/O failures still stop without publishing an incomplete result. Existing formal output is never migrated or overwritten; delete it manually before a fresh formal run.

## Commands

CPU smoke uses the first eight real `MH_01_easy` images and metadata, but replaces both models with deterministic mocks. It verifies `DPVO -> hook_before -> JEPA -> hook_after` ordering and the three-field JEPA payload:

```bash
python -m research.src.phase1_dpvo_feasibility.exp5.run --smoke
```

The formal experiment requires CUDA and the configured DPVO/V-JEPA environments. It processes all three sequences and may take a long time; run it manually:

```bash
python -m research.src.phase1_dpvo_feasibility.exp5.run
```

## Future work

Exp5-1 will decide the actual fusion location. Exp5-0 neither selects nor implements a fusion hook. DPVO internal state recorders, fmap/gmap/imap snapshots, correlation/BA snapshots, dense-token transport, fusion layers, adapters, multi-GPU execution, visualization, and representation-quality experiments remain out of scope.
