# Phase 1 / Exp5-2 and Exp5-Oracle — JEPA–DPVO Feature Validation

The default formal entry is self-contained: it rebuilds Exp5-2 from raw EuRoC
frames, trains and publishes its dense projection, then loads that exact
checkpoint read-only for Exp5-Oracle FNet replacement. No previous Exp5 result
is required or reused.

## Question

Exp5-2 tests whether retaining V-JEPA's 24×24 token grid provides local visual constraints that Exp5-1 global pooling and spatial broadcast did not establish. It is a feasibility experiment, not a final SLAM system.

The experiment does not modify DPVO or V-JEPA upstream, train either backbone, use SLAM/pose/GT losses, add attention, sweep alpha, or retain latent, FNet, pair, or intermediate-trajectory files. Only each method's final `terminate()` trajectory is retained as evaluation evidence.

## Projection and scale contract

```text
V-JEPA [B,576,768]
  -> reshape [B,24,24,768]
  -> permute [B,768,24,24]
  -> Conv2d(768,128,1)
  -> bilinear resize, align_corners=False
  -> projected dense fmap [B,128,H,W]
```

The Conv2d, including bias, has 98,432 trainable parameters. There is no normalization, activation or additional layer.

The teacher and raw-FNet hook use an explicit scale conversion:

```text
teacher = raw_fnet / 4.0
fused_raw = raw_fnet + 0.1 * (4.0 * projected_dense)
downstream_fmap = fused_raw / 4.0
                = raw_fnet / 4.0 + 0.1 * projected_dense
```

`alpha=0.1` is fixed. The hook consumes exactly one dense projection for one `frame_id/timestamp`, preserves the raw output shape/dtype/device and is removed after the run.

## RAM-only training

- Train: `MH_01_easy`
- Validation/checkpoint selection: `MH_03_medium`
- Test: `MH_05_difficult`
- AdamW, 50 epochs, batch size 8, learning rate `1e-4`, weight decay `1e-4`, seed 1234
- Validation at epochs 5, 10, …, 50; highest MH_03 mean spatial cosine selects the checkpoint

MH_01 and MH_03 dense tokens and `raw_fnet/4` teachers are extracted once into preallocated CPU float32 NumPy arrays. The arrays are never written to disk. Before allocation, the runner computes their exact byte size and requires `MemAvailable >= 1.25 × pair_bytes`. Frozen extractors are released before projection training; pair arrays are released before MH_05 and SLAM evaluation.

MH_05 dense alignment is evaluated online, one frame at a time, without retaining a test pair. The loss is `1 - mean spatial cosine` over `[B,H*W,128]` vectors.

The 24×24 V-JEPA grid describes a 384×384 center crop, while DPVO FNet covers the full EuRoC image. This first probe directly resizes the grid across the DPVO feature extent and does not perform crop-aware coordinate mapping. That approximation is an interpretation risk.

## Separate environments

Both experiments run in the configured DPVO Python. The first stage keeps DPVO, FNet extraction, projection training and fusion evaluation in that process; the immediately following Oracle stage loads that projection read-only. The DPVO environment does not import or install V-JEPA dependencies.

Each JEPA-consuming preparation/alignment run or fused method×sequence run starts one persistent subprocess using `environment.vjepa_python`. The worker loads the frozen encoder once, processes the complete sequence synchronously and then exits. It is never reused across methods or sequences.

Per-frame requests contain only `request_id`, `image_path`, `frame_id` and `timestamp`. Responses contain one `[576,768]` float32 token array encoded as base64 in the JSON protocol. Tokens are decoded, consumed and released immediately; no latent cache is retained. There is no Unix socket, shared memory, DPVO worker wrapper, per-frame Python startup or automatic dependency installation.

Baseline does not start a V-JEPA worker or install an injection hook. Worker setup and stdin/stdout transfer overhead are reported separately and are diagnostic only.

## Evaluation and interpretation

The three fixed methods are:

- `dpvo_baseline`
- `random_dense_projection_fusion`
- `jepa_dense_projection_fusion`

All sequences report Sim(3) ATE, tracking/pose coverage, GT association and split runtime. MH_01/MH_03 are diagnostic; MH_05 determines the classification.

- Strong success: learned tracks, beats random ATE and is no more than 5% worse than baseline.
- Feasible success: the same conditions with degradation above 5% and no more than 10%.
- Inconclusive: tracking/degradation remain feasible but learned does not beat random.
- Failure: tracking/coverage fails or degradation exceeds 10%.

Learned ATE is not required to beat baseline. Dense cosine and runtime are diagnostic rather than effectiveness gates.

## Trajectory evidence

The completed Exp5-2 run exports only the final `terminate()` trajectory for baseline, random dense fusion and learned dense fusion on all three fixed sequences. Each sequence also stores the corresponding EuRoC GT subset, independently recomputes the same timestamp-associated Sim(3) translation ATE and generates one 2D XY PNG. Raw JSON poses remain unaligned; Sim(3)-aligned coordinates exist only while drawing the figure.

- `MH_01_easy` trajectory evidence is a training-sequence diagnostic.
- `MH_03_medium` trajectory evidence is a transfer diagnostic.
- `MH_05_difficult` is the only decision-sequence visualization.

MH_01/MH_03 figures help explain in-split behavior and generalization but never affect classification. Trajectory export, ATE verification and plotting do not modify fusion, training or the MH_05-only decision criteria. No PDF, 3D trajectory, per-frame error plot or intermediate pose buffer is retained.

The report reads the existing Exp5-1 metrics and presents global-vector cosine separately from dense-spatial cosine. Their numeric values are not directly compared because they belong to different representation spaces.

## Exp5-Oracle FNet replacement

Exp5-Oracle evaluates whether Oracle dense JEPA can replace non-keyframe
RGB-derived matching features. It does not evaluate complete RGB replacement:
upstream INet/context and color sampling remain unchanged and continue to use
RGB. The controlled variable is only the tensor source at `Patchifier.fnet`
output.

With fixed `keyframe_stride=5`:

```text
keyframe:     raw_fnet + 0.1 * 4.0 * projected_dense
non-keyframe: 4.0 * projected_dense  (original FNet is not called)
```

Both paths expose the same `[B,N,128,H,W]` interface to the unchanged downstream
`/4.0`, gmap, correlation, update and BA graph. The Exp5-2 `projection.pt`
generated by the preceding stage is loaded read-only. Fresh full-RGB baseline and Oracle runs use
the same Git/source snapshot, DPVO checkpoint, calibration, dataset sampling and
seed. MH_01 and MH_03 are diagnostic; only MH_05 determines the Oracle outcome.
Coverage is reported but is not an independent failure criterion.

## Commands and outputs

CPU smoke uses eight real MH_01 images and mock model tensors. It verifies two
RGB-FNet keyframes and six JEPA replacements without training a projection:

```bash
python -m research.src.phase1_dpvo_feasibility.exp5.fusion.run --smoke
```

The user-operated formal command starts from raw data, rebuilds Exp5-2, and then
runs Exp5-Oracle from the newly published checkpoint:

```bash
python -m research.src.phase1_dpvo_feasibility.exp5.fusion.run
```

Formal output is atomically published only when complete:

```text
research/results/phase1-dpvo-feasibility/exp5/dense_fusion_validation/
├── metrics.json
├── REPORT.md
├── projection.pt
├── trajectories/
│   ├── MH_01_easy_{baseline,random,learned,gt}.json
│   ├── MH_03_medium_{baseline,random,learned,gt}.json
│   └── MH_05_difficult_{baseline,random,learned,gt}.json
└── figures/
    ├── MH_01_xy.png
    ├── MH_03_xy.png
    └── MH_05_xy.png
```

Existing result directories are not treated as compatible inputs. The Exp5-2
stage replaces them with a complete newly staged result; Exp5-Oracle then
publishes only its nested directory. Exp5-1 global metrics are not required,
loaded or reconstructed because they are diagnostic and do not affect dense or
Oracle decisions.

The Oracle result is published independently and never rewrites the files above:

```text
dense_fusion_validation/
└── oracle_missing_rgb/
    ├── metrics.json
    ├── REPORT.md
    ├── trajectories/
    │   ├── MH_05_baseline.json
    │   ├── MH_05_oracle.json
    │   └── MH_05_gt.json
    └── figures/
        └── MH_05_oracle_missing_rgb.png
```
