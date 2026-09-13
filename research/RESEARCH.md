# Phase 1 — Feasibility Analysis

## 环境前提

使用 `research/environment.yml` 中的 DPVO 环境（Python 3.10），并安装项目的 CUDA extensions。
扩展编译使用 CUDA Toolkit 12.1。运行前准备 EuRoC 数据、DPVO/V-JEPA 权重及正式 YAML 配置指定的路径。
上游 DPVO 的安装、Demo 和可选后端说明见仓库 README；它们不属于 Phase 1 的正式研究入口。

源码按功能平铺在 `research/src/`，以文件名表达职责，供后续研究阶段复用。Phase 用于科学实验组织和 results，不再作为源码 namespace；Phase 1 的 H0 State、H1 Interface、H2 Prediction 定义不变。正式配置仍为 `research/configs/phase1_feasibility_h0.yaml`、`phase1_feasibility_h1.yaml` 和 `phase1_feasibility_h2.yaml`。

Phase 1 的三个公开研究 CLI 为 `research.src.run_h0`、`research.src.run_h1`、`research.src.run_h2`；Phase 2 新增 `research.src.run_anchor_budget`。`jepa_worker`、`pipeline_worker`、`parallel_runtime` 的 module 启动方式仅供内部进程使用。源码布局迁移不改变科学 lineage 或 canonical checkpoint 兼容性；execution provenance 如实记录当前源码路径与 hash。

在仓库根目录执行 CPU/unit 验证：`CUDA_VISIBLE_DEVICES="" python -m unittest discover -s research/src -t . -v`。测试覆盖 CLI mock dispatch、worker fresh-process/spawn 导入；安装了 canonical artifacts 时还会只读验证 H1/H2 checkpoint。

## H0 State — Latent-State Feasibility

H0 回答“learning-based VSLAM 的 hidden frame 最少需要什么 latent visual state”。conditions 固定为 Full RGB、Sparse RGB 和 True FMap。hidden packet 只保存 `fmap`；`patch_xy` 按 FrameIdentity/seed 确定性派生，`gmap/fmap2` 从 FMap 派生，`imap` 为 zero，colors 删除；pose/depth、factor、update、BA 与 upstream culling 保持正常 DPVO 语义。H0 不加载 JEPA、bridge 或 predictor。

```bash
python -m research.src.run_h0 --sequences MH_01_easy
```

## H1 Interface — Representation-Interface Feasibility

H1 验证 Oracle JEPA 能否经 coordinate-correct block-5→FMap interface 提供 H0 latent state。每次显式运行都在 MH01 fresh 训练一次 `bridge.pt`，并使用本轮刚训练的同一个 bridge fresh 评估全部 requested sequences。H1 不读取 H0 results，MH03/MH05 是本轮 bridge 的 frozen zero-shot evaluation。

```bash
python -m research.src.run_h1 --sequences MH_01_easy
```

## H2 Prediction — Sparse-Anchor Prediction Feasibility

H2 只加载 canonical H1 `bridge.pt`，不会调用 H1 或训练 bridge。加载前验证 bridge 与当前 H1 科学 config/source/protocol lineage 兼容；不兼容时 fail closed 并要求先运行 H1。兼容后，每次 H2 显式运行都在 MH01 fresh 训练一次 `predictor.pt`，其 training lineage 绑定当前 bridge hash，再使用本轮 predictor fresh 评估全部 requested sequences。

strict 在线 capability 只有 uploaded anchor RGB、anchor identity 和 hidden identity/timestamp。anchor RGB 通过 native DPVO FNet/Patchifier 形成 VSLAM observation 并提取 JEPA context；hidden observation 只能来自 predicted JEPA 经 frozen bridge 生成的 FMap-only packet。

A5 必须真实到达并完成 JEPA 编码，随后才按时间戳顺序提交 buffered hidden observations，且每个 candidate 只消费一次。因此 H2 是 delayed/bracketed、non-causal deployment，`timestamp_causal=false`，不声明 causal 或 strict real-time。

```bash
python -m research.src.run_h2 --sequences MH_01_easy
```

三条 CLI 均支持一次请求多个 sequence。需要三序列横向比较时，必须让它们共享同一次 fresh 模型运行：

```bash
python -m research.src.run_h2 \
    --sequences MH_01_easy MH_03_medium MH_05_difficult
```

三条常规 fresh runner 会把性能诊断与 canonical 结果一起持久化：每个 sequence 的
`results.json` 包含独立的 `result.performance_diagnostics`；H1/H2 的训练诊断同时保存在
`INDEX.json -> canonical_checkpoint.training.performance_diagnostics`。sequence 和 aggregate
summary 会显示简要 wall time、逐卡利用率以及 H2 predictor 的 CPU/CUDA outer totals。
这些字段固定标记为 diagnostic-only，不进入评价指标、科学 gate、checkpoint tensor、
模型输入或方法 lineage 决策。H2 还在常规 strict replay 中记录 predictor exclusive
fine stages和transfer/IPC；H1/H2 training记录data/H2D、forward/loss、backward、
optimizer/scaler及独立 synchronization wait。

## 正式执行协议

- H0/H1/H2 的 baseline/control 均在 GPU0 上逐 sequence、逐 condition 执行；每条 trajectory 使用独立 DPVO 进程，退出后再启动下一条，不并行多个 DPVO 实验。
- H2 Predicted JEPA 每次只运行一条 trajectory、一个 DPVO consumer：GPU2 编码 V-JEPA，GPU1 执行 correspondence/transport/predictor，GPU0 执行 native frontend、frozen bridge 和 DPVO。多个 sequence 依次使用同一三卡映射。
- Stage C 与标准 DPVO 使用相同固定 CPU profile；predictor/encoder 使用隔离的 CPU cores 和单线程配置。正式运行不动态选择 profile。
- H1/H2 保留多 GPU preparation 和 GPU-resident train/validation；batch、RNG、AMP、optimizer/scaler、两 pass 或 best-checkpoint selection 均按各模块原 recipe 执行。训练准备与 trajectory 生命周期分离。
- matched trajectory timing 排除训练、离线 preparation、模型加载、worker 启动、warmup 和 artifact I/O；包括在线 observation 处理、pipeline fill/steady/drain、DPVO terminate 及完成所需同步。
- 正式路径保留 identity/order、exactly-once、capability、轻量 timing/transfer/provenance。完整 payload 验证和 decision trace 不进入正式计时。GPU telemetry 仅用于解释执行环境，不作为资源准入 gate。
- scientific lineage 与 execution provenance 分离；执行源码变化不绕过或放宽 checkpoint 科学兼容检查。

## Artifact 与执行策略

Canonical outputs 位于 `research/results/phase1-feasibility/{h0_state,h1_interface,h2_prediction}/`。现阶段采用 fresh current-canonical replace：H0 每次 fresh 执行 requested sequences；H1 每次 fresh 训练 bridge 并 fresh 评估；H2 验证当前 H1 bridge 后 fresh 训练 predictor 并 fresh 评估。

本次 requested sequences 是模块完整的当前有效集合。只有整轮成功且通过 artifact manifest 验证后才替换旧模块，未请求的旧 sequence 不保留。每个模块根 `INDEX.json` 仅作为本轮 canonical provenance/artifact manifest，记录 requested sequences、dataset/config/source/protocol/schedule、适用的 bridge/predictor 及 sequence artifact hashes，不承担 cache 或 reuse 职责。

## Phase 2 — Anchor Budget Sensitivity / Prediction Horizon

Phase 1 — Feasibility 已完成并冻结，H0 State、H1 Interface、H2 Prediction 的既有结论、canonical results、checkpoint 和 H1 bridge 不变。本阶段以该方法为基础，不重新验证三个猜想。

科学问题：降低 anchor upload budget 时，Sparse 因真实观测减少、Ours 因 prediction horizon 延长，各自如何变化；哪个 budget 区间可能出现精度优势？提高 anchor 密度时，两者是否改善？退化速度、单调性、交叉点都必须由后续实测验证，当前不写实验结论。独立变量为 `anchor_stride=K`，第一轮仅 MH_01_easy 的 3/5/10 三点。

每点比较 GT / Full RGB / Sparse RGB / Ours（Predicted JEPA）四条轨迹。GT 仅作为 reference 引用一次；Full RGB 在固定有效 population 上标准 single-GPU DPVO 正式运行一次并共享；Sparse 和 Ours 使用完全相同的 schedule，分别 fresh、串行运行独立 DPVO。Ours 原样复用 GPU2 V-JEPA、GPU1 predictor、GPU0 bridge + DPVO 的正式 pipeline，不增加 Oracle trajectory。

每个 stride 使用 `train_predictor` fresh 初始化独立 predictor，保持相同 architecture、loss、optimizer、30 epochs、AMP、seed 1234、interval batch size 2 和 lowest-validation-total checkpoint selection；H1 bridge 冻结复用。不会使用 stride-5 predictor zero-shot 代替其他 stride 的正式训练，也不会重新执行 Phase 1 tiny-overfit feasibility gate。不同 stride 每 epoch 的 interval/batch 数随数据自然改变，不额外配平训练步数。

Schedule 直接调用 canonical ratio accumulator，令 ratio=1/K，不重新定义 K=5：candidate 0–7 为 bootstrap anchors，candidate 8 为第一个 post-bootstrap anchor，随后 K−1 hidden、closing anchor；没有 closing anchor 的 tail 按原逻辑排除。三个 pilot 点均自然保留 candidate 0–1838（1839 observations），末尾 2 candidates 排除，无需人为截断或提升 tail 为 anchor。

固定时间切分来自冻结 H2 `INDEX.json` 的 split，端点 inclusive。所有 stride 只保留两端 anchor 都落在同一个固定区域内的完整 interval，跨区域或落入原 boundary gap 的 interval 丢弃，不共享 anchor identity，不按新的 interval 数量重新切 60/20/20。

| Split | Candidate range | 原始 frame_id range | Timestamp ns range |
| --- | --- | --- | --- |
| Train | 8–1103 | 16–2206 | 1403636580563555584–1403636690063555584 |
| Validation | 1108–1468 | 2216–2936 | 1403636690563555584–1403636726563555584 |
| Test | 1473–1838 | 2946–3676 | 1403636727063555584–1403636763563555584 |

这些是固定的许可时间区域；不同 stride 的有效 interval 端点可能向区域内部移动，不能把这些端点再当作新的 split 定义。Trajectory evaluation 仍沿用 Phase 1 的整段 in-sequence feasibility protocol；held-out/test 仅用于训练后 representation diagnostics，不声称 held-out trajectory generalization。

只读构造的 population（不是模型或 SLAM 实验结果）：

| Stride | 理论 Anchor % | 实际 Anchor % | Anchors | Hidden | Complete intervals | Dropped split intervals | Train/Val/Test intervals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 33.333 | 33.660 | 619 | 1220 | 610 | 5 | 365 / 119 / 121 |
| 5 | 20.000 | 20.392 | 375 | 1464 | 366 | 2 | 219 / 72 / 73 |
| 10 | 10.000 | 10.440 | 192 | 1647 | 183 | 2 | 109 / 36 / 36 |

实际比例的分母为 1839 个有效 observations，分子包括 bootstrap 上传。Encoded bytes 使用原 PNG 文件大小，和 Phase 1 一致；Full RGB 665,764,973 bytes，stride 3/5/10 分别上传 224,141,816 / 135,800,130 / 69,529,672 bytes，byte reduction 分别为 66.333% / 79.602% / 89.556%。Split boundary drops 只用于训练/验证/test划分，不从完整在线轨迹中删除。

```bash
# 已有正式结果时只读验证并报告 complete；首次运行仅在 /tmp 做 CPU preparation。
python -m research.src.run_anchor_budget --sequence MH_01_easy --strides 3 5 10

# 仅迁移已完成的旧目录；已有 compact 产物时验证后直接返回。
CUDA_VISIBLE_DEVICES="" python -m research.src.run_anchor_budget --consolidate-existing

# 仅凭 results.json + trajectories.npz 重建两张 PNG，不加载模型或数据集。
CUDA_VISIBLE_DEVICES="" python -m research.src.run_anchor_budget --render-figures
```

首轮 3/5/10 实验已由用户完成。本次只聚合已有产物和绘图，没有重新训练或运行 DPVO。正式运行仍需显式添加 `--execute`；已有正式结果或 predictor 时拒绝覆盖/重跑。首次正式执行的 staging、worker 和 training cache 均在 `/tmp/anchor_budget_run_*/`，整轮成功后验证并发布；失败时保留 `/tmp` 现场供诊断，不自动恢复。默认 preparation 也只写 `/tmp/anchor_budget_prepare_*/`。

协议配置为 `research/configs/phase2_anchor_budget.yaml`，只保存实验变量、冻结 split、canonical 引用和 hashes。Runner 读取现有 H2 YAML 并局部覆盖 schedule/split/output，不建立配置继承框架。Checkpoint 的 Phase 2 lineage 绑定 stride、实际 train anchor identities、固定 split、canonical H2 config/scientific contract、H1 hash、DPVO config/checkpoint、seed、训练源码 hashes 与 train-only calibration；加载时精确匹配，拒绝跨 stride 和 Phase 1 checkpoint。

正式产物收敛为以下五个文件，可以独立打包 results 目录而不携带模型：

```text
research/results/phase2-anchor-budget/
├── SUMMARY.md
├── results.json
├── trajectories.npz
└── figures/
    ├── trajectories.png
    └── tradeoffs.png
research/checkpoints/phase2-anchor-budget/
├── predictor_stride_3.pt
├── predictor_stride_5.pt
└── predictor_stride_10.pt
```

`results.json` schema v2 的顶层为 `metadata`、`full_rgb`、`strides`；每个 stride 包含 `schedule`、`communication`、`training`、`sparse`、`ours`、`horizon`、`comparison` 和 `provenance`。保留已有 ATE/RPE、evaluation population、Sim(3)、coverage、graph workload、matched wall、stage timing、context wait、held-out quality。Epoch metrics 和所有 test query diagnostics 用无损 columnar 表保存（`columns` + `rows`），可按任一 anchor 时间距离重新分组。模型 provenance 包括相对仓库路径、SHA256、stride、seed、best epoch 和原始完整 scientific lineage/hash。实验 Git commit 和产物发布 commit 分开记录；当前产物二者均有记录，并标明 dirty worktree。兼容旧目录时，如原 runner 未记录实验 commit 则明确保留 unknown，不拿当前 commit 冒充原实验版本。

`trajectories.npz` 包含 GT、Full_RGB 和每个 stride 的 sparse/ours，共八条轨迹，字段为 `<name>__timestamps_ns`、`__translation`、`__quaternion_xyzw`。七条 SLAM 轨迹的原始 dtype、点数、timestamp、pose 逐项完全相同；GT 保存原 reference 的全部点，遵循 Phase 1 reader 的 timestamp/quaternion 语义。另保留原 schedule 的 roles/intervals/query payload 压缩数组，以免清理旧 schedule JSON 时丢失身份与时间信息。`anchor_budget_artifacts.reconstruct_trajectory` 和 `reconstruct_schedule` 可完整还原。

轨迹图仅应用已保存的 Sim(3)，不重新 fitting 或 evaluation；三个 panel 使用同一 XY 坐标范围、aspect、颜色和起终点标记。Tradeoffs 的六个 panel 只画已有实测点，不插值。所有绘图仅依赖 JSON+NPZ。迁移先在 `/tmp` 验证替代产物、checkpoint 字节、轨迹数组和数值，再发布并删除已审计旧文件；原目录暂存于 `/tmp/anchor_budget_before_publish_*/artifacts` 以便恢复。正式目录不保留逐帧 worker telemetry 或多份 JSON/CSV。

Offline true JEPA / FMap 在 checkpoint selection 之后提取并在 deployment 前关闭，不进入正式 Ours capability。

ATE 和 Sim(3) fitting 使用每个 stride 的 GT-associable anchor population，Sparse/Ours 完全配对；跨 stride 的 scoring timestamps 因 anchor schedule 而不同，必须连同 population hash/count 解释。Full RGB 的一次性 ATE 使用冻结的 stride-5 population，同时保存 dense diagnostics。RPE 保留 1 秒 horizon 和 1 ms tolerance：stride 3 经 GT association 后没有合法 pairs，记录 null 和 pair count 0；stride 5/10 分别有 362/181 pairs。不会插值、改变 horizon 或替换成 frame-offset RPE。默认 Phase 1 无 pair 时仍报错，只有 Phase 2 显式启用空 RPE population。

Stride-5 hard gate 只读比较原 ratio schedule/split 与 canonical artifacts：candidate/anchor/hidden identities、interval/query alpha/delta、timestamp/order、split membership、boundary drops、exactly-once order hash 和 communication counts/bytes。`anchor_budget_canonical_sha256.json` 固定全部 22 个现有 Phase 1 artifacts；正式执行前后验证 SHA256，不更新 legacy lineage。本次迁移再次只读核对这些 artifacts，不重跑 GPU 等价性或正式实验。

实现审计与检查范围见 [ANCHOR_BUDGET_IMPLEMENTATION.md](ANCHOR_BUDGET_IMPLEMENTATION.md)。Phase 2 测试位于 `research/src/test_anchor_budget.py` 和 `test_anchor_budget_artifacts.py`；本次产物收口执行全部 117 条 CPU/unit tests，均通过。未运行 GPU smoke、fresh training 或 SLAM。
