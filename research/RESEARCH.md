# Phase 1 — Feasibility Analysis

## 环境前提

使用 `research/environment.yml` 中的 DPVO 环境（Python 3.10），并安装项目的 CUDA extensions。
扩展编译使用 CUDA Toolkit 12.1。运行前准备 EuRoC 数据、DPVO/V-JEPA 权重及正式 YAML 配置指定的路径。
上游 DPVO 的安装、Demo 和可选后端说明见仓库 README；它们不属于 Phase 1 的正式研究入口。

源码按功能平铺在 `research/src/`，以文件名表达职责，供后续研究阶段复用。Phase 用于科学实验组织和 results，不再作为源码 namespace；Phase 1 的 H0 State、H1 Interface、H2 Prediction 定义不变。正式配置仍为 `research/configs/phase1_feasibility_h0.yaml`、`phase1_feasibility_h1.yaml` 和 `phase1_feasibility_h2.yaml`。

唯一三个公开研究 CLI 为 `research.src.run_h0`、`research.src.run_h1`、`research.src.run_h2`。`jepa_worker`、`pipeline_worker`、`parallel_runtime` 的 module 启动方式仅供内部进程使用。源码布局迁移不改变科学 lineage 或 canonical checkpoint 兼容性；execution provenance 如实记录当前源码路径与 hash。

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
