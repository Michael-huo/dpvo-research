# Research Method

当前正式 Ours/H2：Sparse RGB Anchors → V-JEPA Predictor → Predicted JEPA → current H1 Bridge → Predicted FMap → Uniform Budgeted Admission → Native DPVO。

RGB 只在 anchor 上传。V-JEPA 提取 block-5 representation，predictor 利用已上传的 bracket endpoints 预测 hidden JEPA；frozen H1 bridge 将它映射成 DPVO FMap-only visual state。Anchor 保留 native DPVO frontend，hidden observation 不具有 RGB、offline reference 或 groundtruth capability。Closing anchor 必须真实到达并完成编码，随后按时间戳顺序消费 buffered hidden observations，每个 candidate 接收恰好一次。所有 hidden 完成 predictor、Bridge 和 FMap packet 构造，随后仅在 DPVO insertion 前做固定 Uniform admission；所有 anchor 均提交 native DPVO。因此当前方法是 delayed/bracketed、non-causal deployment，`timestamp_causal=false`，不声明 causal 或 strict real-time。

源码按功能平铺在 `research/src/`。上游 DPVO 安装和 Demo 见仓库 README。Block-5 指 JEPA representation 的提取位置；H0/H1/H2 分别对应正式的观测、Bridge 和预测实验。


Uniform 对完整 bracket 采用 `n = stride - 1`、`K = n // 2`，按 hidden timestamp 升序选择一基 ordinal `1 + ((2*j+1)*n // (2*K))`，`j=0..K-1`。Stride 3/5/10 为 `[2]`、`[2,4]`、`[2,4,6,8]`；K=0 为空集。不存在策略开关。主 ATE 使用固定 anchor population；secondary native evaluation 使用预定 admission population，不要求被丢弃 hidden 存在 pose。JSON 保存计数、规则和 mask hash，NPZ 保存 identity/mask。

H2 固定 predictor 的 state SHA256 为 `4d33a2dfdcc0eb6965d5bb9788ac272dd6a969f5d41bd7169111e7bc77f139c8`。它是已验证 stride-5 predictor 的 CPU promotion，tensor、架构、训练配方及 calibration 完全保持一致。来源 SHA、完整 scientific metadata 和目标 SHA 见 [H2_CANONICAL_PREDICTOR.json](H2_CANONICAL_PREDICTOR.json)。历史验证 ATE median 0.207489 m、IQR 0.003048 m；收口后的 GPU 验证由用户运行，不将历史数值标记为本次新结果。

Observation sampling 保留固定 identity/role hash 派生、frontend 和 track_packet 的独立 RNG scope 及保存/恢复，不共享跨 observation RNG stream。Runtime 验收是相同 matched online timing 不高于当前约 70–80s；进一步变快属于正常收益。DPVO core 审计基准为分支创建时 merge-base `16d5d5fc114778f891fc39b2c21b9dfd62d96377`。

# Experiment Modules

H0 State、H1 Interface、H2 Prediction、Anchor Budget 是解耦实验模块，不是严格线性的阶段流程。允许修改 H2 → 回归 H2 → 回归 Anchor Budget → 需要时回到 H1。H2/Anchor Budget 复用兼容的 H1 canonical bridge，这是模型依赖，不要求读取 H0/H1/H2 旧结果。

## H0 State

回答 latent visual state 是否可以替代 RGB-derived state。比较 Full RGB、Sparse RGB、True FMap；固定 DPVO FMap-only/zero-context state contract。每次 fresh 评估本次请求的 sequences，不产生 checkpoint。

```bash
python -m research.src.run_h0 --sequences MH_01_easy
```

## H1 Interface

回答 JEPA representation 是否可以映射到 DPVO state。每次在 MH01 fresh 训练 bridge，使用本轮同一个 bridge 评估全部 requested sequences。MH03/MH05 是 frozen zero-shot evaluation。比较 Full RGB、Sparse RGB、True FMap、Oracle JEPA→Bridge。

训练保留 seed 1236 初始化、两次 30 epochs、第二 pass seed 1234、batch 4、AMP、AdamW、相同 epoch-local batch order 和 pass 间 optimizer/scaler reset。正式 checkpoint 仍取第二 pass 最后 epoch；最低 validation 的 `best_epoch` 仅作为诊断，结果另记录实际 `selected_epoch` 和 selector。

```bash
python -m research.src.run_h1 --sequences MH_01_easy MH_03_medium MH_05_difficult
```

## H2 Prediction

回答 sparse-anchor predicted hidden JEPA 能否驱动 VSLAM。只加载兼容的 canonical H1 Bridge 和经过权重 SHA 绑定的 H2 canonical predictor，不训练或重新计算 calibration，也不读取 H1 results。评估 Full RGB、Sparse RGB、Anchor JEPA only、Oracle JEPA，以及采用 Uniform Budgeted Admission 的 Predicted JEPA/Ours。通用训练实现仍供 Anchor Budget 逐 stride 训练使用，保持原 30 epochs、seed 1234、interval batch 2、AMP、loss/AdamW 和 lowest-validation-total checkpoint selection。Tiny-overfit 通用实现保留；Anchor Budget 维持既有行为，不重复运行此 gate。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m research.src.run_h2 \
    --sequences MH_01_easy MH_03_medium MH_05_difficult
```

## Anchor Budget

研究 anchor upload budget / prediction horizon 对 communication、accuracy、latency 的影响。默认 MH01、strides 3/5/10，也接受新的 distinct integer strides ≥ 2，例如 5/7/8。每个 stride fresh 训练独立 predictor，复用 H1 canonical bridge；Ours 使用与 H2 相同的固定 Uniform admission；不复制 bridge，不使用其他 stride 的 predictor zero-shot 替代训练。其训练不会覆盖 H2 canonical predictor。

配置直接读取 `h2_prediction.yaml`，再用 `anchor_budget.yaml` 和请求的 stride 局部覆盖 schedule、固定 split、output；不另存 H2 科学配置副本。训练 architecture、loss、optimizer、batch、epochs、AMP 和 seed 均沿用 H2。train-only transport calibration 按各 stride 的 train endpoints 计算。

每次先运行一次标准 Full RGB，再逐 stride fresh train → Sparse → Ours。GT 仅是 reference，Sparse/Ours 使用同一 schedule 和配对的 GT-associable anchor evaluation population。跨 stride population 可以不同，必须结合 hash/count 解读。Full RGB 保留固定 stride-5 evaluation population。RPE 仍使用 1 秒 horizon、1 ms tolerance；这个模块显式允许没有合法 pairs 时记录 null/count 0，不插值或替换 horizon。

Schedule 使用原 ratio accumulator，ratio=1/K。Bootstrap candidate 0–7，第一个 post-bootstrap anchor 为 candidate 8；缺少 closing anchor 的尾段按原完整 interval 规则排除。不同 stride 保留各自自然尾段，不为匹配 Full RGB 人为截断或提升 anchor。固定时间区域直接保存在 YAML；只有两端 anchor 均处于同一区域的完整 interval 才进入该 split。

| Split | Candidate（inclusive） | Frame ID | Timestamp ns |
| --- | --- | --- | --- |
| Train | 8–1103 | 16–2206 | 1403636580563555584–1403636690063555584 |
| Validation | 1108–1468 | 2216–2936 | 1403636690563555584–1403636726563555584 |
| Test | 1473–1838 | 2946–3676 | 1403636727063555584–1403636763563555584 |

Trajectory evaluation 仍为整段 in-sequence feasibility；held-out/test 只用于训练后的 representation diagnostics，不声称 held-out trajectory generalization。Stride-5 guard 直接比较当前 ratio schedule/interval/split、冻结时间区域与 exactly-once order，不依赖已经生成的 H2 results。

```bash
# CPU preparation / dry-run：重新构造本次请求，在 /tmp 输出 PREPARATION.json。
python -m research.src.run_anchor_budget --strides 5 7 8

# 正式运行：fresh 训练与评估，成功后替换本模块 canonical 集合。
CUDA_VISIBLE_DEVICES=0,1,2 python -m research.src.run_anchor_budget --strides 5 7 8 --execute
```

# Configs and Artifacts

```text
research/configs/
  h0_state.yaml
  h1_interface.yaml
  h2_prediction.yaml
  anchor_budget.yaml
research/results/
  h0-state/
  h1-interface/
  h2-prediction/
  anchor-budget/
research/checkpoints/
  h1-interface/bridge.pt
  h2-prediction/predictor.pt
  anchor-budget/predictor_stride_<K>.pt
```

唯一公开实验 CLI 是 `research.src.run_h0`、`run_h1`、`run_h2`、`run_anchor_budget`。`parallel_runtime`、`pipeline_worker`、`jepa_worker` 仅是内部 fresh-process worker 入口。旧产物由用户手动删除；当前代码不读取、迁移或合并它们。

四个正式 runner 的结果发布使用 **fresh-current-replace**（metadata 标识为 `fresh_current_canonical_replace`）。已有产物不会阻止 fresh execution。本次 requested sequences/strides 就是完整的新 canonical 集合；不保留未请求的旧项。

Evaluation、绘图和 artifact 验证在临时 staging 完成；H1 与 Anchor Budget 的训练也使用 staging。Results/checkpoint 的发布 staging 分别创建在目标目录所在文件系统。成功后才 rename：旧目录 → 临时 backup，staging → canonical；涉及两个目录时，在任一 rename 失败后逆序回滚两者。发布完成后删除 backup，staging 在成功或失败时均清理。父目录 advisory lock 串行化 publisher，不留下锁文件或 generation 目录。每个目录 rename 原子；跨 results/checkpoints 的两次 rename 不是一个文件系统级原子操作，因此并发 reader 可能短暂看到路径缺失。该事务处理 Python 异常和中断，不承诺断电/SIGKILL 恢复。

H2 的 fresh execution 仅替换结果集，canonical predictor 是经过 SHA 绑定的只读输入；缺少或不匹配时直接报错，不自动训练、查找其他模型或读取 Anchor Budget 路径。

H0/H1/H2 继续保留 `INDEX.json`、aggregate `SUMMARY_H*.md`、requested `sequences/` 下的结果 JSON、summary、trajectory NPZ/PNG，以及 H1/H2 必要的 feature diagnostics。results 禁止模型文件；H1/H2 的 INDEX 记录 repository-relative checkpoint path、SHA256、scientific lineage、seed、best epoch 和 selector。模型只在 checkpoints 内。

Anchor Budget 只发布：

```text
anchor-budget/
  SUMMARY.md
  results.json
  trajectories.npz
  figures/trajectories.png
  figures/tradeoffs.png
```

`results.json` 是唯一正式数值真源。NPZ 聚合 GT、Full_RGB、每个请求 stride 的 sparse/ours trajectories，以及压缩 schedule identities/intervals。绘图仅消费 JSON/NPZ 和已保存的 Sim(3)，不重新拟合或评价。训练、worker、merge、diagnostic 中间文件不会进入 canonical results。

Predictor training lineage 只绑定 architecture、objective、训练配方、representation/coordinate/transport、seed、实际 train/validation population、split 和 train-only calibration。Deployment lineage 单独绑定 predictor state/training hash、H1 Bridge、DPVO/config、schedule、sampling、evaluation 与 admission。Uniform 变化只影响 deployment hash，不使训练内容相同的 predictor 失效。Anchor Budget 也使用此分离契约，并绑定 stride 与实际 train anchor population。产物路径、可见 GPU 数、物理 GPU ID 和执行源码 hash 不作为科学兼容条件。执行源码 hash、资源分配和硬件信息单独进入 provenance。

# Execution Resources

`CUDA_VISIBLE_DEVICES` 是唯一外部 CUDA 资源选择入口。`CudaDevicePool` 只依据 `torch.cuda.device_count()` 构造可见 logical devices。primary 为 logical cuda:0，worker devices 为其余可见卡；不把 logical ordinal 当成 physical GPU index。重排序 `CUDA_VISIBLE_DEVICES=2,0,1` 后，科学代码仍只使用 logical 0/1/2。

| 可见 GPU 数 | Training / preparation | 当前 H2 / Ours online |
| --- | --- | --- |
| 1 | primary forward/backward + resident store；preparation 串行在 primary | fail closed |
| 2 | primary 训练；preparation 按完整原 batch/独立 interval 分到两卡 | fail closed |
| 3 | primary 训练；preparation 分到三卡 | Stage C=logical 0，predictor=logical 1，V-JEPA=logical 2 |
| 4+ | primary 训练；preparation 使用可见池 | 只用 logical 0/1/2，其余卡不加入 online |

当前 H2/Ours 必须至少 3 visible GPUs；不足时明确报错：`current online runtime requires 3 visible CUDA devices; single-GPU runtime will be evaluated separately`。Step C 尚未实施，当前没有新 single-GPU online candidate。

H0/H1 controls、H2 Full/Sparse/Oracle/Anchor-only、Anchor Budget Full/Sparse 都在 primary 以单 DPVO instance 顺序运行；不并行多个 condition 或 sequence。H2/Ours 仍使用原三阶段 bounded pipeline、CPU shared-memory IPC、pinned host transfer 和固定 CPU profile。CPU/NUMA affinity 优先按 logical device 对应 PCI topology 选择，读取失败采用已有 cpuset fallback；GPU 模型/UUID/PCI 查询均为 best-effort telemetry，不构成 admission gate。

Preparation 维持完整原 JEPA batch，不切碎 batch；correspondence 按独立 interval 分片。合并校验 duplicate/missing identity、native dtype、identity order 和 ordered content hash。H1 train/validation、H2 development store 保持 GPU-resident，不使用 DDP，不改 batch/AMP/RNG/optimizer/selection。显存不足自然 OOM，没有 free-VRAM gate 或自动 batch 修改。

所有 worker 使用现有 fresh Python subprocess（独立解释器，不 fork CUDA coordinator），继承原 visibility mask，在模型/stream 创建前只选择获分配 logical device；嵌套 JEPA sidecar 沿用该 ordinal。物理 UUID/PCI 由不创建 CUDA context 的 driver 查询采集，失败时记录 unknown，不猜测映射。标准 matched online timing 和 offline preparation/training/diagnostic time 继续分开报告。

# Validation

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s research/src -t . -v
python -m compileall -q research/src
git diff --check
python -m research.src.run_h0 --help
python -m research.src.run_h1 --help
python -m research.src.run_h2 --help
python -m research.src.run_anchor_budget --help
```

A+B 的配置逐字段与科学源码 AST 核对、mock/spawn/transaction 验证结果见 [STEP_AB_VALIDATION.md](STEP_AB_VALIDATION.md)。Anchor Budget 的 compact artifact 实现约定见 [ANCHOR_BUDGET_IMPLEMENTATION.md](ANCHOR_BUDGET_IMPLEMENTATION.md)。
