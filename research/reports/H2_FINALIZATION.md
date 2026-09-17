# Ours/H2 收口交付

本次继续实施遵守「只编辑代码和文件」：未再启动 tests、compileall、CLI、dry validation、GPU 实验或训练，未 commit。下面的 promotion 和测试记录来自此前已完成的工作；当前修改仍需用户手动验证。

## 1. 正式数据流

Sparse RGB Anchors → V-JEPA Predictor → Predicted JEPA → current H1 Bridge → Predicted FMap → Uniform Budgeted Admission → Native DPVO。

`run_h2` 只加载 `research/checkpoints/h2-prediction/predictor.pt`，校验固定 state SHA、training lineage、architecture、recipe 和 calibration。缺失或不匹配直接失败，不重新训练、不自动训练回退。H2 发布只替换 results，checkpoint 是只读输入。运行时不读取 Anchor Budget 的 checkpoint、配置或结果；来源路径仅保存在审计 provenance 中。

已完成的 CPU promotion 保持全部 tensor 的 dtype/shape/value、architecture、training recipe 和 train-only calibration 一致。完整原始 metadata、训练 population、split、依赖和历史结果记录见 [H2_CANONICAL_PREDICTOR.json](H2_CANONICAL_PREDICTOR.json)。

| 对象 | SHA256 |
| --- | --- |
| 来源 predictor 文件 | `579aa66be224bb6ae5b92391ee158ac377524a3a425b7f534e46a058f82439b3` |
| 目标 H2 predictor 文件 | `de37a7fd6c5a172b6de969484b24a439b3952cd2cbf4ed4c8fde0595e049054a` |
| 不变的 predictor state_dict | `4d33a2dfdcc0eb6965d5bb9788ac272dd6a969f5d41bd7169111e7bc77f139c8` |
| 来源 training lineage | `b0af5bc3a39ec318134b4a24c1dd163dca0a9381178896d879ee4b321f986875` |
| 整理后的 training lineage | `2e2285a22f869a2674fe4c81428e2a4150abd82fbfd062b7981dd498426277d4` |

来源训练事实：MH01、stride 5、seed 1234、best epoch 12、220 个 train anchors。不存在长期 promotion CLI 或历史 checkpoint loader。

## 2. Exact Uniform rule

完整 bracket 内 `n = stride - 1`，`K = n // 2`。Hidden 按 timestamp 严格升序，选择一基 ordinal：

```text
1 + ((2*j + 1)*n // (2*K)), j = 0 … K-1
```

Stride 3/5/10 分别为 `[2]`、`[2,4]`、`[2,4,6,8]`；K=0 为空集。H2 与 Anchor Budget Ours 共用 `uniform_admission.py`，没有 policy registry、dispatch、CLI 或可选开关。

所有 hidden 完成 predictor、Bridge 和 FMap packet 构造，随后才在 native packet conversion/DPVO insertion 前筛选。所有 anchor 提交 DPVO；后续淘汰由 native culling 决定。JSON 区分 generated/received、inserted/discarded 和最终 node 数；identity/mask 存入压缩 NPZ。主 ATE 固定 anchor population，secondary evaluation 使用预定 admission population。H2 和 Anchor Budget 发布代码均校验保存的 admission 数组。

## 3. 删除的研究代码与 DPVO hooks

已移除 `run_prediction_reliability.py`、`run_backend_study.py`、`prediction_backend.py`、`reliability_admission.py`、`reliability_correlation.py`、`reliability_repeats.py`、`reliability_results.py`、`reliability_selection.py`、`runtime_graph_diagnostics.py` 及其专用 tests。`reliability_rng.py` 的最小通用 RNG 部分迁入 `observation_sampling.py`。

由此删除 random/confidence/oracle-correlation/stratified 选择、motion/graph-span、combined backend、研究结果 writer/IPC/audit 字段和对应入口。没有保留不做 admission 的 Ours condition。Full RGB、Sparse、Oracle、Anchor-only reference 保留。现有探索结果与总结保留，未将其迁移或标记为本次新结果。

DPVO 审计基准为 `16d5d5fc114778f891fc39b2c21b9dfd62d96377`。已恢复本轮新增 protected culling、factor scaling、source restriction、runtime graph diagnostics 和 experimental hooks。只读 diff 显示 `dpvo/`、`config/` 与该基准无差异；保留基准中已有 packet 接口。Correlation、Update、graph construction、BA 和 native culling 均沿用基准。

## 4. 保留的通用机制

保留 identity/role seed 派生的原 namespace/salt、frontend 与 track_packet 独立 RNG scope 和异常恢复；进程级 seed 初始化没有替代 observation 隔离。保留三卡映射、pipeline overlap、shared-memory IPC、pinned transfer、batch、CPU profile、resident/preparation 优化、通用训练和全部 prediction workload。

`predictor.py`、`transport.py`、`pipeline_worker.py`、`h2_pipeline.py`、`h2_training.py` 与指定基准无差异。实际参与运算的 confidence/coverage/warp/difference tensor、计算顺序、dtype、AMP 和 transport 分支未改。基准已有的通用 trace 保留。

## 5. Training / deployment lineage

`predictor_checkpoint.py` 提供 H2 与 Anchor Budget 共用的训练完整性验证。Training lineage 绑定真实 architecture、objective、recipe、seed、population/split、representation、coordinate/transport、JEPA 依赖和 calibration；不含 Uniform admission、H1 部署 Bridge 或执行硬件。

Deployment lineage 单独绑定 predictor state/training SHA、H1 Bridge、DPVO checkpoint/config、schedule、sampling、evaluation 和 exact admission contract。Admission 改变会改变 deployment hash，不改变 training hash。Config 固定 canonical predictor 身份与 admission；INDEX 和 sequence provenance 分别记录训练与部署 lineage。执行源码和硬件仍进入 execution provenance。

## 6. 验证状态

此前最后一次完整 CPU suite 记录为 156 项，155 项通过、1 项 fixture 构造报错：数值回归测试错误地请求了非正式 predictor 参数。此次已将 fixture 改为正式 768-channel 架构，但按最新要求没有重跑。此后补充的 admission round-trip/tampering 测试及源码修改也未运行。

因此不能将此前通过项视作最终工作区全绿。Compileall、diff whitespace、CLI help、dry validation 和最终 MH01 均留给用户执行。没有本次新的 ATE 或 runtime 测量。

## 7. Git diff 统计

编辑完成时 `git diff --stat`：**22 files changed, 399 insertions(+), 694 deletions(-)**。

该统计仅包含已跟踪文件；另有 8 个新增未跟踪文件：本说明、promotion 审计 JSON、`predictor_checkpoint.py`、`observation_sampling.py`、`uniform_admission.py` 及三个相应测试文件。Promoted checkpoint 属于忽略的模型产物。删除的研究源码此前也是未跟踪文件，因此不会显示为相对 HEAD 的删除行。

## 8. 用户手动验证命令

在仓库根目录执行轻量验证：

```bash
DPVO_PY=/home/hx/miniforge3/envs/dpvo/bin/python
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m unittest discover -s research/src -t . -v
"$DPVO_PY" -m compileall -q research/src dpvo
git diff --check
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m research.src.run_h0 --help
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m research.src.run_h1 --help
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m research.src.run_h2 --help
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m research.src.run_anchor_budget --help
CUDA_VISIBLE_DEVICES="" "$DPVO_PY" -m research.src.run_anchor_budget --strides 3 5 10
git diff --stat
```

完整 suite 已包含 canonical checkpoint/lineage 的 CPU 加载、禁止训练的 mock H2 execution、RNG、Uniform、DPVO 基准和 compact artifact 校验。需要单独定位这部分时：

```bash
CUDA_VISIBLE_DEVICES="" /home/hx/miniforge3/envs/dpvo/bin/python -m unittest -v \
  research.src.test_checkpoints research.src.test_canonical_h2 \
  research.src.test_uniform_admission research.src.test_observation_sampling \
  research.src.test_anchor_budget_artifacts research.src.test_architecture
```

最终正式验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 /home/hx/miniforge3/envs/dpvo/bin/python \
  -m research.src.run_h2 --sequences MH_01_easy
```

此命令只评估已验证 canonical predictor，不重新训练。固定同一 checkpoint 和 scientific seed，独立运行三次，并在每次下一轮替换前保存 `research/results/h2-prediction/` 完整 compact results。核对固定 anchor ATE 的 median/IQR，耗时读取 Ours matched online wall；验收不高于原约 70–80s，更快属于正常收益。历史 median 0.207489 m、IQR 0.003048 m 仅作为复现参照。
