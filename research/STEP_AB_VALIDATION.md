# Step A + Step B validation

基线：`refactor/experiment-runtime-architecture`，HEAD `5578ec04698f80d92b06bc89d22ff200fc875622`。未 commit，未执行 Step C，未运行训练、完整 MH01 或正式 Anchor Budget evaluation。所有模拟产物都在 `/tmp`，未发布 canonical 实验结果。

1. **去除工程 Phase 组织。** 当前 configs、results/checkpoint 引用、CLI 标题、runtime schema/exception metadata、scientific contract 顶层组织字段均使用功能名。旧 `phase1_feasibility_h0/h1/h2.yaml`、`phase2_anchor_budget.yaml` 已重命名。`RESEARCH.md` 以 Research Method / Experiment Modules 为主体，明确四模块解耦。唯一保留的活跃 `phase2` 符号是外部 V-JEPA repository 的 `load_phase2_encoder(device)` API；已检查其将 device 传给 `load_vitb_encoder`，不把它误改成当前工程模块名。旧词还会出现在测试的拒绝用例及本次 before/after 审计证据中。

2. **新 configs。** `research/configs/{h0_state,h1_interface,h2_prediction,anchor_budget}.yaml`。逐字段核对得到 196 个科学 leaf 值相同，详情见下表及 `STEP_AB_SOURCE_AUDIT.json`。允许差异仅有 experiment name、output/bridge/config 路径、两个去 Phase 的来源标签；删除了只用于旧产物冻结的 `canonical_h2_index`、`canonical_artifact_manifest`、`canonical_artifact_manifest_sha256`、原整份 YAML 的 `canonical_h2_config_sha256`。当前 scientific config 通过剔除 execution/artifact path 后的 fingerprint 进入严格 lineage。

3. **新 results。** `research/results/{h0-state,h1-interface,h2-prediction,anchor-budget}/`。当前运行完全不依赖旧目录。本轮未创建/替换这些 canonical 结果；用户可在验收后自行删除旧产物并重新生成。

4. **新 checkpoints。** `research/checkpoints/h1-interface/bridge.pt`、`h2-prediction/predictor.pt`、`anchor-budget/predictor_stride_<K>.pt`。H0 无 checkpoint。H2/Anchor Budget 只引用同一个 H1 canonical bridge，不复制它。

5. **公开 CLI。** 仍只有 `python -m research.src.run_h0`、`run_h1`、`run_h2`、`run_anchor_budget`。前三者支持本次 sequence 集合；Anchor Budget 支持 distinct strides ≥2，保留默认 CPU preparation 与显式 `--execute`。标题分别为用户指定的 H0 State / H1 Interface / H2 Prediction / Anchor Budget 文本。内部 worker CLI 不作为实验入口；已删除 `--consolidate-existing` 和直接覆盖 canonical 的 `--render-figures` 分支，绘图函数仍供 staging/独立包使用。

6. **fresh-current-replace。** 四个 runner 使用 `fresh_current_canonical_replace` metadata 标识。公共实现位于 `artifact_runtime.py`；`registry.py` 使用同一 publisher。显式正式运行总是 fresh，移除所有“已有结果禁止 rerun”的 gate。

7. **发布事务。** 训练、evaluation、绘图及 manifest/hash/lineage 验证先完成；results 和 checkpoint staging 分别位于各自目标文件系统。Publisher 取得父目录 advisory locks 后，以 rename 暂存旧目录、替换新目录；任一 rename 失败逆序恢复两棵旧树，最后删除 backup。失败运行清理 staging，成功运行不保留旧 generation。H2 CUDA cleanup 已移到 publish 前。目录 rename 原子，但两个 canonical 目录之间不是单一文件系统原子提交；并发 reader 可能短暂遇到 missing path。没有实现断电/SIGKILL journal recovery。

8. **Anchor Budget compact 保持。** 严格只发布 SUMMARY.md、results.json、trajectories.npz、figures/trajectories.png、figures/tradeoffs.png。JSON 是唯一正式数值真源；NPZ 保存 GT、Full_RGB、请求 stride 的 sparse/ours，以及 schedule payload。测试实际执行 compact build/render/validate/publish，验证 `3/5/10 → 5/7/8 → 7` 后 JSON、NPZ、checkpoint 全部只保留新集合；单 stride 与新 stride 绘图均通过。中间多目录结构仅存在于私有 staging。

9. **删除的旧 compatibility。** 删除 `legacy_scientific_lineage.json` 和 legacy source-hash 映射逻辑、`anchor_budget_canonical_sha256.json`、旧 H2 INDEX/results/trajectory/hash manifest gate、旧产物 relocation/migration/in-place consolidation 分支、旧路径 `.gitignore` 规则。保留严格 exact scientific lineage、architecture、state hash、calibration 和 cross-stride 拒绝检查。没有迁移、复制或删除用户旧结果/checkpoint。

10. **CudaDevicePool。** `cuda_devices.py` 集中提供 visible_devices、device_count、primary_device、worker_devices、preparation_devices、online_mapping。只依据当前进程 `torch.cuda.device_count()` 和 logical ordinal 分配资源。UUID/PCI 由 CUDA driver 的无 context 查询获取；不会将 visibility 数字 selector 猜成 nvidia-smi 物理 index，telemetry 失败记录 unknown/error。

11. **1/2/3/4 卡。** 1 卡 primary=0，preparation 串行；2 卡 preparation 分片，但 H2/Ours fail closed；3 卡使用当前 online 0/1/2；4 卡 preparation 可用 0/1/2/3，而 online 仍只用前三张。小于三卡的正式 H2/Ours 在准备/训练前明确报错：`current online runtime requires 3 visible CUDA devices; single-GPU runtime will be evaluated separately`。

12. **训练与 preparation。** H1/H2/Anchor Budget forward/backward、optimizer、GPU-resident train/validation/development store 都在 primary。Extraction 按完整原 JEPA batch 分到可见池，correspondence 按独立 interval 分片。CPU/mock 验证 1/3 卡合并后 identity/order/native float16 payload/hash 完全一致；执行 provenance 记录 shard→logical device→identity/interval 分配。没有 DDP、batch 调整、显存 hard gate。原 AMP/RNG/optimizer/selection 保留。

13. **当前 H2 online。** Stage C/DPVO=logical cuda:0，predictor=logical cuda:1，V-JEPA=logical cuda:2。原 bounded pipeline、两槽 host IPC、pinned transfer、队列顺序、三阶段计算及单 DPVO 协议保留。控制轨迹始终在 primary 串行，不并行 sequence/condition。此处等价证据是 source/AST 与 mapping/dispatch tests；未宣称完成真实 GPU 的数值 A/B。

14. **重排序。** `CUDA_VISIBLE_DEVICES=2,0,1` 的 mock launcher 与 fresh spawn 测试通过：mask 原样继承，内部仍选择 logical 0/1/2。独立 driver-identity mock 特意令 physical indices 与 mask 数字不同，验证通过 UUID 映射而非猜测物理卡号。

15. **Worker。** 保持现有 fresh Python subprocess start 方式，不 fork 已初始化 CUDA 的 parent。parallel worker 从内部 task、pipeline worker 从内部 logical-device 参数、嵌套 JEPA sidecar 从 worker config 读取分配，在模型/stream 创建前 set_device。Fresh spawn 测试覆盖 1/2/3/4 pool，只调用获分配设备；两个实际 Python 环境（DPVO/V-JEPA）均通过 fresh worker import/CLI help。没有重写子进程 CUDA_VISIBLE_DEVICES。

16. **Lineage。** 去除 Phase 标签导致新 contract hash；不兼容旧 checkpoint 是本轮明确行为。H1/H2 仍严格检查 architecture、state、training input/lineage、split、coordinate、transport calibration、seed/protocol 和依赖模型 hash。Anchor Budget 严格绑定 stride、实际 train anchors 和 scientific config，原全源码 hash 从 scientific compatibility 移到 execution provenance。新增测试证明 runtime GPU count/device/physical ID 与 artifact path 不影响 scientific config；batch 变化仍改变它。

17. **科学源码保护。** 7 文件共 128 个函数的 before/after AST 核对通过。5 文件逐字节相同；h2_training 仅 MPLCONFIGDIR 临时路径标签改变；h2_pipeline 仅 default allocation 改为 pool、cleanup exception 字段去 Phase、usage provenance 的 encoder logical ordinal 改正。规范化只消除这四处明确允许的非科学差异，不忽略数值常量、运算、控制流或 tensor 操作。DPVO/upstream config worktree 无差异。原始/规范化 SHA256、函数数和 runner scientific helper 核对见 JSON 审计。

18. **CPU/unit tests。** 完整 `unittest discover -s research/src -t . -v`：136 tests，OK (skipped=1)，45.682 s。唯一 skip 是新 layout 的 canonical H1/H2 checkpoint 尚不存在的可选只读验证；synthetic checkpoint integrity/lineage 拒绝测试仍执行。还执行了最终 cleanup 顺序调整后的 targeted runner/architecture/checkpoint 回归。覆盖已有结果替换、请求子集、失败保留、每一步 rename 回滚、staging 清理、模型后缀禁入 results、旧路径隔离、全部公开 CLI mock dispatch、正式 runner 准备失败生命周期、pool 与 fresh worker import/spawn。未运行正式实验。

19. **其他验证。** compileall、git diff --check 均通过；四个公开 CLI --help 均 exit 0，无 Phase 标题。真实数据的 Anchor Budget CPU dry-run `--strides 5 7 8` 成功，只在 /tmp 生成准备信息，没有训练/SLAM。只读复核 22 个旧 H0/H1/H2 canonical artifacts，全部匹配重构前已有 manifest。当前 nvidia-smi 无法连接 NVIDIA driver，所以没有执行真实 GPU smoke；这一环境限制没有成为实验代码的 telemetry admission gate。

20. **Diff。** 以下是标准 git diff --stat（只统计 tracked changes，重命名的新文件尚未 git add）。另有四个新 YAML、两个 runtime 文件、两个 test 文件以及本报告和 SOURCE_AUDIT.json，共 10 个 untracked 新文件。未改真实 Git index，未 commit。

21. **Step C 前。** 未发现 CPU/mock/AST 验证中的代码阻塞项。进入真实 GPU A/B 之前应先恢复本环境 GPU driver 可用性，再在 GPU 机器补做本轮允许的极短 1/3 卡 preparation 与当前三卡 worker mapping smoke，确认真实 CUDA/context 与 native payload hash；本轮没有测试或实现 single-GPU online candidate。

## Config field audit

| 新 config | 相同科学 leaf 值 |
| --- | ---: |
| `h0_state.yaml` | 24 |
| `h1_interface.yaml` | 62 |
| `h2_prediction.yaml` | 80 |
| `anchor_budget.yaml` | 30 |

具体字段变化包含在 [STEP_AB_SOURCE_AUDIT.json](STEP_AB_SOURCE_AUDIT.json)，没有隐藏配置覆盖。

## Protected source audit

| 文件 | 函数数 | 结果 |
| --- | ---: | --- |
| `predictor.py` | 17 | 逐字节相同 |
| `transport.py` | 19 | 逐字节相同 |
| `h1_training.py` | 15 | 逐字节相同 |
| `h2_training.py` | 23 | 限定 metadata/device plumbing 后 AST 相同 |
| `jepa_fmap.py` | 18 | 逐字节相同 |
| `h2_deployment.py` | 16 | 逐字节相同 |
| `h2_pipeline.py` | 20 | 限定 metadata/device plumbing 后 AST 相同 |

## git diff --stat

```text
 .gitignore                                         |   6 -
 research/ANCHOR_BUDGET_IMPLEMENTATION.md           |  84 +---------
 research/RESEARCH.md                               | 185 ++++++++++-----------
 .../configs/anchor_budget_canonical_sha256.json    |  24 ---
 research/configs/phase1_feasibility_h0.yaml        |  26 ---
 research/configs/phase1_feasibility_h1.yaml        |  64 -------
 research/configs/phase1_feasibility_h2.yaml        |  89 ----------
 research/configs/phase2_anchor_budget.yaml         |  46 -----
 research/src/anchor_budget.py                      |  98 ++++-------
 research/src/anchor_budget_artifacts.py            | 101 ++++-------
 research/src/anchor_budget_figures.py              |  16 +-
 research/src/anchor_budget_results.py              |   6 +-
 research/src/anchor_budget_training.py             |  19 +--
 research/src/bridge_checkpoint.py                  |   5 +-
 research/src/canonical.py                          |   4 +-
 research/src/dpvo_backend.py                       |   2 +-
 research/src/efficiency_profiling.py               |  94 ++---------
 research/src/evaluation.py                         |   2 +-
 research/src/execution_runtime.py                  | 135 +++++----------
 research/src/h2_pipeline.py                        |   5 +-
 research/src/h2_training.py                        |   2 +-
 research/src/jepa_runtime.py                       |   1 +
 research/src/jepa_worker.py                        |  11 +-
 research/src/legacy_scientific_lineage.json        |  48 ------
 research/src/parallel_runtime.py                   |  68 ++++----
 research/src/pipeline_worker.py                    |  10 +-
 research/src/protocol.py                           |   2 +-
 research/src/registry.py                           |  68 ++++----
 research/src/run_anchor_budget.py                  |  93 ++++-------
 research/src/run_h0.py                             |  18 +-
 research/src/run_h1.py                             |  34 ++--
 research/src/run_h2.py                             |  48 +++---
 research/src/schema.py                             |   2 +-
 research/src/scientific_lineage.py                 |  21 +--
 research/src/staged_transfer.py                    |  22 +--
 research/src/test_anchor_budget.py                 |  30 ++--
 research/src/test_anchor_budget_artifacts.py       |  94 +++++++----
 research/src/test_architecture.py                  |   6 +-
 research/src/test_checkpoints.py                   |  39 ++---
 research/src/test_execution_runtime.py             |  51 +++---
 research/src/test_h0.py                            |   8 +-
 research/src/test_registry.py                      |  19 ++-
 research/src/training_runtime.py                   |   4 +-
 43 files changed, 549 insertions(+), 1161 deletions(-)
```
