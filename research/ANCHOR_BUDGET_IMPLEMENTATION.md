# Anchor Budget implementation audit

本文件记录实现审计与产物收口，不推断 Phase 2 实验结论。分支 `feat/phase2-anchor-budget`。首轮 3/5/10 实验已由用户完成；本次只聚合与绘图，没有 commit、fresh training、GPU smoke 或 SLAM run。

## 修改前 variable hidden-query 审计

| 部件 | 现状 | Phase 2 处理 |
| --- | --- | --- |
| `protocol.post_bootstrap_ratio_roles` | 已支持任意 ratio；默认 .2，accumulator 保留 bootstrap 和第一个 post-bootstrap anchor | `stride_roles` 直接调用原函数，ratio=1/K |
| `run_h2` / canonical H2 YAML | 校验 `post_bootstrap_anchor_interval=5`，训练 ratio=.2，expected split 219/72/73 和 query 876/288/292；主程序固定五个运行 conditions | 不改 Phase 1 runner；新增功能命名 CLI，只有 Full/Sparse/Ours 三个 SLAM runs 加 GT reference |
| `predictor.build_anchor_intervals` | hidden tuple 动态长度；alpha 由真实整数 timestamp 算，delta 是整个 bracket 时长，ordinal 从 1 开始 | 原样复用 |
| `predictor.effective_records` | 最后 complete closing anchor 后的 tail 排除，未写死 4 | 原样复用 |
| `split_anchor_intervals` | 不是写死 4，但每次按 interval 数切 60/20/20，不适用于跨 stride 公平比较 | 仅作为 canonical 提取/等价基准；Phase 2 按冻结 timestamp 区域分配 |
| `h2_training` batch/validation/train | 逐 interval/query 构造 tensor，`len(target)` 聚合；没有固定 4 query shape；seed/optimizer/epoch/loss 硬编码为 canonical recipe | 训练函数不改；只给 held-out diagnostics 增加默认关闭的 horizon scalar collector |
| resident training / correspondence precompute | 动态 identities；endpoint pair 每 interval 计算一次，repeat count=`len(interval.hidden)` | 原样复用，包括 half correspondence storage 和 resident batching |
| predictor architecture | `[B,768,H,W]`、time输入 `[B,2]`；B 动态，768/hidden_dim/blocks 等是冻结模型维度 | 不改 architecture、loss、初始化或训练选择 |
| transport | B 动态；`COARSE_STRIDE=4` 是空间 transport lattice，clamp(4,12) 是 displacement 范围，均不是 hidden 数 | 不改任何 transport/calibration 算法 |
| `h2_deployment` | query count 动态；有历史报错文本 `closing anchor A5`，warmup alpha/delta=.5 是固定合成输入，不限定正式 query 数 | 保留现有实现与 Phase 1 文本兼容，不将 A5 文本视为算法限制 |
| `h2_pipeline` / worker | output shared slot 按 `max(len(i.hidden))` 分配；worker/packet 按实际 alpha 数和 hidden 长度切片 | 三 GPU pipeline 全部不改 |
| packet assembly | 每 query 一个 `[1,1,128,H,W]` FMap packet；5 是 tensor ndim，128 是 channel，不是 stride/count | 原样复用 exactly-once 顺序与 capability 边界 |
| evaluation | population source label 写死 k5；默认必须有合法 1秒 RPE pairs。stride3 GT-associable population 的 pairs=0 | source 参数默认保持 k5；Phase 2 标明实际 stride，显式允许 empty pairs，RPE null，ATE 算法不变 |
| summary/schema/lineage | Phase 1 condition roster、k5 source 字符串、旧 scientific contract 的 H1–H4/A5 名称固定 | 不改 Phase 1 registry/legacy contract；Phase 2 独立结果 schema/lineage 引用冻结方法，并绑定 stride |

没有发现科学计算依赖 batch 必须等于四个 hidden query。必要改动是独立实验编排、固定时间 split、结果与 checkpoint lineage、offline diagnostics 和空 RPE 的可表达性，不涉及新模型。

## 等价与公平性

Stride-5 hard gate 比较新 wrapper、旧 ratio=.2 路径、canonical INDEX split hash 和已保存 trajectories 的输入 identities/timestamps/roles。检查 interval 对象包含完整 query ordinal、alpha、delta，因此这些字段也 exact。Communication gate 比较 anchor count 与 encoded full/anchor bytes、reduction。Online gate 比较 closing-anchor emission 顺序、唯一性和已冻结 runtime 的 order SHA256，不声称做过 GPU packet 数值运行。

固定 split、三个 population 的准确数字见 RESEARCH.md，以及 `results.json` 的 `metadata.fixed_split`、`strides.K.schedule`。原 `PREPARATION.json` 的 canonical guards/stride5 等价性/split/population 已聚合保存。K=5 split 仍为 219/72/73，boundary drops 仍是 219/292。K=3 drops 为 365/366/486/487/488；K=10 drops 为 109/146。三个点的 tail 都是 2 个 candidates。

每次 `fresh_train_budget` 调用现有 `train_predictor` 从新模型开始。唯一科学自变量是 schedule/query horizon；train-only transport thresholds 会按各 stride 的 train endpoint pairs 自然重校准，算法保持相同。H1 不重训，训练 epoch/seed/AMP/optimizer/architecture/loss/batch policy 不变。原 tiny-overfit 是 Phase 1 feasibility gate，本阶段不重跑。所有 test diagnostics 都在 checkpoint freeze 后执行。

Formal execution 是一次 Full RGB，随后每个 stride fresh training → Sparse → Ours。所有 trajectory 都通过现有 `run_sequential_trajectory_jobs` 每次提交一个 task，在前一个进程完成和清理之后启动下一条。每个 predictor 加载既检查原 H2 state/architecture/calibration 完整性，又检查 Phase 2 stride/固定 split/完整 lineage，不接受 zero-shot 替代。正式 artifacts 已存在时禁止自动重跑/覆盖。

## 测试与检查范围

新增 `test_anchor_budget.py` 的 19 条测试覆盖：独立旧 accumulator 与 stride5/split exact；3/5/10 populations 和 queries；固定 timestamps、no shared anchors、tail；实际 ratio 与 encoded bytes；科学配置继承；canonical SHA256 和篡改拒绝；variable predictor batch；transport/correspondence repeat/order；variable packet assembly 和 pipeline shared slots；hidden RGB capability 拒绝；stride/legacy checkpoint 拒绝；empty RPE 与 ATE；horizon aggregation；CLI 默认不执行；Full RGB 一次与 fresh per-stride/串行 Sparse/Ours 编排。

初次实现只做语法/静态核对；本次产物收口已执行全部 117 条 CPU/unit tests，全部通过，包含既有 Phase 1 测试、19 条 anchor budget 测试和新增 9 条 artifact 测试。未删除或放宽 Phase 1 科学断言。未做 GPU predictor/transport/packet 数值 smoke。

新增 `test_anchor_budget_artifacts.py` 覆盖原数值与所有轨迹数组完全还原、checkpoint 字节/lineage 不变、五文件正式结构、未知文件与重复数值不一致时停止清理、checkpoint 冲突拒绝、仅 JSON+NPZ 重建图表且禁止调用模型/评价、应用已保存 Sim(3) 不改原数组、清理后默认 runner 不依赖旧目录、offline CLI 不触发 execution。

Canonical Phase 1 全部 22 个已存在 artifact 以修改前 SHA256 清单固定；代码中的 gate 可重复验证。任何 canonical artifact/config 漂移都会阻止正式运行，不修改原文件来解决不兼容。

## 产物收口与人工验收

Stride 3 没有合法 1 秒 RPE pairs，继续保留原 unavailable/null，没有改配对规则。如果旧目录含 `IMPLEMENTATION_CHECKS.json`，其静态记录原样迁入 `results.json → metadata.prior_implementation_checks`；该历史检查范围不代表本次测试状态。当前正式产物已记录实验 Git commit、发布 commit 和 dirty worktree；兼容缺少实验 commit 的旧产物时保留未知状态。Checkpoint 内原训练源码 hashes 和科学 lineage 全部保留。

新增 `anchor_budget_artifacts.py` 负责无损聚合、验证和发布；`anchor_budget_figures.py` 仅从正式 JSON+NPZ 画两张 PNG 并生成短摘要。`run_anchor_budget` 新增 `--consolidate-existing` 和 `--render-figures`，二者均不会进入 training/SLAM；默认命令可读取清理后的 compact 结果。未来显式执行时在 `/tmp` 完成旧格式 staging，成功后发布五文件结果，并将三个 checkpoint 移至独立目录。Phase 1 与 predictor/transport/schedule/pipeline 科学实现没有在本次收口中修改。

接续验收时五文件结构已经发布，因此未再次迁移或覆盖模型，只刷新图表/摘要并补全文档。旧目录统计取自当前 `results.json → metadata.migration` 的发布前清单；当前文件统计如下：

| 范围 | 文件数 | Bytes |
| --- | ---: | ---: |
| 旧 Phase 2 目录（含 checkpoint） | 36 | 42,960,424 |
| 当前正式 results（不含 checkpoint） | 5 | 2,723,595 |
| 独立 checkpoint 目录 | 3 | 37,252,386 |

旧 INDEX、独立 accuracy/horizon JSON/CSV、per-stride/per-condition JSON、七个独立 trajectory NPZ、独立 schedule/training/query JSON 和 results 内的模型已收口；其逐文件原 hash 仍在 migration 清单。高价值数值保留在 JSON，八条原始轨迹与 schedule payload 保留在 NPZ；只舍弃声明过的 worker/逐帧 debug 明细。摘要为 28 行，两张 PNG 为三 panel 轨迹图和六 panel 实测 tradeoffs 图。

接续只读验证还从聚合文件在内存中还原了七个原 trajectory NPZ、三个 schedule JSON、三个逐 query JSON 和两个全局数值表，重新序列化后的 SHA256 全部与发布前清单一致。三个 checkpoint SHA256/lineage 均通过验证。858 条 test queries、15 个 horizon 分组、每个 stride 的全部 30 个 epoch metrics 均存在。22 个 Phase 1 artifacts 和 37 个本次禁止修改的源码文件与收口前快照完全一致；compileall、git diff --check 及未跟踪文件的空白检查通过。

在仓库根目录和 dpvo 环境中人工验收（均为 CPU；不要添加 `--execute`）：

```bash
CUDA_VISIBLE_DEVICES="" python -m research.src.run_anchor_budget
CUDA_VISIBLE_DEVICES="" python -m research.src.run_anchor_budget --render-figures
CUDA_VISIBLE_DEVICES="" python - <<'PY'
from pathlib import Path
from research.src.anchor_budget_artifacts import FORMAL_FILES, inventory, validate_compact
root = Path('research/results/phase2-anchor-budget')
data = validate_compact(root)
assert set(inventory(root)) == FORMAL_FILES
print('PASS: 5 formal files; trajectory and checkpoint integrity verified')
print({name: row['point_count'] for name, row in data['metadata']['trajectories'].items()})
PY
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s research/src -t . -v
python -m compileall -q research/src
git diff --check
```

最后打开 `research/results/phase2-anchor-budget/figures/trajectories.png` 和 `tradeoffs.png` 查看三组轨迹与六个实测趋势 panel。详细数值只以 `results.json` 为准；PNG/SUMMARY 的显示舍入不改变原始指标。
