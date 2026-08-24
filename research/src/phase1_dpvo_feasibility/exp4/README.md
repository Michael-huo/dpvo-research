# Experiment 4: JEPA-DPVO Representation Bridge Validation

Experiment 4 冻结两个研究问题：frozen V-JEPA representation 能否映射到 DPVO FMap space，以及该映射的瓶颈是否来自 adapter capacity。代码不修改 DPVO upstream 或 `/home/hx/vjepa2-research`，不运行 DPVO trajectory、correlation/update/BA 或 V-JEPA finetuning。

## Fixed protocol

- EuRoC cam0，stride=2，seed=1234。
- Train=`MH_01_easy`，validation=`MH_03_medium`，test=`MH_05_difficult`。
- JEPA：V-JEPA 2.1 ViT-B/16 384，EMA final dense token `[576,768]`。
- Teacher：通过 Exp3 Oracle equivalence gate 的 DPVO Patchifier/FNet FMap `[128,120,188]`。
- Feature 无额外 normalization，float16 临时落盘，Dataset 返回 float32。
- Bridge：small Adapter 与 rank-16 LowRank Linear 均训练 50 epochs；LowRank 是检验 JEPA 与 DPVO FMap 之间简单低维线性对应关系的诊断 baseline。
- Capacity：small/medium/large Adapter 均训练 100 epochs。

完整 index 和 feature 只存在于单次运行的 staging 目录。成功、失败或 smoke 结束后都会删除，不提供 cache、resume、旧路径迁移或旧产物兼容。

## 1. Bridge Validation

```bash
/home/hx/miniforge3/envs/dpvo/bin/python -m \
  research.src.phase1_dpvo_feasibility.exp4.run
```

该入口依次执行 prepare、FMap Oracle sanity/extraction、JEPA extraction、Adapter/LowRank training、Random/Mean/LowRank/Adapter evaluation、单模态 temporal retrieval 和 report generation。

正式结果：

```text
research/results/phase1-dpvo-feasibility/exp4/final/
├── REPORT.md
├── metrics.json
├── figures/
├── adapter/{best.pt,metrics.json}
├── lowrank/{best.pt,metrics.json}
├── baseline/metrics.json
└── retrieval/metrics.json
```

若 `final/` 已存在，命令会在任何 GPU 工作前失败，不会覆盖。Baseline ladder 为 Random → Mean FMap → LowRank Linear → Adapter；报告展示实际 cosine/MSE/norm、LowRank–Adapter gap 和严格数值排序，不用人为阈值定义两者是否“接近”。

## 2. Capacity Scaling

```bash
/home/hx/miniforge3/envs/dpvo/bin/python -m \
  research.src.phase1_dpvo_feasibility.exp4.capacity_benchmark
```

该入口只比较 hidden dim 256/512/768 的 Adapter，固定 100 epochs，不包含 LowRank。每档只保留 `metrics.json`，不保留 checkpoint 或完整 history：

```text
research/results/phase1-dpvo-feasibility/exp4/capacity_scaling/
├── small/metrics.json
├── medium/metrics.json
├── large/metrics.json
├── summary.json
└── REPORT.md
```

报告以 best validation total loss 比较 large 与 small。相对改善 `<1%` 表示无明确 capacity 收益；`>=1%` 表示增加容量可能有效。实际数值和排序始终保留。
