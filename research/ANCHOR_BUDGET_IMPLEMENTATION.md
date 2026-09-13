# Anchor Budget implementation

Anchor Budget 是独立实验模块，直接使用 H2 scientific settings 和 H1 canonical bridge。`anchor_budget.py` 构造 stride override、固定时间 split、原始 PNG 通信计数和 stride-5 schedule/split guard；不依赖任何旧 H2 result 或 checkpoint layout。

`anchor_budget_training.py` 每个 stride fresh 调用原 `train_predictor`，保留 architecture、loss、30 epochs、seed 1234、AMP、interval batch 2、AdamW 和 best-validation selection。开发集 preparation 使用 logical device pool，训练和 GPU-resident development store 在 primary；test diagnostics 只在 checkpoint freeze 后执行。Lineage 绑定 stride、实际 train anchors、固定 split、scientific config、H1/DPVO/V-JEPA hashes 和 train-only calibration。

`run_anchor_budget.py` 在私有 staging 内先运行一次 Full RGB，再逐 stride fresh train → Sparse → Ours。所有 DPVO trajectory 顺序执行。Sparse 在 primary；Ours 复用当前 H2 三阶段 pipeline：logical 0 Stage C、logical 1 predictor、logical 2 V-JEPA，至少三张可见 GPU。默认命令仅作本次请求的 CPU preparation；`--execute` 才运行训练/evaluation。

`anchor_budget_results.py` 只为本轮 staging 生成中间数值表、训练引用和 sequence/stride artifacts。`anchor_budget_artifacts.py` 无损聚合到一个 JSON/NPZ，验证原轨迹 dtype、timestamp、pose、schedule/query payload、checkpoint bytes/hash/lineage、epoch/horizon 数值及重复表一致性；不训练、重新拟合 alignment 或计算评价指标。

`anchor_budget_figures.py` 从 JSON/NPZ 和已保存 Sim(3) 绘制两张总图，支持任意请求 stride 和单 stride。正式结果严格为 `SUMMARY.md`、`results.json`、`trajectories.npz` 和两张 figures。训练日志、模型、worker 输出和中间表不发布。Canonical checkpoint 为 `research/checkpoints/anchor-budget/predictor_stride_<K>.pt`；H1 bridge 仅引用，绝不复制。

`artifact_runtime.py` 负责 staging 生命周期和跨 results/checkpoints 的 rename/rollback。已有结果/模型不会阻止重跑，成功后的集合严格等于请求集合。所有构建/验证失败保留旧 canonical。无旧目录 migration、旧 checkpoint 兼容表、consolidate-existing CLI 或冻结旧产物 manifest。

完整方法、CLI、结果布局和资源协议见 [RESEARCH.md](RESEARCH.md)；本次验证证据见 [STEP_AB_VALIDATION.md](STEP_AB_VALIDATION.md)。
