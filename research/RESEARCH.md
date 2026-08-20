# RESEARCH

本文件用于记录基于 DPVO
的后续研究工作，包括环境部署、实验复现、算法改进和实验结果分析。

# 1. DPVO 环境部署与 Demo 复现

## 1.1 环境部署
说明：

-   使用 Miniforge/Conda 管理环境；
-   Python 版本固定为 3.10；
-   DPVO 自定义 CUDA extension 编译需要 CUDA Toolkit 12.1；
``` bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version
```
-   DPVO 使用源码安装，方便后续研究修改。

``` bash
#Setup and Installation
git clone --recursive https://github.com/Michael-huo/dpvo-research.git
cd dpvo-research

mamba env create -f research/environment.yml
conda activate dpvo

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install . --no-build-isolation

#Recommended - Install the Pangolin Viewer
./Pangolin/scripts/install_prerequisites.sh recommended
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
sudo ldconfig
cd ../..
pip install ./DPViewer --no-build-isolation
rm -rf Pangolin/build

# Classical Backend (optional)
sudo apt-get install -y libopencv-dev
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
pip install ./DPRetrieval
rm -rf DBoW2/build
```

## 1.2 下载运行 Demo 所需文件

### EuRoC 数据集

例如：
```
research/assets/datasets/euroc/MH_01_easy/
```
### 网络权重和其他配置文件

-   DPVO 模型：
dpvo-research/dpvo.pth

-   ORB Vocabulary：
dpvo-research/ORBvoc.txt

-   DPV-SLAM 长程回环相关权重：
```
~/.cache/torch/hub/checkpoints/
├── depth-save.pth
└── disk_lightglue_v0-1_arxiv.pth
```

## 1.3 运行 Demo

DPVO 的 `demo.py` 可直接处理图像序列或视频文件。基本调用格式如下：

```bash
python demo.py \
    --imagedir=<图像目录或视频文件> \
    --calib=<相机标定文件> \
    [其他选项]
```

常用参数：

* `--imagedir`：输入图像目录或视频文件路径。
* `--calib`：相机内参文件路径。
* `--stride`：输入帧采样间隔，例如 `--stride=2` 表示每隔 2 帧处理一次。
* `--viz`：启动 DPViewer，实时显示相机轨迹与三维重建结果。
* `--plot`：运行结束后保存轨迹图。
* `--save_trajectory`：将估计轨迹保存为 TUM 格式的 `.txt` 文件。
* `--save_ply`：将重建点云保存为 `.ply` 文件。
* `--save_colmap`：将轨迹和点云保存为 COLMAP 文本格式。

### EuRoC 示例

以 `MH_01_easy` 为例：

```bash
python demo.py \
    --imagedir=research/assets/datasets/euroc/MH_01_easy/mav0/cam0/data \
    --calib=calib/euroc.txt \
    --stride=2 \
    --plot \
    --viz \
    --save_trajectory
```

该命令使用 EuRoC 左目相机图像运行基础 DPVO，并启用实时可视化、轨迹绘制和轨迹保存。

### 开启 DPV-SLAM 后端

基础 `demo.py` 默认运行 DPVO 视觉里程计。若需要启用 DPV-SLAM 的 SLAM 后端和回环检测功能，在命令末尾增加：

```bash
--opts LOOP_CLOSURE True
```

例如：

```bash
python demo.py \
    --imagedir=research/assets/datasets/euroc/MH_01_easy/mav0/cam0/data \
    --calib=calib/euroc.txt \
    --stride=2 \
    --plot \
    --viz \
    --save_trajectory \
    --opts LOOP_CLOSURE True
```

### 开启 Classical Loop Closure

若已经按照前文安装 DBoW2、DPRetrieval，并准备好 ORB Vocabulary 和 LightGlue 权重，可进一步启用 Classical Backend：

```bash
--opts CLASSIC_LOOP_CLOSURE True
```

该后端主要用于处理较大的长程回环。基础实验中可优先使用 DPVO 或 DPV-SLAM，仅在需要测试大尺度回环时启用 Classical Backend。

# 2. Phase 1 / Experiment 1

## 2.1 研究问题与固定协议

Experiment 1 在不修改原始 `dpvo/`、不引入 JEPA 的前提下，描述 DPVO 的 correlation、confidence、patch/factor lifetime 与局部轨迹误差如何在 EuRoC `MH_01_easy`、`MH_03_medium`、`MH_05_difficult` 上变化。固定使用 stride=2、seed=1234、`config/default.yaml`、同一 checkpoint，并关闭 loop closure。

唯一正式复现命令为：

```bash
python -m research.src.phase1_dpvo_feasibility.run_exp1
```

入口先执行临时 MH_01 200-frame sanity，验证 hook cleanup、stable ID、verified correlation layout、deterministic sampling、有限值与 input/output contract。正式 sequence 均运行 uninstrumented baseline 和 Probe：baseline 是 ATE、FPS、peak VRAM 及 `report/trajectory_est.tum` 的唯一来源；Probe 只负责内部状态和行为分析。

## 2.2 Probe perturbation diagnostic

baseline 与 Probe 仍记录 ATE difference、relative ATE difference、global-Sim(3) aligned position RMSE、rotation RMSE、tracking completion 与 timestamp/frame contract。这是 feasibility-stage engineering diagnostic，不是统计显著性检验或 numerical acceptance gate：`<5%` 为 `small`/normal，`5%–<10%` 为 `moderate`/caution，`≥10%` 为 `large`/low representativeness。三种情况都保留结果；large 的 Probe artifacts 不得作为强 mechanism-level evidence。只有 tracking failure、trajectory 缺失、NaN/Inf 或 public contract 破坏才使 Probe 无效。

## 2.3 输出与核心分析

结果位于 `research/results/phase1-dpvo-feasibility/exp1/`，每条 sequence 的 `report/` 保存 standalone report、manifest、summary、metrics、baseline trajectory 与五张图；`artifacts/` 仅保存压缩的 `frames.npz`、`patches.npz`、`observations.npz`、`windows.npz`。comparison 引用三条 artifacts，不复制 NPZ，不创建 raw 或 zip。

五类核心分析为：

1. patch spatial distribution、coverage、border 与 nearest-neighbor；
2. weight、raw correlation peak/margin、fixed-temperature normalized entropy 与 delta distribution；
3. correlation/confidence Spearman 与 frozen-threshold mismatch；
4. blur、texture、GT relative rotation 与 model-derived apparent motion quintile response；
5. Probe trajectory 的唯一 global Sim(3) 后，内部状态与 20-frame local RMSE 的描述性 association。

MH_01 → MH_03 → MH_05 是 EuRoC nominal difficulty order，不是单变量控制实验；ordinal trend、difficulty proxy 和 trajectory association 都仅是描述性证据，不作因果解释。patch residency 与 factor-observation lifetime 受 graph/keyframe policy 影响，不能直接作为视觉质量标签。

## 2.4 结果与 Decision Gate

正式复现已完成，三条 baseline ATE RMSE 分别为：MH_01 0.09864 m、MH_03 0.18218 m、MH_05 0.14325 m。MH_01/MH_03/MH_05 的 Probe relative ATE change 分别为 0.47%、1.00%、0.80%，均为 `small`，且 `probe_representativeness = normal`；所有 tracking、timestamp、finite、stable-ID、hook cleanup 与 artifact checks 均通过。

描述性结果显示 raw correlation peak（2.093→1.994→1.960）和 margin（0.244→0.214→0.195）呈单调下降，delta norm（2.463→2.526→2.722）、low-confidence ratio（20.0%→23.1%→24.4%）及 large-delta ratio（20.0%→20.9%→23.0%）呈单调上升；weight mean 整体下降（0.116→0.112→0.106）。correlation entropy ↔ weight Spearman 为 -0.814、-0.816、-0.810，属于 `no_meaningful_change`：confidence 对 matching degradation 的描述性响应在三序列中稳定。极端 poor-correlation + high-confidence mismatch 仍极少（0.0048%、0.0084%、0.0052%），因此 confidence fusion 暂不是主要矛盾。

low-confidence + weak-correlation patch 比例由 11.7%→16.6%→19.4% 增加，说明 nominally harder 条件下固定 random patch budget 中低质量 correspondence 增多。Figure 4 中 texture 与 model-derived motion 下的 matching degradation 最清晰；这仍是 difficulty proxy 的描述性结果。internal state 与 local trajectory RMSE 没有稳定的跨序列关联，且 global ATE 为 non-monotonic（MH_03 高于 MH_05）。MH_01 → MH_03 → MH_05 仅是 EuRoC nominal difficulty order，不是单变量控制实验，因此这些结果不构成难度、内部量或轨迹误差之间的因果结论。

完整 evidence、五张 comparison figures 及 Q1–Q7 讨论位于 `research/results/phase1-dpvo-feasibility/exp1/comparison/`。Experiment 1 的 Decision Gate 是优先验证 correspondence representation，而不是 confidence fusion。Experiment 1 在此暂停，不进入 Experiment 2 或 JEPA。

## Phase 1 / Experiment 2

### Research Question

Experiment 2 验证：当 DPVO local correspondence representation 已经退化时，当前固定的 V-JEPA dense representation 是否仍然保留具有几何一致性的互补 correspondence information？

本实验不修改 DPVO、不进行 feature fusion、也不训练模型；它只对 DPVO 与 V-JEPA representation 进行离线 feasibility comparison。

### Protocol

固定序列为 `MH_01_easy` 与 `MH_05_difficult`。DPVO grouping 使用 Experiment 1 的 `corr_margin_l0`：MH_01 Q20=`0.0`、Q80=`0.3967015743255615`；good=`corr_margin_l0 >= Q80`，bad=`corr_margin_l0 <= Q20`。Q20 的 tie 保留，bad 组称为 `MH_01 Q20 frozen-threshold bad group`，不视为严格 bottom 20%。

正式实验使用每组 1000 samples、四组共 4000 correct temporal pairs，以及 200 temporal-shuffled controls；seed=`1234`，共享 source-frame cap=`8`。V-JEPA 固定为 V-JEPA 2.1 ViT-B/16 384 的 final EMA encoder dense tokens、framewise representation、24×24 token grid、768-dimensional feature 与 full-grid cosine retrieval。核心指标为 peak margin、epipolar error、cycle consistency 和 geometry consistency；epipolar geometry 在已验证的 EuRoC cam0 undistorted pinhole domain 中计算。

### Main Results

| Group | Peak margin | Epipolar error (tokens) | Cycle success | Geometry consistency |
|---|---:|---:|---:|---:|
| MH01 good | 0.01849 | 2.1349 | 58.4% | 20.9% |
| MH01 bad | 0.00881 | 4.5702 | 14.1% | 3.0% |
| MH05 good | 0.01899 | 1.9491 | 58.5% | 22.9% |
| MH05 bad | 0.00832 | 4.2228 | 15.8% | 3.5% |

DPVO good→bad 时，两条序列均稳定表现为 JEPA peak margin 下降、epipolar error 上升、cycle success 大幅下降，以及 geometry consistency 大幅下降。`geometry consistency` 是 geometry+cycle 一致的 feasibility operational criterion，不宣称已经恢复真实 correspondence。

### Temporal-gap Control

固定使用 `|dt| <= 2 s`、`2 < |dt| <= 5 s` 与 `|dt| > 5 s` 三个 bin。全部 6 个 sequence×gap 的 good→bad 对照中，peak margin 均下降、epipolar error 均上升、cycle success 与 geometry consistency 均下降。因此 formal 结果不能主要归因于 good/bad 的 temporal-gap 分布差异。完整 12-cell table 位于 `research/results/phase1-dpvo-feasibility/exp2/formal/report/REPORT.md`。

### Shuffled Null Control

200 个 paired joint-valid controls 的 correct / temporal-shuffled 指标分别为：peak margin 0.01326 / 0.00781，epipolar error 3.5358 / 5.7875 tokens，cycle success 33.5% / 15.5%，geometry consistency 12.0% / 2.0%。这表明当前 V-JEPA representation 确实包含一定 temporal correspondence signal，因此 negative 结果不能解释为 measurement pipeline 完全失效。

但 DPVO-bad 时 JEPA correspondence quality 同样明显下降，没有形成稳定、明显规模的 DPVO-bad / JEPA-good population。Shuffled 仅是 descriptive null sanity，并非 NEGATIVE 的必要判据。

### Decision Gate

**Experiment 2: NEGATIVE**

“当前固定的 V-JEPA 2.1 ViT-B framewise final dense representation 未表现出对 DPVO local correspondence 的明显互补性。”

该结论不表示 V-JEPA 没有 correspondence information，也不表示 JEPA 无法用于 VSLAM。Experiment 2 只否定当前最直接的 off-the-shelf V-JEPA dense token → DPVO local correspondence 融合方式；它不否定 temporal/world-state representation、task-aligned latent、latent prediction，或 sparse-image + predicted latent hybrid VSLAM 的后续可能性。

### Next Research Direction

Experiment 3 将不继续尝试直接 local-token fusion，而首先验证：“Sparse real images + Oracle task latent 是否能够恢复 sparse-image VSLAM 的精度损失？”这是 feature-domain / latent-domain VSLAM 的 upper-bound feasibility test；本轮不实现 Experiment 3。

## Phase 1 / Experiment 3

### Research Question

Experiment 3 验证：在真实 RGB 严重稀疏时，`Sparse real RGB + perfect DPVO task latent` 是否存在恢复 sparse-image VSLAM 精度损失的 upper bound？Oracle latent 由缺失帧的隐藏 RGB 直接提取 DPVO target FMap，仅用于判断 task representation 的潜力，不代表可部署的预测模型。

固定使用 stride=2、seed=1234、`config/default.yaml`、同一 checkpoint，并关闭 `LOOP_CLOSURE` 与 `CLASSIC_LOOP_CLOSURE`。最终正式复现命令只有：

```bash
python -m research.src.phase1_dpvo_feasibility.exp3.run_exp3
```

该入口独立执行 MH_01_easy、MH_03_medium、MH_05_difficult 的 K=8 representative protocol；历史 formal artifacts 仅作为经过 provenance/hash 验证的可选 cache。K=8 表示 severe-sparsity representative setting，用于展示 Sparse RGB 已显著退化时的 upper-bound behavior，不是调参得到的最佳参数。

### Smoke

400-candidate MH_01_easy smoke 使用 nominal 25% RGB upload。common-anchor global Sim(3) ATE 为：Full RGB 0.00411 m、Sparse RGB 0.09130 m、Oracle FMap 0.00490 m，preliminary gap recovery 约 99.1%。Native packet replay 与正常 DPVO 基本等价，Oracle extractor 通过 FMap equivalence microcheck；294/294 latent nodes 获得 factor、correlation、update 与 BA coverage，latent-source factors 和 placeholder violations 均为 0，最终为 `INTERFACE_READY`。

该结果只说明短窗口中存在强 Oracle upper-bound signal，不能代表完整序列稳定性。

### Formal Sweep

正式实验覆盖 MH_01_easy、MH_05_difficult 的 nominal 50%/25%/12.5%。12.5% severe-sparsity 结果为：

| Sequence | Full Anchor ATE [m] | Sparse Anchor ATE [m] | Oracle Anchor ATE [m] | Gap Recovery |
|---|---:|---:|---:|---:|
| MH_01_easy | 0.0992 | 3.7418 | 1.4082 | 64.1% |
| MH_05_difficult | 0.1446 | 5.8487 | 1.1215 | 82.9% |

但收益并不跨 ratio 稳定：MH01 50% 仅从 0.1221 m 改善到 0.1154 m；MH05 50% 从 0.1740 m 恶化到 0.2703 m；MH01 25% 从 0.0935 m 恶化到 2.8207 m，MH05 25% 从 0.1881 m 恶化到 0.3448 m。因此不能只保留 severe-sparsity 正结果，也不能宣称 Oracle FMap 能稳定替代缺失 RGB。Formal Decision 为 `PARTIAL`。

Full RGB 使用 upstream normal culling，Sparse RGB 与 Oracle FMap 使用共同 no-culling policy；核心公平比较始终是 Sparse RGB 与 Sparse + Oracle FMap。Full 只作为性能 reference，Oracle 优于 Full 或 gap recovery 大于 1 不能解释为 latent 强于真实 RGB observation。

### Auxiliary Topology Diagnosis

Naive Oracle 将 latent 当作普通 raw DPVO state，导致 latent-inclusive frame distance 消耗 `PATCH_LIFETIME` 与 `REMOVAL_WINDOW`，削弱 real-anchor connectivity。Auxiliary diagnosis 恢复了 100% Sparse-like cumulative anchor-factor identities，同时保留原 anchor→latent policy，结果为：

| Setting | Naive Oracle ATE [m] | Auxiliary Oracle ATE [m] | Anchor Topology Coverage |
|---|---:|---:|---:|
| MH01 25% | 2.8207 | 0.7248 | 100% |
| MH05 25% | 0.3448 | 0.8988 | 100% |
| MH05 12.5% | 1.1215 | 5.6853 | 100% |

MH01 25% 明显改善，但 MH05 25% 和 12.5% 未保留 naive Oracle 的收益，最终为 `CONDITIONAL`。这说明 anchor-topology dilution 是 integration instability 的影响因素之一，但不能单独解释不稳定性；恢复 Sparse anchor topology 也不是稳定解决方案。Auxiliary 仅是 graph-semantics diagnosis，不作为最终 representative trajectory method。

### Final Three-Sequence Representative Result

最终使用 MH01/MH03/MH05、K=8，展示 Full RGB、Sparse RGB 与原始 naive Oracle FMap 三种方法。MH01/MH05 直接复用通过 hash/provenance 检查的 formal trajectories；MH03 新运行完整 1350 candidates 的三个 methods。

| Sequence | Actual RGB Upload | Full Anchor ATE [m] | Sparse Anchor ATE [m] | Oracle Anchor ATE [m] | Gap Recovery |
|---|---:|---:|---:|---:|---:|
| MH_01_easy | 12.93% | 0.0992 | 3.7418 | 1.4082 | 64.1% |
| MH_03_medium | 14.74% | 0.1817 | 3.3845 | 2.9395 | 13.9% |
| MH_05_difficult | 16.18% | 0.1446 | 5.8487 | 1.1215 | 82.9% |

MH03 表现为 intermediate recovery：Oracle 相对 Sparse 改善约 13.15%，但恢复幅度远小于 MH01/MH05。这进一步支持 Oracle upper bound 的 sequence dependence，不改变整体 Decision。

三张轨迹图位于 `research/results/phase1-dpvo-feasibility/exp3/final/report/figures/`。Full RGB 为 dense trajectory，Oracle FMap 为 dense hybrid trajectory；Sparse RGB 只按时间顺序连接实际 accepted RGB-anchor poses 方便观察，没有插值，也不是 dense estimated trajectory。三种估计均使用相同 common-anchor评价约定，各自在 common anchors 上拟合 global Sim(3) 后应用于其可用轨迹。

### Exp3 Final Decision

**PARTIAL / CONDITIONAL UPPER-BOUND EVIDENCE**

Experiment 3 provides PARTIAL / CONDITIONAL upper-bound evidence that perfect task latent can bridge severe RGB sparsity, but robust measurement–prediction graph integration remains unsolved.

Experiment 3 证明完美 DPVO task latent 在严重图像稀疏条件下具有显著的精度恢复潜力，但该收益依赖序列、稀疏程度和 latent graph integration。真实 measurement 与预测 latent 不能简单使用完全相同的图语义，稳定的 measurement–prediction coupling 仍是后续需要解决的问题。

### Next Direction

下一阶段不继续 Oracle graph ablation。Experiment 4 将首先脱离 trajectory/runtime，验证 `V-JEPA current-frame state → lightweight task adapter → DPVO FMap-like task latent` 能否逼近 Oracle task representation；本轮不实现 Experiment 4。
