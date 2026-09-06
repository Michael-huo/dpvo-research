# RESEARCH

本文件用于记录基于 DPVO
的后续研究工作，包括环境部署、实验复现、算法改进和实验结果分析。

# 1. DPVO 环境部署与 Demo 复现

## 1.1 环境部署

说明：

- 使用 Miniforge/Conda 管理环境；
- Python 版本固定为 3.10；
- DPVO 自定义 CUDA extension 编译需要 CUDA Toolkit 12.1；

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version
```

- DPVO 使用源码安装，方便后续研究修改。

```bash
# Setup and Installation
git clone --recursive https://github.com/Michael-huo/dpvo-research.git
cd dpvo-research

mamba env create -f research/environment.yml
conda activate dpvo

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install . --no-build-isolation

# Recommended - Install the Pangolin Viewer
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

```text
research/assets/datasets/euroc/MH_01_easy/
```

### 网络权重和其他配置文件

- DPVO 模型：`dpvo-research/dpvo.pth`
- ORB Vocabulary：`dpvo-research/ORBvoc.txt`
- DPV-SLAM 长程回环相关权重：

```text
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

- `--imagedir`：输入图像目录或视频文件路径。
- `--calib`：相机内参文件路径。
- `--stride`：输入帧采样间隔，例如 `--stride=2` 表示每隔 2 帧处理一次。
- `--viz`：启动 DPViewer，实时显示相机轨迹与三维重建结果。
- `--plot`：运行结束后保存轨迹图。
- `--save_trajectory`：将估计轨迹保存为 TUM 格式的 `.txt` 文件。
- `--save_ply`：将重建点云保存为 `.ply` 文件。
- `--save_colmap`：将轨迹和点云保存为 COLMAP 文本格式。

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

# 2. Phase 1 — Feasibility Analysis

早期 Exp1–5/Exp6 探索中的有效结论已经提炼并整合进当前 H0/H1/H2；旧实现、命令和 artifacts 不再作为正式研究接口，历史细节可通过 Git history 追溯。

当前实现位于 `research/src/phase1_feasibility/`，正式配置位于 `research/configs/phase1_feasibility_h0.yaml`、`phase1_feasibility_h1.yaml` 和 `phase1_feasibility_h2.yaml`。三个 runner 不依赖旧实验源码、配置、checkpoint 或 results。

## H0 State — Latent-State Feasibility

H0 回答“learning-based VSLAM 的 hidden frame 最少需要什么 latent visual state”。conditions 固定为 Full RGB、Sparse RGB 和 True FMap。hidden packet 只保存 `fmap`；`patch_xy` 按 FrameIdentity/seed 确定性派生，`gmap/fmap2` 从 FMap 派生，`imap` 为 zero，colors 删除；pose/depth、factor、update、BA 与 upstream culling 保持正常 DPVO 语义。H0 不加载 JEPA、bridge 或 predictor。

```bash
python -m research.src.phase1_feasibility.run_h0 --sequences MH_01_easy
```

## H1 Interface — Representation-Interface Feasibility

H1 验证 Oracle JEPA 能否经 coordinate-correct block-5→FMap interface 提供 H0 latent state。每次显式运行都在 MH01 fresh 训练一次 `bridge.pt`，并使用本轮刚训练的同一个 bridge fresh 评估全部 requested sequences。H1 不读取 H0 results，MH03/MH05 是本轮 bridge 的 frozen zero-shot evaluation。

```bash
python -m research.src.phase1_feasibility.run_h1 --sequences MH_01_easy
```

## H2 Prediction — Sparse-Anchor Prediction Feasibility

H2 只加载 canonical H1 `bridge.pt`，不会调用 H1 或训练 bridge。加载前验证 bridge 与当前 H1 科学 config/source/protocol lineage 兼容；不兼容时 fail closed 并要求先运行 H1。兼容后，每次 H2 显式运行都在 MH01 fresh 训练一次 `predictor.pt`，其 training lineage 绑定当前 bridge hash，再使用本轮 predictor fresh 评估全部 requested sequences。

strict 在线 capability 只有 uploaded anchor RGB、anchor identity 和 hidden identity/timestamp。anchor RGB 通过 native DPVO FNet/Patchifier 形成 VSLAM observation 并提取 JEPA context；hidden observation 只能来自 predicted JEPA 经 frozen bridge 生成的 FMap-only packet。

A5 必须真实到达并完成 JEPA 编码，随后才按时间戳顺序提交 buffered hidden observations，且每个 candidate 只消费一次。因此 H2 是 delayed/bracketed、non-causal deployment，`timestamp_causal=false`，不声明 causal 或 strict real-time。

```bash
python -m research.src.phase1_feasibility.run_h2 --sequences MH_01_easy
```

三条 CLI 均支持一次请求多个 sequence。需要三序列横向比较时，必须让它们共享同一次 fresh 模型运行：

```bash
python -m research.src.phase1_feasibility.run_h2 \
    --sequences MH_01_easy MH_03_medium MH_05_difficult
```

## Artifact 与执行策略

Canonical outputs 位于 `research/results/phase1-feasibility/{h0_state,h1_interface,h2_prediction}/`。现阶段采用 fresh current-canonical replace：H0 每次 fresh 执行 requested sequences；H1 每次 fresh 训练 bridge 并 fresh 评估；H2 验证当前 H1 bridge 后 fresh 训练 predictor 并 fresh 评估。

本次 requested sequences 是模块完整的当前有效集合。只有整轮成功且通过 artifact manifest 验证后才替换旧模块，未请求的旧 sequence 不保留。每个模块根 `INDEX.json` 仅作为本轮 canonical provenance/artifact manifest，记录 requested sequences、dataset/config/source/protocol/schedule、适用的 bridge/predictor 及 sequence artifact hashes，不承担 cache 或 reuse 职责。
