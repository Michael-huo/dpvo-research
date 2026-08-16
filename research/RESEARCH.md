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

