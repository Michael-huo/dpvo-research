# RESEARCH

本文件用于记录基于 DPVO
的后续研究工作，包括环境部署、实验复现、算法改进和实验结果分析。

当前章节记录从零开始部署 DPVO 并运行官方 EuRoC demo 的过程。

------------------------------------------------------------------------

# 1. DPVO 环境部署与 Demo 复现

## 1.1 环境部署

创建独立 Conda 环境：

``` bash
#Setup and Installation
git clone https://github.com/Michael-huo/dpvo-research.git
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
cd ../..
pip install ./DPViewer --no-build-isolation

# Classical Backend (optional)
sudo apt-get install -y libopencv-dev
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
pip install ./DPRetrieval
```

说明：

-   使用 Miniforge/Conda 管理环境；
-   Python 版本固定为 3.10；
-   PyTorch CUDA 版本根据设备驱动调整；
-   DPVO 使用源码安装，方便后续研究修改。

------------------------------------------------------------------------

## 1.2 下载运行 Demo 所需文件

### EuRoC 数据集

例如：

    research/datasets/euroc/MH_01_easy/

### 网络权重

DPVO 默认需要：

-   DPVO 模型权重：dpvo.pth
-   LightGlue 特征匹配模型（用于 DPV-SLAM 回环检测）：depth-save.pth, disk_lightglue_v0-1_arxiv-pth

## 1.3 运行 EuRoC Demo

基础 DPVO：

``` bash
python demo.py \
--imagedir=research/datasets/euroc/MH_01_easy/mav0/cam0/data \
--calib=euroc.txt \
--stride=2 \
--plot \
--viz \
--save_trajectory
```

功能：

-   加载 EuRoC 单目图像；
-   执行 DPVO 视觉里程计；
-   打开可视化窗口；
-   保存估计轨迹。

------------------------------------------------------------------------

## 1.4 部署验证结果

完成上述步骤后，应能够：

-   成功运行官方 EuRoC demo；
-   正常显示 DPViewer 可视化界面；
-   输出轨迹结果；
-   使用 DPV-SLAM 扩展模块进行长期回环检测。

当前版本作为后续研究工作的 baseline。

------------------------------------------------------------------------
