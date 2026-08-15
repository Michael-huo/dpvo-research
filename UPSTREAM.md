# UPSTREAM

## 1. 项目来源

本研究仓库基于开源项目 **DPVO (Deep Patch Visual Odometry)** 构建。

-   上游项目：DPVO
-   上游仓库：https://github.com/princeton-vl/DPVO
-   原始作者：Princeton Vision & Learning Lab
-   项目类型：深度学习驱动的视觉里程计 / Visual Odometry

本仓库不修改上游项目的核心源码结构，主要用于基于 DPVO
的后续研究、实验复现与算法扩展。

------------------------------------------------------------------------

## 2. 仓库维护原则

本仓库遵循以下原则：

1.  **保持上游代码完整性**

    -   上游 DPVO 源码保持原始结构。
    -   不直接修改官方实现文件。
    -   避免影响后续与官方仓库同步。

2.  **研究代码隔离**

    所有针对本研究新增的代码、配置、实验脚本和实验结果统一放置于：

        research/

    目录。

    推荐结构：

        research/
        ├── RESEARCH.md
        ├── configs/
        ├── scripts/
        ├── src/
        ├── assets/
        └── results/

3.  **实验环境独立记录**

    Python 环境配置文件、依赖版本和部署过程记录放置于：

        research/environment.yml

    以保证实验环境可复现。

------------------------------------------------------------------------

## 3. 上游同步方式

本仓库通过 Git remote 保留官方仓库：

``` bash
upstream:
https://github.com/princeton-vl/DPVO.git
```

个人研究仓库：

``` bash
origin:
https://github.com/Michael-huo/dpvo-research.git
```

同步上游更新：

``` bash
git fetch upstream
git merge upstream/main
```

如需引入官方更新，应优先同步 upstream，再在 research
目录中适配新的实验代码。

------------------------------------------------------------------------

## 4. 第三方依赖

DPVO 使用以下第三方组件：

-   DBoW2
-   Pangolin
-   vcpkg
-   pybind11

这些组件通过 Git submodule 管理，保持与上游一致。

不要直接修改 submodule
内部代码，除非研究需求明确要求，并应单独记录修改原因。

------------------------------------------------------------------------

## 5. 数据集与模型文件

以下内容不上传至 GitHub：

-   数据集文件
-   网络下载权重
-   编译产生的中间文件
-   CUDA/C++ 编译产物
-   实验输出结果

例如：

    research/datasets/
    research/results/
    *.pth
    *.pt
    build/

等内容应通过 `.gitignore` 排除。

------------------------------------------------------------------------

## 6. 研究目标

本仓库用于探索：

-   DPVO 视觉里程计性能分析；
-   基于深度视觉表示的 SLAM 方法改进；
-   与世界模型、视觉表征学习方法结合的可能性；
-   面向边云协同 VSLAM 系统的算法研究。

所有研究过程、实验设计和结果分析记录于：

    research/RESEARCH.md

------------------------------------------------------------------------

## 7. License

上游代码遵循 DPVO 官方许可证。

本仓库新增研究内容遵循个人研究仓库管理规范，并尊重所有上游项目版权和许可证要求。
