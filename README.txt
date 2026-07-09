================================================================================
  GPU 大作业 — Jacobi 集 CUDA 并行计算
  项目文件说明
================================================================================

一、项目简介
--------------------------------------------------------------------------------

本仓库对应 GPU 课程大作业，主题为：在理解论文
《Jacobi Sets of Multiple Morse Functions》的基础上，对照串行参考实现
JacobiSetComputation，开发二维三角网格上两个 Morse 函数的 Jacobi 集
CUDA 并行版本，并完成论文翻译、代码分析与结果验证。

核心思路：
  - 串行版本逐条遍历网格内部边，判断其是否属于 Jacobi 集；
  - 并行版本将「每条内部边的判定」映射为「一个 GPU 线程处理一条边」；
  - CPU 负责网格读取、拓扑预处理、退化边 SoS 精确回退和结果写出；
  - GPU 负责大规模内部边的并行 Jacobi 判定。

运行环境：Windows + WSL2 (Ubuntu)，CUDA Toolkit 12.0，NVIDIA GPU
（已在 RTX 4060 上验证）。


二、目录总览
--------------------------------------------------------------------------------

大作业/
├── README.txt                          ← 本文件，项目文件说明
├── .gitignore                          ← Git 忽略规则（编译产物、第三方依赖等）
├── .gitattributes                      ← Git 属性配置
│
├── CODE/                               ← 全部代码
│   ├── JacobiSetComputation-master/    ← 串行参考工程（上游开源项目）
│   └── mycode/                         ← 本项目 CUDA 并行实现（主要开发目录）
│
└── docs/                               ← 课程文档、论文、分析报告
    ├── 要求.txt                        ← 课程作业四项要求
    ├── analysis/                       ← 代码实现思路分析与验收材料
    └── paper/                          ← 论文原文、翻译稿与插图


三、根目录文件
--------------------------------------------------------------------------------

README.txt
  项目总览与各目录、文件说明（本文件）。

.gitignore
  忽略编译目录 build/、第三方依赖 trimesh2/ 与 Detri_2.6.a/、benchmark
  生成文件、Python 缓存、IDE 配置等。原则：提交源码/脚本/文档，忽略可
  重新生成的文件。

.gitattributes
  Git 行尾与文本属性配置。


四、CODE/JacobiSetComputation-master/ — 串行参考工程
--------------------------------------------------------------------------------

来源：https://github.com/bhatiaharsh/JacobiSetComputation
作用：论文算法的串行 C++ 实现，作为 CUDA 并行版本的算法对照与正确性基准。

--- 顶层文件 ---

README.md
  上游项目说明：算法背景、依赖、编译与使用方法。

LICENSE
  Lawrence Livermore National Laboratory 开源许可证。

CMakeLists.txt
  串行程序 CMake 构建配置（链接 trimesh2 与 SoS 库）。

install_deps.sh
  上游提供的依赖安装脚本（macOS 导向；本项目主要使用 mycode/setup_deps_wsl.sh）。

patch_SOS.txt
  SoS（Simulation of Simplicity）库的补丁说明，用于修复老版本 C 代码在
  现代编译器下的兼容问题。

--- 源码 ---

include/
  TriMeshJ.h      在 trimesh2 基础上扩展边集合与 link 查询
  JacobiSet.h     Jacobi 集判定核心接口（POS1/POS2/is_lowerLink/compute）
  sos_utils.h     SoS 相关工具头文件

src/
  main.cpp        串行程序入口：读 mesh、构造 f=y/g=x、计算、写结果
  TriMeshJ.cpp    网格边生成（need_edges）、邻接（need_neighbors）、
                  边 link 查询（get_e_link）
  JacobiSet.cpp   Jacobi 集核心算法：lower-link 判定、alignment、compute 循环

utilities/
  jstovtk.py      将 *_jacobi.txt 转为 VTK 可视化文件

--- 测试数据与参考输出 ---

test_torus.obj
  标准测试网格（torus，512 顶点 / 1536 边 / 1024 面）。

test_torus_jacobi.txt
  串行 SoS 版在 test_torus.obj 上的参考输出（64 条 Jacobi 边）。

test_torus_jacobi.vtk
  串行参考结果的 VTK 可视化文件。

--- 第三方依赖（由 setup_deps_wsl.sh 下载构建，.gitignore 忽略） ---

trimesh2/
  三角网格读写与拓扑库。编译后生成 lib.Linux64/libtrimesh.a。

Detri_2.6.a/
  Simulation of Simplicity 符号计算库（SoS）。编译后生成
  build/lib/libSoS.a。用于退化情况下精确的符号判定。

--- 编译产物（.gitignore 忽略，本地存在） ---

build/
  串行可执行文件 JacobiSetComputation 及 CMake 中间文件。


五、CODE/mycode/ — CUDA 并行实现（本项目核心）
--------------------------------------------------------------------------------

--- 核心源码 ---

main_cuda.cpp
  CPU 端主程序。支持两种模式：
    --preprocess-only <mesh.obj> [--dump-edges <edges.txt>]
      仅做 CPU 拓扑预处理，验证边表是否正确。
    <mesh.obj> [--output <jacobi.txt>] [--dump-degenerate <edges.txt>]
      完整流程：预处理 → GPU 计算 → CPU SoS 回退 → 写结果。

jacobi_cuda.cu
  CUDA 核心实现：
    - __device__ 版 POS1、POS2、is_lowerLink、alignment
    - compute_jacobi_kernel：一线程处理一条内部边
    - compute_jacobi_gpu()：host 端内存管理、kernel 启动与计时

jacobi_gpu.h
  CPU/GPU 共享数据结构（EdgeRecord、JacobiGpuResult、GpuTiming）及
  compute_jacobi_gpu() 接口声明。

CMakeLists.txt
  CUDA 工程构建配置。链接串行工程的 TriMeshJ.cpp、JacobiSet.cpp，
  以及 trimesh2、SoS、OpenMP。定义 USE_SOS 以启用 CPU 回退路径。

--- 构建与依赖脚本 ---

build_cuda_wsl.sh
  WSL 下一键 CMake 构建，生成 build/jacobi_cuda。

setup_deps_wsl.sh
  WSL 依赖一键安装：系统包、trimesh2、Detri/SoS、串行程序编译。
  新环境首次使用时应先运行此脚本。

fix_detri_download.sh
  Detri/SoS 下载补救脚本（setup_deps_wsl.sh 下载失败时手动使用）。

fix_detri_layout.sh
  Detri 目录结构修复与 libSoS.a 编译补救脚本（功能已并入 setup_deps_wsl.sh）。

--- Python 工具 ---

compare_jacobi.py
  正确性对拍：顺序无关地比较两个 *_jacobi.txt 的边集。

benchmark_jacobi.py
  多规模 torus 性能测试：自动生成网格、跑串行/GPU、对拍、输出 CSV。

jacobi_to_vtk.py
  将 *_jacobi.txt 转为 VTK 文件，供 ParaView 等工具可视化。

--- 自动化测试脚本 ---

test_preprocess.sh       验证 CPU 边表预处理（torus: 512/1536/1024/1536）
test_gpu_basic.sh        验证 GPU 基础计算（64 条 Jacobi 边）
test_degenerate_dump.sh  验证退化边导出（16 条退化边）
test_sos_fallback.sh     验证 CPU SoS 回退（sos_fallback_edges: 16）
test_compare_jacobi.sh   验证 GPU+SoS 与串行参考边集一致（match: yes）
test_benchmark_smoke.sh  验证 benchmark 脚本能生成 CSV 且 match=yes

--- 开发记录 ---

DEVELOPMENT_LOG.md
  按里程碑（M1–M6）记录的开发过程：环境配置、预处理、GPU kernel、
  退化标记、SoS 回退、对拍与性能测试。含测试命令与预期输出。

--- 测试输出示例 ---

test_torus_gpu_sos_jacobi.vtk
  GPU+SoS 版 torus 结果的 VTK 可视化文件。

--- 本地生成目录（.gitignore 忽略） ---

build/
  CUDA 可执行文件 jacobi_cuda 及编译中间文件。

benchmark_meshes/
  benchmark 自动生成的多规模 torus 网格与对应 jacobi 输出。

benchmark_results.csv
  性能测试 CSV 结果。


六、docs/ — 文档与报告材料
--------------------------------------------------------------------------------

要求.txt
  课程作业四项要求原文：
    1) 翻译论文并整理为 Word 文档
    2) 对照串行代码开发 CUDA 并行版本
    3) 详细分析代码实现思路
    4) 以上内容合并为一个带封面的文档

--- docs/paper/ — 论文材料 ---

2004-02-JacobiSets.pdf / jacobi_paper.pdf
  论文原文 PDF。

jacobi_paper.md
  论文翻译稿（Markdown 格式）。

jacobi_paper - 副本.md
  论文翻译稿备份。

images/
  figure1.png ~ figure4.png   论文插图

--- docs/analysis/ — 分析与验收 ---

cuda_jacobi_code_analysis.md
  最终分析报告主文档，分三部分：
    第一部分 — 论文翻译（待补充）
    第二部分 — CUDA 并行核心代码摘录
    第三部分 — 代码实现思路详细分析（串行对照、算法理解、并行设计、
                测试验证、性能分析）

cuda_jacobi_code_analysis.docx
  上述分析报告的最终 Word 排版版本。

cuda_jacobi_code_analysis copy.md
cuda_jacobi_code_analysis copy 2.md
cuda_jacobi_code_analysis copy 3.md
  分析报告撰写过程中的草稿备份。

cuda_jacobi_acceptance_walkthrough.md
  现场验收讲解稿：按报告顺序组织，含演示命令、预期输出与解说词。

1.png / 2.png / 3.png
  分析报告或验收演示用配图。


七、快速上手（WSL）
--------------------------------------------------------------------------------

1. 安装依赖（首次）：
   cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
   chmod +x setup_deps_wsl.sh
   ./setup_deps_wsl.sh

2. 构建 CUDA 程序：
   ./build_cuda_wsl.sh

3. 运行标准测试（torus）：
   ./build/jacobi_cuda ../JacobiSetComputation-master/test_torus.obj

4. 运行全部自动化测试：
   bash test_preprocess.sh
   bash test_gpu_basic.sh
   bash test_degenerate_dump.sh
   bash test_sos_fallback.sh
   bash test_compare_jacobi.sh

5. 性能测试：
   python3 benchmark_jacobi.py --sizes 32x16,64x32,128x64 --output benchmark_results.csv

6. 可视化：
   python3 jacobi_to_vtk.py <输出文件.txt>


八、程序数据流简述
--------------------------------------------------------------------------------

  读 mesh (.obj)
    → CPU: need_edges + need_neighbors → 生成 EdgeRecord[] 内部边表
    → 构造 f=y, g=x 函数数组
    → cudaMemcpy 到 GPU
    → CUDA kernel: 每个线程判定一条边（device_is_lower_link × 4）
    → cudaMemcpy 结果回 CPU
    → CPU SoS 回退：对 degenerate 边用串行 JacobiSet 重新计算
    → 写出 *_jacobi.txt（格式与串行版一致）
    → compare_jacobi.py 与串行参考对拍验证


九、关键数字（test_torus.obj 标准测试）
--------------------------------------------------------------------------------

  顶点数：512        边数：1536        面数：1024
  内部边：1536       Jacobi 边：64      退化边：16
  GPU+SoS 输出与串行 test_torus_jacobi.txt 完全一致（match: yes）


十、文件阅读建议
--------------------------------------------------------------------------------

  想了解课程要求        → docs/要求.txt
  想了解开发过程        → CODE/mycode/DEVELOPMENT_LOG.md
  想了解算法与实现细节    → docs/analysis/cuda_jacobi_code_analysis.md
  想现场验收讲解        → docs/analysis/cuda_jacobi_acceptance_walkthrough.md
  想读串行参考算法      → CODE/JacobiSetComputation-master/src/JacobiSet.cpp
  想读 CUDA 并行实现    → CODE/mycode/jacobi_cuda.cu + main_cuda.cpp
  想一键跑通            → CODE/mycode/setup_deps_wsl.sh → build_cuda_wsl.sh
                          → test_compare_jacobi.sh

================================================================================
  最后更新：2026-07-09
================================================================================
