# CUDA Jacobi 开发记录

本文档用于记录并行 CUDA 版本的每个里程碑，包括计划、开发过程、测试过程和当前结论。后续每完成一个阶段，都在这里补充，便于项目管理和最终报告整理。

## 总体目标

根据论文 `Jacobi Sets of Multiple Morse Functions` 和串行工程 `CODE/JacobiSetComputation-master`，实现对应的 CUDA 并行版本。

核心并行点：

- 串行版本逐条遍历 mesh edge 判断是否属于 Jacobi set。
- CUDA 版本将“每条内部边的 Jacobi 判定”映射为“一个 GPU 线程处理一条边”。
- 对数值退化边保留 CPU SoS 回退，兼顾并行速度和精确性。

## 里程碑总览

| 里程碑 | 目标 | 状态 |
| --- | --- | --- |
| M1 | WSL2 CUDA 环境配置与最小 CUDA 程序验证 | 已完成 |
| M2 | CPU 预处理：复用串行工程导出内部边表 | 已完成 |
| M3 | GPU kernel 基础版：非 SoS 判定并行化 | 已完成 |
| M4 | GPU 退化边标记 | 已完成 |
| M5 | CPU SoS 回退合并 | 已完成 |
| M6-1 | 正确性对拍脚本 | 已完成 |
| M6-2 | 性能测试 | 已完成 |

## M1：WSL2 CUDA 环境配置

### 计划

先让 WSL Ubuntu 中的 CUDA 编译环境可用，再开始 Jacobi 代码开发。通过最小 CUDA 程序验证：

- `nvcc --version` 可用。
- `nvidia-smi` 能看到 RTX 4060。
- 一个最小 kernel 能编译并在 GPU 上运行。

### 开发 / 配置过程

用户在 WSL Ubuntu 中安装 CUDA Toolkit 后，检查工具链：

```bash
nvcc --version
nvidia-smi
```

然后创建最小 CUDA 程序：

```bash
mkdir -p ~/cuda_test && cd ~/cuda_test

cat > hello.cu << 'EOF'
#include <stdio.h>
__global__ void hello() { printf("CUDA OK from GPU\n"); }
int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc hello.cu -o hello
./hello
```

### 测试结果

已确认：

- `nvcc` 版本：CUDA compilation tools, release 12.0, V12.0.140。
- `nvidia-smi` 能看到 RTX 4060，Windows 驱动 CUDA Version 为 13.0。
- 最小程序输出：

```text
CUDA OK from GPU
```

### 结论

M1 通过。WSL2 CUDA 编译与运行环境可用于后续 CUDA 开发。

## M2：CPU 预处理导出内部边表

### 计划

在进入 GPU kernel 前，先实现 CPU 预处理程序，确认后续 GPU 所需输入数据是正确的。

预处理要完成：

- 复用 `TriMeshJ` 读取 mesh。
- 调用 `need_edges()` 建立边集合。
- 调用 `need_neighbors()` 建立顶点邻接。
- 对每条内部边调用 `get_e_link()` 得到两个 link 顶点。
- 导出或统计 `e1, e2, link1, link2`。

torus 测试网格的预期结果：

```text
vertices: 512
edges: 1536
faces: 1024
interior_edges: 1536
```

### 开发过程

新增文件：

- `CODE/mycode/main_cuda.cpp`
  - 当前只实现 `--preprocess-only` 模式。
  - 读取 mesh 后输出顶点数、边数、面数、内部边数。
  - 可选 `--dump-edges <path>` 导出边表。
- `CODE/mycode/CMakeLists.txt`
  - 构建 `jacobi_cuda` 可执行文件。
  - 复用串行工程的 `TriMeshJ.cpp` 与 `trimesh2/lib.Linux64/libtrimesh.a`。
  - 链接 `OpenMP::OpenMP_CXX`，解决 `trimesh2` 静态库依赖 OpenMP 的链接问题。
- `CODE/mycode/build_cuda_wsl.sh`
  - WSL 下一键 CMake 构建。
- `CODE/mycode/test_preprocess.sh`
  - 自动运行 torus 预处理测试。
  - 检查输出中是否包含预期统计值。

### 测试过程

先写测试，再运行，确认初始失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_preprocess.sh
```

初始失败原因：

```text
missing executable: .../CODE/mycode/build/jacobi_cuda
```

随后实现 M2 程序并构建：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash build_cuda_wsl.sh
bash test_preprocess.sh
```

自测通过输出包含：

```text
vertices: 512
edges: 1536
faces: 1024
interior_edges: 1536
```

额外测试边表导出：

```bash
./build/jacobi_cuda --preprocess-only "../JacobiSetComputation-master/test_torus.obj" --dump-edges /tmp/test_torus_edges.txt
wc -l /tmp/test_torus_edges.txt
sed -n '1,5p' /tmp/test_torus_edges.txt
```

自测结果：

```text
1537 /tmp/test_torus_edges.txt
# edge_id e1 e2 link1 link2
0 0 1 17 496
1 0 15 16 511
2 0 16 17 15
3 0 17 1 16
```

`1537` 行表示 1 行表头 + 1536 条内部边。

### 结论

M2 通过。GPU 后续需要的边表输入数据已经能正确生成。

## M3：GPU kernel 基础版

### 计划

实现 `jacobi_cuda.cu` 与 `jacobi_gpu.h`：

- 把串行非 SoS 分支的 `POS1`、`POS2`、`is_lowerLink`、`alignment` 改写成 `__device__` 函数。
- 每个 GPU 线程处理一条内部边。
- 输出 `is_jacobi`、`min_f`、`min_g`、`pos_align`。
- 先不做 CPU SoS 回退，只验证 GPU 主流程跑通。

### 开发过程

新增 / 修改文件：

- `CODE/mycode/jacobi_gpu.h`
  - 定义 `EdgeRecord`、`JacobiGpuResult`、`GpuTiming`。
  - 声明 host 接口 `compute_jacobi_gpu()`。
- `CODE/mycode/jacobi_cuda.cu`
  - 实现 device 版 `are_equal`、`sort3`、`POS1`、`POS2`、`is_lowerLink`、`alignment`。
  - 实现 `compute_jacobi_kernel`，每个线程处理一条内部边。
  - 实现 host 端 `compute_jacobi_gpu()`，负责 `cudaMalloc`、`cudaMemcpy`、kernel launch、结果拷回、计时。
- `CODE/mycode/main_cuda.cpp`
  - 保留 `--preprocess-only` 模式。
  - 新增默认 GPU 计算模式：
    - 读取 mesh。
    - 构造 `f=y`、`g=x`。
    - 调用 GPU kernel。
    - 输出 `*_gpu_jacobi.txt` 或 `--output` 指定文件。
    - 打印 `jacobi_edges`、`degenerate_edges`、`gpu_kernel_ms`、`gpu_total_ms`。
- `CODE/mycode/CMakeLists.txt`
  - 启用 CUDA 语言。
  - 加入 `jacobi_cuda.cu`。
  - 设置 `CMAKE_CUDA_ARCHITECTURES 75 86`，兼容 CUDA 12.0 与 RTX 4060。
- `CODE/mycode/test_gpu_basic.sh`
  - 自动测试 GPU 基础版在 torus 上输出 64 条 Jacobi 边。

注意：M3 中已经统计 `degenerate_edges`，但只是报告数量，不执行 CPU SoS 覆盖。真正的回退合并放到 M5。

### 测试过程

先写测试并确认失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_gpu_basic.sh
```

初始失败原因：

```text
Usage:
  .../build/jacobi_cuda --preprocess-only <mesh.obj> [--dump-edges <edges.txt>]
```

说明默认 GPU 计算模式尚未实现，测试有效。

实现 M3 后重新构建并运行回归测试：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash build_cuda_wsl.sh
bash test_preprocess.sh
bash test_gpu_basic.sh
```

自测通过输出包含：

```text
vertices: 512
edges: 1536
faces: 1024
interior_edges: 1536
jacobi_edges: 64
degenerate_edges: 16
gpu_kernel_ms: 1.76333
gpu_total_ms: 6.27133
output: /tmp/test_torus_gpu_jacobi.txt
```

其中：

- `jacobi_edges: 64` 与串行 torus 示例一致。
- `degenerate_edges: 16` 表示有 16 条边在 double 判定中触发了近退化标记，后续 M5 会用 CPU SoS 回退覆盖这些边。

### 结论

M3 通过。CUDA kernel 基础版已能在 GPU 上完成 Jacobi 边判断，并在 torus 示例上输出 64 条 Jacobi 边。

## M4：GPU 退化边标记

### 计划

在 GPU 判定过程中，当出现近似退化情况时标记该边：

- `POS1` 中 `abs(X) < eps`。
- `POS2` 或 `alignment` 中函数值近似相等。

M4 的目标不是执行 CPU SoS 回退，而是把 GPU 标记出的退化边稳定导出，为 M5 做准备。

### 开发过程

新增 / 修改文件：

- `CODE/mycode/test_degenerate_dump.sh`
  - 自动测试 `--dump-degenerate` 参数。
  - 要求 torus 示例输出 `degenerate_edges: 16`。
  - 要求退化边文件为 17 行：1 行表头 + 16 条退化边。
- `CODE/mycode/main_cuda.cpp`
  - 新增命令行参数 `--dump-degenerate <edges.txt>`。
  - 新增 `write_degenerate_edges()`。
  - 输出字段为 `edge_id e1 e2 link1 link2 is_jacobi`，其中 `edge_id` 对应内部边表中的下标，后续 M5 可按该下标覆盖 GPU 结果。

### 测试过程

先写测试并确认失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_degenerate_dump.sh
```

初始失败原因：

```text
Usage:
  .../jacobi_cuda --preprocess-only <mesh.obj> [--dump-edges <edges.txt>]
  .../jacobi_cuda <mesh.obj> [--output <jacobi.txt>]
```

说明程序尚不支持 `--dump-degenerate`，测试有效。

实现 M4 后，运行 M2-M4 回归：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash build_cuda_wsl.sh
bash test_preprocess.sh
bash test_gpu_basic.sh
bash test_degenerate_dump.sh
```

自测通过输出包含：

```text
jacobi_edges: 64
degenerate_edges: 16
output: /tmp/test_torus_gpu_jacobi.txt
degenerate_dump: /tmp/test_torus_degenerate_edges.txt
```

检查退化边文件：

```bash
wc -l /tmp/test_torus_degenerate_edges.txt
sed -n '1,10p' /tmp/test_torus_degenerate_edges.txt
```

自测结果：

```text
17 /tmp/test_torus_degenerate_edges.txt
# edge_id e1 e2 link1 link2 is_jacobi
0 0 1 17 496 0
1 0 15 16 511 0
6 1 2 18 497 0
11 2 3 19 498 0
16 3 4 20 499 0
21 4 5 21 500 0
26 5 6 22 501 0
31 6 7 23 502 0
36 7 8 24 503 0
```

### 结论

M4 通过。GPU 近退化标记稳定，torus 示例中共有 16 条退化边可导出。下一步 M5 将对这些边执行 CPU SoS 回退覆盖。

## M5：CPU SoS 回退合并

### 计划

对 GPU 标记为退化的边，在 CPU 端复用串行 SoS 逻辑重新计算并覆盖 GPU 结果。

目标：

- 普通边由 GPU 并行计算。
- 退化边由 CPU SoS 保证与串行版本一致。

### 开发过程

新增 / 修改文件：

- `CODE/mycode/test_sos_fallback.sh`
  - 自动测试默认计算路径是否执行 SoS 回退。
  - 要求 torus 示例输出 `sos_fallback_edges: 16`。
  - 要求最终 Jacobi 输出文件仍为 64 条边。
- `CODE/mycode/main_cuda.cpp`
  - 引入 `JacobiSet.h`。
  - 新增 `apply_sos_fallback()`。
  - 对 GPU 结果中 `degenerate == 1` 的边，复用串行 `JacobiSet::is_lowerLink()` 与 `JacobiSet::alignment()` 重新计算。
  - 回退后覆盖 `is_jacobi`、`min_f`、`min_g`、`pos_align`。
  - 新增输出 `sos_fallback_edges`。
- `CODE/mycode/CMakeLists.txt`
  - 加入串行工程 `JacobiSet.cpp`。
  - 增加 SoS include 路径：`Detri_2.6.a/basic`、`lia`、`sos`。
  - 链接 `Detri_2.6.a/build/lib/libSoS.a`。
  - 定义 `USE_SOS`，确保回退路径使用串行 SoS 分支。

说明：M5 只对 GPU 标记为退化的边回退。普通边仍保留 GPU 计算结果。

### 测试过程

先写测试并确认失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_sos_fallback.sh
```

初始失败现象：

```text
jacobi_edges: 64
degenerate_edges: 16
gpu_kernel_ms: ...
gpu_total_ms: ...
output: /tmp/test_torus_gpu_sos_jacobi.txt
```

缺少 `sos_fallback_edges: 16`，说明 CPU SoS 回退尚未实现，测试有效。

实现 M5 后运行 M2-M5 回归：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash build_cuda_wsl.sh
bash test_preprocess.sh
bash test_gpu_basic.sh
bash test_degenerate_dump.sh
bash test_sos_fallback.sh
```

自测通过输出包含：

```text
-------------- Creating SOS Matrix ........... (#fix=15.14) scale = 0.000000000000010
SoS: matrix[512,2] @ 5 Lia digits; lia_length (10); 0.047 Mb.
jacobi_edges: 64
degenerate_edges: 16
sos_fallback_edges: 16
output: /tmp/test_torus_gpu_sos_jacobi.txt
```

额外对比串行参考输出：

```bash
diff -u "/mnt/d/金介然/大三下/gpu/大作业/CODE/JacobiSetComputation-master/test_torus_jacobi.txt" \
        /tmp/test_torus_gpu_sos_jacobi.txt
```

自测结果：`diff` 无输出，说明 torus 示例中 GPU+SoS 输出与串行参考文件完全一致。

### 结论

M5 通过。GPU 普通边并行计算 + CPU SoS 退化边回退已经合并完成，torus 示例输出与串行参考完全一致。

## M6-1：正确性对拍脚本

### 计划

实现顺序无关的 Jacobi 边集对拍：

- GPU 输出与串行 `*_jacobi.txt` 做顺序无关比较。
- torus 结果应为 64 条 Jacobi 边。

### 开发过程

新增文件：

- `CODE/mycode/compare_jacobi.py`
  - 读取两个 `*_jacobi.txt` 文件。
  - 校验头部 `JacobiSet` 和边数。
  - 提取每条 Jacobi 边的两个顶点编号。
  - 将 `(a,b)` 规范化为小编号在前。
  - 用集合做顺序无关比较。
  - 输出 `reference_edges`、`candidate_edges`、`match`。
- `CODE/mycode/test_compare_jacobi.sh`
  - 先运行 GPU+SoS 输出。
  - 再调用 `compare_jacobi.py` 与串行参考结果对拍。

### 测试过程

先写测试并确认失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_compare_jacobi.sh
```

初始失败原因：

```text
python3: can't open file '.../compare_jacobi.py': [Errno 2] No such file or directory
```

说明测试正在检查尚未实现的对拍脚本。

实现脚本后重新运行：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_compare_jacobi.sh
```

自测通过输出：

```text
jacobi_edges: 64
degenerate_edges: 16
sos_fallback_edges: 16
output: /tmp/test_torus_gpu_sos_jacobi.txt
reference_edges: 64
candidate_edges: 64
match: yes
```

### 结论

M6-1 通过。GPU+SoS 输出与串行参考结果的 Jacobi 边集完全一致，对拍过程已自动化。

## M6-2：性能测试

### 计划

多尺寸网格测试 CPU 串行时间、GPU kernel 时间、GPU 总时间。

测试内容：

- 自动生成不同规模 torus 网格。
- 运行串行 `JacobiSetComputation`。
- 运行 CUDA hybrid 版本。
- 对拍串行/GPU 输出，确保 `match=yes`。
- 输出 CSV，记录 wall time、GPU kernel time、GPU total time、端到端加速比。

### 开发过程

新增文件：

- `CODE/mycode/benchmark_jacobi.py`
  - 支持 `--sizes 32x16,64x32,...`。
  - 调用 `trimesh2/bin.Linux64/mesh_make` 生成 torus。
  - 调用串行程序和 GPU hybrid 程序。
  - 使用 `compare_jacobi.py` 的读取逻辑做边集对拍。
  - 输出 CSV 字段：
    - `u,v,vertices,edges`
    - `jacobi_edges,degenerate_edges,sos_fallback_edges`
    - `cpu_wall_ms,gpu_wall_ms,gpu_kernel_ms,gpu_total_ms`
    - `speedup_wall,match`
- `CODE/mycode/test_benchmark_smoke.sh`
  - 冒烟测试：只跑 `32x16`，确认 CSV 生成、字段正确、`match=yes`。
- `.gitignore`
  - 忽略 `CODE/mycode/benchmark_meshes/` 与 `CODE/mycode/benchmark_results.csv`，避免提交生成网格和临时结果。

### 测试过程

先写冒烟测试并确认失败：

```bash
cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
bash test_benchmark_smoke.sh
```

初始失败原因：

```text
python3: can't open file '.../benchmark_jacobi.py': [Errno 2] No such file or directory
```

实现 benchmark 脚本后，运行冒烟测试：

```bash
bash test_benchmark_smoke.sh
```

自测通过输出：

```text
32x16: cpu_wall=44.638 ms, gpu_wall=329.112 ms, speedup=0.136x, match=yes
wrote: /tmp/jacobi_benchmark_smoke.csv
```

运行默认 3 个规模：

```bash
python3 benchmark_jacobi.py --sizes 32x16,64x32,128x64 --output /tmp/jacobi_benchmark.csv
```

自测结果：

```text
32x16: cpu_wall=47.496 ms, gpu_wall=370.139 ms, speedup=0.128x, match=yes
64x32: cpu_wall=83.758 ms, gpu_wall=281.042 ms, speedup=0.298x, match=yes
128x64: cpu_wall=204.979 ms, gpu_wall=335.847 ms, speedup=0.610x, match=yes
```

运行更大规模：

```bash
python3 benchmark_jacobi.py --sizes 256x128,512x256 --output /tmp/jacobi_benchmark_large.csv
```

自测结果：

```text
256x128: cpu_wall=563.048 ms, gpu_wall=548.318 ms, speedup=1.027x, match=yes
512x256: cpu_wall=1914.722 ms, gpu_wall=1231.177 ms, speedup=1.555x, match=yes
```

其中 `gpu_kernel_ms` 在大规模下约 2 ms，说明核心逐边判定 kernel 很快；端到端时间还包含 mesh 读取、CPU 预处理、数据传输和 SoS 回退，因此小规模时 wall speedup 小于 1，大规模时开始体现加速。

用户手动二次验证结果：

```text
32x16: cpu_wall=42.031 ms, gpu_wall=312.991 ms, speedup=0.134x, match=yes
64x32: cpu_wall=82.968 ms, gpu_wall=303.732 ms, speedup=0.273x, match=yes
128x64: cpu_wall=190.101 ms, gpu_wall=327.330 ms, speedup=0.581x, match=yes
256x128: cpu_wall=550.632 ms, gpu_wall=497.479 ms, speedup=1.107x, match=yes
512x256: cpu_wall=1874.647 ms, gpu_wall=1199.303 ms, speedup=1.563x, match=yes
```

对应 CSV：

```text
u,v,vertices,edges,jacobi_edges,degenerate_edges,sos_fallback_edges,cpu_wall_ms,gpu_wall_ms,gpu_kernel_ms,gpu_total_ms,speedup_wall,match
32,16,512,1536,64,16,16,42.031,312.991,1.819650,4.905220,0.134,yes
64,32,2048,6144,128,32,32,82.968,303.732,0.907264,3.720830,0.273,yes
128,64,8192,24576,256,64,64,190.101,327.330,0.969728,4.178270,0.581,yes
256,128,32768,98304,512,128,128,550.632,497.479,1.221790,5.263550,1.107,yes
512,256,131072,393216,1024,260,260,1874.647,1199.303,1.281860,7.102780,1.563,yes
```

### 结论

M6-2 通过。benchmark 已自动化，所有测试规模 `match=yes`，并在大规模网格上观察到端到端加速。

## 手动测试约定

每完成一个里程碑，流程固定为：

1. 我先在本机工具中构建和自测。
2. 自测通过后，给出用户终端中的手动测试命令。
3. 用户运行并反馈输出。
4. 二次确认通过后，再进入下一个里程碑。
