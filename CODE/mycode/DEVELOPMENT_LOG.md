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
| M4 | GPU 退化边标记 | 未开始 |
| M5 | CPU SoS 回退合并 | 未开始 |
| M6 | 正确性对拍与性能测试 | 未开始 |

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

### 开发过程

未开始。

### 测试过程

未开始。

### 结论

未开始。

## M5：CPU SoS 回退合并

### 计划

对 GPU 标记为退化的边，在 CPU 端复用串行 SoS 逻辑重新计算并覆盖 GPU 结果。

目标：

- 普通边由 GPU 并行计算。
- 退化边由 CPU SoS 保证与串行版本一致。

### 开发过程

未开始。

### 测试过程

未开始。

### 结论

未开始。

## M6：正确性对拍与性能测试

### 计划

实现输出对拍和性能统计：

- GPU 输出与串行 `*_jacobi.txt` 做顺序无关比较。
- torus 结果应为 64 条 Jacobi 边。
- 多尺寸网格测试 CPU 串行时间、GPU kernel 时间、GPU 总时间。

### 开发过程

未开始。

### 测试过程

未开始。

### 结论

未开始。

## 手动测试约定

每完成一个里程碑，流程固定为：

1. 我先在本机工具中构建和自测。
2. 自测通过后，给出用户终端中的手动测试命令。
3. 用户运行并反馈输出。
4. 二次确认通过后，再进入下一个里程碑。
