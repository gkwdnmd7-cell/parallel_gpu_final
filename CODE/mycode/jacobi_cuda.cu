#include "jacobi_gpu.h"

#include <cuda_runtime.h>

#include <sstream>

namespace {

constexpr double kEps = 1e-12;

static std::string cuda_error(const char *what, cudaError_t err) {
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    return os.str();
}

__device__ bool device_are_equal(double a, double b) {
    double diff = a - b;
    if (diff < 0.0) {
        diff = -diff;
    }
    return diff < kEps;
}

__device__ void device_sort3(int *a, int *b, int *c) {
    if (*a > *b) {
        int t = *a;
        *a = *b;
        *b = t;
    }
    if (*b > *c) {
        int t = *b;
        *b = *c;
        *c = t;
        if (*a > *b) {
            t = *a;
            *a = *b;
            *b = t;
        }
    }
}

__device__ bool device_pos1(int a,
                            int b,
                            int v,
                            const double *f,
                            const double *g,
                            int *degenerate) {
    const double x =
        (f[a] * (g[v] - g[b])) +
        (f[b] * (g[a] - g[v])) +
        (f[v] * (g[b] - g[a]));

    if (!device_are_equal(x, 0.0)) {
        return x > 0.0;
    }

    *degenerate = 1;
    if (!device_are_equal(g[v], g[b])) {
        return g[v] > g[b];
    }
    if (!device_are_equal(g[a], g[v])) {
        return g[a] > g[v];
    }
    if (!device_are_equal(f[b], f[v])) {
        return f[b] > f[v];
    }
    if (!device_are_equal(f[v], f[a])) {
        return f[v] > f[a];
    }
    return true;
}

__device__ bool device_pos2(int a,
                            int b,
                            const double *field,
                            int *degenerate) {
    if (device_are_equal(field[a], field[b])) {
        *degenerate = 1;
        return a < b;
    }
    return field[a] < field[b];
}

__device__ bool device_is_lower_link(int i,
                                     int j,
                                     int k,
                                     const double *f,
                                     const double *g,
                                     const double *field,
                                     int *degenerate) {
    int a = i;
    int b = j;
    int v = k;
    device_sort3(&a, &b, &v);

    const int index_i = (i == a) ? 0 : ((i == b) ? 1 : 2);
    const int index_j = (j == a) ? 0 : ((j == b) ? 1 : 2);

    const bool is_pos1 = device_pos1(a, b, v, f, g, degenerate);
    bool is_pos2 = false;
    if (index_j == (index_i + 1) % 3) {
        is_pos2 = device_pos2(i, j, field, degenerate);
    } else {
        is_pos2 = device_pos2(j, i, field, degenerate);
    }

    return is_pos1 == is_pos2;
}

__device__ bool device_alignment(int v1,
                                 int v2,
                                 const double *f,
                                 const double *g,
                                 int *degenerate) {
    if (device_are_equal(f[v2], f[v1]) || device_are_equal(g[v2], g[v1])) {
        *degenerate = 1;
    }
    return (f[v2] > f[v1]) == (g[v2] > g[v1]);
}

__global__ void compute_jacobi_kernel(const EdgeRecord *edges,
                                      int edge_count,
                                      const double *f,
                                      const double *g,
                                      JacobiGpuResult *results) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count) {
        return;
    }

    const EdgeRecord edge = edges[idx];
    int degenerate = 0;

    const bool f_lower_v1 = device_is_lower_link(edge.e1, edge.e2, edge.link1, f, g, g, &degenerate);
    const bool f_lower_v2 = device_is_lower_link(edge.e1, edge.e2, edge.link2, f, g, g, &degenerate);
    const bool g_lower_v1 = device_is_lower_link(edge.e1, edge.e2, edge.link1, f, g, f, &degenerate);
    const bool g_lower_v2 = device_is_lower_link(edge.e1, edge.e2, edge.link2, f, g, f, &degenerate);

    const bool f_crit = (f_lower_v1 == f_lower_v2);
    const bool g_crit = (g_lower_v1 == g_lower_v2);
    const bool is_jacobi = f_crit;

    JacobiGpuResult out;
    out.fv1 = edge.e1;
    out.fv2 = edge.e2;
    out.min_f = !f_lower_v1;
    out.min_g = g_lower_v1;
    out.pos_align = device_alignment(edge.e1, edge.e2, f, g, &degenerate);
    out.is_jacobi = is_jacobi;
    out.degenerate = degenerate || (f_crit != g_crit);
    results[idx] = out;
}

} // namespace

bool compute_jacobi_gpu(const std::vector<EdgeRecord> &edges,
                        const std::vector<double> &f,
                        const std::vector<double> &g,
                        std::vector<JacobiGpuResult> *results,
                        GpuTiming *timing,
                        std::string *error_message) {
    if (results == nullptr || timing == nullptr || error_message == nullptr) {
        return false;
    }

    results->assign(edges.size(), JacobiGpuResult());
    timing->kernel_ms = 0.0f;
    timing->total_ms = 0.0f;
    error_message->clear();

    if (edges.empty()) {
        return true;
    }

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    EdgeRecord *d_edges = nullptr;
    double *d_f = nullptr;
    double *d_g = nullptr;
    JacobiGpuResult *d_results = nullptr;

    auto fail = [&](const std::string &message) {
        *error_message = message;
        if (d_edges) cudaFree(d_edges);
        if (d_f) cudaFree(d_f);
        if (d_g) cudaFree(d_g);
        if (d_results) cudaFree(d_results);
        if (total_start) cudaEventDestroy(total_start);
        if (total_stop) cudaEventDestroy(total_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (kernel_stop) cudaEventDestroy(kernel_stop);
        return false;
    };

    cudaError_t err = cudaEventCreate(&total_start);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventCreate(total_start)", err));
    err = cudaEventCreate(&total_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventCreate(total_stop)", err));
    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventCreate(kernel_start)", err));
    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventCreate(kernel_stop)", err));

    cudaEventRecord(total_start);

    err = cudaMalloc(&d_edges, edges.size() * sizeof(EdgeRecord));
    if (err != cudaSuccess) return fail(cuda_error("cudaMalloc(edges)", err));
    err = cudaMalloc(&d_f, f.size() * sizeof(double));
    if (err != cudaSuccess) return fail(cuda_error("cudaMalloc(f)", err));
    err = cudaMalloc(&d_g, g.size() * sizeof(double));
    if (err != cudaSuccess) return fail(cuda_error("cudaMalloc(g)", err));
    err = cudaMalloc(&d_results, edges.size() * sizeof(JacobiGpuResult));
    if (err != cudaSuccess) return fail(cuda_error("cudaMalloc(results)", err));

    err = cudaMemcpy(d_edges, edges.data(), edges.size() * sizeof(EdgeRecord), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail(cuda_error("cudaMemcpy(edges)", err));
    err = cudaMemcpy(d_f, f.data(), f.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail(cuda_error("cudaMemcpy(f)", err));
    err = cudaMemcpy(d_g, g.data(), g.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail(cuda_error("cudaMemcpy(g)", err));

    const int block_size = 256;
    const int grid_size = static_cast<int>((edges.size() + block_size - 1) / block_size);

    cudaEventRecord(kernel_start);
    compute_jacobi_kernel<<<grid_size, block_size>>>(d_edges,
                                                     static_cast<int>(edges.size()),
                                                     d_f,
                                                     d_g,
                                                     d_results);
    err = cudaGetLastError();
    if (err != cudaSuccess) return fail(cuda_error("compute_jacobi_kernel launch", err));
    err = cudaEventRecord(kernel_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventRecord(kernel_stop)", err));
    err = cudaEventSynchronize(kernel_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventSynchronize(kernel_stop)", err));
    cudaEventElapsedTime(&timing->kernel_ms, kernel_start, kernel_stop);

    err = cudaMemcpy(results->data(), d_results, edges.size() * sizeof(JacobiGpuResult), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail(cuda_error("cudaMemcpy(results)", err));

    err = cudaEventRecord(total_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventRecord(total_stop)", err));
    err = cudaEventSynchronize(total_stop);
    if (err != cudaSuccess) return fail(cuda_error("cudaEventSynchronize(total_stop)", err));
    cudaEventElapsedTime(&timing->total_ms, total_start, total_stop);

    cudaFree(d_edges);
    cudaFree(d_f);
    cudaFree(d_g);
    cudaFree(d_results);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    return true;
}
