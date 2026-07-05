#ifndef JACOBI_GPU_H
#define JACOBI_GPU_H

#include <string>
#include <vector>

struct EdgeRecord {
    int e1;
    int e2;
    int link1;
    int link2;
};

struct JacobiGpuResult {
    int fv1;
    int fv2;
    int min_f;
    int min_g;
    int pos_align;
    int is_jacobi;
    int degenerate;
};

struct GpuTiming {
    float kernel_ms;
    float total_ms;
};

bool compute_jacobi_gpu(const std::vector<EdgeRecord> &edges,
                        const std::vector<double> &f,
                        const std::vector<double> &g,
                        std::vector<JacobiGpuResult> *results,
                        GpuTiming *timing,
                        std::string *error_message);

#endif
