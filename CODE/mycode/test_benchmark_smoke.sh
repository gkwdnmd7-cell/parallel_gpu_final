#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="/tmp/jacobi_benchmark_smoke.csv"

rm -f "${CSV}"
python3 "${SCRIPT_DIR}/benchmark_jacobi.py" --sizes 32x16 --output "${CSV}"

if [[ ! -f "${CSV}" ]]; then
    echo "missing benchmark CSV: ${CSV}" >&2
    exit 1
fi

grep -q "u,v,vertices,edges,jacobi_edges,degenerate_edges,sos_fallback_edges,cpu_wall_ms,gpu_wall_ms,gpu_kernel_ms,gpu_total_ms,speedup_wall,match" "${CSV}"
grep -q "32,16,512,1536,64,16,16" "${CSV}"
grep -q "yes" "${CSV}"
