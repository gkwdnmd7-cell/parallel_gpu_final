#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXE="${BUILD_DIR}/jacobi_cuda"
MESH="${PROJECT_ROOT}/CODE/JacobiSetComputation-master/test_torus.obj"
OUT="/tmp/test_torus_gpu_jacobi.txt"
DEG="/tmp/test_torus_degenerate_edges.txt"

if [[ ! -x "${EXE}" ]]; then
    echo "missing executable: ${EXE}" >&2
    exit 1
fi

rm -f "${OUT}" "${DEG}"
output="$("${EXE}" "${MESH}" --output "${OUT}" --dump-degenerate "${DEG}")"
echo "${output}"

grep -q "jacobi_edges: 64" <<<"${output}"
grep -q "degenerate_edges: 16" <<<"${output}"
grep -q "degenerate_dump: ${DEG}" <<<"${output}"

if [[ ! -f "${DEG}" ]]; then
    echo "missing degenerate dump: ${DEG}" >&2
    exit 1
fi

wc -l "${DEG}" | grep -q "^17 "
head -n 1 "${DEG}" | grep -q "^# edge_id e1 e2 link1 link2 is_jacobi$"
