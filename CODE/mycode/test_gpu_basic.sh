#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXE="${BUILD_DIR}/jacobi_cuda"
MESH="${PROJECT_ROOT}/CODE/JacobiSetComputation-master/test_torus.obj"
OUT="/tmp/test_torus_gpu_jacobi.txt"

if [[ ! -x "${EXE}" ]]; then
    echo "missing executable: ${EXE}" >&2
    exit 1
fi

rm -f "${OUT}"
output="$("${EXE}" "${MESH}" --output "${OUT}")"
echo "${output}"

grep -q "jacobi_edges: 64" <<<"${output}"

if [[ ! -f "${OUT}" ]]; then
    echo "missing output file: ${OUT}" >&2
    exit 1
fi

head -n 1 "${OUT}" | grep -q "^JacobiSet$"
sed -n '2p' "${OUT}" | grep -q "^64$"
