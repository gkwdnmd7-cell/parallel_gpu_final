#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXE="${BUILD_DIR}/jacobi_cuda"
MESH="${PROJECT_ROOT}/CODE/JacobiSetComputation-master/test_torus.obj"
REF="${PROJECT_ROOT}/CODE/JacobiSetComputation-master/test_torus_jacobi.txt"
OUT="/tmp/test_torus_gpu_sos_jacobi.txt"

if [[ ! -x "${EXE}" ]]; then
    echo "missing executable: ${EXE}" >&2
    exit 1
fi

"${EXE}" "${MESH}" --output "${OUT}"
python3 "${SCRIPT_DIR}/compare_jacobi.py" "${REF}" "${OUT}"
