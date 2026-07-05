#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXE="${BUILD_DIR}/jacobi_cuda"
MESH="${PROJECT_ROOT}/CODE/JacobiSetComputation-master/test_torus.obj"

if [[ ! -x "${EXE}" ]]; then
    echo "missing executable: ${EXE}" >&2
    exit 1
fi

if [[ ! -f "${MESH}" ]]; then
    echo "missing mesh: ${MESH}" >&2
    exit 1
fi

output="$("${EXE}" --preprocess-only "${MESH}")"
echo "${output}"

grep -q "vertices: 512" <<<"${output}"
grep -q "edges: 1536" <<<"${output}"
grep -q "faces: 1024" <<<"${output}"
grep -q "interior_edges: 1536" <<<"${output}"
