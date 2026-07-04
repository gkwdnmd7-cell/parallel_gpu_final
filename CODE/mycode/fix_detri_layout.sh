#!/usr/bin/env bash
# 修正 Detri 目录 + 修复 basic.h + 编译 libSoS.a
# 用法: ./fix_detri_layout.sh
set -euo pipefail

JS_DIR="/mnt/d/金介然/大三下/gpu/大作业/CODE/JacobiSetComputation-master"
DETRI="${JS_DIR}/Detri_2.6.a"
SRC_NEST="${DETRI}/Detri_2.6.b"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info(){ echo -e "${GREEN}[INFO]${NC} $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
err(){ echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ -d "${DETRI}" ]] || err "找不到 ${DETRI}"
[[ -d "${SRC_NEST}/basic" ]] || err "找不到 ${SRC_NEST}/basic，请确认 Detri 已下载"

# ---------- 1. 从 Detri_2.6.b 重新复制干净源码（覆盖被 sed 破坏的文件） ----------
info "从 Detri_2.6.b/ 复制干净源码 ..."
for d in basic lia sos; do
    rm -rf "${DETRI}/${d}"
    cp -a "${SRC_NEST}/${d}" "${DETRI}/${d}"
done

# ---------- 2. 修复 basic.h：保留 AND/OR/NOT/mod，并添加小写别名 ----------
info "修复 basic.h（添加 and/or/not 别名，勿破坏 #define mod）..."
BH="${DETRI}/basic/basic.h"
    grep -q '#ifndef __cplusplus' "${BH}" 2>/dev/null && grep -A3 '#ifndef __cplusplus' "${BH}" | grep -q '#define and' && return 0
    # 删除旧的无 guard 别名（若存在）
    sed -i '/^#define and[[:space:]]*AND$/d;/^#define or[[:space:]]*OR$/d;/^#define or[[:space:]]*OR$/d;/^#define not[[:space:]]*NOT$/d' "${BH}" 2>/dev/null || true
    sed -i '/#define mod[[:space:]]*%/a\
\
#ifndef __cplusplus\
#define and    AND\
#define or     OR\
#define not    NOT\
#endif' "${BH}"

# 若曾被错误 sed 成 #define %，恢复 mod 行
if grep -q '^#define %' "${BH}"; then
    warn "检测到损坏的 #define %，正在恢复 #define mod ..."
    sed -i 's/^#define %[[:space:]]*%/#define mod    %/' "${BH}"
fi

# ---------- 3. 写入 CMakeLists ----------
write_cmake() {
    cat > "${DETRI}/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.5)
project(SoS C)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(BASIC_SRC
        ./basic/arg.c ./basic/cb.c ./basic/counter.c ./basic/getarg.c
        ./basic/istaque.c ./basic/math2.c ./basic/qsort.c ./basic/tokenize.c
        ./basic/basic.c ./basic/cb_doprnt.c ./basic/files.c ./basic/isort.c
        ./basic/malloc.c ./basic/prime.c ./basic/uhash.c
)
SET(LIA_SRC ./lia/aux.c ./lia/chars.c ./lia/det.c ./lia/lia.c ./lia/pool.c ./lia/stack.c)
SET(SOS_SRC
        ./sos/in_sphere.c ./sos/lambda3.c ./sos/lambda4.c ./sos/lambda5.c
        ./sos/minor.c ./sos/positive3.c ./sos/primitive.c ./sos/smaller.c ./sos/sos.c
)
include_directories(./basic/ ./lia/ ./sos/)
add_library(SoS ${BASIC_SRC} ${LIA_SRC} ${SOS_SRC})
EOF
}
write_cmake

# ---------- 4. 编译 ----------
info "编译 libSoS.a ..."
rm -rf "${DETRI}/build"
cmake -S "${DETRI}" -B "${DETRI}/build"
cmake --build "${DETRI}/build" -j"$(nproc)"

[[ -f "${DETRI}/build/lib/libSoS.a" ]] || err "libSoS.a 未生成"
touch "${DETRI}/.sos_patches_applied"

info "=========================================="
ls -lh "${DETRI}/build/lib/libSoS.a"
info "SoS 编译成功！接下来:"
info "  cd $(dirname "$0") && ./setup_deps_wsl.sh"
info "=========================================="
