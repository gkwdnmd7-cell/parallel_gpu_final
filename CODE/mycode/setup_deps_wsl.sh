#!/usr/bin/env bash
# =============================================================================
# WSL (Ubuntu) 依赖安装脚本 — trimesh2 + SoS(Detri) + JacobiSetComputation 串行版
#
# 用法（在 Ubuntu 终端里）：
#   cd "/mnt/d/金介然/大三下/gpu/大作业/CODE/mycode"
#   chmod +x setup_deps_wsl.sh
#   ./setup_deps_wsl.sh
#
# 前提：已安装 Ubuntu WSL，且能访问 /mnt/d/...
# =============================================================================
set -euo pipefail

# ---------- 配置：按你的实际路径修改（默认指向本项目） ----------
PROJECT_ROOT="/mnt/d/金介然/大三下/gpu/大作业"
JS_DIR="${PROJECT_ROOT}/CODE/JacobiSetComputation-master"
PATCH_FILE="${JS_DIR}/patch_SOS.txt"
TRIMESH_DIR="${JS_DIR}/trimesh2"
DETRI_DIR="${JS_DIR}/Detri_2.6.a"
DETRI_TAR="${JS_DIR}/Detri_2.6.a.tar.gz"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# Clash 等代理端口（Windows 侧）。若走代理，需在 Clash 里开启「允许局域网连接」
CLASH_PORT="${CLASH_PORT:-7890}"

# 获取 Windows 主机在 WSL 中的 IP（用作 HTTP 代理地址）
windows_host_ip() {
    ip route show default 2>/dev/null | awk '{print $3}' | head -1
}

# 尝试用 curl 下载；支持直连 / 国内 GitHub 镜像 / Windows 代理
fetch_url() {
    local out="$1"
    shift
    local url="$1"
    shift
    local mirrors=("$@")
    local host proxy_url

    host="$(windows_host_ip)"
    if [[ -n "${host}" ]]; then
        proxy_url="http://${host}:${CLASH_PORT}"
    fi

    # 1) 直连
    if curl -fsSL --connect-timeout 15 --max-time 120 -o "${out}" "${url}"; then
        return 0
    fi
    warn "直连失败: ${url}"

    # 2) 国内 GitHub 镜像
    for m in "${mirrors[@]}"; do
        info "尝试镜像: ${m}"
        if curl -fsSL --connect-timeout 15 --max-time 300 -o "${out}" "${m}"; then
            return 0
        fi
    done

    # 3) 经 Windows Clash 代理（需 Allow LAN）
    if [[ -n "${proxy_url}" ]]; then
        info "尝试经 Windows 代理 ${proxy_url} ..."
        if curl -fsSL --connect-timeout 15 --max-time 300 \
            -x "${proxy_url}" -o "${out}" "${url}"; then
            return 0
        fi
    fi

    return 1
}

clone_github_repo() {
    local dest="$1"
    local repo="$2"   # 形如 Forceflow/trimesh2
    local host proxy_url

    if [[ -d "${dest}/.git" ]]; then
        return 0
    fi
    rm -rf "${dest}"

    host="$(windows_host_ip)"
    proxy_url="http://${host}:${CLASH_PORT}"

    info "克隆 ${repo} ..."

    # 1) gitclone 镜像（国内常用）
    if git clone --depth 1 "https://gitclone.com/github.com/${repo}.git" "${dest}" 2>/dev/null; then
        return 0
    fi
    warn "gitclone 克隆失败，尝试其他方式..."

    # 2) 经 Windows Clash 代理直连 GitHub
    if [[ -n "${host}" ]]; then
        if HTTPS_PROXY="${proxy_url}" HTTP_PROXY="${proxy_url}" \
            git -c http.proxy="${proxy_url}" -c https.proxy="${proxy_url}" \
            clone --depth 1 "https://github.com/${repo}.git" "${dest}" 2>/dev/null; then
            return 0
        fi
        warn "经代理 ${proxy_url} 克隆失败（请确认 Clash 已开且允许局域网）"
    fi

    # 3) 下载 tarball（gitclone）
    local tmp="${dest}.tar.gz"
    if fetch_url "${tmp}" \
        "https://github.com/${repo}/archive/refs/heads/main.tar.gz" \
        "https://gitclone.com/github.com/${repo}/archive/refs/heads/main.tar.gz" \
        "https://gitclone.com/github.com/${repo}/archive/refs/heads/master.tar.gz"; then
        rm -rf "${dest}"
        tar -xzf "${tmp}" -C "$(dirname "${dest}")"
        rm -f "${tmp}"
        local extracted
        extracted="$(find "$(dirname "${dest}")" -maxdepth 1 -type d -name "$(basename "${repo}")-*" | head -1)"
        [[ -n "${extracted}" ]] || return 1
        mv "${extracted}" "${dest}"
        return 0
    fi

    return 1
}

# ---------- 0. 基本检查 ----------
if [[ ! -d "${JS_DIR}" ]]; then
    error "找不到串行代码目录: ${JS_DIR}\n请修改脚本顶部的 PROJECT_ROOT。"
fi

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    warn "当前似乎不在 WSL 里运行。建议在 Ubuntu(WSL) 终端执行本脚本。"
fi

info "项目目录: ${JS_DIR}"

# ---------- 1. 安装系统依赖 ----------
info "安装编译工具与 trimesh2 依赖..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    patch \
    mesa-common-dev \
    libglu1-mesa-dev \
    libxi-dev

# ---------- 2. 编译 trimesh2 ----------
if [[ -f "${TRIMESH_DIR}/lib.Linux64/libtrimesh.a" ]]; then
    info "trimesh2 已编译，跳过。"
else
    info "获取 trimesh2 (Forceflow 维护版)..."
    if [[ ! -d "${TRIMESH_DIR}" ]] || [[ ! -f "${TRIMESH_DIR}/Makefile" ]]; then
        clone_github_repo "${TRIMESH_DIR}" "Forceflow/trimesh2" \
            || error "trimesh2 下载失败。可手动: git clone https://gitclone.com/github.com/Forceflow/trimesh2.git"
    fi
    info "编译 trimesh2..."
    make -C "${TRIMESH_DIR}" -j"$(nproc)"
    [[ -f "${TRIMESH_DIR}/lib.Linux64/libtrimesh.a" ]] || error "trimesh2 编译失败。"
    info "trimesh2 完成: ${TRIMESH_DIR}/lib.Linux64/libtrimesh.a"
fi

# ---------- 3. 获取 Detri / SoS ----------
download_detri_original() {
    local urls=(
        "https://web.archive.org/web/2020id_/http://www.geom.uiuc.edu/software/cglist/GeomDir/Detri_2.6.a.tar.gz"
        "https://web.archive.org/web/2019id_/http://www.geom.uiuc.edu/software/cglist/GeomDir/Detri_2.6.a.tar.gz"
    )
    for url in "${urls[@]}"; do
        info "尝试下载 Detri 原版: ${url}"
        if wget -q --timeout=30 -O "${DETRI_TAR}" "${url}" 2>/dev/null; then
            if file "${DETRI_TAR}" | grep -q gzip; then
                return 0
            fi
        fi
        rm -f "${DETRI_TAR}"
    done
    return 1
}

download_detri_github() {
    info "Wayback 失败，改用 GitHub 镜像 pkuwwt/Detri ..."
    local tmp="${JS_DIR}/Detri-master.tar.gz"

    # ghfast / gitclone 等国内加速（ghfast 通常比 gitclone tarball 更稳）
    fetch_url "${tmp}" \
        "https://github.com/pkuwwt/Detri/archive/refs/heads/master.tar.gz" \
        "https://ghfast.top/https://github.com/pkuwwt/Detri/archive/refs/heads/master.tar.gz" \
        "https://gitclone.com/github.com/pkuwwt/Detri/archive/refs/heads/master.tar.gz" \
        || {
            info "tar 下载失败，尝试 git clone 镜像 ..."
            rm -rf "${DETRI_DIR}"
            if git clone --depth 1 "https://gitclone.com/github.com/pkuwwt/Detri.git" "${DETRI_DIR}" 2>/dev/null; then
                :
            elif git clone --depth 1 "https://ghfast.top/https://github.com/pkuwwt/Detri.git" "${DETRI_DIR}" 2>/dev/null; then
                :
            else
                error "Detri 下载失败。请手动执行 fix_detri_download.sh 或见文档。"
            fi
            [[ -d "${DETRI_DIR}/basic" && -d "${DETRI_DIR}/sos" ]] || error "GitHub Detri 目录结构异常。"
            return 0
        }
    rm -rf "${DETRI_DIR}"
    mkdir -p "${JS_DIR}"
    tar -xzf "${tmp}" -C "${JS_DIR}"
    # GitHub 解压后目录名通常是 Detri-master，重命名为 Detri_2.6.a
    if [[ -d "${JS_DIR}/Detri-master" ]]; then
        mv "${JS_DIR}/Detri-master" "${DETRI_DIR}"
    fi
    rm -f "${tmp}"
    [[ -d "${DETRI_DIR}/basic" && -d "${DETRI_DIR}/sos" ]] || error "GitHub Detri 目录结构异常。"
}

if [[ -f "${DETRI_DIR}/build/lib/libSoS.a" ]]; then
    info "libSoS.a 已存在，跳过 Detri 编译。"
else
    if [[ ! -d "${DETRI_DIR}" ]]; then
        if [[ ! -f "${DETRI_TAR}" ]]; then
            download_detri_original || true
        fi
        if [[ -f "${DETRI_TAR}" ]]; then
            info "解压 Detri_2.6.a.tar.gz ..."
            tar -xzf "${DETRI_TAR}" -C "${JS_DIR}"
        else
            download_detri_github
        fi
    fi

# 源码应在 Detri_2.6.a/{basic,lia,sos}；Wayback 原版可能在 Detri_2.6.b/ 子目录
normalize_detri_layout() {
    if [[ -d "${DETRI_DIR}/basic" && -d "${DETRI_DIR}/sos" ]]; then
        return 0
    fi
    if [[ -d "${DETRI_DIR}/Detri_2.6.b/basic" ]]; then
        warn "发现嵌套 Detri_2.6.b/，提升到 Detri_2.6.a/ ..."
        for d in basic lia sos; do
            rm -rf "${DETRI_DIR}/${d}"
            cp -a "${DETRI_DIR}/Detri_2.6.b/${d}" "${DETRI_DIR}/${d}"
        done
    else
        error "Detri 目录结构异常：缺少 basic/lia/sos"
    fi
}

# 从 Detri_2.6.b 复制干净源码；修复 basic.h；不对 .h 做 sed
apply_detri_fixes() {
    normalize_detri_layout
    info "配置 Detri/SoS 编译（非交互）..."

    local BH="${DETRI_DIR}/basic/basic.h"
    if [[ -f "${BH}" ]]; then
        if ! grep -q '#ifndef __cplusplus' "${BH}" 2>/dev/null || ! grep -A5 '#ifndef __cplusplus' "${BH}" | grep -q '#define and'; then
            sed -i '/^#define and[[:space:]]*AND$/d;/^#define or[[:space:]]*OR$/d;/^#define not[[:space:]]*NOT$/d' "${BH}" 2>/dev/null || true
            sed -i '/#define mod[[:space:]]*%/a\
\
#ifndef __cplusplus\
#define and    AND\
#define or     OR\
#define not    NOT\
#endif' "${BH}"
        fi
        if grep -q '^#define %' "${BH}" 2>/dev/null; then
            sed -i 's/^#define %[[:space:]]*%/#define mod    %/' "${BH}"
        fi
    fi

    cat > "${DETRI_DIR}/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.5)
project(SoS C)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(BASIC_SRC
        ./basic/arg.c       ./basic/cb.c        ./basic/counter.c
        ./basic/getarg.c    ./basic/istaque.c   ./basic/math2.c
        ./basic/qsort.c     ./basic/tokenize.c  ./basic/basic.c
        ./basic/cb_doprnt.c ./basic/files.c     ./basic/isort.c
        ./basic/malloc.c    ./basic/prime.c     ./basic/time.c
        ./basic/uhash.c
)
SET(LIA_SRC
        ./lia/aux.c         ./lia/chars.c       ./lia/det.c
        ./lia/lia.c         ./lia/pool.c        ./lia/stack.c
)
SET(SOS_SRC
        ./sos/in_sphere.c   ./sos/lambda3.c     ./sos/lambda4.c
        ./sos/lambda5.c     ./sos/minor.c       ./sos/positive3.c
        ./sos/primitive.c   ./sos/smaller.c     ./sos/sos.c
)
include_directories(./basic/ ./lia/ ./sos/)
add_library(SoS ${BASIC_SRC} ${LIA_SRC} ${SOS_SRC})
EOF

    rm -f "${DETRI_DIR}/.sos_patches_applied"
    # 不再对 .h/.c 批量 sed（会破坏 #define mod）
}

    apply_detri_fixes

    info "编译 libSoS.a ..."
    rm -rf "${DETRI_DIR}/build"
    mkdir -p "${DETRI_DIR}/build"
    cmake -S "${DETRI_DIR}" -B "${DETRI_DIR}/build"
    cmake --build "${DETRI_DIR}/build" -j"$(nproc)"
    [[ -f "${DETRI_DIR}/build/lib/libSoS.a" ]] || error "libSoS.a 编译失败。"
    info "SoS 完成: ${DETRI_DIR}/build/lib/libSoS.a"
fi

# ---------- 5. 修正 JacobiSetComputation 的 CMakeLists（Darwin64 -> Linux64） ----------
CMAKE_JS="${JS_DIR}/CMakeLists.txt"
if grep -q 'lib.Darwin64' "${CMAKE_JS}" 2>/dev/null; then
    info "将 CMakeLists 中 trimesh 库路径改为 lib.Linux64 ..."
    sed -i 's|lib\.Darwin64/libtrimesh\.a|lib.Linux64/libtrimesh.a|g' "${CMAKE_JS}"
fi

# ---------- 6. 编译 JacobiSetComputation 串行程序 ----------
info "编译 JacobiSetComputation ..."
rm -rf "${JS_DIR}/build"
mkdir -p "${JS_DIR}/build"
cmake -S "${JS_DIR}" -B "${JS_DIR}/build"
cmake --build "${JS_DIR}/build" -j"$(nproc)"

BIN="${JS_DIR}/build/JacobiSetComputation"
if [[ -x "${BIN}" ]]; then
    info "=========================================="
    info "全部成功！可执行文件:"
    info "  ${BIN}"
    info "测试（需自备 .obj 网格）:"
    info "  ${BIN} /path/to/mesh.obj"
    info "=========================================="
else
    error "JacobiSetComputation 未生成，请查看上方编译错误。"
fi
