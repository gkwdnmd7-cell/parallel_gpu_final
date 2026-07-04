#!/usr/bin/env bash
# 单独下载并解压 Detri (SoS)，供 setup_deps_wsl.sh 失败后补救
set -euo pipefail

JS_DIR="/mnt/d/金介然/大三下/gpu/大作业/CODE/JacobiSetComputation-master"
DETRI_DIR="${JS_DIR}/Detri_2.6.a"
TMP="${JS_DIR}/Detri-master.tar.gz"

echo "[1/3] 从 ghfast 下载 Detri ..."
curl -fsSL --connect-timeout 30 --max-time 600 \
  -o "${TMP}" \
  "https://ghfast.top/https://github.com/pkuwwt/Detri/archive/refs/heads/master.tar.gz"

echo "[2/3] 解压 ..."
rm -rf "${DETRI_DIR}"
tar -xzf "${TMP}" -C "${JS_DIR}"
mv "${JS_DIR}/Detri-master" "${DETRI_DIR}"
rm -f "${TMP}"

echo "[3/3] 校验目录 ..."
test -d "${DETRI_DIR}/basic"
test -d "${DETRI_DIR}/lia"
test -d "${DETRI_DIR}/sos"
echo "OK: ${DETRI_DIR}"
echo ""
echo "接下来请重新运行: ./setup_deps_wsl.sh"
