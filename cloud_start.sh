#!/usr/bin/env bash
# ---------------------------------------------------------------
# 云端一键启动：装依赖 → 解析数据 → 起 API（127.0.0.1:8000）→ 开 Cloudflare 隧道
# 用法（在 GitHub Codespace / 任意 Linux 云主机 / Colab 里）：
#     bash cloud_start.sh
# 隧道起来后终端会打印 https://xxxx.trycloudflare.com，复制它即可提交。
# ---------------------------------------------------------------
set -e
cd "$(dirname "$0")"

echo "==> [1/4] 安装依赖"
pip install --quiet fastapi uvicorn 2>&1 | tail -2 || pip install fastapi uvicorn

# 网页上传有 25 MiB 单文件限制，所以原始数据以 .gz 形式进仓库；这里自动解压
if [ ! -f judgments.jsonl ] && [ -f judgments.jsonl.gz ]; then
  echo "    解压 judgments.jsonl.gz -> judgments.jsonl"
  gunzip -k -f judgments.jsonl.gz
fi

echo "==> [2/4] 解析判决书（生成 parsed.jsonl，约 10 秒）"
if [ ! -f parsed.jsonl ]; then
  python parse.py
else
  echo "    parsed.jsonl 已存在，跳过"
fi

echo "==> [3/4] 启动 API（仅监听 127.0.0.1:8000，带崩溃自愈）"
pkill -f "python api.py" 2>/dev/null || true
pkill -f "guard-api" 2>/dev/null || true
# 守护循环：API 万一崩了自动拉起，避免老师的检查程序撞上 502
nohup bash -c 'while true; do python api.py; echo "[guard] API 退出，2 秒后重启" >> api.log; sleep 2; done' \
  > guard.out 2>&1 &
sleep 7
echo -n "    /health -> "
curl -s -m 10 http://127.0.0.1:8000/health || { echo "API 未起来，看 api.log"; tail -20 api.log; exit 1; }
echo

echo "==> [4/4] 下载 cloudflared 并开隧道"
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) BIN=cloudflared-linux-amd64 ;;
  aarch64|arm64) BIN=cloudflared-linux-arm64 ;;
  *) echo "未知架构 $ARCH"; exit 1 ;;
esac
if [ ! -x ./cloudflared ]; then
  curl -sL -o cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/$BIN"
  chmod +x cloudflared
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true

echo
echo "================================================================"
echo " 下面的 https://xxxx.trycloudflare.com 就是公网地址，复制提交："
echo "================================================================"
echo
exec ./cloudflared tunnel --url http://127.0.0.1:8000 --protocol http2
