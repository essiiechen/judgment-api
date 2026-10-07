#!/usr/bin/env bash
# ---------------------------------------------------------------
# 免敲命令版启动：装依赖 → 解压数据 → 解析 → 起 API → 开隧道
# Codespace 创建/重启时由 devcontainer 自动调用，无需任何手动输入。
# 公网地址会写进 PUBLIC_URL.txt —— 在左侧文件列表点开它就能看到，不用开终端。
# ---------------------------------------------------------------
set -u
cd "$(dirname "$0")"
LOG=start.log
echo "===== autostart $(date -u +%FT%TZ) =====" >> "$LOG"

# Python / pip 命令名在不同镜像里不一样（python 或 python3），统一探测
PY=$(command -v python || command -v python3)
PIP=$(command -v pip || command -v pip3)
echo "    python=$PY  pip=$PIP" | tee -a "$LOG"

echo "==> [1/5] 安装依赖" | tee -a "$LOG"
"$PIP" install --quiet fastapi uvicorn 2>&1 | tail -2 | tee -a "$LOG"

echo "==> [2/5] 解压数据" | tee -a "$LOG"
if [ ! -f judgments.jsonl ] && [ -f judgments.jsonl.gz ]; then
  gunzip -k -f judgments.jsonl.gz
fi

echo "==> [3/5] 解析判决书（3520 份，约 10 秒）" | tee -a "$LOG"
if [ ! -f parsed.jsonl ]; then
  "$PY" parse.py 2>&1 | tail -5 | tee -a "$LOG"
else
  echo "    parsed.jsonl 已存在，跳过" | tee -a "$LOG"
fi

echo "==> [4/5] 启动 API（127.0.0.1:8000，带崩溃自愈）" | tee -a "$LOG"
pkill -f "api.py" 2>/dev/null || true
setsid nohup bash -c "while true; do \"$PY\" api.py; echo '[guard] API 退出，2 秒后重启' >> api.log; sleep 2; done" \
  > guard.out 2>&1 &
sleep 10
curl -s -m 10 http://127.0.0.1:8000/health | tee -a "$LOG"
echo >> "$LOG"

echo "==> [5/5] 开 Cloudflare 隧道" | tee -a "$LOG"
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) BIN=cloudflared-linux-amd64 ;;
  aarch64|arm64) BIN=cloudflared-linux-arm64 ;;
  *) BIN=cloudflared-linux-amd64 ;;
esac
if [ ! -x ./cloudflared ]; then
  curl -sL -o cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/$BIN"
  chmod +x cloudflared
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
rm -f tunnel.log PUBLIC_URL.txt
setsid nohup ./cloudflared tunnel --url http://127.0.0.1:8000 --protocol http2 > tunnel.log 2>&1 &

# 最多等 40 秒，把公网地址写进 PUBLIC_URL.txt
for i in $(seq 1 40); do
  U=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' tunnel.log | head -1 || true)
  if [ -n "$U" ]; then
    echo "$U" > PUBLIC_URL.txt
    echo "    公网地址已写入 PUBLIC_URL.txt -> $U" | tee -a "$LOG"
    break
  fi
  sleep 1
done

if [ ! -f PUBLIC_URL.txt ]; then
  echo "    未拿到地址，请看 tunnel.log / start.log 排查" | tee -a "$LOG"
fi
exit 0
