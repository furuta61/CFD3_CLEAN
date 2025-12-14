#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/Users/otomi/Desktop/vs code/CFD3_AutoSystem/CFD3_CLEAN"
SCRIPT="app_streamlit_v2.py"
PORT=8501
LOG="$PROJECT_DIR/streamlit.log"

cd "$PROJECT_DIR"

# 優しく停止
pkill -f "streamlit run" || true
sleep 1

# 強制終了されて残っている PID を kill（念のため）
pids=$(pgrep -f "streamlit run" || true)
if [ -n "$pids" ]; then
  kill -9 $pids || true
fi

# 仮想環境がある場合はここで有効化（必要に応じて編集）
# if [ -f ".venv/bin/activate" ]; then
#   source .venv/bin/activate
# fi

# 起動（バックグラウンド、ログ出力）
nohup streamlit run "$SCRIPT" --server.port $PORT > "$LOG" 2>&1 &

echo "Streamlit restarted (port $PORT). Log -> $LOG"
