#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/Users/otomi/Desktop/vs code/CFD3_AutoSystem/CFD3_CLEAN"
OUT="$PROJECT_DIR/streamlit_diagnose.txt"
LOG="$PROJECT_DIR/streamlit_diagnose.log"
: > "$OUT"
{
  echo "==== DATE ===="
  date
  echo "\n==== HOST ===="
  uname -a
  echo "\n==== PYTHON ===="
  python3 --version 2>&1 || python --version 2>&1
  echo "\n==== which streamlit ===="
  which streamlit || true
  echo "\n==== streamlit version (pip show) ===="
  pip show streamlit 2>/dev/null || true
  echo "\n==== STREAMLIT PROCESSES ===="
  ps aux | grep -i "streamlit run" | grep -v grep || true
  echo "\n==== pgrep -af streamlit ===="
  pgrep -af streamlit || true
  echo "\n==== LSOF :8501 ===="
  lsof -i :8501 || true
  echo "\n==== NETSTAT (port 8501) ===="
  netstat -anv | grep 8501 || true
  echo "\n==== streamlit.log (if exists) ===="
  if [ -f "$PROJECT_DIR/streamlit.log" ]; then
    ls -l "$PROJECT_DIR/streamlit.log"
    echo "\n--- tail streamlit.log ---"
    tail -n 200 "$PROJECT_DIR/streamlit.log"
  else
    echo "no streamlit.log found"
  fi
  echo "\n==== LAST 200 LINES OF SYSLOG (may require sudo) ===="
  # macOS system log might be verbose; try last lines from unified log
  if command -v log >/dev/null 2>&1; then
    log show --last 1h --predicate 'process == "streamlit"' 2>/dev/null || true
  fi
  echo "\n==== RECOMMENDED: start streamlit in foreground to see errors ===="
  echo "cd '$PROJECT_DIR' && streamlit run app_streamlit_v2.py --server.port 8501 --server.address 0.0.0.0"
  echo "\n==== CURL CHECK ===="
  curl -I http://localhost:8501 || true
} >> "$OUT" 2>&1

echo "Diagnostic data written to: $OUT"

echo "(Also: streamlit log path: $LOG - create or inspect if you used restart script)"
exit 0
