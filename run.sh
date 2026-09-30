#!/usr/bin/env bash
# Launch robohand, creating the Python 3.12 virtualenv on first run.
#
#   ./run.sh                 track camera 0
#   ./run.sh --demo          scripted motion, no camera needed
#   ./run.sh --source 1      use a different camera
#   ./run.sh --list-cameras  scan for video devices
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating the virtualenv (Python 3.12; mediapipe needs <= 3.12)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -r requirements.txt
  else
    python3.12 -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
  fi
fi

exec "$VENV/bin/python" -m robohand "$@"
