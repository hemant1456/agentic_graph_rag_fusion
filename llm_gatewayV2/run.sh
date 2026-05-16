#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PORT="${GATEWAY_V2_PORT:-8100}"
# Free the port if something is already bound to it
lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r requirements.txt
fi
exec ./.venv/bin/python main.py
