#!/usr/bin/env bash
set -euo pipefail

# Allow overriding by env
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Starting API on ${HOST}:${PORT}"
exec uvicorn api.main:app --host "$HOST" --port "$PORT" --workers 1 --reload




