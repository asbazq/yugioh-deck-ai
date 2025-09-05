#!/bin/sh
set -eu

# Optional: auto-import snapshot for local chroma mode
python scripts/entrypoint.py || true

exec uvicorn server:app --host 0.0.0.0 --port 8000
