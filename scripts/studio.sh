#!/usr/bin/env bash
# Start Bio-Bot Studio: fetch assets if missing, build the web app if needed,
# then run the server on http://localhost:8765
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x $PY ]]; then
  uv venv --python 3.12 .venv
  uv pip install --python $PY -e .
fi

[[ -d third_party/mujoco_menagerie/flybody && -f studio/web/public/hdri/sky_2k.hdr ]] || $PY scripts/fetch_assets.py

if [[ ! -f studio/web/dist/index.html || -n "$(find studio/web/src studio/web/index.html -newer studio/web/dist/index.html 2>/dev/null | head -1)" ]]; then
  (cd studio/web && { [[ -d node_modules ]] || npm install --no-audit --no-fund; } && npm run build)
fi

echo "Bio-Bot Studio -> http://localhost:${PORT:-8765}"
exec $PY -m uvicorn studio.server.app:app --host 127.0.0.1 --port "${PORT:-8765}"
