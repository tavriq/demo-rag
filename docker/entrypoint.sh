#!/bin/sh
set -eu

# Build the index into the volume on first start or when the corpus changed.
# If that fails the server still starts and reports the problem on / and /api/health.
python -m app.build_index --if-missing || echo "index build failed; starting in degraded mode" >&2

exec uvicorn --factory app.main:create_app \
  --host 0.0.0.0 --port 8000 \
  --workers 1 \
  --no-proxy-headers \
  --no-server-header
