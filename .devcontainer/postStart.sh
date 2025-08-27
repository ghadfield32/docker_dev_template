#!/usr/bin/env bash
set -euo pipefail

DEV_DIR="/workspace/.devcontainer"
export UV_PROJECT_ENVIRONMENT=/app/.venv

# Normalize CRLF to LF for any .sh files (common Windows pitfall)
if command -v dos2unix >/dev/null 2>&1; then
  find "${DEV_DIR}" -maxdepth 1 -type f -name "*.sh" -exec dos2unix {} \; || true
fi

# Your existing bootstrap
"${DEV_DIR}/gpu_bootstrap.sh"

# Safety net: if postCreate sentinel missing/empty, run it now (idempotent)
if [ ! -s "${DEV_DIR}/.postcreate.hash" ]; then
  echo "[postStart] postCreate sentinel missing; running postCreate now"
  bash "${DEV_DIR}/scripts/postcreate.sh" || true
fi

python "${DEV_DIR}/verify_env.py" || true

echo "[postStart] DONE"
