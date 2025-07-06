#!/usr/bin/env bash
# Copy the template only on first run so local secrets are not overwritten
set -euo pipefail

# Ensure we're in the workspace directory
cd /workspace

# Copy template if it doesn't exist
if [ ! -f .env ]; then
  if [ -f .env.template ]; then
    echo "📝  Generating default .env from template"
    cp .env.template .env
  else
    echo "⚠️  Warning: .env.template not found"
  fi
fi
