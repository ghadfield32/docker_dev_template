#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=".devcontainer/.env.runtime"

# Only generate once per local checkout
if [[ ! -f "${ENV_FILE}" ]]; then
  # workspaceFolderBasename → "docker-dev-template", devcontainerId → random but stable
  RAND="${devcontainerId:-$(date +%s | tail -c 6 | tr -d '\n')}"
  BASENAME="${localWorkspaceFolderBasename:-$(basename "$PWD")}"
  
  # Clean up basename to be docker-compose friendly
  BASENAME=$(echo "$BASENAME" | sed 's/[^a-zA-Z0-9_-]/-/g' | tr '[:upper:]' '[:lower:]')
  
  PROJECT_NAME="${BASENAME}-${RAND}"
  
  echo "COMPOSE_PROJECT_NAME=${PROJECT_NAME}" > "${ENV_FILE}"
  echo "ENV_NAME=${PROJECT_NAME}"           >> "${ENV_FILE}"
  echo "🔧  Generated unique project name: ${PROJECT_NAME}"
  echo "🔧  Wrote ${ENV_FILE}: $(cat ${ENV_FILE})"
fi 
