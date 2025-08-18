#!/usr/bin/env bash
set -euxo pipefail

BASE_PY="/app/.venv/bin/python"
UV="${UV:-uv}"

# venv locations
JAX_ENV="/opt/venvs/jax"
TORCH_ENV="/opt/venvs/torch-cu121"
TF_ENV="/opt/venvs/tf"

mkdir -p /opt/venvs

create_env() {
  local path="$1"
  local pyver
  pyver="$($BASE_PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if [ ! -d "$path" ]; then
    echo "[frameworks] creating venv: $path (py $pyver)"
    $UV venv "$path" --python "$pyver"
  fi
}

install_kernel() {
  local venv="$1"
  local name="$2"
  local label="$3"
  "$venv/bin/python" -m pip install --upgrade pip ipykernel >/dev/null
  "$venv/bin/python" -m ipykernel install --name "$name" --display-name "$label" --user
}

# --- JAX (CUDA12 via PJRT plugin) ---
create_env "$JAX_ENV"
$UV pip --python "$JAX_ENV/bin/python" install --no-cache-dir \
  "jax[cuda12]==0.6.0" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
install_kernel "$JAX_ENV" "jax-cu12" "Python (JAX CUDA12)"

# --- PyTorch cu121 (official index) ---
create_env "$TORCH_ENV"
$UV pip --python "$TORCH_ENV/bin/python" install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
install_kernel "$TORCH_ENV" "torch-cu121" "Python (PyTorch cu121)"

# --- TensorFlow (GPU) – optional, guarded by env var ---
if [ "${ENABLE_TF:-0}" = "1" ]; then
  create_env "$TF_ENV"
  # Follow TF pip guidance for GPU-enabled install on Linux
  $UV pip --python "$TF_ENV/bin/python" install --no-cache-dir "tensorflow[and-cuda]"
  install_kernel "$TF_ENV" "tf-gpu" "Python (TensorFlow GPU)"
fi

echo "[frameworks] summary:"
echo "  JAX venv:    $JAX_ENV"
echo "  Torch venv:  $TORCH_ENV"
if [ -d "$TF_ENV" ]; then echo "  TF venv:     $TF_ENV"; fi
