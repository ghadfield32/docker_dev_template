#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf "\n[%s] %s\n" "$(date +'%F %T')" "$*" >&2; }
die() { printf "\n[ERROR] %s\n" "$*" >&2; exit 1; }

# Parse boolean-ish env values: 1/true/yes/on → 1, else 0
parse_env_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) echo "1" ;;
    *) echo "0" ;;
  esac
}

# Load /workspace/.env if present, exporting its variables (safe for comments & inline comments)
read_dotenv_if_present() {
  if [ -f "/workspace/.env" ]; then
    log "Sourcing /workspace/.env into the environment"
    # normalize CRLF just in case
    sed -i 's/\r$//' /workspace/.env || true
    # export all simple assignments; shell will ignore comments safely
    # and treat 'VAR=value # comment' correctly.
    set -a
    . /workspace/.env
    set +a
  else
    log "No /workspace/.env to source"
  fi
}

# --- 0) Normalize line endings for any .sh files we might call later ---
normalize_line_endings() {
  log "Normalizing CRLF -> LF for .devcontainer/*.sh (safety against sh\\r)."
  # sed works without needing extra packages like dos2unix
  find .devcontainer -maxdepth 1 -type f -name "*.sh" -print0 | \
    xargs -0 -I{} bash -lc 'sed -i "s/\r$//" "{}"'
}

# --- 1) Ensure git safe.directory to avoid dubious ownership errors ---
fix_git_safety() {
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    local root
    root="$(git rev-parse --show-toplevel || echo /workspace)"
    log "Marking ${root} as a safe.directory for git."
    git config --global --add safe.directory "${root}" || true
  else
    log "Not a git repo or git unavailable—skipping safe.directory."
  fi
}

# --- 2) Ensure uv targets the project venv we expect ---
ensure_uv_env() {
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/app/.venv}"
  log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  if [[ ! -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
    log "Creating uv venv at ${UV_PROJECT_ENVIRONMENT}"
    uv venv "${UV_PROJECT_ENVIRONMENT}" --python "3.10" --prompt "docker_dev_template"
  fi
  # Make sure activation is easy in interactive shells too
  echo ". ${UV_PROJECT_ENVIRONMENT}/bin/activate" > /etc/profile.d/10-uv-activate.sh
}

# --- 3) Sync core project deps (respect lockfile) ---
sync_project_deps() {
  if [[ -f "/workspace/.devcontainer/pyproject.toml" ]]; then
    log "Syncing project deps with uv (frozen)."
    (cd /workspace/.devcontainer && uv sync --frozen --no-dev) || {
      log "Lock out-of-date; refreshing…"
      (cd /workspace/.devcontainer && uv sync --no-dev && uv lock)
    }
  else
    log "No /workspace/.devcontainer/pyproject.toml found—skipping uv sync."
  fi
}

# --- 4) Enhanced RTX 5080 Memory Management Setup ---
setup_rtx5080_memory_management() {
  log "Setting up enhanced RTX 5080 memory management for tcache double free prevention..."

  # Critical: Use jemalloc instead of glibc malloc to prevent tcache conflicts
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

  # Verify jemalloc is working and available
  if ! python -c "import ctypes; ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2')" 2>/dev/null; then
    log "❌ jemalloc not available - this is critical for preventing tcache double free errors"
    log "Installing jemalloc..."
    apt-get update && apt-get install -y libjemalloc2 libjemalloc-dev || {
      log "⚠️  Failed to install jemalloc, falling back to system malloc (may cause tcache issues)"
      unset LD_PRELOAD
    }
  else
    log "✅ jemalloc memory allocator loaded successfully"
  fi

  # Enhanced memory management environment variables for RTX 5080
  export MALLOC_ARENA_MAX=1
  export MALLOC_MMAP_THRESHOLD_=131072
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1

  # Additional memory management for preventing double free
  export MALLOC_TCACHE_MAX=0  # Disable tcache entirely
  export MALLOC_MMAP_MAX=0    # Disable mmap threshold

  # GPU framework specific memory settings
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  export TF_FORCE_GPU_ALLOW_GROWTH=true

  log "✅ Memory management environment configured for RTX 5080"
}

# --- 5) Detect CUDA and map to PyTorch wheel channel ---
detect_cuda_channel() {
  # RTX 5080 requires CUDA 12.8, use nightly builds for latest compatibility
  echo "cu128"
}

# --- 6) Enhanced GPU Framework Installation (TensorFlow-Gated) ---
install_accelerators() {
  log "Installing GPU frameworks (RTX 5080)… honoring INSTALL_TF toggle"

  read_dotenv_if_present
  local INSTALL_TF_VAL
  INSTALL_TF_VAL="$(parse_env_bool "${INSTALL_TF:-0}")"
  log "INSTALL_TF=${INSTALL_TF:-<unset>} (parsed=${INSTALL_TF_VAL})"

  # Clear any forced JAX platform – we want autodetect
  log "Clearing any JAX platform forcing…"
  unset JAX_PLATFORM_NAME || true
  unset JAX_PLATFORMS || true

  # Avoid mixing pip 'nvidia-*' CUDA stacks with system CUDA
  log "Ensuring no pip nvidia-* CUDA libs remain (prevents double loads)…"
  uv pip uninstall -y \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
    nvidia-cuda-cupti-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    || true

  # ---- PyTorch (nightly cu128 for Blackwell) ----
  if python -c "import torch" >/dev/null 2>&1; then
    log "PyTorch importable — checking CUDA availability…"
    if ! python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    then
      log "Upgrading PyTorch to nightly cu128…"
      uv pip uninstall -y torch torchvision torchaudio || true
      uv pip install --no-cache-dir --pre torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/nightly/cu128"
    else
      log "PyTorch with CUDA OK — leaving as-is"
    fi
  else
    log "Installing PyTorch nightly cu128…"
    uv pip install --no-cache-dir --pre torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/nightly/cu128"
  fi

  # ---- JAX (CUDA via local plugin) ----
  log "Reconciling JAX backends and installing CUDA plugin…"
  uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true
  # Prefer local plugin tied to system CUDA:
  uv pip install --no-cache-dir "jax[cuda12-local]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # ---- TensorFlow (optional) ----
  if [ "${INSTALL_TF_VAL}" = "1" ]; then
    log "INSTALL_TF=1 → installing TensorFlow nightly for RTX 5080"
    uv pip uninstall -y tensorflow tensorboard keras || true
    uv pip install -U --pre tf-nightly tb-nightly keras-nightly
  else
    log "INSTALL_TF=0 → skipping TensorFlow installation"
  fi
}

# --- 7) Enhanced Framework Initialization (JAX+PyTorch Only) ---
initialize_frameworks() {
  log "Initializing JAX and PyTorch (no TensorFlow)…"

  python3 -c "
import os, gc
print('🔧 Initializing JAX…')
try:
    import jax, importlib.util as u
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    devs = jax.devices()
    print('   JAX devices:', devs)
    from jax import numpy as jnp
    x = jnp.ones((128,128)); y = (x @ x.T).sum(); _ = y.block_until_ready()
    print('   JAX small matmul: OK')
except Exception as e:
    print('❌ JAX init failed:', e)

print('🔧 Initializing PyTorch…')
try:
    import torch
    print('   torch:', torch.__version__, 'CUDA avail:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('   device:', torch.cuda.get_device_name(0))
        a = torch.randn(128,128, device='cuda')
        b = torch.randn(128,128, device='cuda')
        _ = (a@b).sum().item()
        print('   PyTorch small matmul: OK')
except Exception as e:
    print('❌ PyTorch init failed:', e)

gc.collect()
"
}

# --- 8) Register a Jupyter kernel pointing at this venv ---
register_kernel() {
  log "Registering Jupyter ipykernel for this environment…"
  python -c "
import json, sys, subprocess
name = 'docker_dev_template'
display = 'Python (docker_dev_template)'
try:
    subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install',
                           '--sys-prefix', '--name', name, '--display-name', display])
    print('Kernel registered:', display)
except subprocess.CalledProcessError as e:
    print('Kernel registration failed:', e)
    sys.exit(1)
"
}

# --- 9) Enhanced Accelerator Verification with Memory Testing (TensorFlow-Gated) ---
verify_accelerators() {
  read_dotenv_if_present
  local INSTALL_TF_VAL STRICT_VERIFY_VAL
  INSTALL_TF_VAL="$(parse_env_bool "${INSTALL_TF:-0}")"
  STRICT_VERIFY_VAL="$(parse_env_bool "${STRICT_VERIFY:-0}")"
  log "Verification policy: INSTALL_TF=${INSTALL_TF:-<unset>} (parsed=${INSTALL_TF_VAL}), STRICT_VERIFY=${STRICT_VERIFY:-0}"

  local failed=()
  local attempted=()
  local required_failed=0

  log "Verifying PyTorch CUDA with memory testing…"
  if python - <<'PY'
import sys, gc
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda.is_available:', torch.cuda.is_available())
    print('cuda.device_count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('cuda.device_name:', torch.cuda.get_device_name(0))
        print('Testing PyTorch memory management...')
        for _ in range(5):
            x = torch.randn(1000,1000, device='cuda')
            del x; torch.cuda.empty_cache(); gc.collect()
        print('✅ PyTorch CUDA verification passed with memory testing')
        sys.exit(0)
    else:
        print('⚠️  PyTorch installed but CUDA not available'); sys.exit(2)
except Exception as e:
    print('❌ Torch import/verify failed:', e); sys.exit(1)
PY
  then
    attempted+=("PyTorch")
  else
    attempted+=("PyTorch"); failed+=("PyTorch"); required_failed=$((required_failed+1))
  fi

  log "Verifying JAX CUDA with memory testing…"
  if python - <<'PY'
import sys, gc
try:
    import jax, jax.numpy as jnp
    print('jax:', jax.__version__)
    print('devices:', jax.devices())
    has_gpu = any('GPU' in str(d).upper() or 'CUDA' in str(d).upper() for d in jax.devices())
    if has_gpu:
        print('Testing JAX memory management...')
        for _ in range(5):
            x = jnp.ones((1000,1000)); del x; gc.collect()
        print('✅ JAX CUDA verification passed with memory testing'); sys.exit(0)
    else:
        print('⚠️  JAX installed but no GPU devices detected'); sys.exit(2)
except Exception as e:
    print('❌ JAX import/verify failed:', e); sys.exit(1)
PY
  then
    attempted+=("JAX")
  else
    attempted+=("JAX"); failed+=("JAX"); required_failed=$((required_failed+1))
  fi

  log "TensorFlow verification policy check…"
  if [ "${INSTALL_TF_VAL}" = "1" ]; then
    log "INSTALL_TF=1 → attempting TensorFlow verification…"
    if python - <<'PY'
import sys, gc
try:
    import tensorflow as tf
    print('tensorflow:', tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print('tf GPUs:', gpus)
    if gpus:
        print('Testing TensorFlow memory management...')
        for _ in range(5):
            x = tf.ones((1000,1000)); del x; gc.collect()
        print('✅ TensorFlow CUDA verification passed with memory testing'); sys.exit(0)
    else:
        print('⚠️  TensorFlow installed but no GPU devices detected'); sys.exit(2)
except Exception as e:
    print('❌ TF import/verify failed:', e); sys.exit(1)
PY
    then
      attempted+=("TensorFlow")
    else
      attempted+=("TensorFlow"); failed+=("TensorFlow"); required_failed=$((required_failed+1))
    fi
  else
    # Not requested and likely not installed; don't attempt or mark as failed
    log "INSTALL_TF=0 → skipping TensorFlow verification (not requested)."
  fi

  if [ "${STRICT_VERIFY_VAL}" = "1" ]; then
    if [ "${required_failed}" -gt 0 ]; then
      log "⚠️  Some requested frameworks failed verification: ${failed[*]}"
      return 1
    fi
  else
    if [ "${#failed[@]}" -gt 0 ]; then
      log "⚠️  Some requested frameworks failed verification: ${failed[*]}"
      # Non-strict: still return success to avoid blocking container usage
      return 0
    fi
  fi

  log "✅ All requested frameworks verified successfully with memory testing"
  return 0
}

# --- 10) Memory Allocator Verification ---
verify_memory_allocator() {
  log "Verifying memory allocator configuration..."

  # Check if jemalloc is loaded
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    log "LD_PRELOAD is set to: $LD_PRELOAD"

    if [[ -f "$LD_PRELOAD" ]]; then
      log "✅ jemalloc library file exists"
    else
      log "❌ jemalloc library file not found: $LD_PRELOAD"
      return 1
    fi

    # Test if jemalloc is actually being used
    if python -c "import ctypes; ctypes.CDLL('$LD_PRELOAD')" 2>/dev/null; then
      log "✅ jemalloc is loadable by Python"
    else
      log "❌ jemalloc is not loadable by Python"
      return 1
    fi
  else
    log "⚠️  LD_PRELOAD not set - using system malloc (may cause tcache issues)"
  fi

  # Check memory management environment variables
  local mem_vars=("MALLOC_ARENA_MAX" "MALLOC_MMAP_THRESHOLD_" "PYTORCH_NO_CUDA_MEMORY_CACHING" "MALLOC_TCACHE_MAX")
  for var in "${mem_vars[@]}"; do
    if [[ -n "${!var:-}" ]]; then
      log "✅ $var=${!var}"
    else
      log "⚠️  $var not set"
    fi
  done
}

# --- ENTRYPOINT ---
main() {
  # Normalize scripts early
  normalize_line_endings

  # Load environment variables early (after CRLF normalization)
  read_dotenv_if_present

  # Enable debug mode if requested
  if [[ "${DEBUG:-0}" == "1" ]]; then
    log "DEBUG mode enabled - enabling shell tracing"
    set -x
  fi

  fix_git_safety
  ensure_uv_env
  setup_rtx5080_memory_management
  verify_memory_allocator

  if [[ "${1:-}" == "--verify-only" ]]; then
    verify_accelerators || true
    return 0
  fi

  sync_project_deps
  install_accelerators
  initialize_frameworks
  register_kernel
  verify_accelerators
  log "Opening sequence completed with enhanced RTX 5080 memory management."
}

main "$@"
