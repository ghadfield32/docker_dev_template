# .devcontainer/Dockerfile — uv‑based replacement for the previous Conda image
# -----------------------------------------------------------------------------
# CUDA + cuDNN base with drivers already installed --------------------------------
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# ----------------------------------------------------------------------------
# --- Build‑time ARGs ---------------------------------------------------------
ARG PYTHON_VER=3.10               # system Python to seed the venv
ARG ENV_NAME=docker_dev_template  # prompt shown when the venv is active
# JAX GPU memory tuning (kept 1‑for‑1 with the old Conda image)
ARG JAX_PREALLOCATE=true
ARG JAX_MEM_FRAC=0.95
ARG JAX_ALLOCATOR=platform
ARG JAX_PREALLOC_LIMIT=8589934592

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 1) Core OS deps, build tools, & Python (system) -----------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git procps htop util-linux build-essential \
        python3 python3-venv python3-pip python3-dev \
        autoconf automake libtool m4 cmake pkg-config \
        jags \
        && pkg-config --modversion jags \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 2) Copy a *pinned* uv & uvx binary pair from the official distroless image --
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# ----------------------------------------------------------------------------
# 3) Create project dir & copy only the lock/manifest for best layer‑caching --
WORKDIR /app
COPY pyproject.toml uv.lock* ./

# ----------------------------------------------------------------------------
# 4) Create an in‑project venv & sync dependencies (no project‑source yet) ----
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}" && \
    uv sync --locked

# Promote venv for all later layers ------------------------------------------------
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# ----------------------------------------------------------------------------
# 5) Install CUDA wheels (PyTorch nightly + JAX CUDA 12) ----------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --pre --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    uv pip install --no-cache-dir \
        "jax[cuda12-local]==0.4.38" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ----------------------------------------------------------------------------
# 6) Install PyJAGS with the cstdint header work‑around -----------------------
RUN CPPFLAGS="-include cstdint" uv pip install --no-build-isolation pyjags==1.3.8

# ----------------------------------------------------------------------------
# 7) Copy *rest* of the project after deps → fast rebuild when code changes ---
COPY . /app

# ----------------------------------------------------------------------------
# 8) GPU‑tuning env vars (carried forward from Conda‑based image) -------------
ENV XLA_PYTHON_CLIENT_PREALLOCATE=${JAX_PREALLOCATE}
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRAC}
ENV XLA_PYTHON_CLIENT_ALLOCATOR=${JAX_ALLOCATOR}
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_force_host_platform_device_count=1"
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=${JAX_PREALLOC_LIMIT}

# ----------------------------------------------------------------------------
# 9) Library path so PyJAGS & CUDA libs resolve correctly ---------------------
ENV LD_LIBRARY_PATH="/app/.venv/lib:${LD_LIBRARY_PATH}"

# ----------------------------------------------------------------------------
# 10) Final working directory & default command ------------------------------
WORKDIR /workspace
CMD ["bash"]
