# .devcontainer/Dockerfile
ARG CUDA_TAG=12.8.0
FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=docker_dev_template
ENV DEBIAN_FRONTEND=noninteractive

# OS dependencies with jemalloc for memory management
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates git procps htop util-linux build-essential \
        python3 python3-venv python3-pip python3-dev \
        autoconf automake libtool m4 cmake pkg-config \
        jags iproute2 net-tools lsof \
        libjemalloc2 libjemalloc-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Node.js for VS Code remote development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get update && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

WORKDIR /app

# Create virtual environment
RUN uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}"

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Critical: Use jemalloc instead of glibc malloc to prevent tcache conflicts
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Install base requirements
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir jupyterlab ipykernel

# CUDA symlink setup
RUN set -e; \
    CUDA_REAL="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
    if [ -z "$CUDA_REAL" ] && [ -d /usr/local/cuda ]; then CUDA_REAL="/usr/local/cuda"; fi; \
    if [ -z "$CUDA_REAL" ]; then echo '❌ No CUDA toolkit found.' >&2; exit 1; fi; \
    if [ "$CUDA_REAL" != "/usr/local/cuda" ]; then ln -sfn "$CUDA_REAL" /usr/local/cuda; fi; \
    echo "🟢 CUDA toolkit: $CUDA_REAL"

# GPU frameworks will be installed in postCreate via opening.sh
# This prevents uv sync from removing them later

# Copy project files and install dependencies
COPY pyproject.toml /workspace/.devcontainer/
COPY uv.lock* /workspace/.devcontainer/

# Install project dependencies at build time (frozen first; refresh lock if needed)
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace/.devcontainer && \
    (uv sync --frozen --no-dev || (echo "🔁 Lock out-of-date; refreshing…" && uv sync --no-dev && uv lock))

# RTX 5080 optimized environment variables for memory management
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=8589934592
ENV LD_LIBRARY_PATH="/app/.venv/lib"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# Additional memory management for RTX 5080
ENV MALLOC_ARENA_MAX=1
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Note: XLA_FLAGS and CUDA-specific LD_LIBRARY_PATH will be set by set_cuda_policy() in opening.sh

WORKDIR /workspace
RUN echo 'cd /workspace' > /etc/profile.d/99-workspace-cd.sh
RUN mkdir -p /root/.ipython/profile_default/startup && \
    printf "import os, sys\nos.chdir('/workspace')\nsys.path.append('/workspace')\n" \
      > /root/.ipython/profile_default/startup/00-cd-workspace.py
RUN echo '. /app/.venv/bin/activate' > /etc/profile.d/10-uv-activate.sh

CMD ["bash"]
