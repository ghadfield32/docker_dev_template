# Dockerfile: RTX 4090 devcontainer with UV, JAX, and PyTorch (CUDA 12.x)

ARG CUDA_TAG=12.4.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=docker_dev_template
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates git procps htop \
        python3 python3-venv python3-pip python3-dev \
        build-essential cmake pkg-config \
        libjemalloc2 libjemalloc-dev \
        iproute2 net-tools lsof wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV package manager
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

WORKDIR /app

# Create venv managed by UV
RUN uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}"

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH="/workspace"

# Memory and allocator settings
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TCACHE_MAX=0 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

# GPU‑relevant environment
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True \
    JAX_PREALLOCATION_SIZE_LIMIT_BYTES=17179869184

# Bring in project descriptors and tests
COPY pyproject.toml /workspace/
COPY uv.lock* /workspace/
COPY .devcontainer/validate_gpu.py /app/validate_gpu.py
COPY .devcontainer/tests/ /app/tests/

# Resolve project dependencies with UV
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    cd /workspace && (uv sync --frozen --no-dev || (uv sync --no-dev && uv lock))

# CRITICAL FIX 2: Install PyTorch first to establish CUDA environment
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Installing PyTorch with CUDA 12.4..." && \
    uv pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# CRITICAL FIX 3: Install compatible CuDNN 9.8.0 to satisfy JAX requirements
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Upgrading CuDNN to 9.8.0 for JAX compatibility..." && \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12==9.8.0.69 || \
    uv pip install --no-cache-dir --upgrade nvidia-cudnn-cu12>=9.8.0

# CRITICAL FIX 4: Install JAX after CuDNN upgrade with proper dependency resolution
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    echo "Removing any existing JAX installations..." && \
    (uv pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true) && \
    echo "Installing JAX with CUDA 12 support..." && \
    (uv pip install --no-cache-dir "jax[cuda12-local]>=0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
     || uv pip install --no-cache-dir "jax[cpu]>=0.4.26")

# Jupyter kernel support
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install ipykernel jupyter-client jupyterlab

# CUDA libs in path - include both system and package CUDA libraries
ENV LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Shell activation helper with updated library paths
RUN echo '#!/bin/bash' > /app/activate_uv.sh && \
    echo 'export VIRTUAL_ENV="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PATH="/app/.venv/bin:$PATH"' >> /app/activate_uv.sh && \
    echo 'export UV_PROJECT_ENVIRONMENT="/app/.venv"' >> /app/activate_uv.sh && \
    echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /app/activate_uv.sh && \
    echo 'export LD_LIBRARY_PATH="/app/.venv/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"' >> /app/activate_uv.sh && \
    echo 'cd /workspace' >> /app/activate_uv.sh && \
    chmod +x /app/activate_uv.sh && \
    echo 'source /app/activate_uv.sh' > /etc/profile.d/10-uv-activate.sh && \
    echo 'source /app/activate_uv.sh' >> /root/.bashrc && \
    chmod +x /etc/profile.d/10-uv-activate.sh

# Enhanced healthcheck script with CuDNN diagnostics
RUN echo '#!/bin/bash' > /app/healthcheck.sh && \
    echo 'source /app/.venv/bin/activate' >> /app/healthcheck.sh && \
    echo 'echo "=== CuDNN Version Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import torch; print(f\"PyTorch CuDNN: {torch.backends.cudnn.version()}\")" || echo "PyTorch CuDNN check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== JAX Device Check ==="' >> /app/healthcheck.sh && \
    echo 'python -c "import jax; print(f\"JAX devices: {jax.devices()}\")" || echo "JAX device check failed"' >> /app/healthcheck.sh && \
    echo 'echo "=== GPU Validation ==="' >> /app/healthcheck.sh && \
    echo 'python /app/validate_gpu.py --quick' >> /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

WORKDIR /workspace
CMD ["bash", "-l"]
