#!/bin/bash
set -euo pipefail

echo "🔧 Testing GPU frameworks from host..."

# Get container ID
CONTAINER_ID=$(docker compose -p nba_player_valuation ps -q datascience)
if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Container not found. Make sure the container is running."
    exit 1
fi

echo "📦 Container ID: $CONTAINER_ID"

# Test nvidia-smi
echo ""
echo "🔍 Testing nvidia-smi..."
docker exec "$CONTAINER_ID" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || {
    echo "❌ nvidia-smi failed - GPU may not be accessible"
}

# Test PyTorch
echo ""
echo "🔍 Testing PyTorch..."
docker exec "$CONTAINER_ID" python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    
    # Test basic computation
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    z = torch.mm(x, y)
    print(f'Computation test: {z.shape}')
    print('✅ PyTorch GPU test passed')
else:
    print('❌ PyTorch CUDA not available')
"

# Test JAX
echo ""
echo "🔍 Testing JAX..."
docker exec "$CONTAINER_ID" python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices()
print(f'JAX devices: {devices}')

gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print(f'GPU devices found: {len(gpu_devices)}')
    
    # Test basic computation
    import jax.numpy as jnp
    x = jnp.ones((100, 100))
    y = jnp.ones((100, 100))
    z = jnp.dot(x, y)
    print(f'Computation test: {z.shape}')
    print('✅ JAX GPU test passed')
else:
    print('❌ JAX GPU devices not found')
"

# Test TensorFlow
echo ""
echo "🔍 Testing TensorFlow..."
docker exec "$CONTAINER_ID" python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow GPUs: {gpus}')

if gpus:
    print(f'GPU devices found: {len(gpus)}')
    
    # Configure memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Test basic computation
    x = tf.ones((100, 100))
    y = tf.ones((100, 100))
    z = tf.matmul(x, y)
    print(f'Computation test: {z.shape}')
    print('✅ TensorFlow GPU test passed')
else:
    print('❌ TensorFlow GPU devices not found')
"

# Test ml-dtypes version
echo ""
echo "🔍 Testing ml-dtypes..."
docker exec "$CONTAINER_ID" python -c "
import ml_dtypes
print(f'ml-dtypes version: {ml_dtypes.__version__}')

# Check for float8_e3m4 attribute (requires ml-dtypes 0.5.3+)
try:
    from ml_dtypes import float8_e3m4
    print('✅ float8_e3m4 available')
except ImportError:
    print('❌ float8_e3m4 not available - ml-dtypes version too old')
"

# Environment variables snapshot
echo ""
echo "🔍 Environment variables snapshot..."
docker exec "$CONTAINER_ID" bash -c "
echo 'LD_PRELOAD: $LD_PRELOAD'
echo 'LD_LIBRARY_PATH: $LD_LIBRARY_PATH'
echo 'MALLOC_ARENA_MAX: $MALLOC_ARENA_MAX'
echo 'PYTORCH_NO_CUDA_MEMORY_CACHING: $PYTORCH_NO_CUDA_MEMORY_CACHING'
echo 'XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE'
echo 'TF_FORCE_GPU_ALLOW_GROWTH: $TF_FORCE_GPU_ALLOW_GROWTH'
"

echo ""
echo "✅ GPU testing complete!"

