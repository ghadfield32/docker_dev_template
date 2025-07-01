#!/usr/bin/env python3
"""
Test script to verify that PyTensor can access the GPU and perform GPU computations.
"""

import sys
import time


def test_pytensor_gpu():
    """Test PyTensor GPU availability and basic operations."""
    print("\n=== Testing PyTensor GPU ===")
    try:
        import pytensor
        import pytensor.tensor as pt
        import numpy as np
        
        print(f"PyTensor version: {pytensor.__version__}")
        print(f"Active device: {pytensor.config.device}")
        print(f"Float type: {pytensor.config.floatX}")
        
        # Check if GPU device is configured
        if 'cuda' not in pytensor.config.device.lower():
            print("❌ PyTensor is not configured to use CUDA!")
            print(f"   Current device: {pytensor.config.device}")
            print("   Expected: cuda0 or similar")
            return False
        
        print("✅ PyTensor is configured for CUDA")
        
        # Test a simple computation on GPU
        print("Running GPU computation test...")
        
        # Create symbolic variables
        x = pt.matrix("x", dtype="float32")
        y = pt.matrix("y", dtype="float32")
        
        # Define computation: matrix multiplication + addition
        result = pt.dot(x, y) + 1.0
        
        # Compile function (this should create CUDA kernels)
        print("Compiling PyTensor function...")
        f = pytensor.function([x, y], result)
        print("✅ Function compiled successfully")
        
        # Prepare test data
        size = 2000
        a = np.random.randn(size, size).astype("float32")
        b = np.random.randn(size, size).astype("float32")
        
        # First call (includes kernel compilation time)
        print(f"Running matrix multiplication ({size}x{size})...")
        t0 = time.time()
        output1 = f(a, b)
        t1 = time.time()
        print(f"✅ First call completed: {t1-t0:.3f}s (includes compilation)")
        
        # Second call (should be faster, using cached kernels)
        t0 = time.time()
        output2 = f(a, b)
        t1 = time.time()
        print(f"✅ Second call completed: {t1-t0:.3f}s (cached)")
        
        # Verify outputs are consistent
        if np.allclose(output1, output2):
            print("✅ Results are consistent between calls")
        else:
            print("❌ Results differ between calls - potential issue")
            return False
        
        # Check result shape and basic properties
        expected_shape = (size, size)
        if output1.shape == expected_shape:
            print(f"✅ Output shape correct: {output1.shape}")
        else:
            print(f"❌ Wrong output shape. Expected {expected_shape}, got {output1.shape}")
            return False
        
        # Performance check - GPU should be fast for large matrices
        if t1-t0 < 1.0:  # Should complete in less than 1 second for 2000x2000
            print(f"✅ GPU performance looks good: {t1-t0:.3f}s for {size}x{size} matrices")
        else:
            print(f"⚠️  Performance seems slow: {t1-t0:.3f}s - might be running on CPU")
        
        print("✅ PyTensor GPU test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ PyTensor not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during PyTensor GPU test: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_pytensor_config():
    """Test PyTensor configuration and environment."""
    print("\n=== PyTensor Configuration ===")
    try:
        import pytensor
        
        print("PyTensor configuration:")
        print(f"  device: {pytensor.config.device}")
        print(f"  floatX: {pytensor.config.floatX}")
        print(f"  force_device: {pytensor.config.force_device}")
        print(f"  print_active_device: {pytensor.config.print_active_device}")
        
        # Check environment variables
        import os
        pytensor_flags = os.environ.get('PYTENSOR_FLAGS', 'Not set')
        print(f"  PYTENSOR_FLAGS: {pytensor_flags}")
        
        cuda_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
        print(f"  LD_LIBRARY_PATH: {cuda_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking PyTensor config: {e}")
        return False


def test_gpu_libraries():
    """Test that required GPU libraries are available."""
    print("\n=== GPU Libraries Check ===")
    
    success = True
    
    # Test pygpu
    try:
        import pygpu
        print(f"✅ pygpu version: {pygpu.__version__}")
    except ImportError:
        print("❌ pygpu not found - required for PyTensor GPU support")
        success = False
    except Exception as e:
        print(f"❌ Error importing pygpu: {e}")
        success = False
    
    # Test that we can create a GPU context
    try:
        import pygpu
        # Try to create a CUDA context
        ctx = pygpu.init('cuda0')
        print("✅ CUDA context created successfully")
        print(f"   Device name: {ctx.devname}")
    except Exception as e:
        print(f"❌ Could not create CUDA context: {e}")
        success = False
    
    return success


if __name__ == "__main__":
    print("🔍 Testing PyTensor GPU Support")
    print("=" * 50)
    
    config_success = test_pytensor_config()
    libs_success = test_gpu_libraries()
    pytensor_success = test_pytensor_gpu()
    
    print("\n" + "=" * 50)
    print("=== Test Summary ===")
    print(f"Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"GPU Libraries: {'✅ PASS' if libs_success else '❌ FAIL'}")
    print(f"PyTensor GPU:  {'✅ PASS' if pytensor_success else '❌ FAIL'}")
    
    if config_success and libs_success and pytensor_success:
        print("\n🎉 All PyTensor GPU tests passed!")
        print("PyTensor is ready for GPU-accelerated computations!")
        sys.exit(0)
    else:
        print("\n❌ Some PyTensor GPU tests failed.")
        print("Check the output above for specific issues.")
        sys.exit(1) 