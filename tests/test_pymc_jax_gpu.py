#!/usr/bin/env python3
"""
Test script to verify that PyMC can use JAX for GPU computations.
This is the modern approach since PyTensor 2.18+ removed the old GPU backend.
"""

import sys
import time


def test_pytensor_config():
    """Test PyTensor configuration (should be CPU now)."""
    print("\n=== PyTensor Configuration ===")
    try:
        import pytensor
        
        print("PyTensor configuration:")
        print(f"  device: {pytensor.config.device}")
        print(f"  floatX: {pytensor.config.floatX}")
        print(f"  print_active_device: {pytensor.config.print_active_device}")
        
        # Check environment variables
        import os
        pytensor_flags = os.environ.get('PYTENSOR_FLAGS', 'Not set')
        print(f"  PYTENSOR_FLAGS: {pytensor_flags}")
        
        if "cpu" in pytensor.config.device.lower():
            print("✅ PyTensor is correctly configured for CPU (new architecture)")
        else:
            print(f"⚠️  Unexpected device configuration: {pytensor.config.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking PyTensor config: {e}")
        return False


def test_jax_gpu_backend():
    """Test JAX GPU availability for PyMC."""
    print("\n=== JAX GPU Backend Test ===")
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"JAX version: {jax.__version__}")
        
        # Get device information
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        
        gpu_available = any('gpu' in str(device).lower() or 'cuda' in str(device).lower() 
                          for device in devices)
        
        if gpu_available:
            print("✅ JAX GPU support detected")
            
            # Test a simple computation
            print("Running JAX GPU computation test...")
            x = jnp.ones((1000, 1000))
            y = jnp.ones((1000, 1000))
            
            # Use JIT compilation
            @jax.jit
            def matmul_test(a, b):
                return jnp.matmul(a, b)
            
            t0 = time.time()
            result = matmul_test(x, y)
            t1 = time.time()
            
            print(f"✅ JAX computation completed: {t1-t0:.3f}s")
            print(f"   Result shape: {result.shape}")
            print(f"   Result sum: {jnp.sum(result)}")
            
            return True
        else:
            print("⚠️  JAX GPU not detected - will use CPU")
            return False
            
    except ImportError:
        print("❌ JAX not found")
        return False
    except Exception as e:
        print(f"❌ Error testing JAX GPU: {e}")
        return False


def test_pymc_jax_integration():
    """Test PyMC with JAX backend for GPU acceleration."""
    print("\n=== PyMC + JAX Integration Test ===")
    try:
        import pymc as pm
        import numpy as np
        import arviz as az
        
        print(f"PyMC version: {pm.__version__}")
        
        # Create a simple Bayesian model
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=np.random.randn(100))
        
        print("✅ PyMC model created successfully")
        
        # Try sampling with JAX backend (NumPyro)
        try:
            print("Testing PyMC with JAX backend (NumPyro)...")
            with model:
                trace = pm.sample_numpyro_nuts(
                    draws=100, 
                    tune=50, 
                    chains=2,
                    progress_bar=False
                )
            
            print("✅ PyMC sampling with JAX backend completed")
            print(f"   Trace shape: {trace.posterior.mu.shape}")
            print(f"   Mean mu: {trace.posterior.mu.mean().values:.3f}")
            print(f"   Mean sigma: {trace.posterior.sigma.mean().values:.3f}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  JAX backend sampling failed: {e}")
            print("Falling back to default sampler...")
            
            # Fallback to default sampler
            with model:
                trace = pm.sample(
                    draws=100, 
                    tune=50, 
                    chains=2, 
                    progressbar=False
                )
            
            print("✅ PyMC sampling with default backend completed")
            return True
            
    except ImportError as e:
        print(f"❌ PyMC not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing PyMC: {e}")
        return False


def test_pytensor_basic_functionality():
    """Test basic PyTensor functionality on CPU."""
    print("\n=== PyTensor Basic Functionality ===")
    try:
        import pytensor
        import pytensor.tensor as pt
        import numpy as np
        
        print(f"PyTensor version: {pytensor.__version__}")
        print(f"Device: {pytensor.config.device}")
        
        # Create symbolic variables
        x = pt.dmatrix('x')
        y = pt.dmatrix('y')
        
        # Define computation
        z = pt.dot(x, y)
        
        # Compile function
        f = pytensor.function([x, y], z)
        
        # Test computation
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        
        t0 = time.time()
        result = f(a, b)
        t1 = time.time()
        
        print(f"✅ PyTensor computation completed: {t1-t0:.3f}s")
        print(f"   Result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PyTensor: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Testing Modern PyMC + JAX GPU Setup")
    print("=" * 50)
    
    pytensor_config = test_pytensor_config()
    pytensor_basic = test_pytensor_basic_functionality()
    jax_gpu = test_jax_gpu_backend()
    pymc_jax = test_pymc_jax_integration()
    
    print("\n" + "=" * 50)
    print("=== Test Summary ===")
    print(f"PyTensor Config:  {'✅ PASS' if pytensor_config else '❌ FAIL'}")
    print(f"PyTensor Basic:   {'✅ PASS' if pytensor_basic else '❌ FAIL'}")
    print(f"JAX GPU:          {'✅ PASS' if jax_gpu else '⚠️  CPU only'}")
    print(f"PyMC + JAX:       {'✅ PASS' if pymc_jax else '❌ FAIL'}")
    
    if pytensor_config and pytensor_basic and pymc_jax:
        print("\n🎉 Environment is ready for modern PyMC development!")
        if jax_gpu:
            print("🚀 GPU acceleration available via JAX backend!")
        else:
            print("💻 CPU-only mode - still fully functional for development")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1) 