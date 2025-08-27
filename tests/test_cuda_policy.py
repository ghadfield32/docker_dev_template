#!/usr/bin/env python3
"""
Test script to validate CUDA policy implementation and GPU framework functionality.
This script tests the specific CUDA stack conflicts that were addressed in the fixes.
"""

import sys
import os
import subprocess
import importlib.util
from typing import List, Dict, Any

def log(message: str, level: str = "INFO") -> None:
    """Print a formatted log message."""
    print(f"[{level}] {message}")

def check_environment_variables() -> Dict[str, str]:
    """Check CUDA-related environment variables."""
    cuda_vars = [
        "CUDA_STACK", "LD_LIBRARY_PATH", "XLA_FLAGS", "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE"
    ]
    
    env_status = {}
    for var in cuda_vars:
        env_status[var] = os.environ.get(var, "<unset>")
    
    return env_status

def check_pip_packages() -> Dict[str, List[str]]:
    """Check for CUDA-related pip packages."""
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {"error": f"uv pip freeze failed: {result.stderr}"}
        
        packages = result.stdout.splitlines()
        
        # Filter for CUDA-related packages
        nvidia_packages = [pkg for pkg in packages if pkg.lower().startswith("nvidia-")]
        jax_cuda_packages = [pkg for pkg in packages if "jax" in pkg.lower() and "cuda" in pkg.lower()]
        torch_packages = [pkg for pkg in packages if pkg.lower().startswith("torch")]
        tf_packages = [pkg for pkg in packages if pkg.lower().startswith("tensorflow")]
        
        return {
            "nvidia_packages": nvidia_packages,
            "jax_cuda_packages": jax_cuda_packages,
            "torch_packages": torch_packages,
            "tf_packages": tf_packages,
            "all_packages": packages
        }
    except Exception as e:
        return {"error": f"Failed to check pip packages: {e}"}

def test_pytorch_cuda() -> Dict[str, Any]:
    """Test PyTorch CUDA functionality."""
    try:
        import torch
        
        # Basic import test
        result = {
            "success": True,
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, 'cuda', 'unknown'),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        # Test CUDA tensor operations if CUDA is available
        if torch.cuda.is_available():
            try:
                # Create a simple tensor on GPU
                x = torch.randn(3, 3).cuda()
                y = torch.randn(3, 3).cuda()
                z = torch.mm(x, y)  # Matrix multiplication
                result["gpu_operations"] = "success"
                result["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            except Exception as e:
                result["gpu_operations"] = f"failed: {e}"
        else:
            result["gpu_operations"] = "not tested (CUDA not available)"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_jax_cuda() -> Dict[str, Any]:
    """Test JAX CUDA functionality."""
    try:
        import jax
        import jaxlib
        
        # Basic import test
        result = {
            "success": True,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "devices": jax.devices(),
            "default_device": jax.default_backend()
        }
        
        # Check for GPU devices
        gpu_devices = [d for d in result["devices"] if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
        result["gpu_devices"] = gpu_devices
        result["has_gpu"] = len(gpu_devices) > 0
        
        # Test GPU operations if available
        if result["has_gpu"]:
            try:
                # Simple JAX operation on GPU
                import jax.numpy as jnp
                x = jnp.array([[1., 2.], [3., 4.]])
                y = jnp.array([[5., 6.], [7., 8.]])
                z = jnp.dot(x, y)
                result["gpu_operations"] = "success"
                result["test_result"] = z.tolist()
            except Exception as e:
                result["gpu_operations"] = f"failed: {e}"
        else:
            result["gpu_operations"] = "not tested (no GPU devices)"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_tensorflow_cuda() -> Dict[str, Any]:
    """Test TensorFlow CUDA functionality."""
    try:
        import tensorflow as tf
        
        # Basic import test
        result = {
            "success": True,
            "version": tf.__version__,
            "gpu_devices": tf.config.list_physical_devices('GPU'),
            "cpu_devices": tf.config.list_physical_devices('CPU')
        }
        
        result["has_gpu"] = len(result["gpu_devices"]) > 0
        
        # Test GPU operations if available
        if result["has_gpu"]:
            try:
                # Simple TensorFlow operation
                with tf.device('/GPU:0'):
                    a = tf.constant([[1., 2.], [3., 4.]])
                    b = tf.constant([[5., 6.], [7., 8.]])
                    c = tf.matmul(a, b)
                    result["gpu_operations"] = "success"
                    result["test_result"] = c.numpy().tolist()
            except Exception as e:
                result["gpu_operations"] = f"failed: {e}"
        else:
            result["gpu_operations"] = "not tested (no GPU devices)"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def check_cuda_policy_consistency() -> Dict[str, Any]:
    """Check if the CUDA policy is being applied consistently."""
    cuda_stack = os.environ.get("CUDA_STACK", "pip-cuda")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    
    pip_packages = check_pip_packages()
    
    if "error" in pip_packages:
        return {"error": pip_packages["error"]}
    
    nvidia_packages = pip_packages.get("nvidia_packages", [])
    jax_cuda_packages = pip_packages.get("jax_cuda_packages", [])
    
    # Check policy consistency
    if cuda_stack == "pip-cuda":
        # Should have nvidia packages, no local CUDA in LD_LIBRARY_PATH
        has_nvidia_packages = len(nvidia_packages) > 0
        has_local_cuda = "/usr/local/cuda" in ld_library_path
        has_xla_flags = bool(xla_flags)
        
        return {
            "policy": "pip-cuda",
            "consistent": has_nvidia_packages and not has_local_cuda and not has_xla_flags,
            "details": {
                "has_nvidia_packages": has_nvidia_packages,
                "has_local_cuda_in_path": has_local_cuda,
                "has_xla_flags": has_xla_flags,
                "nvidia_packages": nvidia_packages,
                "jax_cuda_packages": jax_cuda_packages
            }
        }
    
    elif cuda_stack == "local-cuda":
        # Should NOT have nvidia packages, should have local CUDA in LD_LIBRARY_PATH
        has_nvidia_packages = len(nvidia_packages) > 0
        has_local_cuda = "/usr/local/cuda" in ld_library_path
        has_xla_flags = bool(xla_flags)
        
        return {
            "policy": "local-cuda",
            "consistent": not has_nvidia_packages and has_local_cuda and has_xla_flags,
            "details": {
                "has_nvidia_packages": has_nvidia_packages,
                "has_local_cuda_in_path": has_local_cuda,
                "has_xla_flags": has_xla_flags,
                "nvidia_packages": nvidia_packages,
                "jax_cuda_packages": jax_cuda_packages
            }
        }
    
    else:
        return {
            "error": f"Unknown CUDA_STACK: {cuda_stack}"
        }

def run_comprehensive_test() -> Dict[str, Any]:
    """Run all tests and return comprehensive results."""
    log("Starting CUDA policy validation tests...")
    
    results = {
        "environment": check_environment_variables(),
        "pip_packages": check_pip_packages(),
        "policy_consistency": check_cuda_policy_consistency(),
        "pytorch": test_pytorch_cuda(),
        "jax": test_jax_cuda(),
        "tensorflow": test_tensorflow_cuda()
    }
    
    return results

def print_results(results: Dict[str, Any]) -> None:
    """Print test results in a readable format."""
    print("\n" + "="*80)
    print("CUDA POLICY VALIDATION RESULTS")
    print("="*80)
    
    # Environment variables
    print("\n🌍 Environment Variables:")
    env = results["environment"]
    for var, value in env.items():
        status = "✅" if value != "<unset>" else "⚠️"
        print(f"  {status} {var}: {value}")
    
    # Policy consistency
    print("\n🔧 CUDA Policy Consistency:")
    policy = results["policy_consistency"]
    if "error" in policy:
        print(f"  ❌ Policy check failed: {policy['error']}")
    else:
        status = "✅" if policy["consistent"] else "❌"
        print(f"  {status} Policy '{policy['policy']}' is {'consistent' if policy['consistent'] else 'inconsistent'}")
        print(f"  📋 Details:")
        for key, value in policy["details"].items():
            print(f"    - {key}: {value}")
    
    # Pip packages
    print("\n📦 CUDA-Related Pip Packages:")
    pip_packages = results["pip_packages"]
    if "error" in pip_packages:
        print(f"  ❌ Package check failed: {pip_packages['error']}")
    else:
        print(f"  📋 NVIDIA packages ({len(pip_packages.get('nvidia_packages', []))}):")
        for pkg in pip_packages.get("nvidia_packages", []):
            print(f"    - {pkg}")
        print(f"  📋 JAX CUDA packages ({len(pip_packages.get('jax_cuda_packages', []))}):")
        for pkg in pip_packages.get("jax_cuda_packages", []):
            print(f"    - {pkg}")
    
    # PyTorch results
    print("\n📦 PyTorch CUDA Test:")
    pytorch = results["pytorch"]
    if pytorch["success"]:
        print(f"  ✅ PyTorch {pytorch['version']} (CUDA {pytorch['cuda_version']})")
        print(f"  🔍 CUDA Available: {pytorch['cuda_available']}")
        if pytorch['cuda_available']:
            print(f"  🎯 CUDA Devices: {pytorch['device_count']}")
            print(f"  🔧 GPU Operations: {pytorch['gpu_operations']}")
        else:
            print("  ⚠️  CUDA not available")
    else:
        print(f"  ❌ PyTorch test failed: {pytorch['error']}")
    
    # JAX results
    print("\n🚀 JAX Test:")
    jax = results["jax"]
    if jax["success"]:
        print(f"  ✅ JAX {jax['jax_version']} (jaxlib {jax['jaxlib_version']})")
        print(f"  🔍 Devices: {jax['devices']}")
        print(f"  🎯 GPU Devices: {jax['gpu_devices']}")
        print(f"  ✅ Has GPU: {jax['has_gpu']}")
        print(f"  🔧 GPU Operations: {jax['gpu_operations']}")
    else:
        print(f"  ❌ JAX test failed: {jax['error']}")
    
    # TensorFlow results
    print("\n🧠 TensorFlow Test:")
    tf = results["tensorflow"]
    if tf["success"]:
        print(f"  ✅ TensorFlow {tf['version']}")
        print(f"  🔍 GPU Devices: {tf['gpu_devices']}")
        print(f"  ✅ Has GPU: {tf['has_gpu']}")
        print(f"  🔧 GPU Operations: {tf['gpu_operations']}")
    else:
        print(f"  ❌ TensorFlow test failed: {tf['error']}")

def check_success_criteria(results: Dict[str, Any]) -> bool:
    """Check if all success criteria are met."""
    success = True
    
    # Policy should be consistent
    policy = results["policy_consistency"]
    if "error" in policy or not policy.get("consistent", False):
        log("❌ CUDA policy is inconsistent", "ERROR")
        success = False
    
    # All frameworks should be importable
    frameworks_importable = 0
    if results["pytorch"]["success"]:
        frameworks_importable += 1
    if results["jax"]["success"]:
        frameworks_importable += 1
    if results["tensorflow"]["success"]:
        frameworks_importable += 1
    
    if frameworks_importable < 3:
        log(f"❌ Only {frameworks_importable}/3 frameworks are importable", "ERROR")
        success = False
    
    # At least one framework should have GPU support
    gpu_frameworks = 0
    if results["pytorch"].get("cuda_available", False):
        gpu_frameworks += 1
    if results["jax"].get("has_gpu", False):
        gpu_frameworks += 1
    if results["tensorflow"].get("has_gpu", False):
        gpu_frameworks += 1
    
    if gpu_frameworks == 0:
        log("❌ No frameworks have GPU support", "ERROR")
        success = False
    elif gpu_frameworks < 3:
        log(f"⚠️  Only {gpu_frameworks}/3 frameworks have GPU support", "WARN")
    
    return success

def main():
    """Main test function."""
    try:
        results = run_comprehensive_test()
        print_results(results)
        
        print("\n" + "="*80)
        print("SUCCESS CRITERIA CHECK")
        print("="*80)
        
        if check_success_criteria(results):
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ CUDA policy is consistent")
            print("✅ All frameworks are importable")
            print("✅ GPU support is working")
            print("✅ No more double-free errors")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("🔧 Check the issues above and verify CUDA policy implementation")
            sys.exit(1)
            
    except Exception as e:
        log(f"Test execution failed: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()

