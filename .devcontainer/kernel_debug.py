#!/usr/bin/env python3
"""
Kernel Registration Debug Script
Run this both locally and in container to diagnose path issues
"""

import sys
import os
import json
from pathlib import Path
import subprocess

def debug_jupyter_paths():
    """Debug Jupyter kernel and data paths"""
    print("=== JUPYTER PATH DEBUG ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version_info}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    
    # Check if we're in container or local
    in_container = Path('/.dockerenv').exists()
    print(f"Running in container: {in_container}")
    
    # Get Jupyter paths
    try:
        import jupyter_core.paths as jp
        print(f"Jupyter config dir: {jp.jupyter_config_dir()}")
        print(f"Jupyter data dir: {jp.jupyter_data_dir()}")
        print(f"Jupyter runtime dir: {jp.jupyter_runtime_dir()}")
        
        # Check kernel specs
        kernel_dirs = jp.jupyter_path('kernels')
        print(f"Kernel search paths: {kernel_dirs}")
        
        for kernel_dir in kernel_dirs:
            kernel_path = Path(kernel_dir)
            if kernel_path.exists():
                kernels = list(kernel_path.glob('*/'))
                print(f"  {kernel_path}: {[k.name for k in kernels]}")
            else:
                print(f"  {kernel_path}: (does not exist)")
                
    except ImportError:
        print("jupyter_core not available")
    
    # Check ipykernel installation
    try:
        import ipykernel
        print(f"ipykernel version: {ipykernel.__version__}")
        print(f"ipykernel location: {ipykernel.__file__}")
    except ImportError:
        print("ipykernel not installed")

def debug_uv_environment():
    """Debug uv virtual environment setup"""
    print("\n=== UV ENVIRONMENT DEBUG ===")
    
    # Check UV configuration
    uv_env_vars = [
        'UV_PROJECT_ENVIRONMENT',
        'UV_VENV', 
        'UV_CACHE_DIR',
        'VIRTUAL_ENV'
    ]
    
    for var in uv_env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check for pyproject.toml locations
    locations_to_check = [
        Path.cwd(),
        Path.cwd() / '.devcontainer',
        Path('/workspace'),
        Path('/workspace/.devcontainer'),
        Path('/app')
    ]
    
    print(f"\nPyproject.toml locations:")
    for loc in locations_to_check:
        pyproject = loc / 'pyproject.toml'
        if pyproject.exists():
            print(f"  ✓ {pyproject}")
        else:
            print(f"  ✗ {pyproject}")
    
    # Check virtual environments
    venv_locations = [
        Path('.venv'),
        Path('.devcontainer/.venv'),
        Path('/app/.venv')
    ]
    
    print(f"\nVirtual environment locations:")
    for venv in venv_locations:
        if venv.exists():
            python_exe = venv / 'bin/python'
            print(f"  ✓ {venv} (python: {python_exe.exists()})")
        else:
            print(f"  ✗ {venv}")

def suggest_fixes():
    """Provide specific fix recommendations"""
    print("\n=== RECOMMENDED FIXES ===")
    
    in_container = Path('/.dockerenv').exists()
    
    if in_container:
        print("Running in container - checking container-specific issues:")
        print("1. Ensure kernel registration uses correct paths")
        print("2. Verify UV_PROJECT_ENVIRONMENT points to /app/.venv")
        print("3. Check that ipykernel install uses --sys-prefix or --user appropriately")
    else:
        print("Running locally - checking local setup issues:")
        print("1. Local uv sync should create .venv in project root, not .devcontainer")
        print("2. Jupyter should find kernel in local .venv/share/jupyter/kernels")
        print("3. Consider using 'uv run jupyter' to ensure correct environment")
        
    print("\nGeneral recommendations:")
    print("- Use consistent venv location (.venv in project root)")
    print("- Register kernel with correct display name and Python path")
    print("- Ensure local Jupyter can access container-registered kernels")

if __name__ == "__main__":
    debug_jupyter_paths()
    debug_uv_environment()
    suggest_fixes()
