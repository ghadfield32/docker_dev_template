#!/usr/bin/env python3
"""
Fixed Kernel Registration Script
This addresses the core issue with Jupyter kernels not being found locally
"""

import sys
import os
import subprocess
import json
from pathlib import Path


def register_kernels_properly():
    """
    Register kernels in both local and container contexts properly.
    This fixes the path mismatch issue you're experiencing.
    """
    # Detect environment
    in_container = Path('/.dockerenv').exists()
    venv_path = Path(sys.executable).parent.parent
    
    # Get environment name from environment variable
    env_name = os.environ.get('ENV_NAME', 'docker_dev_template')
    
    print(f"Environment: {'Container' if in_container else 'Local'}")
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {venv_path}")
    print(f"Environment name: {env_name}")
    
    # Base kernel configuration using ENV_NAME
    kernel_name = env_name
    display_name = f"Python ({env_name})"
    
    if in_container:
        # Container registration - register both user and system
        register_container_kernels(kernel_name, display_name, sys.executable)
    else:
        # Local registration - ensure it points to correct local venv
        register_local_kernels(kernel_name, display_name, sys.executable)


def register_container_kernels(kernel_name, display_name, python_path):
    """Register kernels in container for both user and system access"""
    
    # User kernel (accessible when mounting home directory)
    try:
        subprocess.check_call([
            python_path, '-m', 'ipykernel', 'install',
            '--user', '--name', kernel_name, '--display-name', display_name
        ])
        print(f"User kernel registered: {display_name}")
    except subprocess.CalledProcessError as e:
        print(f"User kernel registration failed: {e}")
    
    # System kernel (for container-wide access)
    try:
        subprocess.check_call([
            python_path, '-m', 'ipykernel', 'install',
            '--sys-prefix', '--name', f"{kernel_name}_sys", 
            '--display-name', f"{display_name} (System)"
        ])
        print(f"System kernel registered")
    except subprocess.CalledProcessError as e:
        print(f"System kernel registration failed: {e}")


def register_local_kernels(kernel_name, display_name, python_path):
    """Register kernels locally with proper path mapping"""
    
    # For local development, register user kernel that points to local venv
    try:
        subprocess.check_call([
            python_path, '-m', 'ipykernel', 'install',
            '--user', '--name', kernel_name, '--display-name', display_name
        ])
        print(f"Local kernel registered: {display_name}")
        
        # Verify the kernel spec points to correct Python
        import jupyter_core.paths as jp
        kernel_spec_dir = Path(jp.jupyter_data_dir()) / 'kernels' / kernel_name
        kernel_spec_file = kernel_spec_dir / 'kernel.json'
        
        if kernel_spec_file.exists():
            with open(kernel_spec_file) as f:
                spec = json.load(f)
            print(f"Kernel spec python path: {spec['argv'][0]}")
            
            # Validate the Python path is correct
            if Path(spec['argv'][0]).exists():
                print("Kernel Python path is valid")
            else:
                print(f"WARNING: Kernel Python path does not exist: {spec['argv'][0]}")
                
        else:
            print("WARNING: Kernel spec file not found after registration")
            
    except subprocess.CalledProcessError as e:
        print(f"Local kernel registration failed: {e}")


def create_workspace_kernel_script():
    """
    Create a script that can be run from workspace root to register kernels properly.
    This addresses your specific issue with cd .devcontainer; uv sync
    """
    
    # Get environment name
    env_name = os.environ.get('ENV_NAME', 'docker_dev_template')
    
    script_content = f'''#!/usr/bin/env bash
# Run this from workspace root to register kernels properly after uv sync

set -euo pipefail

echo "Registering Jupyter kernels for workspace..."

# Determine if we have a local .venv or need to use container env
if [ -f "pyproject.toml" ] && [ -d ".venv" ]; then
    echo "Using local .venv in workspace root"
    PYTHON_PATH="$(pwd)/.venv/bin/python"
elif [ -f ".devcontainer/pyproject.toml" ] && [ -d ".devcontainer/.venv" ]; then
    echo "Using .devcontainer/.venv"
    PYTHON_PATH="$(pwd)/.devcontainer/.venv/bin/python"
else
    echo "No virtual environment found. Run 'uv sync' first."
    exit 1
fi

if [ ! -f "$PYTHON_PATH" ]; then
    echo "Python not found at $PYTHON_PATH"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"

# Register the kernel with environment name
"$PYTHON_PATH" -m ipykernel install --user --name "{env_name}" --display-name "Python ({env_name})"

echo "Kernel registered successfully!"
echo "You can now select this kernel in Jupyter Lab/Notebook"
'''
    
    script_path = Path('register_kernels.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    print(f"Created {script_path} - run this from workspace root after uv sync")


def diagnose_current_setup():
    """Diagnose the current kernel setup to identify issues"""
    print("\n=== KERNEL DIAGNOSIS ===")
    
    # Check available kernels
    try:
        result = subprocess.run([
            sys.executable, '-m', 'jupyter', 'kernelspec', 'list'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Available kernels:")
            print(result.stdout)
        else:
            print(f"Error listing kernels: {result.stderr}")
    except Exception as e:
        print(f"Could not list kernels: {e}")
    
    # Check if ipykernel is installed
    try:
        import ipykernel
        print(f"ipykernel version: {ipykernel.__version__}")
    except ImportError:
        print("ipykernel not installed - this is the problem!")
        return False
    
    # Check jupyter installation
    try:
        import jupyterlab
        print(f"jupyterlab version: {jupyterlab.__version__}")
    except ImportError:
        print("jupyterlab not installed")
        return False
    
    return True


if __name__ == "__main__":
    print("Jupyter Kernel Registration Fix")
    print("=" * 40)
    
    # First diagnose current state
    setup_ok = diagnose_current_setup()
    
    if not setup_ok:
        print("Please install missing packages first:")
        print("uv pip install jupyterlab ipykernel")
        sys.exit(1)
    
    # Register kernels appropriately
    register_kernels_properly()
    
    # Create helper script for future use
    create_workspace_kernel_script()
    
    print("\nNext steps:")
    print("1. If running locally: Use ./register_kernels.sh after uv sync")
    print("2. If in container: Kernels should now be available")
    print("3. Restart Jupyter Lab to see the new kernels")
