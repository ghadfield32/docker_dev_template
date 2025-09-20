#!/usr/bin/env python3
"""
System Commands Debug Script
Diagnoses missing system commands and provides fixes
"""
import os
import sys
import subprocess
from pathlib import Path


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_system_commands() -> dict:
    """Test essential system commands."""
    print_section("SYSTEM COMMANDS DIAGNOSTIC")
    
    essential_commands = [
        'groups', 'uname', 'dircolors', 'whoami', 'id', 'ls', 'cp', 'mv', 'rm',
        'mkdir', 'rmdir', 'chmod', 'chown', 'find', 'grep', 'sed', 'awk'
    ]
    
    results = {}
    
    for cmd in essential_commands:
        try:
            result = subprocess.run(['which', cmd], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {cmd}: {result.stdout.strip()}")
                results[cmd] = True
            else:
                print(f"❌ {cmd}: Not found")
                results[cmd] = False
        except Exception as e:
            print(f"❌ {cmd}: Error - {e}")
            results[cmd] = False
    
    return results


def check_installed_packages() -> dict:
    """Check if essential packages are installed."""
    print_section("PACKAGE INSTALLATION CHECK")
    
    packages_to_check = [
        'coreutils', 'util-linux', 'findutils', 'grep', 'sed', 'gawk'
    ]
    
    results = {}
    
    for package in packages_to_check:
        try:
            result = subprocess.run(['dpkg', '-l', package], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and package in result.stdout:
                print(f"✅ {package}: Installed")
                results[package] = True
            else:
                print(f"❌ {package}: Not installed")
                results[package] = False
        except Exception as e:
            print(f"❌ {package}: Error checking - {e}")
            results[package] = False
    
    return results


def attempt_package_fix() -> bool:
    """Attempt to install missing packages."""
    print_section("ATTEMPTING PACKAGE FIX")
    
    try:
        print("Updating package lists...")
        result = subprocess.run(['apt-get', 'update'], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"❌ apt-get update failed: {result.stderr}")
            return False
        
        print("Installing essential packages...")
        packages = ['coreutils', 'util-linux', 'findutils', 'grep', 'sed', 'gawk']
        result = subprocess.run(['apt-get', 'install', '-y'] + packages, 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Essential packages installed successfully")
            return True
        else:
            print(f"❌ Package installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Package fix failed: {e}")
        return False


def test_environment_variables() -> dict:
    """Test environment variables."""
    print_section("ENVIRONMENT VARIABLES CHECK")
    
    env_vars = {
        'PATH': os.getenv('PATH', ''),
        'VIRTUAL_ENV': os.getenv('VIRTUAL_ENV', ''),
        'UV_PROJECT_ENVIRONMENT': os.getenv('UV_PROJECT_ENVIRONMENT', ''),
        'PYTHONPATH': os.getenv('PYTHONPATH', ''),
        'HOME': os.getenv('HOME', ''),
        'USER': os.getenv('USER', ''),
        'SHELL': os.getenv('SHELL', '')
    }
    
    results = {}
    
    for var, value in env_vars.items():
        if value:
            print(f"✅ {var}: {value}")
            results[var] = True
        else:
            print(f"❌ {var}: Not set")
            results[var] = False
    
    return results


def test_python_environment() -> dict:
    """Test Python environment."""
    print_section("PYTHON ENVIRONMENT CHECK")
    
    results = {}
    
    # Test Python executable
    try:
        python_path = sys.executable
        print(f"✅ Python executable: {python_path}")
        results['python_executable'] = True
        
        # Check if in virtual environment
        venv = os.getenv('VIRTUAL_ENV')
        if venv:
            print(f"✅ Virtual environment: {venv}")
            results['virtual_env'] = True
        else:
            print("❌ Virtual environment not set")
            results['virtual_env'] = False
            
    except Exception as e:
        print(f"❌ Python environment error: {e}")
        results['python_executable'] = False
        results['virtual_env'] = False
    
    # Test UV
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ UV: {result.stdout.strip()}")
            results['uv'] = True
        else:
            print(f"❌ UV error: {result.stderr}")
            results['uv'] = False
    except Exception as e:
        print(f"❌ UV test failed: {e}")
        results['uv'] = False
    
    return results


def generate_diagnostic_report(all_results: dict) -> str:
    """Generate comprehensive diagnostic report."""
    print_section("DIAGNOSTIC REPORT")
    
    total_commands = len(all_results.get('commands', {}))
    available_commands = sum(1 for v in all_results.get('commands', {}).values() if v)
    
    total_packages = len(all_results.get('packages', {}))
    installed_packages = sum(1 for v in all_results.get('packages', {}).values() if v)
    
    total_env_vars = len(all_results.get('env_vars', {}))
    set_env_vars = sum(1 for v in all_results.get('env_vars', {}).values() if v)
    
    print(f"System Commands: {available_commands}/{total_commands} available")
    print(f"Essential Packages: {installed_packages}/{total_packages} installed")
    print(f"Environment Variables: {set_env_vars}/{total_env_vars} set")
    
    # Recommendations
    print("\nRecommendations:")
    if available_commands < total_commands * 0.8:
        print("🔧 CRITICAL: Install missing system packages (coreutils, util-linux)")
    
    if installed_packages < total_packages:
        print("🔧 Install missing essential packages")
    
    if not all_results.get('python', {}).get('virtual_env', False):
        print("🔧 Fix virtual environment activation")
    
    return f"Commands: {available_commands}/{total_commands}, Packages: {installed_packages}/{total_packages}"


def main() -> int:
    """Main diagnostic function."""
    print("System Commands Debug Script")
    print(f"Working directory: {os.getcwd()}")
    
    all_results = {}
    
    try:
        all_results['commands'] = test_system_commands()
        all_results['packages'] = check_installed_packages()
        all_results['env_vars'] = test_environment_variables()
        all_results['python'] = test_python_environment()
        
        # Attempt fix if needed
        if sum(1 for v in all_results['commands'].values() if v) < len(all_results['commands']) * 0.8:
            print("\n⚠️ Many commands missing, attempting fix...")
            fix_success = attempt_package_fix()
            if fix_success:
                print("\n🔄 Re-testing after fix...")
                all_results['commands_after_fix'] = test_system_commands()
        
        # Generate report
        report = generate_diagnostic_report(all_results)
        
        # Determine exit code
        command_success_rate = sum(1 for v in all_results['commands'].values() if v) / len(all_results['commands'])
        return 0 if command_success_rate > 0.8 else 1
        
    except Exception as e:
        print(f"Diagnostic script failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

