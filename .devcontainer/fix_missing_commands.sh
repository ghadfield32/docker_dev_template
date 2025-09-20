#!/bin/bash
"""
Fix Missing Commands Script
Comprehensive fix for missing system commands in Docker container
"""
set -e

echo "=========================================="
echo "  FIXING MISSING SYSTEM COMMANDS"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install missing packages
install_missing_packages() {
    echo "🔧 Installing missing system packages..."
    
    # Update package lists
    echo "📦 Updating package lists..."
    apt-get update
    
    # Install essential packages
    echo "📦 Installing essential packages..."
    apt-get install -y \
        coreutils \
        util-linux \
        findutils \
        grep \
        sed \
        gawk \
        procps \
        psmisc
    
    echo "✅ Package installation completed"
}

# Function to verify commands
verify_commands() {
    echo "🔍 Verifying essential commands..."
    
    commands=("groups" "uname" "dircolors" "whoami" "id" "ls" "cp" "mv" "rm" "mkdir")
    all_good=true
    
    for cmd in "${commands[@]}"; do
        if command_exists "$cmd"; then
            echo "✅ $cmd: $(which $cmd)"
        else
            echo "❌ $cmd: MISSING"
            all_good=false
        fi
    done
    
    if [ "$all_good" = true ]; then
        echo "🎉 All essential commands are available!"
        return 0
    else
        echo "⚠️ Some commands are still missing"
        return 1
    fi
}

# Function to test system functionality
test_system_functionality() {
    echo "🧪 Testing system functionality..."
    
    # Test basic commands
    echo "Testing uname..."
    uname -a
    
    echo "Testing groups..."
    groups
    
    echo "Testing whoami..."
    whoami
    
    echo "Testing dircolors..."
    dircolors --help >/dev/null 2>&1 || echo "dircolors help not available (but command exists)"
    
    echo "✅ System functionality tests completed"
}

# Main execution
main() {
    echo "Starting system command fix process..."
    
    # Check current state
    echo "📊 Current system state:"
    verify_commands || true
    
    # Install missing packages if needed
    if ! command_exists "groups" || ! command_exists "uname" || ! command_exists "dircolors"; then
        echo "⚠️ Missing essential commands detected, installing packages..."
        install_missing_packages
    else
        echo "✅ All essential commands already available"
    fi
    
    # Verify after installation
    echo "📊 System state after fix:"
    if verify_commands; then
        echo "🎉 SUCCESS: All essential commands are now available!"
        
        # Test functionality
        test_system_functionality
        
        echo ""
        echo "=========================================="
        echo "  SYSTEM COMMANDS FIX COMPLETED"
        echo "=========================================="
        echo "✅ groups: $(which groups)"
        echo "✅ uname: $(which uname)"
        echo "✅ dircolors: $(which dircolors)"
        echo "✅ whoami: $(which whoami)"
        echo "✅ id: $(which id)"
        echo ""
        echo "Your container should now work properly!"
        
        return 0
    else
        echo "❌ FAILED: Some commands are still missing after installation"
        echo "This may indicate a deeper system issue."
        return 1
    fi
}

# Run main function
main "$@"

