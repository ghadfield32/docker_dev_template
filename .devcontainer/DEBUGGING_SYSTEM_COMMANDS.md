# System Commands Debugging Guide

This guide provides comprehensive solutions for fixing missing system commands in your Docker container.

## Problem Analysis

### **Root Cause**
The error messages indicate missing essential Unix commands:
```bash
bash: groups: command not found
bash: dircolors: command not found  
bash: uname: command not found
```

These commands are part of the `coreutils` package in Ubuntu and are essential for:
- Shell startup scripts
- Environment setup
- Basic system operations

### **Why This Happens**
1. **Incomplete Base Image**: CUDA base image may be missing essential packages
2. **Package Installation Failure**: `coreutils` package installation failed silently
3. **Docker Cache Issues**: Corrupted build cache preventing proper installation
4. **Minimal Installation**: Using `--no-install-recommends` without including essential packages

## Solutions

### **Solution 1: Enhanced Dockerfile (Recommended)**

The Dockerfile has been updated with:
1. **Explicit Package Installation**: `coreutils`, `util-linux`, `findutils`
2. **Command Verification**: Build fails if essential commands are missing
3. **Better Error Handling**: Clear error messages when packages fail to install

### **Solution 2: Runtime Fix Script**

Use the `fix_missing_commands.sh` script to fix issues in running containers:

```bash
# Run the fix script in your container
docker exec -it your_container_name /app/fix_missing_commands.sh
```

### **Solution 3: Manual Package Installation**

If the automated fix doesn't work:

```bash
# Connect to your container
docker exec -it your_container_name bash

# Install missing packages manually
apt-get update
apt-get install -y coreutils util-linux findutils grep sed gawk

# Verify installation
groups && uname -a && dircolors --help
```

## Debugging Tools

### **1. System Commands Diagnostic Script**

Run the diagnostic script to identify issues:

```bash
python /app/debug_system_commands.py
```

This script will:
- Test all essential system commands
- Check package installation status
- Verify environment variables
- Test Python environment
- Attempt automatic fixes
- Generate comprehensive reports

### **2. Command Verification**

Test essential commands manually:

```bash
# Test basic commands
which groups uname dircolors whoami id ls cp mv rm mkdir

# Test command functionality
groups
uname -a
whoami
id
```

### **3. Package Status Check**

Check if packages are installed:

```bash
# Check package installation
dpkg -l | grep -E "(coreutils|util-linux|findutils)"

# Check package files
dpkg -L coreutils | head -10
```

## Prevention

### **1. Enhanced Dockerfile Verification**

The updated Dockerfile includes:
```dockerfile
# Verify essential commands during build
for cmd in groups uname dircolors whoami id ls cp mv rm mkdir; do \
    if command -v $cmd >/dev/null 2>&1; then \
        echo "✅ $cmd: $(which $cmd)"; \
    else \
        echo "❌ CRITICAL ERROR: $cmd command missing"; \
        exit 1; \
    fi; \
done
```

### **2. Build Cache Management**

Clear Docker build cache if issues persist:

```bash
# Clear all build cache
docker builder prune --all --force

# Rebuild without cache
docker-compose build --no-cache
```

### **3. Base Image Verification**

Ensure you're using a complete base image:

```dockerfile
# Use full CUDA development image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
```

## Testing and Validation

### **1. Build-Time Testing**

The Dockerfile now includes comprehensive testing:
- Package installation verification
- Command availability checks
- Build failure on missing commands

### **2. Runtime Testing**

Use the diagnostic script to test running containers:

```bash
# Full system test
python /app/debug_system_commands.py

# Quick command test
/app/fix_missing_commands.sh
```

### **3. Integration Testing**

Test the complete environment:

```bash
# Test shell activation
source /app/activate_uv.sh

# Test Python environment
python --version
uv --version

# Test system commands
groups && uname -a && whoami
```

## Troubleshooting Workflow

### **Step 1: Identify the Issue**
```bash
python /app/debug_system_commands.py
```

### **Step 2: Apply Fix**
```bash
/app/fix_missing_commands.sh
```

### **Step 3: Verify Fix**
```bash
# Test essential commands
groups && uname -a && dircolors --help

# Test environment
source /app/activate_uv.sh
```

### **Step 4: Rebuild if Necessary**
```bash
docker builder prune --all --force
docker-compose build --no-cache
docker-compose up -d
```

## Expected Results

After applying the fixes, you should see:

```bash
✅ groups: /usr/bin/groups
✅ uname: /bin/uname
✅ dircolors: /usr/bin/dircolors
✅ whoami: /usr/bin/whoami
✅ id: /usr/bin/id

🐍 UV Environment activated: Python 3.10.12
🖥️  System: Linux 5.15.0-91-generic x86_64
👤 User: root (Groups: root)
📁 Working directory: /workspace
```

## Common Issues and Solutions

### **Issue: Commands Still Missing After Fix**
**Solution**: Rebuild container with `--no-cache` flag

### **Issue: Package Installation Fails**
**Solution**: Check network connectivity and package repositories

### **Issue: Permission Denied**
**Solution**: Ensure running as root or with proper permissions

### **Issue: Base Image Issues**
**Solution**: Use complete CUDA development image instead of runtime image

## Best Practices

1. **Always verify package installation** during Docker build
2. **Use explicit package names** instead of relying on dependencies
3. **Test essential commands** before proceeding with application setup
4. **Clear build cache** when experiencing persistent issues
5. **Use diagnostic scripts** to identify problems early

The enhanced debugging system provides comprehensive tools to identify, fix, and prevent missing system command issues in your Docker container environment.

