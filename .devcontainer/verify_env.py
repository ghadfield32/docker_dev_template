#!/usr/bin/env python3
import encodings, jupyterlab, torch, jax, sys, os, subprocess

print("## Python & library diagnostics ##")
print("Python:", sys.executable, sys.version.split()[0])
print("🟢 encodings OK")
print("🟢 jupyterlab OK")
print("🟢 torch", torch.__version__, "CUDA:", torch.cuda.is_available())
print("🟢 jax", jax.__version__, "devices:", jax.devices())

# Check Railway CLI
try:
    railway_version = subprocess.check_output(["railway", "--version"], stderr=subprocess.STDOUT).decode().strip()
    print("🛤️  railway", railway_version)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("⚠️  railway CLI not found")
