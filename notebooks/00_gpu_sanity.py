# %% [markdown]
# # GPU Sanity Checks
# Run this on both machines. It prints driver info, Torch/JAX builds, and runs tiny GPU ops.

# %%
import subprocess, sys, os, json, textwrap

def run(cmd):
    print(f"\n$ {cmd}")
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        print(out.strip())
        return out.strip()
    except subprocess.CalledProcessError as e:
        print(e.output.strip())
        return ""

# %%
print("# System / Driver")
run("nvidia-smi || echo 'nvidia-smi not available'")

# %%
print("# Torch Info")
try:
    import torch
    print("torch:", torch.__version__, " cuda:", torch.version.cuda, " available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(" -", i, torch.cuda.get_device_name(i))
        x = torch.ones(1024, 1024, device="cuda")
        y = torch.ones(1024, 1024, device="cuda")
        z = x @ y
        print("matmul OK:", z.shape, float(z[0,0].item()))
except Exception as e:
    print("Torch error:", e)

# %%
print("# JAX Info")
try:
    import jax, jax.numpy as jnp
    print("jax:", jax.__version__)
    print("devices:", jax.devices())
    x = jnp.ones((1024, 1024))
    y = jnp.ones((1024, 1024))
    z = jax.jit(lambda a,b: a @ b)(x, y)
    print("JAX matmul OK:", z.shape, float(z[0,0]))
except Exception as e:
    print("JAX error:", e)

# %%
print("# Conclusion")
print("If both Torch and JAX reported GPU devices and ran matrix multiplies, you're good.")
