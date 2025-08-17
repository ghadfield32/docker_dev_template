#!/usr/bin/env python3
import sys, importlib, json

def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

def main():
    print("## Python & library diagnostics ##")
    print("Python:", sys.version.split()[0])

    mods = ["jupyterlab", "torch", "jax"]
    status = {}
    for m in mods:
        ok = _have(m)
        status[m] = ok
        print(f" - {m}: {'OK' if ok else 'MISSING'}")

    # Only probe deeper if present (no papering over)
    if status.get("torch"):
        import torch
        print("   torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
    if status.get("jax"):
        import jax
        print("   jax", jax.__version__, "devices:", jax.devices())

    # Exit nonzero if anything critical is missing
    missing = [m for m in ["jupyterlab","jax"] if not status.get(m)]
    if missing:
        print("❌ Missing critical packages:", ", ".join(missing))
        sys.exit(1)

    print("✅ verify_env ok")
    sys.exit(0)

if __name__ == "__main__":
    main()
