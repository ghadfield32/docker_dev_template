from pathlib import Path
import os

_added_src_flag: bool = False          # module-level cache

def project_root() -> Path:
    """
    Return the absolute path to the repo root *without* relying on __file__.

    • If running from a .py file, use that file's parent/parent (…/src/..)
    • If running interactively (no __file__), fall back to CWD.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    return Path.cwd()

def ensure_src_on_path(verbose: bool = True) -> None:
    """
    Ensure <repo-root>/src is the *first* entry in sys.path exactly once.
    The verbose flag prints the helper line the first time only.
    """
    import sys
    global _added_src_flag
    root = project_root()
    src_path = root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        if verbose and not _added_src_flag:
            print(f"🔧 Added {src_path} to sys.path")
        _added_src_flag = True

