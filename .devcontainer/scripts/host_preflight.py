#!/usr/bin/env python3
"""Host-side validation to catch devcontainer path issues before build."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    print("=== initialize: host context check ===")
    cwd = Path.cwd()
    print(f"Host CWD: {cwd}")
    workspace = Path(os.environ.get("LOCAL_WORKSPACE_FOLDER", cwd))
    print(f"LOCAL_WORKSPACE_FOLDER: {workspace}")

    expected = [
        workspace / ".devcontainer" / "Dockerfile",
        workspace / ".devcontainer" / "docker-compose.yml",
    ]

    missing = [path for path in expected if not path.exists()]
    if missing:
        print("Missing critical files:")
        for path in missing:
            print(f"  - {path}")
        print("Open VS Code at the project root so the devcontainer files are in scope.")
        return 1

    print("All required devcontainer files located; continuing with build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
