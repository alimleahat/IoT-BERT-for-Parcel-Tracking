#!/usr/bin/env python3
"""Run the existing host tests without modifying the sample parcel database."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="parcel-host-tests-") as temporary:
        server = Path(temporary) / "server"
        shutil.copytree(ROOT / "server", server, ignore=shutil.ignore_patterns("__pycache__", ".env", ".venv"))
        subprocess.run([sys.executable, str(server / "test_server.py")], cwd=server, check=True, timeout=60)


if __name__ == "__main__":
    main()
