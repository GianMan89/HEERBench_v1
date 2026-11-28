"""Offline structural and provenance validation of the repository's corpus."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, "-m", "heerbench", "validate"],
    [sys.executable, "data/heerbench_v05/scripts/validate_dataset.py"],
    [sys.executable, "-m", "pytest", "-q"],
]
for command in commands:
    subprocess.run(command, cwd=root, check=True)
