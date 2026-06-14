import sys
from pathlib import Path

# Make the repo root importable as a package root so both `pytest` and
# `python -m` invocations find `core` and `tasks` without extra setup.
sys.path.insert(0, str(Path(__file__).parent))
