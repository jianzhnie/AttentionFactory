"""Make the project root and tests/ importable when running pytest from anywhere.

The tests/ entry lets test modules in subdirectories keep using
``from helpers import ...`` without an ``__init__.py`` (pytest rootdir mode).
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))
