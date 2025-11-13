"""
Minimal conftest for lifecycle pattern tests.

This conftest isolates pattern tests from the main codegen conftest
to avoid dependency issues.
"""

import sys
from pathlib import Path

# Ensure src is in path
repo_root = Path(__file__).parent.parent.parent.parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
