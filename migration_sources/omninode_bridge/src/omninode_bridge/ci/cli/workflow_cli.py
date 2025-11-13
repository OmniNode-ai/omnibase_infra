#!/usr/bin/env python3
"""
Entry point script for the workflow CLI.

This script uses proper Python package imports instead of path manipulation
for enhanced security. The package should be properly installed using:
    pip install -e .
or the PYTHONPATH environment variable should be configured.
"""

import os
import sys
from pathlib import Path

# Secure package import - use PYTHONPATH environment variable instead of sys.path.insert()
if __name__ == "__main__":
    # Check if running in development mode and PYTHONPATH is not set
    project_root = Path(__file__).parent.parent.parent.parent
    if "PYTHONPATH" not in os.environ and not any(
        str(project_root) in p for p in sys.path
    ):
        print("Warning: Package not properly installed.", file=sys.stderr)
        print(
            "Please install the package using 'pip install -e .' or set PYTHONPATH",
            file=sys.stderr,
        )
        print(f"Example: export PYTHONPATH={project_root}:$PYTHONPATH", file=sys.stderr)
        sys.exit(1)

# Import after path validation
try:
    from omninode_bridge.ci.cli.main import cli
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    print(
        "Ensure the package is properly installed or PYTHONPATH is configured.",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    cli()
