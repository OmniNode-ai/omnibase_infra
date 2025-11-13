#!/usr/bin/env python3
"""
Script to run the metadata stamping service with proper module path configuration.

This script ensures that the PYTHONPATH is correctly set so that all modules
can be imported properly.
"""

import os
import sys
from pathlib import Path


def main():
    """Run the metadata stamping service with correct PYTHONPATH."""

    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"

    # Add src directory to Python path
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        os.environ["PYTHONPATH"] = f"{src_dir}:{current_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = str(src_dir)

    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

    # Change to project root directory
    os.chdir(project_root)

    try:
        # Import and run the service
        from omninode_bridge.services.metadata_stamping.main import main as service_main

        print("Successfully imported metadata stamping service")
        print("Starting service...")
        service_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  poetry install")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
