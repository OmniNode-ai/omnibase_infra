#!/usr/bin/env python3
"""
CLI module entry point for OmniNode Bridge workflow submission.

Allows running the CLI with: python -m omninode_bridge.cli.workflow_submit
"""

import asyncio
import sys

from .workflow_submit import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
