"""
Main entry point for ONEX node generation CLI.

Provides command-line interface for generating ONEX nodes
via event-driven orchestration.
"""

import sys

from .commands import generate_command


def main() -> int:
    """
    Main entry point for omninode-generate CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Click handles argument parsing and command execution
        return generate_command()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
