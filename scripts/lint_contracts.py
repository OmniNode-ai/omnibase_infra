#!/usr/bin/env python3
"""
ONEX Infrastructure Contract Linter CLI.

Validates contract.yaml files against ONEX infrastructure requirements.
Designed for CI integration with proper exit codes.

Usage:
    poetry run python scripts/lint_contracts.py
    poetry run python scripts/lint_contracts.py --directory src/omnibase_infra/nodes/
    poetry run python scripts/lint_contracts.py --strict --verbose
    poetry run python scripts/lint_contracts.py --no-import-check  # Skip import validation

Exit Codes:
    0: All contracts valid
    1: Validation failures found
    2: Runtime error (file system, invalid arguments)

Examples:
    # Lint all contracts in default location
    python scripts/lint_contracts.py

    # Lint with strict mode (warnings become errors)
    python scripts/lint_contracts.py --strict

    # Skip module import validation (faster, for CI without full deps)
    python scripts/lint_contracts.py --no-import-check

    # Verbose output including INFO-level suggestions
    python scripts/lint_contracts.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> int:
    """Main entry point for contract linting CLI."""
    parser = argparse.ArgumentParser(
        description="ONEX Infrastructure Contract Linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="src/omnibase_infra/nodes/",
        help="Directory to search for contract.yaml files (default: src/omnibase_infra/nodes/)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Lint a single contract.yaml file instead of a directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 if any warnings)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all violations including INFO-level suggestions",
    )
    parser.add_argument(
        "--no-import-check",
        action="store_true",
        help="Skip validation that model modules are importable (faster)",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only check contract.yaml in the specified directory, not subdirectories",
    )

    args = parser.parse_args()

    try:
        # Import after path setup
        from omnibase_infra.validation.linter_contract import (
            ContractLinter,
            EnumContractViolationSeverity,
        )

        linter = ContractLinter(
            check_imports=not args.no_import_check,
            strict_mode=args.strict,
        )

        # Lint single file or directory
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"ERROR: File not found: {file_path}", file=sys.stderr)
                return 2
            result = linter.lint_file(file_path)
        else:
            directory = Path(args.directory)
            if not directory.exists():
                print(f"ERROR: Directory not found: {directory}", file=sys.stderr)
                return 2
            result = linter.lint_directory(directory, recursive=not args.non_recursive)

        # Print summary
        print("=" * 60)
        print("ONEX Contract Linting Results")
        print("=" * 60)
        print(f"Files checked: {result.files_checked}")
        print(f"Files valid:   {result.files_valid}")
        print(f"Files with errors: {result.files_with_errors}")
        print(f"Total errors:   {result.error_count}")
        print(f"Total warnings: {result.warning_count}")
        print()

        # Print violations by severity
        errors = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.ERROR
        ]
        warnings = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.WARNING
        ]
        infos = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.INFO
        ]

        if errors:
            print("ERRORS (must fix):")
            for v in errors:
                print(f"  {v}")
            print()

        if warnings:
            print("WARNINGS (should fix):")
            for v in warnings:
                print(f"  {v}")
            print()

        if args.verbose and infos:
            print("INFO (suggestions):")
            for v in infos:
                print(f"  {v}")
            print()

        # Final status
        print("=" * 60)
        if result.is_valid:
            print("PASSED: All contracts are valid")
            return 0
        else:
            print("FAILED: Contract validation errors found")
            return 1

    except ImportError as e:
        print(f"ERROR: Failed to import contract linter: {e}", file=sys.stderr)
        print("Make sure omnibase_infra is installed: poetry install", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
