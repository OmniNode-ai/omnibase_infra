#!/usr/bin/env python3
"""
Validate Generated Files Script.

Validates actual generated files on disk to ensure:
- Valid Python syntax
- No stub patterns remain
- All expected files exist

This script is meant to be run AFTER code generation to verify
that files were actually written correctly.

Usage:
    # Validate specific directory
    python scripts/validate_generated_files.py generated_nodes/vault_secrets_effect

    # Validate multiple directories
    python scripts/validate_generated_files.py generated_nodes/vault_secrets_effect generated_nodes/postgres_crud

    # Validate with strict mode (fail on warnings)
    python scripts/validate_generated_files.py --strict generated_nodes/vault_secrets_effect

Example Integration:
    # In generation script, after writing files:
    result = await service.generate_node(...)

    # Write files
    for filename, content in result.artifacts.get_all_files().items():
        (output_dir / filename).write_text(content)

    # Validate files were written correctly
    validator = FileValidator()
    validation_result = await validator.validate_generated_files(
        file_paths=[output_dir / f for f in result.artifacts.get_all_files().keys()]
    )

    if not validation_result.passed:
        print(validation_result.summary)
        sys.exit(1)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.codegen.file_validator import FileValidator


async def validate_path(
    path: Path,
    strict_mode: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Validate Python files in a path (file or directory).

    Args:
        path: File or directory containing generated files
        strict_mode: Fail on warnings if True
        verbose: Print detailed output

    Returns:
        True if validation passed, False otherwise
    """
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        return False

    # Find all Python files
    if path.is_file():
        # Single file
        if path.suffix == ".py":
            python_files = [path]
        else:
            print(f"⚠️  Not a Python file: {path}")
            return True
    else:
        # Directory - find all Python files recursively
        python_files = list(path.rglob("*.py"))

    if not python_files:
        print(f"⚠️  No Python files found in {path}")
        return True

    print(f"\n{'=' * 80}")
    print(f"Validating {len(python_files)} files in: {path}")
    print(f"{'=' * 80}\n")

    # Validate files
    validator = FileValidator()
    result = await validator.validate_generated_files(
        file_paths=python_files,
        strict_mode=strict_mode,
    )

    # Print report
    if verbose or not result.passed:
        report = validator.format_validation_report(
            result,
            include_file_paths=True,
        )
        print(report)
    else:
        print(f"✅ All {result.files_validated} files passed validation")
        print(f"   Execution time: {result.execution_time_ms:.0f}ms\n")

    return result.passed


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate generated files for syntax and stub patterns"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing generated files to validate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed validation report",
    )

    args = parser.parse_args()

    # Validate each path
    all_passed = True
    for path in args.paths:
        passed = await validate_path(
            path=path,
            strict_mode=args.strict,
            verbose=args.verbose,
        )
        if not passed:
            all_passed = False

    # Exit with appropriate code
    if all_passed:
        print("✅ All validations passed!")
        sys.exit(0)
    else:
        print("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
