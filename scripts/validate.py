#!/usr/bin/env python3
"""
ONEX Infrastructure Validation Script.

Run all validators with infrastructure-specific defaults.
Can be used standalone or as part of pre-commit hooks.

Usage:
    python scripts/validate.py [--verbose] [--quick]
    python scripts/validate.py architecture
    python scripts/validate.py contracts
    python scripts/validate.py patterns
    python scripts/validate.py unions
    python scripts/validate.py imports
    python scripts/validate.py all
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_architecture(verbose: bool = False) -> bool:
    """Run architecture validation."""
    try:
        from omnibase_core.validation import validate_architecture

        from omnibase_infra.validation.infra_validators import INFRA_MAX_VIOLATIONS

        result = validate_architecture(
            "src/omnibase_infra/", max_violations=INFRA_MAX_VIOLATIONS
        )
        if verbose or not result.is_valid:
            print(f"Architecture: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
            if hasattr(result, "metadata") and result.metadata:
                meta = result.metadata
                print(
                    f"  Files processed: {meta.files_processed}, "
                    f"violations: {meta.violations_found}/{meta.max_violations}"
                )
        return bool(result.is_valid)
    except ImportError as e:
        print(f"Skipping architecture validation: {e}")
        return True


def run_contracts(verbose: bool = False) -> bool:
    """Run contract validation."""
    nodes_dir = Path("src/omnibase_infra/nodes")
    if not nodes_dir.exists():
        if verbose:
            print("Contracts: SKIP (no nodes directory)")
        return True

    try:
        from omnibase_core.validation import validate_contracts

        result = validate_contracts("src/omnibase_infra/nodes/")
        if verbose or not result.is_valid:
            print(f"Contracts: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
            if hasattr(result, "metadata") and result.metadata:
                meta = result.metadata
                if meta.yaml_files_found is not None:
                    print(
                        f"  YAML files found: {meta.yaml_files_found}, "
                        f"violations: {meta.violations_found}"
                    )
        return bool(result.is_valid)
    except ImportError as e:
        print(f"Skipping contract validation: {e}")
        return True


def run_patterns(verbose: bool = False) -> bool:
    """Run pattern validation with infrastructure-specific exemptions."""
    try:
        from omnibase_core.validation import validate_patterns

        from omnibase_infra.validation.infra_validators import INFRA_PATTERNS_STRICT

        result = validate_patterns("src/omnibase_infra/", strict=INFRA_PATTERNS_STRICT)
        if verbose or not result.is_valid:
            print(f"Patterns: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
            if hasattr(result, "metadata") and result.metadata:
                meta = result.metadata
                print(
                    f"  Files processed: {meta.files_processed}, "
                    f"strict mode: {meta.strict_mode}, "
                    f"violations: {meta.violations_found}"
                )
        return bool(result.is_valid)
    except ImportError as e:
        print(f"Skipping pattern validation: {e}")
        return True


def run_unions(verbose: bool = False) -> bool:
    """Run union usage validation."""
    try:
        from omnibase_core.validation import validate_union_usage

        from omnibase_infra.validation.infra_validators import (
            INFRA_MAX_UNIONS,
            INFRA_UNIONS_STRICT,
        )

        result = validate_union_usage(
            "src/omnibase_infra/",
            max_unions=INFRA_MAX_UNIONS,
            strict=INFRA_UNIONS_STRICT,
        )
        if verbose or not result.is_valid:
            print(f"Unions: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
            if hasattr(result, "metadata") and result.metadata:
                meta = result.metadata
                if hasattr(meta, "total_unions"):
                    print(
                        f"  Total unions: {meta.total_unions}, max allowed: {meta.max_unions}"
                    )
        return bool(result.is_valid)
    except ImportError as e:
        print(f"Skipping union validation: {e}")
        return True


def run_imports(verbose: bool = False) -> bool:
    """Run circular import check."""
    src_path = Path("src/omnibase_infra/")
    if not src_path.exists():
        if verbose:
            print("Imports: SKIP (no src directory)")
        return True

    try:
        from omnibase_core.models.errors.model_onex_error import ModelOnexError
        from omnibase_core.validation.circular_import_validator import (
            CircularImportValidator,
        )

        validator = CircularImportValidator(source_path=src_path)
        result = validator.validate()
        passed = not result.has_circular_imports

        if verbose or not passed:
            print(f"Imports: {'PASS' if passed else 'FAIL'}")

            # Show circular imports if found
            if result.has_circular_imports:
                print("  Circular import cycles detected:")
                for module in result.circular_imports[:10]:
                    print(f"    - {module}")
                if len(result.circular_imports) > 10:
                    print(f"    ... and {len(result.circular_imports) - 10} more")
                print("\n  Fix: Break circular dependencies by:")
                print("    1. Moving shared code to a common module")
                print("    2. Using TYPE_CHECKING imports for type hints")
                print("    3. Restructuring module dependencies")

            # Show import errors even if no circular imports (helps diagnose issues)
            if result.has_errors and (verbose or result.failure_count > 0):
                print(
                    f"  Import validation: {result.success_count} succeeded, {result.failure_count} failed"
                )
                if result.import_errors:
                    # Show more errors in verbose mode
                    max_errors = len(result.import_errors) if verbose else 5
                    print("  Module import errors (may indicate missing dependencies):")
                    for err in result.import_errors[:max_errors]:
                        print(f"    - {err.module_name}: {err.error_message}")
                    if len(result.import_errors) > max_errors:
                        print(
                            f"    ... and {len(result.import_errors) - max_errors} more (use --verbose for all)"
                        )
                if result.unexpected_errors:
                    max_unexpected = len(result.unexpected_errors) if verbose else 5
                    print("  Unexpected errors during validation:")
                    for err in result.unexpected_errors[:max_unexpected]:
                        print(f"    - {err}")
                    if len(result.unexpected_errors) > max_unexpected:
                        print(
                            f"    ... and {len(result.unexpected_errors) - max_unexpected} more (use --verbose for all)"
                        )

            # Show summary statistics
            if hasattr(result, "total_files"):
                print(
                    f"  Summary: {result.total_files} files analyzed, "
                    f"success rate: {result.success_rate:.1%}"
                )

        return passed

    except ImportError as e:
        # CircularImportValidator not available (omnibase_core not installed)
        print(f"Imports: SKIP (CircularImportValidator not available: {e})")
        print("  Fix: Install omnibase_core with: poetry add omnibase-core")
        return True
    except ModelOnexError as e:
        # Path validation or configuration errors from validator initialization
        print(f"Imports: ERROR (Configuration error: {e})")
        print(f"  Fix: Verify source path exists and is readable: {src_path}")
        # Fail validation - configuration errors should be fixed
        return False
    except AttributeError as e:
        # Validator result missing expected attributes (API incompatibility)
        # This indicates integration bug between omnibase_infra and omnibase_core
        print(f"Imports: ERROR (Validator API incompatible: {e})")
        print("  Fix: Update omnibase_core to compatible version")
        print("    poetry update omnibase-core")
        print("    or check omnibase_core version requirements")
        # Fail validation on API incompatibility - this is a real integration bug
        return False
    except PermissionError as e:
        # File system permission issues
        print(f"Imports: ERROR (Permission denied: {e})")
        print(f"  Fix: Ensure read permissions for: {src_path}")
        return False
    except Exception as e:
        # Unexpected errors during validation (file system issues, bugs in validator, etc.)
        # Log with full exception type to help debugging
        exception_type = type(e).__name__
        print(f"Imports: ERROR (Unexpected {exception_type}: {e})")
        print("  This may indicate a bug in the validator or unexpected file structure")
        print("  Fix: Report this error with full output if it persists")
        # Fail validation on unexpected errors - these may hide real bugs
        return False


def run_all(verbose: bool = False, quick: bool = False) -> bool:
    """Run all validations."""
    print("Running ONEX Infrastructure Validations...")
    print("=" * 50)

    validators = [
        ("Architecture", run_architecture),
        ("Contracts", run_contracts),
        ("Patterns", run_patterns),
    ]

    if not quick:
        validators.extend(
            [
                ("Unions", run_unions),
                ("Imports", run_imports),
            ]
        )

    results = {}
    for name, func in validators:
        results[name] = func(verbose)

    print("=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Summary: {passed}/{total} passed")

    if all(results.values()):
        print("All validations PASSED")
        return True
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"FAILED: {', '.join(failed)}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ONEX Infrastructure Validation Script"
    )
    parser.add_argument(
        "validator",
        nargs="?",
        default="all",
        choices=["all", "architecture", "contracts", "patterns", "unions", "imports"],
        help="Which validator to run (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick mode (skip medium priority)"
    )

    args = parser.parse_args()

    validator_map = {
        "architecture": run_architecture,
        "contracts": run_contracts,
        "patterns": run_patterns,
        "unions": run_unions,
        "imports": run_imports,
    }

    if args.validator == "all":
        success = run_all(args.verbose, args.quick)
    else:
        success = validator_map[args.validator](args.verbose)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
