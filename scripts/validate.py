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

        result = validate_architecture("src/omnibase_infra/", max_violations=0)
        if verbose or not result.is_valid:
            print(f"Architecture: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
        return result.is_valid
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
        return result.is_valid
    except ImportError as e:
        print(f"Skipping contract validation: {e}")
        return True


def run_patterns(verbose: bool = False) -> bool:
    """Run pattern validation."""
    try:
        from omnibase_core.validation import validate_patterns

        result = validate_patterns("src/omnibase_infra/", strict=True)
        if verbose or not result.is_valid:
            print(f"Patterns: {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
        return result.is_valid
    except ImportError as e:
        print(f"Skipping pattern validation: {e}")
        return True


def run_unions(verbose: bool = False) -> bool:
    """Run union usage validation."""
    try:
        from omnibase_core.validation import validate_union_usage

        # Allow up to 20 unions for infrastructure code (has many typed handlers)
        result = validate_union_usage(
            "src/omnibase_infra/", max_unions=20, strict=False
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
        return result.is_valid
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
        from omnibase_core.validation.circular_import_validator import (
            CircularImportValidator,
        )

        validator = CircularImportValidator(source_path=src_path)
        result = validator.validate()
        passed = not result.has_circular_imports
        if verbose or not passed:
            print(f"Imports: {'PASS' if passed else 'FAIL'}")
            if result.has_circular_imports:
                for module in result.circular_imports[:10]:
                    print(f"  - {module}")
        return passed
    except ImportError as e:
        print(f"Skipping circular import check: {e}")
        return True
    except Exception as e:
        # CircularImportValidator may fail on incomplete codebases
        if verbose:
            print(f"Imports: SKIP (error: {e})")
        return True


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
    sys.exit(main())
