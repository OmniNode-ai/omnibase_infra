#!/usr/bin/env python3
"""
ONEX Infrastructure Validation Script.

Run all validators with infrastructure-specific defaults.
Can be used standalone or as part of pre-commit hooks.

Usage:
    python scripts/validate.py [--verbose] [--quick]
    python scripts/validate.py architecture
    python scripts/validate.py architecture_layers
    python scripts/validate.py contracts
    python scripts/validate.py patterns
    python scripts/validate.py unions
    python scripts/validate.py any_types
    python scripts/validate.py imports
    python scripts/validate.py all
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_architecture(verbose: bool = False) -> bool:
    """Run architecture validation with infrastructure-specific exemptions."""
    try:
        # Use the infrastructure validator which includes exemption filtering
        # for domain-grouped protocols per CLAUDE.md convention
        from omnibase_infra.validation.infra_validators import (
            validate_infra_architecture,
        )

        result = validate_infra_architecture()
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


# =============================================================================
# Known Issues Registry
# =============================================================================
# Track known architecture violations with Linear ticket references.
# Format: dict mapping import_name to (ticket_id, description)
#
# These violations will still cause the check to fail, but the reporting
# will include ticket links for visibility and tracking purposes.
KNOWN_ISSUES: dict[str, tuple[str, str]] = {
    "aiohttp": (
        "OMN-1015",
        "async HTTP client usage in core - needs migration to infra",
    ),
    "redis": ("OMN-1295", "Redis client usage in core - needs migration to infra"),
}


def run_architecture_layers(verbose: bool = False) -> bool:
    """Run architecture layer validation.

    Verifies that omnibase_core does not contain infrastructure dependencies
    (kafka, httpx, asyncpg, etc.) to maintain proper layer separation.

    This wraps scripts/check_architecture.sh for consistent validation interface.

    Known issues are tracked with Linear ticket IDs in KNOWN_ISSUES above.
    The validator will report ticket links for any known violations found.

    LIMITATIONS:
        This validation uses grep-based pattern matching which cannot detect:
        - Inline imports (imports inside functions/methods)
        - Dynamic imports using __import__() or importlib
        - Imports hidden behind conditional logic (if statements)
        - String-based import references

        For comprehensive AST-based analysis, use the Python tests:
            pytest tests/ci/test_architecture_compliance.py
    """
    import subprocess

    script_path = Path(__file__).parent / "check_architecture.sh"

    if not script_path.exists():
        print(f"Architecture Layers: SKIP (script not found: {script_path})")
        return True

    try:
        # Build command with appropriate flags
        cmd = ["bash", str(script_path), "--no-color"]
        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,  # 120 second timeout for large codebases
            shell=False,
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        passed = result.returncode == 0

        if not passed and result.returncode == 2:
            # Exit code 2 means script error (path not found, etc.)
            # This is not a violation, just skip
            if verbose:
                print("Architecture Layers: SKIP (omnibase_core not found)")
            return True

        # Note: The bash script already reports known issues with ticket links
        # when violations are found, so we don't duplicate the reporting here.
        # The _report_known_issues function is available for programmatic use.

        return passed

    except subprocess.TimeoutExpired:
        print("Architecture Layers: ERROR (timeout after 120s)")
        print("  Fix: Check if omnibase_core path is accessible")
        print("  Fix: Try running with --verbose to see progress")
        return False
    except FileNotFoundError:
        print("Architecture Layers: SKIP (bash not available)")
        return True
    except PermissionError as e:
        print(f"Architecture Layers: ERROR (Permission denied: {e})")
        print("  Fix: Ensure execute permissions on check_architecture.sh")
        return False
    except OSError as e:
        print(f"Architecture Layers: ERROR (OS error: {e})")
        print("  Fix: Check file system access and disk space")
        return False


def _report_known_issues(output: str, verbose: bool) -> None:
    """Report known issues with ticket links.

    Parses the validator output to identify known violations and provides
    helpful links to the corresponding Linear tickets.

    Args:
        output: The stdout from check_architecture.sh
        verbose: If True, show additional context
    """
    found_known_issues = []

    for import_name, (ticket_id, description) in KNOWN_ISSUES.items():
        # Check if this import was flagged in the output
        if f"'{import_name}'" in output or f'"{import_name}"' in output:
            found_known_issues.append((import_name, ticket_id, description))

    if found_known_issues:
        print("\n" + "=" * 60)
        print("KNOWN ISSUES (tracked in Linear)")
        print("=" * 60)
        for import_name, ticket_id, description in found_known_issues:
            print(f"\n  {import_name}:")
            print(f"    Ticket: {ticket_id}")
            print(f"    Description: {description}")
            print(f"    Link: https://linear.app/onex/issue/{ticket_id}")
        print("\n" + "-" * 60)
        print("These violations are known and tracked. Fix by resolving the")
        print("corresponding Linear tickets listed above.")
        print("=" * 60 + "\n")


def run_contracts(verbose: bool = False) -> bool:
    """Run contract validation with infrastructure-specific linting.

    Uses two-phase validation:
    1. Basic YAML validation from omnibase_core
    2. Infrastructure contract linting for required fields and type consistency
    """
    nodes_dir = Path("src/omnibase_infra/nodes")
    if not nodes_dir.exists():
        if verbose:
            print("Contracts: SKIP (no nodes directory)")
        return True

    all_passed = True

    # Phase 1: Basic YAML validation from omnibase_core
    try:
        from omnibase_core.validation import validate_contracts

        result = validate_contracts("src/omnibase_infra/nodes/")
        if verbose or not result.is_valid:
            print(f"Contracts (YAML): {'PASS' if result.is_valid else 'FAIL'}")
            for e in result.errors:
                print(f"  - {e}")
        if not result.is_valid:
            all_passed = False
    except ImportError as e:
        print(f"Skipping YAML validation: {e}")

    # Phase 2: Infrastructure contract linting
    try:
        from omnibase_infra.validation.linter_contract import (
            EnumContractViolationSeverity,
            lint_contracts_in_directory,
        )

        lint_result = lint_contracts_in_directory(
            "src/omnibase_infra/nodes/",
            check_imports=True,
            strict_mode=False,
        )

        if verbose or not lint_result.is_valid:
            print(f"Contracts (Lint): {'PASS' if lint_result.is_valid else 'FAIL'}")
            print(
                f"  Files: {lint_result.files_checked}, "
                f"errors: {lint_result.error_count}, "
                f"warnings: {lint_result.warning_count}"
            )
            # Show errors and warnings
            for v in lint_result.violations:
                if v.severity in (
                    EnumContractViolationSeverity.ERROR,
                    EnumContractViolationSeverity.WARNING,
                ):
                    print(f"  - {v}")
        if not lint_result.is_valid:
            all_passed = False

    except ImportError as e:
        print(f"Skipping contract linting: {e}")

    if verbose or not all_passed:
        print(f"Contracts: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


def run_patterns(verbose: bool = False) -> bool:
    """Run pattern validation with infrastructure-specific exemptions."""
    try:
        # Use the infrastructure validator which includes exemption filtering
        from omnibase_infra.validation.infra_validators import validate_infra_patterns

        result = validate_infra_patterns()

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
    """Run union usage validation.

    Counts total unions in the codebase.
    Valid `X | None` patterns are counted but not flagged as violations.
    """
    try:
        # Use infrastructure wrapper which includes exemption filtering
        # for documented infrastructure patterns
        from omnibase_infra.validation.infra_validators import (
            INFRA_MAX_UNIONS,
            INFRA_UNIONS_STRICT,
            validate_infra_union_usage,
        )

        result = validate_infra_union_usage(
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
                        f"  Total unions: {meta.total_unions}, max allowed: {INFRA_MAX_UNIONS}"
                    )
        return bool(result.is_valid)
    except ImportError as e:
        print(f"Skipping union validation: {e}")
        return True


def run_any_types(verbose: bool = False) -> bool:
    """Run Any type usage validation.

    Checks for forbidden Any type usage in function signatures and type annotations.
    Valid usages (Pydantic Field() with NOTE comment) are allowed.
    """
    src_path = Path("src/omnibase_infra")
    if not src_path.exists():
        if verbose:
            print("Any Types: SKIP (no src/omnibase_infra directory)")
        return True

    try:
        from omnibase_infra.validation.validator_any_type import validate_any_types_ci

        result = validate_any_types_ci(src_path)

        if verbose or not result.passed:
            print(f"Any Types: {'PASS' if result.passed else 'FAIL'}")
            print(
                f"  Files checked: {result.files_checked}, "
                f"blocking violations: {result.blocking_count}, "
                f"total violations: {result.total_violations}"
            )
            # Show violations
            for v in result.violations:
                print(f"  - {v.file_path}:{v.line_number}: {v.violation_type.value}")
                if verbose:
                    print(f"      {v.code_snippet}")
                    print(f"      Suggestion: {v.suggestion}")

        return result.passed

    except ImportError as e:
        print(f"Skipping Any type validation: {e}")
        return True


def run_imports(verbose: bool = False) -> bool:
    """Run circular import check."""
    # Use src/ as the source path so module names are fully qualified
    # (e.g., "omnibase_infra.clients" instead of just "clients").
    # The CircularImportValidator creates module names relative to source_path,
    # so using src/ ensures Python can import them correctly.
    src_path = Path("src/")
    if not src_path.exists():
        if verbose:
            print("Imports: SKIP (no src directory)")
        return True

    try:
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        # CircularImportValidator re-exported from omnibase_core.validation in 0.6.2+
        # (moved from circular_import_validator to validator_circular_import submodule)
        from omnibase_core.validation import CircularImportValidator

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
    """Run all validations.

    Runs all ONEX infrastructure validators in sequence. The architecture_layers
    check is included to verify omnibase_core maintains proper layer separation.

    Args:
        verbose: If True, show detailed output for each validator
        quick: If True, skip medium priority validators (unions, imports)

    Returns:
        True if all validations pass, False if any fail
    """
    print("Running ONEX Infrastructure Validations...")
    print("=" * 50)

    validators = [
        ("Architecture", run_architecture),
        ("Architecture Layers", run_architecture_layers),
        ("Contracts", run_contracts),
        ("Patterns", run_patterns),
    ]

    if not quick:
        validators.extend(
            [
                ("Unions", run_unions),
                ("Any Types", run_any_types),
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
        choices=[
            "all",
            "architecture",
            "architecture_layers",
            "contracts",
            "patterns",
            "unions",
            "any_types",
            "imports",
        ],
        help="Which validator to run (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick mode (skip medium priority)"
    )

    args = parser.parse_args()

    validator_map = {
        "architecture": run_architecture,
        "architecture_layers": run_architecture_layers,
        "contracts": run_contracts,
        "patterns": run_patterns,
        "unions": run_unions,
        "any_types": run_any_types,
        "imports": run_imports,
    }

    if args.validator == "all":
        success = run_all(args.verbose, args.quick)
    else:
        success = validator_map[args.validator](args.verbose)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
