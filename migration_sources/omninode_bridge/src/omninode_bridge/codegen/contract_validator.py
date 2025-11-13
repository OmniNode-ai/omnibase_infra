#!/usr/bin/env python3
"""
Contract Validator CLI for ONEX v2.0 Contracts.

Validates contract YAML files against the ONEX v2.0 schema, including:
- JSON Schema validation
- Mixin declaration validation
- Configuration validation
- Backward compatibility checking

Usage:
    python -m omninode_bridge.codegen.contract_validator \\
        --contract path/to/contract.yaml

    # Or with custom schema
    python -m omninode_bridge.codegen.contract_validator \\
        --schema schemas/contract_schema_v2.json \\
        --contract path/to/contract.yaml

Exit codes:
    0 - Validation successful
    1 - Validation failed (errors found)
    2 - CLI usage error (missing arguments, file not found, etc.)
"""

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"\033[92m✅ {message}\033[0m")


def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"\033[91m❌ {message}\033[0m", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    print(f"\033[93m⚠️  {message}\033[0m")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"[INFO] {message}")


def validate_contract(
    contract_path: Path, schema_dir: Path | None = None
) -> tuple[bool, list[str], list[str]]:
    """
    Validate contract YAML file.

    Args:
        contract_path: Path to contract YAML file
        schema_dir: Optional directory containing JSON schemas

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    # Initialize parser
    try:
        if schema_dir:
            parser = YAMLContractParser(schema_dir=schema_dir)
        else:
            parser = YAMLContractParser()
    except FileNotFoundError as e:
        return False, [f"Schema loading failed: {e}"], []
    except Exception as e:
        return False, [f"Parser initialization failed: {e}"], []

    # Parse contract
    try:
        contract = parser.parse_contract_file(contract_path)
    except FileNotFoundError:
        return False, [f"Contract file not found: {contract_path}"], []
    except Exception as e:
        return False, [f"Contract parsing failed: {e}"], []

    # Collect errors
    errors = contract.validation_errors.copy()

    # Check mixin errors
    for mixin in contract.mixins:
        if mixin.validation_errors:
            errors.extend(
                [f"Mixin '{mixin.name}': {err}" for err in mixin.validation_errors]
            )

    # Collect warnings
    warnings = []

    # Check for deprecated fields
    if contract.has_deprecated_error_handling():
        warnings.append(
            "Contract uses deprecated 'error_handling' field - "
            "consider migrating to 'advanced_features'"
        )

    # Check if v1.0 contract (suggest upgrading)
    if not contract.is_v2:
        warnings.append(
            f"Contract is using schema version {contract.schema_version} - "
            "consider upgrading to v2.0.0 for mixin support"
        )

    return contract.is_valid, errors, warnings


def print_validation_results(
    contract_path: Path,
    is_valid: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Print validation results with formatting.

    Args:
        contract_path: Path to validated contract
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
    """
    print(f"\nValidating contract: {contract_path}\n")

    if is_valid:
        print_success(f"Contract validation successful ({contract_path.name})")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print_warning(warning)
    else:
        print_error(f"Contract validation failed ({contract_path.name})")

        if errors:
            print("\nErrors:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print_warning(warning)

    print()  # Empty line at end


def main() -> NoReturn:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Validate ONEX v2.0 contract YAML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a contract file
  python -m omninode_bridge.codegen.contract_validator --contract contract.yaml

  # Validate with custom schema directory
  python -m omninode_bridge.codegen.contract_validator \\
      --schema schemas/ \\
      --contract contract.yaml

Exit codes:
  0 - Validation successful
  1 - Validation failed (errors found)
  2 - CLI usage error
        """,
    )

    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to contract YAML file to validate",
    )

    parser.add_argument(
        "--schema",
        type=Path,
        help="Directory containing JSON schemas (default: auto-detect)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output, only set exit code",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation information",
    )

    args = parser.parse_args()

    # Validate contract path exists
    if not args.contract.exists():
        if not args.quiet:
            print_error(f"Contract file not found: {args.contract}")
        sys.exit(2)

    # Validate schema dir if provided
    if args.schema and not args.schema.is_dir():
        if not args.quiet:
            print_error(f"Schema directory not found: {args.schema}")
        sys.exit(2)

    # Run validation
    try:
        is_valid, errors, warnings = validate_contract(args.contract, args.schema)
    except Exception as e:
        if not args.quiet:
            print_error(f"Validation error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)

    # Print results (unless quiet)
    if not args.quiet:
        print_validation_results(args.contract, is_valid, errors, warnings)

    # Exit with appropriate code
    if is_valid:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
