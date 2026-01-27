#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validate all handler contracts before migration.

This script validates that all handlers previously registered via _KNOWN_HANDLERS
now have valid contract.yaml files with proper configuration.

Run this BEFORE deleting _KNOWN_HANDLERS to ensure no handlers are orphaned.

Part of OMN-1518: Migration from hardcoded _KNOWN_HANDLERS to contract-driven
handler registration.

Usage:
    python scripts/validate_handler_contracts.py
    python scripts/validate_handler_contracts.py --verbose
    python scripts/validate_handler_contracts.py --strict

Exit codes:
    0 - All handlers validated successfully
    1 - Validation failures detected

Note:
    To make this script executable directly, run:
        chmod +x scripts/validate_handler_contracts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

# =============================================================================
# Expected Handlers from _KNOWN_HANDLERS
# =============================================================================
# These must all have valid contract.yaml files before _KNOWN_HANDLERS can be
# deleted from util_wiring.py

EXPECTED_HANDLERS: dict[str, str] = {
    "consul": "HashiCorp Consul service discovery handler",
    "db": "PostgreSQL database handler",
    "graph": "Graph database (Memgraph/Neo4j) handler",
    "http": "HTTP REST protocol handler",
    "intent": "Intent storage and query handler for demo",
    "mcp": "Model Context Protocol handler for AI agents",
    "vault": "HashiCorp Vault secret management handler",
}

# Contract locations (relative to src/omnibase_infra/)
HANDLER_CONTRACT_PATHS: dict[str, Path] = {
    "consul": Path("nodes/handlers/consul/contract.yaml"),
    "db": Path("nodes/handlers/db/contract.yaml"),
    "graph": Path("nodes/handlers/graph/contract.yaml"),
    "http": Path("nodes/handlers/http/contract.yaml"),
    "intent": Path("nodes/handlers/intent/contract.yaml"),
    "mcp": Path("nodes/handlers/mcp/contract.yaml"),
    "vault": Path("nodes/handlers/vault/contract.yaml"),
}

# =============================================================================
# Validation Functions
# =============================================================================


def validate_contract(
    handler_type: str,
    contract_path: Path,
    verbose: bool = False,
    strict: bool = False,
) -> list[str]:
    """Validate a single handler contract.

    Checks for:
    - Contract file existence
    - Valid YAML syntax
    - Required fields (name, node_type, contract_version)
    - handler_routing section with valid handlers
    - operation_bindings validation (if present and using loader)

    Args:
        handler_type: The handler type identifier (e.g., "consul", "db").
        contract_path: Path to the contract.yaml file.
        verbose: If True, print additional diagnostic information.
        strict: If True, apply stricter validation rules.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Check file existence
    if not contract_path.exists():
        errors.append(f"Contract file not found: {contract_path}")
        return errors

    # Load YAML
    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_type = type(e).__name__
        errors.append(f"YAML parse error ({error_type}): check YAML syntax")
        return errors

    if not contract:
        errors.append("Contract is empty")
        return errors

    # Check required fields
    if "name" not in contract:
        errors.append("Missing 'name' field")

    if "node_type" not in contract:
        errors.append("Missing 'node_type' field")
    elif contract["node_type"] != "EFFECT_GENERIC":
        errors.append(
            f"Expected node_type 'EFFECT_GENERIC', got '{contract['node_type']}'"
        )

    if "contract_version" not in contract:
        errors.append("Missing 'contract_version' field")
    else:
        version = contract["contract_version"]
        if isinstance(version, dict):
            if "major" not in version:
                errors.append("contract_version missing 'major' field")
            if "minor" not in version:
                errors.append("contract_version missing 'minor' field")
            if "patch" not in version:
                errors.append("contract_version missing 'patch' field")

    # Check handler_routing section
    handler_routing = contract.get("handler_routing", {})
    if not handler_routing:
        errors.append("Missing 'handler_routing' section")
    else:
        handlers = handler_routing.get("handlers", [])
        if not handlers:
            errors.append("No handlers defined in handler_routing")
        else:
            # Check at least one handler has the expected type
            handler_types = [h.get("handler_type") for h in handlers]
            if handler_type not in handler_types:
                # Also check handler_key field (alternative format)
                handler_keys = [h.get("handler_key") for h in handlers]
                if handler_type not in handler_keys:
                    errors.append(
                        f"Expected handler_type or handler_key '{handler_type}' "
                        f"not found in handlers"
                    )

            # Check supported_operations for each handler
            for handler_entry in handlers:
                h_type = handler_entry.get("handler_type") or handler_entry.get(
                    "handler_key"
                )
                ops = handler_entry.get("supported_operations", [])
                if not ops:
                    errors.append(f"Handler '{h_type}' has no supported_operations")

        # Check routing_strategy (optional but validate if present)
        routing_strategy = handler_routing.get("routing_strategy")
        if routing_strategy and routing_strategy not in {
            "payload_type_match",
            "first_match",
            "all_match",
        }:
            if strict:
                errors.append(f"Unknown routing_strategy: '{routing_strategy}'")

    # Check operation_bindings (optional but validate if present)
    operation_bindings = contract.get("operation_bindings")
    if operation_bindings:
        try:
            from omnibase_infra.runtime.contract_loaders import (
                load_operation_bindings_subcontract,
            )

            # Use the loader to validate the bindings
            load_operation_bindings_subcontract(contract_path)
            if verbose:
                print("    operation_bindings: validated successfully")
        except ImportError:
            # Loader not available - skip advanced validation
            if verbose:
                print("    operation_bindings: skipped (loader not available)")
        except Exception as e:
            errors.append(f"operation_bindings validation failed: {e}")

    # Check io_operations (optional but validate structure if present)
    io_operations = contract.get("io_operations", [])
    if io_operations and verbose:
        print(f"    io_operations: {len(io_operations)} operations defined")

    return errors


def validate_all_contracts(
    verbose: bool = False,
    strict: bool = False,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Validate contracts for all expected handlers.

    Args:
        verbose: If True, print additional diagnostic information.
        strict: If True, apply stricter validation rules.

    Returns:
        Tuple of (validated_count, failed_count, list of (handler_type, error)).
    """
    base_path = Path(__file__).parent.parent / "src" / "omnibase_infra"

    total_errors: list[tuple[str, str]] = []
    validated = 0
    failed = 0

    for handler_type, description in EXPECTED_HANDLERS.items():
        contract_rel_path = HANDLER_CONTRACT_PATHS[handler_type]
        contract_path = base_path / contract_rel_path

        print(f"Validating: {handler_type} ({description})")
        print(f"  Path: {contract_rel_path}")

        errors = validate_contract(
            handler_type, contract_path, verbose=verbose, strict=strict
        )

        if errors:
            print("  Status: FAILED")
            for error in errors:
                print(f"    - {error}")
                total_errors.append((handler_type, error))
            failed += 1
        else:
            print("  Status: OK")
            validated += 1

        print()

    return validated, failed, total_errors


def main() -> int:
    """Run validation for all handler contracts."""
    parser = argparse.ArgumentParser(
        description="Validate handler contracts before migration from _KNOWN_HANDLERS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/validate_handler_contracts.py
    python scripts/validate_handler_contracts.py --verbose
    python scripts/validate_handler_contracts.py --strict

Exit codes:
    0 - All handlers validated successfully
    1 - Validation failures detected

This script validates that all handlers in _KNOWN_HANDLERS have valid
contract.yaml files. Run this BEFORE deleting _KNOWN_HANDLERS to ensure
no handlers are orphaned during migration.
""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print additional diagnostic information",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Apply stricter validation rules",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Handler Contract Validation")
    print("=" * 60)
    print()
    print(f"Validating {len(EXPECTED_HANDLERS)} handlers from _KNOWN_HANDLERS")
    print()

    validated, failed, total_errors = validate_all_contracts(
        verbose=args.verbose,
        strict=args.strict,
    )

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Validated: {validated}/{len(EXPECTED_HANDLERS)}")
    print(f"  Failed: {failed}/{len(EXPECTED_HANDLERS)}")

    if total_errors:
        print()
        print("Errors:")
        for handler_type, error in total_errors:
            print(f"  [{handler_type}] {error}")
        print()
        print("VALIDATION FAILED - Do NOT delete _KNOWN_HANDLERS yet!")
        print()
        print("Next steps:")
        print("  1. Create missing contract.yaml files for failed handlers")
        print("  2. Re-run this validation script")
        print("  3. Once all handlers pass, proceed with migration")
        return 1

    print()
    print("ALL HANDLERS VALIDATED - Safe to proceed with migration")
    print()
    print("Next steps:")
    print("  1. Remove _KNOWN_HANDLERS dict from util_wiring.py")
    print("  2. Update wire_default_handlers() to use contract-driven loading")
    print("  3. Run full test suite to verify migration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
