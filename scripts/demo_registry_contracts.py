#!/usr/bin/env -S poetry run python
"""Demo script for Registry-Based Contract Discovery (OMN-1100).

This script demonstrates contract materialization from Consul KV:
1. Stores sample handler contracts in Consul
2. Discovers them using RegistryContractSource
3. Shows the materialized contracts

Usage:
    # Direct execution (uses shebang with poetry run - preferred)
    ./scripts/demo_registry_contracts.py

    # Explicit poetry run (equivalent, works without executable bit)
    poetry run python scripts/demo_registry_contracts.py

    # List existing contracts only
    ./scripts/demo_registry_contracts.py --list-only

    # Clean up (remove demo contracts)
    ./scripts/demo_registry_contracts.py --cleanup

Environment:
    Configure via CONSUL_HOST, CONSUL_PORT, CONSUL_TOKEN, CONSUL_SCHEME env vars.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Fallback path injection for edge cases where poetry environment isn't active.
# Preferred: use shebang (#!/usr/bin/env -S poetry run python) or explicit poetry run.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnibase_infra.runtime import (
    DEFAULT_CONTRACT_PREFIX,
    RegistryContractSource,
    delete_contract_from_consul,
    list_contracts_in_consul,
    store_contract_in_consul,
)

# Sample contracts for demo
# NOTE: handler_class is stored in metadata (not a core contract field)
SAMPLE_CONTRACTS = {
    "effect.filesystem.handler": """
handler_id: effect.filesystem.handler
name: Filesystem Handler
contract_version:
  major: 1
  minor: 0
  patch: 0
description: Filesystem operations handler for file I/O
input_model: omnibase_infra.models.types.JsonDict
output_model: omnibase_core.models.dispatch.ModelHandlerOutput
descriptor:
  handler_kind: effect
metadata:
  handler_class: omnibase_infra.handlers.HandlerFilesystem
tags:
  - infrastructure
  - filesystem
  - io
""",
    "compute.auth.validator": """
handler_id: compute.auth.validator
name: Auth Token Validator
contract_version:
  major: 2
  minor: 1
  patch: 0
description: Validates authentication tokens and extracts claims
input_model: omnibase_infra.models.auth.ModelAuthRequest
output_model: omnibase_infra.models.auth.ModelAuthResponse
descriptor:
  handler_kind: compute
metadata:
  handler_class: omnibase_infra.handlers.compute.HandlerAuthValidator
tags:
  - security
  - auth
  - compute
""",
    "effect.http.client": """
handler_id: effect.http.client
name: HTTP Client Handler
contract_version:
  major: 1
  minor: 2
  patch: 0
description: HTTP/REST client for external API calls
input_model: omnibase_infra.models.http.ModelHttpRequest
output_model: omnibase_infra.models.http.ModelHttpResponse
descriptor:
  handler_kind: effect
metadata:
  handler_class: omnibase_infra.handlers.handler_http_rest.HandlerHttpRest
tags:
  - infrastructure
  - http
  - external
""",
}


def store_demo_contracts() -> int:
    """Store sample contracts in Consul KV (uses env vars for connection)."""
    host = os.environ.get("CONSUL_HOST", "localhost")
    port = os.environ.get("CONSUL_PORT", "8500")
    print(f"\n[STORE] Storing {len(SAMPLE_CONTRACTS)} contracts in Consul...")
    print(f"        Target: {host}:{port}")
    print(f"        Prefix: {DEFAULT_CONTRACT_PREFIX}")
    print()

    success_count = 0
    for handler_id, yaml_content in SAMPLE_CONTRACTS.items():
        success = store_contract_in_consul(
            contract_yaml=yaml_content.strip(),
            handler_id=handler_id,
        )
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {handler_id}")
        if success:
            success_count += 1

    return success_count


async def discover_contracts() -> None:
    """Discover and display contracts from Consul KV (uses env vars for connection)."""
    host = os.environ.get("CONSUL_HOST", "localhost")
    port = os.environ.get("CONSUL_PORT", "8500")
    print("\n[DISCOVER] Discovering contracts from Consul...")
    print(f"           Source: consul://{host}:{port}")
    print()

    source = RegistryContractSource(
        graceful_mode=True,  # Continue on errors
    )

    result = await source.discover_handlers()

    print(f"  Source Type: {source.source_type}")
    print(f"  Discovered:  {len(result.descriptors)} handlers")
    print(f"  Errors:      {len(result.validation_errors)}")
    print()

    if result.descriptors:
        print("  Materialized Contracts:")
        print("  " + "-" * 60)
        for desc in result.descriptors:
            print(f"  Handler ID:    {desc.handler_id}")
            print(f"  Name:          {desc.name}")
            print(f"  Version:       {desc.version}")
            print(f"  Kind:          {desc.handler_kind}")
            print(f"  Handler Class: {desc.handler_class}")
            print(f"  Contract Path: {desc.contract_path}")
            print("  " + "-" * 60)

    if result.validation_errors:
        print("\n  Validation Errors:")
        for err in result.validation_errors:
            print(f"    [{err.rule_id}] {err.file_path or 'unknown'}")
            print(f"      {err.message}")


def list_existing_contracts() -> None:
    """List existing contracts in Consul KV (uses env vars for connection)."""
    host = os.environ.get("CONSUL_HOST", "localhost")
    port = os.environ.get("CONSUL_PORT", "8500")
    print("\n[LIST] Existing contracts in Consul...")
    print(f"       Source: consul://{host}:{port}")
    print()

    handler_ids = list_contracts_in_consul()

    if handler_ids:
        print(f"  Found {len(handler_ids)} contracts:")
        for hid in handler_ids:
            print(f"    - {hid}")
    else:
        print("  No contracts found.")


def cleanup_demo_contracts() -> None:
    """Remove demo contracts from Consul KV (uses env vars for connection)."""
    print("\n[CLEANUP] Removing demo contracts from Consul...")

    for handler_id in SAMPLE_CONTRACTS:
        success = delete_contract_from_consul(handler_id)
        status = "OK" if success else "FAILED"
        print(f"  [{status}] Deleted: {handler_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo registry-based contract discovery. "
        "Configure via CONSUL_HOST, CONSUL_PORT, CONSUL_TOKEN, CONSUL_SCHEME env vars."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list existing contracts",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove demo contracts",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" OMN-1100: Registry-Based Contract Discovery Demo")
    print("=" * 70)

    if args.cleanup:
        cleanup_demo_contracts()
        return

    if args.list_only:
        list_existing_contracts()
        return

    # Full demo: store, list, and discover
    stored = store_demo_contracts()

    if stored > 0:
        list_existing_contracts()
        asyncio.run(discover_contracts())
    else:
        print("\n[ERROR] No contracts were stored. Check Consul connection.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" Demo complete!")
    print(" Run with --cleanup to remove demo contracts")
    print("=" * 70)


if __name__ == "__main__":
    main()
