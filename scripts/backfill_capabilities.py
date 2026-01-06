#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Backfill Capability Fields from Existing Capabilities JSONB.

This script populates the new capability fields (contract_type, intent_types,
protocols, capability_tags, contract_version) from the existing capabilities
JSONB column for all registration projections.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities

Usage:
    # Dry run (shows what would be updated)
    python scripts/backfill_capabilities.py --dry-run

    # Execute backfill
    python scripts/backfill_capabilities.py

    # With custom connection
    POSTGRES_HOST=localhost POSTGRES_PORT=5432 python scripts/backfill_capabilities.py

Environment Variables:
    POSTGRES_HOST: Database host (default: localhost)
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (required)

Example:
    >>> # From capabilities JSONB:
    >>> # {"postgres": true, "read": true, "write": true, "config": {"contract_type": "effect"}}
    >>> # Extracts:
    >>> #   contract_type: "effect" (from config.contract_type or node_type)
    >>> #   capability_tags: ["postgres", "read", "write"] (from boolean true fields)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any
from uuid import UUID

import asyncpg


async def get_connection() -> asyncpg.Connection:
    """Create database connection from environment variables."""
    dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{os.getenv('POSTGRES_PASSWORD', '')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DATABASE', 'omninode_bridge')}"
    )
    return await asyncpg.connect(dsn)


def extract_capability_tags(capabilities: dict[str, Any]) -> list[str]:
    """Extract capability tags from capabilities dict.

    Converts boolean capability flags to string tags.

    Args:
        capabilities: The capabilities JSONB dict

    Returns:
        List of capability tag strings

    Example:
        >>> extract_capability_tags({"postgres": True, "read": True, "write": False})
        ['postgres', 'read']
    """
    tags = []
    # Known boolean capability flags
    bool_fields = [
        "postgres",
        "read",
        "write",
        "database",
        "transactions",
        "processing",
        "routing",
        "feature",
    ]
    for field in bool_fields:
        if capabilities.get(field) is True:
            tags.append(field)

    # Add any custom capability tags from config
    config = capabilities.get("config", {})
    if isinstance(config, dict):
        if "capability_tags" in config and isinstance(config["capability_tags"], list):
            tags.extend(config["capability_tags"])

    return list(set(tags))  # Deduplicate


def extract_contract_type(capabilities: dict[str, Any], node_type: str) -> str | None:
    """Extract contract type from capabilities or fallback to node_type.

    Args:
        capabilities: The capabilities JSONB dict
        node_type: The node_type column value

    Returns:
        Contract type string or None
    """
    config = capabilities.get("config", {})
    if isinstance(config, dict) and "contract_type" in config:
        return str(config["contract_type"])

    # Fallback to node_type if it's a valid contract type
    if node_type in ("effect", "compute", "reducer", "orchestrator"):
        return node_type

    return None


def extract_protocols(capabilities: dict[str, Any]) -> list[str]:
    """Extract protocol list from capabilities.

    Args:
        capabilities: The capabilities JSONB dict

    Returns:
        List of protocol names
    """
    config = capabilities.get("config", {})
    if isinstance(config, dict) and "protocols" in config:
        protocols = config["protocols"]
        if isinstance(protocols, list):
            return [str(p) for p in protocols]
    return []


def extract_intent_types(capabilities: dict[str, Any]) -> list[str]:
    """Extract intent types from capabilities.

    Args:
        capabilities: The capabilities JSONB dict

    Returns:
        List of intent type strings
    """
    config = capabilities.get("config", {})
    if isinstance(config, dict) and "intent_types" in config:
        intent_types = config["intent_types"]
        if isinstance(intent_types, list):
            return [str(it) for it in intent_types]
    return []


def extract_contract_version(capabilities: dict[str, Any]) -> str | None:
    """Extract contract version from capabilities.

    Args:
        capabilities: The capabilities JSONB dict

    Returns:
        Contract version string or None
    """
    config = capabilities.get("config", {})
    if isinstance(config, dict) and "contract_version" in config:
        return str(config["contract_version"])
    return None


async def backfill(dry_run: bool = False) -> int:
    """Backfill capability fields from existing capabilities JSONB.

    Args:
        dry_run: If True, only print what would be done

    Returns:
        Number of records updated
    """
    conn = await get_connection()
    try:
        # Fetch all registrations with capabilities
        rows = await conn.fetch(
            """
            SELECT entity_id, domain, node_type, capabilities
            FROM registration_projections
            WHERE contract_type IS NULL
               OR intent_types = ARRAY[]::TEXT[]
               OR protocols = ARRAY[]::TEXT[]
               OR capability_tags = ARRAY[]::TEXT[]
            """
        )

        print(f"Found {len(rows)} registrations to process")

        updated = 0
        for row in rows:
            entity_id: UUID = row["entity_id"]
            domain: str = row["domain"]
            node_type: str = row["node_type"]
            capabilities_raw = row["capabilities"]

            # Parse capabilities JSONB
            if isinstance(capabilities_raw, str):
                capabilities = json.loads(capabilities_raw)
            elif isinstance(capabilities_raw, dict):
                capabilities = capabilities_raw
            else:
                capabilities = {}

            # Extract fields
            contract_type = extract_contract_type(capabilities, node_type)
            intent_types = extract_intent_types(capabilities)
            protocols = extract_protocols(capabilities)
            capability_tags = extract_capability_tags(capabilities)
            contract_version = extract_contract_version(capabilities)

            if dry_run:
                print(
                    f"Would update {entity_id} ({domain}):\n"
                    f"  contract_type: {contract_type}\n"
                    f"  intent_types: {intent_types}\n"
                    f"  protocols: {protocols}\n"
                    f"  capability_tags: {capability_tags}\n"
                    f"  contract_version: {contract_version}"
                )
            else:
                await conn.execute(
                    """
                    UPDATE registration_projections
                    SET contract_type = $3,
                        intent_types = $4,
                        protocols = $5,
                        capability_tags = $6,
                        contract_version = $7
                    WHERE entity_id = $1 AND domain = $2
                    """,
                    entity_id,
                    domain,
                    contract_type,
                    intent_types,
                    protocols,
                    capability_tags,
                    contract_version,
                )
                updated += 1

                if updated % 100 == 0:
                    print(f"Updated {updated} records...")

        print(f"{'Would update' if dry_run else 'Updated'} {updated} registrations")
        return updated

    finally:
        await conn.close()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill capability fields from existing capabilities JSONB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()

    if not os.getenv("POSTGRES_PASSWORD"):
        print("ERROR: POSTGRES_PASSWORD environment variable is required")
        return 1

    try:
        updated = asyncio.run(backfill(dry_run=args.dry_run))
        return 0 if updated >= 0 else 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
