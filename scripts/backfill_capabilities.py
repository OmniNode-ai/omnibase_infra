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
    >>> # {"postgres": true, "read": true, "write": true,
    >>> #  "config": {"contract_type": "effect"}}
    >>> # Extracts:
    >>> #   contract_type: "effect" (from config.contract_type or node_type)
    >>> #   capability_tags: ["postgres", "read", "write"] (from boolean true fields)
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import sys
from uuid import UUID

import asyncpg


class ConfigurationError(Exception):
    """Raised when environment configuration is invalid."""


def _validate_hostname(value: str) -> str:
    """Validate hostname or IP address format.

    Args:
        value: The hostname or IP address to validate

    Returns:
        The validated value

    Raises:
        ConfigurationError: If the value is not a valid hostname or IP
    """
    # Try to parse as IP address first
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass

    # Validate as hostname (RFC 1123)
    # - Max 253 characters total
    # - Labels separated by dots, each 1-63 chars
    # - Labels contain only alphanumerics and hyphens
    # - Labels cannot start or end with hyphen
    if len(value) > 253:
        raise ConfigurationError("POSTGRES_HOST: hostname exceeds 253 characters")

    hostname_pattern = re.compile(
        r"^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(?:\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*$"
    )
    if not hostname_pattern.match(value):
        raise ConfigurationError(
            "POSTGRES_HOST: invalid hostname format (must be valid hostname or IP)"
        )

    return value


def _validate_port(value: str) -> int:
    """Validate port number.

    Args:
        value: The port string to validate

    Returns:
        The validated port as integer

    Raises:
        ConfigurationError: If the value is not a valid port number
    """
    try:
        port = int(value)
    except ValueError:
        raise ConfigurationError("POSTGRES_PORT: must be a valid integer")

    if not 1 <= port <= 65535:
        raise ConfigurationError("POSTGRES_PORT: must be between 1 and 65535")

    return port


def _validate_identifier(value: str, name: str) -> str:
    """Validate database identifier (user or database name).

    Args:
        value: The identifier to validate
        name: The name of the parameter (for error messages)

    Returns:
        The validated value

    Raises:
        ConfigurationError: If the value contains invalid characters
    """
    # PostgreSQL identifiers: alphanumerics, underscores, max 63 chars
    # First character must be letter or underscore
    if not value:
        raise ConfigurationError(f"{name}: cannot be empty")

    if len(value) > 63:
        raise ConfigurationError(f"{name}: exceeds 63 characters")

    identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if not identifier_pattern.match(value):
        raise ConfigurationError(
            f"{name}: invalid format (must start with letter or underscore, "
            "contain only alphanumerics and underscores)"
        )

    return value


def _get_validated_config() -> dict[str, str | int]:
    """Get and validate database connection configuration from environment.

    Returns:
        Dictionary with validated connection parameters

    Raises:
        ConfigurationError: If any configuration is invalid
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port_str = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    password = os.getenv("POSTGRES_PASSWORD", "")

    return {
        "host": _validate_hostname(host),
        "port": _validate_port(port_str),
        "user": _validate_identifier(user, "POSTGRES_USER"),
        "database": _validate_identifier(database, "POSTGRES_DATABASE"),
        "password": password,
    }


async def get_connection() -> asyncpg.Connection:
    """Create database connection from validated environment variables.

    Returns:
        Asyncpg connection object

    Raises:
        ConfigurationError: If environment configuration is invalid
        asyncpg.PostgresError: If connection fails
    """
    config = _get_validated_config()

    # Use explicit parameters instead of DSN string for safer construction
    return await asyncpg.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        database=config["database"],
        password=config["password"],
    )


def extract_capability_tags(capabilities: dict[str, object]) -> list[str]:
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


def extract_contract_type(
    capabilities: dict[str, object], node_type: str
) -> str | None:
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


def extract_protocols(capabilities: dict[str, object]) -> list[str]:
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


def extract_intent_types(capabilities: dict[str, object]) -> list[str]:
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


def extract_contract_version(capabilities: dict[str, object]) -> str | None:
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

    All updates are wrapped in a transaction for atomicity - either all
    records are updated or none are (in case of failure).

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

        if dry_run:
            # Dry run: just show what would be done
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

                print(
                    f"Would update {entity_id} ({domain}):\n"
                    f"  contract_type: {contract_type}\n"
                    f"  intent_types: {intent_types}\n"
                    f"  protocols: {protocols}\n"
                    f"  capability_tags: {capability_tags}\n"
                    f"  contract_version: {contract_version}"
                )
                updated += 1

                if updated % 100 == 0:
                    print(f"Would update {updated} records...")
        else:
            # Execute updates within a transaction for atomicity
            async with conn.transaction():
                for row in rows:
                    entity_id = row["entity_id"]
                    domain = row["domain"]
                    node_type = row["node_type"]
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
    except ConfigurationError as e:
        # Configuration errors are safe to display - they don't contain secrets
        print(f"ERROR: Configuration invalid - {e}")
        return 1
    except asyncpg.PostgresError:
        # Database errors may contain sensitive info - use generic message
        print("ERROR: Database connection or query failed")
        return 1
    except Exception:
        # Generic errors - don't expose details that might leak sensitive info
        print("ERROR: An unexpected error occurred during backfill")
        return 1


if __name__ == "__main__":
    sys.exit(main())
