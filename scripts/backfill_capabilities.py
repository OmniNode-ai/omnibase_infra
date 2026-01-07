#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Backfill Capability Fields from Existing Capabilities JSONB.

This script populates the new capability fields (contract_type, intent_types,
protocols, capability_tags, contract_version) from the existing capabilities
JSONB column for all registration projections.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities

Idempotency Strategy:
    This script is idempotent and safe to run multiple times. The idempotency
    mechanism uses `contract_type IS NULL` as the SOLE indicator that a record
    needs processing.

    Why contract_type IS NULL (not array fields)?
        The migration (008_add_capability_fields.py) creates GIN indexes on 4
        columns for efficient querying:
        - capability_tags (GIN index for array containment queries)
        - intent_types (GIN index for array containment queries)
        - protocols (GIN index for array containment queries)
        - contract_type + current_state (B-tree composite index)

        However, these indexes are for QUERY PERFORMANCE, not for backfill
        tracking. Empty arrays (capability_tags = '{}') are valid states for
        nodes that simply don't have those capabilities - they do NOT indicate
        "needs processing".

        The contract_type column is the correct idempotency marker because:
        - It is ALWAYS set to a non-NULL value after processing
        - Even records with no determinable type get 'unknown' as a marker
        - NULL contract_type unambiguously means "never processed"

    Once processed:
        - contract_type is ALWAYS set to a non-NULL value:
          - Extracted from capabilities.config.contract_type, OR
          - Derived from node_type (if valid: effect/compute/reducer/orchestrator), OR
          - Set to 'unknown' as a fallback marker
        - The record will NOT be selected on subsequent runs
        - Running again will only process newly inserted (unprocessed) records
        - Array fields (intent_types, protocols, capability_tags) may be empty
          but this does NOT indicate "needs processing"

Usage:
    # Dry run (shows what would be updated)
    python scripts/backfill_capabilities.py --dry-run

    # Execute backfill
    python scripts/backfill_capabilities.py

    # With custom batch size (default: 1000)
    python scripts/backfill_capabilities.py --batch-size 500

    # With custom connection
    POSTGRES_HOST=localhost POSTGRES_PORT=5432 python scripts/backfill_capabilities.py

    # Enable debug logging (for troubleshooting)
    BACKFILL_DEBUG=1 python scripts/backfill_capabilities.py

Environment Variables:
    POSTGRES_HOST: Database host (default: localhost)
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (required)
    BACKFILL_DEBUG: Enable debug logging to stderr (optional)

Error Codes:
    The script uses error codes for debugging and actionable error messages:

    Configuration Errors (CFG_*):
        CFG_AUTH_001: Missing POSTGRES_PASSWORD
        CFG_HOST_001: Invalid POSTGRES_HOST format
        CFG_PORT_001: Invalid POSTGRES_PORT value
        CFG_USER_001: Invalid POSTGRES_USER format
        CFG_DB_001: Invalid POSTGRES_DATABASE format

    Database Errors (DB_*):
        DB_CONN_001: Connection refused (host/port unreachable)
        DB_AUTH_001: Authentication failed (invalid credentials)
        DB_NOTFOUND_001: Database not found
        DB_TIMEOUT_001: Connection timeout
        DB_QUERY_001: Query execution failed
        DB_ERR_001: Generic database error

    Internal Errors (INT_*):
        INT_ERR_001: Unexpected internal error

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
import logging
import os
import re
import sys
from typing import NoReturn
from uuid import UUID

import asyncpg

# Configure logging - controlled by BACKFILL_DEBUG environment variable
# When enabled, logs to stderr with detailed information (never secrets)
_log_level = logging.DEBUG if os.getenv("BACKFILL_DEBUG") else logging.WARNING
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# Error codes for categorization and debugging
class ErrorCode:
    """Error codes for actionable error messages.

    Format: <CATEGORY>_<SUBCATEGORY>_<NUMBER>
    Categories:
        CFG: Configuration errors
        DB: Database errors
        INT: Internal/unexpected errors
    """

    # Configuration errors (CFG_xxx_xxx)
    CFG_MISSING_PASSWORD = "CFG_AUTH_001"
    CFG_INVALID_HOST = "CFG_HOST_001"
    CFG_INVALID_PORT = "CFG_PORT_001"
    CFG_INVALID_USER = "CFG_USER_001"
    CFG_INVALID_DATABASE = "CFG_DB_001"

    # Database errors (DB_xxx_xxx)
    DB_CONNECTION_REFUSED = "DB_CONN_001"
    DB_AUTH_FAILED = "DB_AUTH_001"
    DB_NOT_FOUND = "DB_NOTFOUND_001"
    DB_TIMEOUT = "DB_TIMEOUT_001"
    DB_QUERY_FAILED = "DB_QUERY_001"
    DB_GENERIC = "DB_ERR_001"

    # Internal errors (INT_xxx_xxx)
    INT_UNEXPECTED = "INT_ERR_001"


class ConfigurationError(Exception):
    """Raised when environment configuration is invalid.

    Attributes:
        error_code: Categorized error code for debugging
        message: Human-readable error message
    """

    def __init__(self, message: str, error_code: str = "CFG_ERR_001") -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


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
        raise ConfigurationError(
            "POSTGRES_HOST: hostname exceeds 253 characters",
            error_code=ErrorCode.CFG_INVALID_HOST,
        )

    hostname_pattern = re.compile(
        r"^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(?:\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*$"
    )
    if not hostname_pattern.match(value):
        raise ConfigurationError(
            "POSTGRES_HOST: invalid hostname format (must be valid hostname or IP). "
            "Check the POSTGRES_HOST environment variable.",
            error_code=ErrorCode.CFG_INVALID_HOST,
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
        raise ConfigurationError(
            "POSTGRES_PORT: must be a valid integer. "
            "Check the POSTGRES_PORT environment variable.",
            error_code=ErrorCode.CFG_INVALID_PORT,
        )

    if not 1 <= port <= 65535:
        raise ConfigurationError(
            "POSTGRES_PORT: must be between 1 and 65535. "
            "Check the POSTGRES_PORT environment variable.",
            error_code=ErrorCode.CFG_INVALID_PORT,
        )

    return port


def _validate_identifier(value: str, name: str, error_code: str) -> str:
    """Validate database identifier (user or database name).

    Args:
        value: The identifier to validate
        name: The name of the parameter (for error messages)
        error_code: Error code to use for validation failures

    Returns:
        The validated value

    Raises:
        ConfigurationError: If the value contains invalid characters
    """
    # PostgreSQL identifiers: alphanumerics, underscores, max 63 chars
    # First character must be letter or underscore
    if not value:
        raise ConfigurationError(
            f"{name}: cannot be empty. Check the {name} environment variable.",
            error_code=error_code,
        )

    if len(value) > 63:
        raise ConfigurationError(
            f"{name}: exceeds 63 characters. Check the {name} environment variable.",
            error_code=error_code,
        )

    identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if not identifier_pattern.match(value):
        raise ConfigurationError(
            f"{name}: invalid format (must start with letter or underscore, "
            f"contain only alphanumerics and underscores). "
            f"Check the {name} environment variable.",
            error_code=error_code,
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

    logger.debug(
        "Validating configuration (host=%s, port=%s, user=%s, database=%s)",
        host,
        port_str,
        user,
        database,
    )

    return {
        "host": _validate_hostname(host),
        "port": _validate_port(port_str),
        "user": _validate_identifier(user, "POSTGRES_USER", ErrorCode.CFG_INVALID_USER),
        "database": _validate_identifier(
            database, "POSTGRES_DATABASE", ErrorCode.CFG_INVALID_DATABASE
        ),
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

    logger.debug(
        "Attempting database connection to %s:%s/%s",
        config["host"],
        config["port"],
        config["database"],
    )

    # Use explicit parameters instead of DSN string for safer construction
    return await asyncpg.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        database=config["database"],
        password=config["password"],
        timeout=30.0,  # 30 second connection timeout
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
            tags.extend(str(tag) for tag in config["capability_tags"])

    # Sorted for deterministic output across runs (ensures idempotency in tests
    # and consistent ordering regardless of dict iteration order)
    return sorted(set(tags))


def extract_contract_type(capabilities: dict[str, object], node_type: str) -> str:
    """Extract contract type from capabilities or fallback to node_type.

    This function ALWAYS returns a non-NULL value to ensure idempotency.
    The fallback chain is:
    1. capabilities.config.contract_type (if present)
    2. node_type (if valid: effect/compute/reducer/orchestrator)
    3. 'unknown' (marker for records without determinable type)

    Args:
        capabilities: The capabilities JSONB dict
        node_type: The node_type column value

    Returns:
        Contract type string (never None - 'unknown' is the final fallback)
    """
    config = capabilities.get("config", {})
    if isinstance(config, dict) and "contract_type" in config:
        return str(config["contract_type"])

    # Fallback to node_type if it's a valid contract type
    if node_type in ("effect", "compute", "reducer", "orchestrator"):
        return node_type

    # Final fallback: 'unknown' marker ensures idempotency
    # (record will not be re-selected on subsequent runs)
    return "unknown"


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


def _handle_database_error(exc: BaseException) -> NoReturn:
    """Handle database errors with actionable messages.

    Categorizes asyncpg exceptions and provides actionable error messages
    without exposing sensitive information like credentials or full
    connection strings.

    Args:
        exc: The asyncpg exception to handle

    Raises:
        SystemExit: Always exits with code 1 after printing error
    """
    # Log detailed error for debugging (only visible with BACKFILL_DEBUG)
    logger.debug("Database error type: %s", type(exc).__name__)
    logger.debug("Database error details: %s", str(exc))

    # Categorize the error and provide actionable guidance
    error_code: str
    message: str
    guidance: str

    if isinstance(exc, asyncpg.InvalidPasswordError):
        error_code = ErrorCode.DB_AUTH_FAILED
        message = "Database authentication failed"
        guidance = "Verify POSTGRES_PASSWORD is correct for the specified user."
    elif isinstance(exc, asyncpg.InvalidCatalogNameError):
        error_code = ErrorCode.DB_NOT_FOUND
        message = "Database not found"
        guidance = "Verify POSTGRES_DATABASE exists and is spelled correctly."
    elif isinstance(exc, asyncpg.CannotConnectNowError):
        error_code = ErrorCode.DB_CONNECTION_REFUSED
        message = "Database server not ready for connections"
        guidance = (
            "The database server is starting up or shutting down. "
            "Wait and retry, or check server status."
        )
    elif isinstance(exc, OSError | ConnectionRefusedError):
        # Connection refused at network level
        error_code = ErrorCode.DB_CONNECTION_REFUSED
        message = "Connection refused"
        guidance = (
            "Verify POSTGRES_HOST and POSTGRES_PORT are correct. "
            "Ensure the database server is running and accepting connections."
        )
    elif isinstance(exc, asyncpg.PostgresConnectionError):
        error_code = ErrorCode.DB_CONNECTION_REFUSED
        message = "Database connection failed"
        guidance = (
            "Verify POSTGRES_HOST and POSTGRES_PORT are correct. "
            "Check network connectivity and firewall rules."
        )
    elif isinstance(exc, asyncpg.InterfaceError):
        error_code = ErrorCode.DB_TIMEOUT
        message = "Database interface error"
        guidance = (
            "Connection may have timed out or been interrupted. Retry the operation."
        )
    elif isinstance(exc, asyncpg.PostgresSyntaxError):
        error_code = ErrorCode.DB_QUERY_FAILED
        message = "Query syntax error"
        guidance = (
            "This may indicate a schema mismatch. "
            "Ensure the database schema is up to date."
        )
    elif isinstance(exc, asyncpg.UndefinedTableError):
        error_code = ErrorCode.DB_QUERY_FAILED
        message = "Table 'registration_projections' not found"
        guidance = (
            "Run database migrations to create required tables. "
            "Check POSTGRES_DATABASE is the correct database."
        )
    elif isinstance(exc, asyncpg.UndefinedColumnError):
        error_code = ErrorCode.DB_QUERY_FAILED
        message = "Required column not found in table"
        guidance = (
            "Run database migrations to add required columns. "
            "The schema may be outdated."
        )
    else:
        error_code = ErrorCode.DB_GENERIC
        message = "Database operation failed"
        guidance = (
            "Check database connectivity and configuration. "
            "Enable BACKFILL_DEBUG=1 for detailed error logging."
        )

    print(f"ERROR [{error_code}]: {message}")
    print(f"  Action: {guidance}")
    sys.exit(1)


def _handle_unexpected_error(exc: Exception) -> NoReturn:
    """Handle unexpected errors with actionable messages.

    Args:
        exc: The unexpected exception to handle

    Raises:
        SystemExit: Always exits with code 1 after printing error
    """
    # Log detailed error for debugging (only visible with BACKFILL_DEBUG)
    logger.debug("Unexpected error type: %s", type(exc).__name__)
    logger.debug("Unexpected error details: %s", str(exc))
    logger.exception("Full traceback:")

    error_code = ErrorCode.INT_UNEXPECTED
    print(f"ERROR [{error_code}]: An unexpected error occurred during backfill")
    print("  Action: Enable BACKFILL_DEBUG=1 and check stderr for detailed logging.")
    print("  If the problem persists, report the error with the debug output.")
    sys.exit(1)


def _parse_capabilities(
    capabilities_raw: str | dict[str, object] | None,
) -> dict[str, object]:
    """Parse capabilities from raw database value.

    Args:
        capabilities_raw: Raw capabilities value from database (JSONB)

    Returns:
        Parsed capabilities dictionary (empty dict on parse failure)
    """
    if isinstance(capabilities_raw, str):
        try:
            result = json.loads(capabilities_raw)
            if isinstance(result, dict):
                return result
            logger.warning("Capabilities JSON is not a dict: %s", type(result).__name__)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse capabilities JSON: %s", e)
            return {}
    elif isinstance(capabilities_raw, dict):
        return capabilities_raw
    else:
        return {}


async def _get_total_count(conn: asyncpg.Connection) -> int:
    """Get total count of records needing processing.

    Args:
        conn: Database connection

    Returns:
        Total count of unprocessed records
    """
    result = await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM registration_projections
        WHERE contract_type IS NULL
        """
    )
    return int(result) if result else 0


async def _process_batch_dry_run(
    rows: list[asyncpg.Record],
    batch_num: int,
    total_batches: int,
    processed_so_far: int,
    total_rows: int,
) -> tuple[int, int]:
    """Process a batch of rows in dry-run mode.

    Args:
        rows: Batch of database records
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches
        processed_so_far: Count of records processed before this batch
        total_rows: Total records to process

    Returns:
        Tuple of (records_analyzed, unknown_count)
    """
    analyzed = 0
    unknown_count = 0

    for row in rows:
        entity_id: UUID = row["entity_id"]
        domain: str = row["domain"]
        # Explicit str() coercion for safety - database may return unexpected types
        node_type: str = str(row["node_type"]) if row["node_type"] else ""
        capabilities = _parse_capabilities(row["capabilities"])

        # Extract fields
        contract_type = extract_contract_type(capabilities, node_type)
        intent_types = extract_intent_types(capabilities)
        protocols = extract_protocols(capabilities)
        capability_tags = extract_capability_tags(capabilities)
        contract_version = extract_contract_version(capabilities)

        # Track records that will get 'unknown' as fallback
        if contract_type == "unknown":
            unknown_count += 1
            type_note = " (fallback - no type determinable)"
        else:
            type_note = ""

        print(
            f"Would update {entity_id} ({domain}):\n"
            f"  contract_type: {contract_type}{type_note}\n"
            f"  intent_types: {intent_types}\n"
            f"  protocols: {protocols}\n"
            f"  capability_tags: {capability_tags}\n"
            f"  contract_version: {contract_version}"
        )
        analyzed += 1

    current_total = processed_so_far + analyzed
    print(
        f"[Batch {batch_num}/{total_batches}] "
        f"Analyzed {current_total}/{total_rows} records"
    )

    return analyzed, unknown_count


async def _process_batch_update(
    conn: asyncpg.Connection,
    rows: list[asyncpg.Record],
    batch_num: int,
    total_batches: int,
    processed_so_far: int,
    total_rows: int,
) -> int:
    """Process a batch of rows and execute updates.

    Args:
        conn: Database connection
        rows: Batch of database records
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches
        processed_so_far: Count of records processed before this batch
        total_rows: Total records to process

    Returns:
        Number of records updated in this batch
    """
    updated = 0

    # Execute updates within a transaction for batch atomicity
    async with conn.transaction():
        for row in rows:
            entity_id = row["entity_id"]
            domain = row["domain"]
            # Explicit str() coercion for safety - database may return unexpected types
            node_type = str(row["node_type"]) if row["node_type"] else ""
            capabilities = _parse_capabilities(row["capabilities"])

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

    current_total = processed_so_far + updated
    print(
        f"[Batch {batch_num}/{total_batches}] "
        f"Updated {current_total}/{total_rows} records"
    )

    return updated


async def _fetch_batch_with_cursor(
    conn: asyncpg.Connection,
    batch_size: int,
    cursor_entity_id: UUID | None,
    cursor_domain: str | None,
) -> list[asyncpg.Record]:
    """Fetch a batch of records using cursor-based pagination.

    Uses keyset pagination with (entity_id, domain) composite cursor to avoid
    the LIMIT/OFFSET bug where records are skipped when the WHERE clause
    result set changes between batches (e.g., when contract_type is updated
    from NULL to non-NULL).

    Args:
        conn: Database connection
        batch_size: Maximum number of records to fetch
        cursor_entity_id: Last processed entity_id (None for first batch)
        cursor_domain: Last processed domain (None for first batch)

    Returns:
        List of database records for this batch
    """
    if cursor_entity_id is None:
        # First batch - no cursor, start from beginning
        logger.debug("Fetching first batch (limit=%d)", batch_size)
        return await conn.fetch(  # type: ignore[no-any-return]
            """
            SELECT entity_id, domain, node_type, capabilities
            FROM registration_projections
            WHERE contract_type IS NULL
            ORDER BY entity_id, domain
            LIMIT $1
            """,
            batch_size,
        )
    else:
        # Subsequent batches - use cursor to continue from last position
        logger.debug(
            "Fetching batch with cursor (entity_id > %s, domain > %s, limit=%d)",
            cursor_entity_id,
            cursor_domain,
            batch_size,
        )
        return await conn.fetch(  # type: ignore[no-any-return]
            """
            SELECT entity_id, domain, node_type, capabilities
            FROM registration_projections
            WHERE contract_type IS NULL
              AND (entity_id, domain) > ($2, $3)
            ORDER BY entity_id, domain
            LIMIT $1
            """,
            batch_size,
            cursor_entity_id,
            cursor_domain,
        )


async def backfill(dry_run: bool = False, batch_size: int = 1000) -> int:
    """Backfill capability fields from existing capabilities JSONB.

    Idempotency:
        This function is idempotent - running it multiple times is safe.
        Records are selected using `contract_type IS NULL` as the sole
        indicator of "needs processing". Once processed, contract_type
        is always set to a non-NULL value ('unknown' if no type can be
        determined), so the record will not be selected on subsequent runs.

    Batch Processing:
        For large datasets, records are processed in configurable batches
        to avoid loading all rows into memory at once. Each batch is
        wrapped in its own transaction for atomicity.

        Uses cursor-based (keyset) pagination instead of LIMIT/OFFSET to
        ensure no records are skipped when records are updated between
        batches. The cursor is the (entity_id, domain) composite key.

    Args:
        dry_run: If True, only print what would be done
        batch_size: Number of records to process per batch (default: 1000)

    Returns:
        Number of records updated
    """
    logger.info("Starting backfill (dry_run=%s, batch_size=%d)", dry_run, batch_size)

    conn = await get_connection()
    try:
        # First, get total count without loading all rows into memory
        total_rows = await _get_total_count(conn)
        print(f"Found {total_rows} registrations needing processing")
        logger.info("Found %d registrations to process", total_rows)

        if total_rows == 0:
            print("No unprocessed records found (script is idempotent)")
            return 0

        # Calculate total batches for progress reporting
        total_batches = (total_rows + batch_size - 1) // batch_size
        print(f"Processing in {total_batches} batch(es) of up to {batch_size} records")

        total_updated = 0
        total_unknown = 0
        batch_num = 0

        # Cursor-based pagination state
        # Track last processed (entity_id, domain) to avoid LIMIT/OFFSET bug
        cursor_entity_id: UUID | None = None
        cursor_domain: str | None = None

        # Process in batches using cursor-based (keyset) pagination
        # This avoids the LIMIT/OFFSET bug where records are skipped when
        # the WHERE clause result set changes between batches
        while True:
            batch_num += 1
            logger.debug(
                "Fetching batch %d (cursor_entity_id=%s, cursor_domain=%s)",
                batch_num,
                cursor_entity_id,
                cursor_domain,
            )

            rows = await _fetch_batch_with_cursor(
                conn, batch_size, cursor_entity_id, cursor_domain
            )

            if not rows:
                # No more rows to process
                break

            if dry_run:
                analyzed, unknown = await _process_batch_dry_run(
                    rows,
                    batch_num,
                    total_batches,
                    total_updated,
                    total_rows,
                )
                total_updated += analyzed
                total_unknown += unknown
            else:
                updated = await _process_batch_update(
                    conn,
                    rows,
                    batch_num,
                    total_batches,
                    total_updated,
                    total_rows,
                )
                total_updated += updated

            # Update cursor to last row in this batch for next iteration
            last_row = rows[-1]
            cursor_entity_id = last_row["entity_id"]
            cursor_domain = last_row["domain"]
            logger.debug(
                "Updated cursor to (entity_id=%s, domain=%s)",
                cursor_entity_id,
                cursor_domain,
            )

        # Final summary
        if dry_run:
            print(f"\nDry-run summary: Would update {total_updated} registrations")
            print(f"  Initial count (from query): {total_rows}")
            print(f"  Total records analyzed: {total_updated}")
            print(f"  Batches processed: {batch_num}")
            if total_unknown > 0:
                print(
                    f"  Note: {total_unknown} records will use 'unknown' as "
                    "contract_type (no type could be determined from capabilities "
                    "or node_type)"
                )
            print(
                "After backfill, these records will NOT be selected on subsequent runs"
            )
        else:
            logger.info("All batches committed successfully")
            print(f"\nBackfill complete: Updated {total_updated} registrations")
            print(f"  Total records processed: {total_updated}")
            print(f"  Batches committed: {batch_num}")

        return total_updated

    finally:
        logger.debug("Closing database connection")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of records to process per batch (default: 1000)",
    )
    args = parser.parse_args()

    # Validate batch size
    if args.batch_size < 1:
        print("ERROR: --batch-size must be at least 1")
        return 1
    if args.batch_size > 100000:
        print("WARNING: Large batch sizes (>100000) may cause memory issues")

    if not os.getenv("POSTGRES_PASSWORD"):
        print(
            f"ERROR [{ErrorCode.CFG_MISSING_PASSWORD}]: "
            "POSTGRES_PASSWORD environment variable is required"
        )
        print("  Action: Set POSTGRES_PASSWORD before running this script.")
        return 1

    try:
        updated = asyncio.run(
            backfill(dry_run=args.dry_run, batch_size=args.batch_size)
        )
        logger.info("Backfill completed successfully (updated=%d)", updated)
        return 0 if updated >= 0 else 1
    except ConfigurationError as e:
        # Configuration errors are safe to display - they don't contain secrets
        print(f"ERROR: Configuration invalid - {e}")
        print("  Action: Check the environment variable mentioned above.")
        return 1
    except asyncpg.PostgresError as exc:
        # Database errors - handle with specific actionable messages
        _handle_database_error(exc)
    except (OSError, ConnectionRefusedError) as exc:
        # Network-level connection errors
        _handle_database_error(exc)
    except Exception as exc:
        # Generic errors - log details but show safe message
        _handle_unexpected_error(exc)


if __name__ == "__main__":
    sys.exit(main())
