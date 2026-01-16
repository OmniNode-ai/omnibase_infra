# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for database performance tests.

This module provides fixtures for testing PostgreSQL query performance
using EXPLAIN ANALYZE to verify index usage and query efficiency.

PRODUCTION SAFETY WARNING:
    This module uses EXPLAIN ANALYZE which EXECUTES the query to get actual
    timing and row counts. While this is safe for SELECT queries (no mutations),
    running performance tests against production databases is NOT recommended:

    1. EXPLAIN ANALYZE can add load to production databases
    2. Test data seeding modifies the database (creates/deletes test records)
    3. Query statistics may differ from production due to test data patterns
    4. Index usage decisions depend on table statistics (ANALYZE freshness)

    For production monitoring, consider:
    - Using EXPLAIN (without ANALYZE) for plan inspection only
    - Using pg_stat_statements for real query performance metrics
    - Running tests against a staging replica with production-like data

Fixture Dependency Graph:
    postgres_pool (module-scoped)
        -> schema_initialized
            -> seeded_test_data
                -> query_analyzer

Environment Requirements:
    POSTGRES_HOST: PostgreSQL server hostname
    POSTGRES_PORT: PostgreSQL server port (default: 5436)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (required)

Related:
    - PR #101: Query performance tests with EXPLAIN ANALYZE
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from omnibase_infra.utils import sanitize_error_message

_logger = logging.getLogger(__name__)

# Load environment configuration
_project_root = Path(__file__).parent.parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# =============================================================================
# Cross-Module Import: Infrastructure Configuration
# =============================================================================
# From tests/infrastructure_config.py:
#   - DEFAULT_POSTGRES_PORT: Standard PostgreSQL port (5436) on infrastructure server
#
# This import ensures consistent port configuration across all database
# performance tests. See tests/infrastructure_config.py for full documentation.
# =============================================================================
from tests.infrastructure_config import (
    DEFAULT_POSTGRES_PORT,
)

if TYPE_CHECKING:
    import asyncpg


# =============================================================================
# Infrastructure Availability
# =============================================================================

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", str(DEFAULT_POSTGRES_PORT)))
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Normalize empty password to None
if POSTGRES_PASSWORD and not POSTGRES_PASSWORD.strip():
    POSTGRES_PASSWORD = None


def _check_postgres_reachable() -> bool:
    """Check if PostgreSQL server is reachable via TCP connection.

    This function verifies actual network connectivity to the PostgreSQL server,
    not just whether environment variables are set. This prevents tests from
    failing with connection errors when the database is unreachable (e.g., when
    running outside the Docker network where hostname resolution may fail).

    Returns:
        bool: True if PostgreSQL is reachable, False otherwise.
    """
    if not POSTGRES_HOST or not POSTGRES_PASSWORD:
        return False

    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            result = sock.connect_ex((POSTGRES_HOST, POSTGRES_PORT))
            return result == 0
    except (OSError, TimeoutError, socket.gaierror):
        return False


# Check PostgreSQL reachability at module import time
POSTGRES_AVAILABLE = _check_postgres_reachable()

# =============================================================================
# Module-Level Markers
# =============================================================================
# NOTE: loop_scope="module" is CRITICAL for pytest-asyncio 0.25+ compatibility.
# This ensures all async fixtures and tests in this module share the same event loop.
# Without this, module-scoped fixtures like postgres_pool get created on one event loop
# but function-scoped fixtures try to use them on a different loop, causing:
# "RuntimeError: Task ... got Future ... attached to a different loop"

pytestmark = [
    pytest.mark.database,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason=(
            "PostgreSQL not reachable for database performance tests. "
            f"POSTGRES_HOST: {'set' if POSTGRES_HOST else 'MISSING'}, "
            f"POSTGRES_PASSWORD: {'set' if POSTGRES_PASSWORD else 'MISSING'}, "
            f"TCP connection to {POSTGRES_HOST}:{POSTGRES_PORT}: "
            f"{'reachable' if POSTGRES_AVAILABLE else 'UNREACHABLE'}."
        ),
    ),
]


# =============================================================================
# Migration Paths
# =============================================================================

MIGRATIONS_DIR = Path(__file__).parent.parent.parent.parent / "docker" / "migrations"


# =============================================================================
# Database Fixtures
# =============================================================================


def _build_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
    )


@pytest_asyncio.fixture(scope="module")
async def postgres_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create asyncpg connection pool for database performance tests.

    Module-scoped for performance: creating a connection pool is expensive, and
    performance tests in the same module can safely share a single pool. This
    avoids the overhead of creating a new pool for every test function.

    Note:
        Uses @pytest_asyncio.fixture(scope="module") to ensure proper event loop
        handling with async fixtures at module scope.

    Yields:
        asyncpg.Pool: Connection pool for database operations.

    Raises:
        RuntimeError: If pool creation fails (with sanitized error message).
    """
    import asyncpg

    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL not available")

    dsn = _build_postgres_dsn()
    try:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=5,
            command_timeout=60.0,
        )
    except Exception as e:
        # Sanitize exception to prevent DSN/credential leakage in error messages
        _logger.warning(
            "Failed to create PostgreSQL connection pool: %s",
            sanitize_error_message(e),
        )
        raise RuntimeError(
            f"Failed to create PostgreSQL connection pool: {sanitize_error_message(e)}"
        ) from e

    yield pool

    await pool.close()


@pytest_asyncio.fixture(scope="module")
async def schema_initialized(
    postgres_pool: asyncpg.Pool,
) -> asyncpg.Pool:
    """Ensure all migrations are applied before tests run.

    Applies all SQL migration files in order (001, 002, etc.).
    Uses a migration tracking table for idempotency and logs migration status.

    Args:
        postgres_pool: Database connection pool.

    Returns:
        asyncpg.Pool: Same pool, with schema guaranteed initialized.

    Raises:
        RuntimeError: If a migration fails (non-idempotent migration that conflicts).

    Note:
        Migrations are expected to be idempotent (use IF NOT EXISTS, ON CONFLICT, etc.).
        The migration_history table tracks applied migrations to prevent re-application.
    """
    # Apply all migrations in order
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_files:
        _logger.warning("No migration files found in %s", MIGRATIONS_DIR)
        return postgres_pool

    async with postgres_pool.acquire() as conn:
        # Create migration tracking table for idempotency
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migration_history (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                checksum TEXT
            )
        """)

        for migration_file in migration_files:
            filename = migration_file.name

            # Skip validation scripts (not migrations)
            if filename == "validate_migration_state.sql":
                _logger.debug("Skipping validation script: %s", filename)
                continue

            # Check if migration was already applied
            already_applied = await conn.fetchval(
                "SELECT 1 FROM _migration_history WHERE filename = $1",
                filename,
            )

            if already_applied:
                _logger.debug("Migration %s already applied, skipping", filename)
                continue

            # Read and apply migration
            try:
                sql = migration_file.read_text()

                # Skip migrations containing CONCURRENTLY - these are for production
                # and cannot run inside a transaction block. The non-concurrent
                # versions (e.g., 003_capability_fields.sql) create the same indexes.
                if "CONCURRENTLY" in sql.upper():
                    _logger.debug(
                        "Skipping %s: contains CREATE INDEX CONCURRENTLY "
                        "(production-only migration, not compatible with test transactions)",
                        filename,
                    )
                    # Mark as applied so we don't try again
                    await conn.execute(
                        """
                        INSERT INTO _migration_history (filename, checksum)
                        VALUES ($1, $2)
                        ON CONFLICT (filename) DO NOTHING
                        """,
                        filename,
                        "skipped-concurrent",
                    )
                    continue

                # Calculate simple checksum for tracking
                checksum = hashlib.md5(sql.encode()).hexdigest()

                _logger.info("Applying migration: %s", filename)
                await conn.execute(sql)

                # Record successful migration
                await conn.execute(
                    """
                    INSERT INTO _migration_history (filename, checksum)
                    VALUES ($1, $2)
                    ON CONFLICT (filename) DO NOTHING
                    """,
                    filename,
                    checksum,
                )
                _logger.info("Migration %s applied successfully", filename)

            except Exception as e:
                # Use warning instead of exception to avoid credential exposure
                # in tracebacks (DSN may contain password in connection errors)
                _logger.warning(
                    "Migration %s failed: %s",
                    filename,
                    sanitize_error_message(e),
                )
                raise RuntimeError(
                    f"Migration {filename} failed: {sanitize_error_message(e)}"
                ) from e

        # Ensure required indexes exist for query performance tests.
        # This handles cases where:
        # 1. The migration was marked "applied" but indexes weren't created
        # 2. The database existed before the migration system
        # 3. Indexes were dropped
        _logger.debug("Ensuring required indexes exist...")
        await conn.execute("""
            -- Index for time-range audit queries (from 002_updated_at_audit_index.sql)
            CREATE INDEX IF NOT EXISTS idx_registration_updated_at
                ON registration_projections (updated_at DESC);

            -- Composite index for state-based audit queries
            CREATE INDEX IF NOT EXISTS idx_registration_state_updated_at
                ON registration_projections (current_state, updated_at DESC);

            -- Index for state filtering (from 001_registration_projection.sql)
            CREATE INDEX IF NOT EXISTS idx_registration_current_state
                ON registration_projections (current_state);

            -- Index for domain + state filtering
            CREATE INDEX IF NOT EXISTS idx_registration_domain_state
                ON registration_projections (domain, current_state);
        """)

        # Update statistics so PostgreSQL's query planner can use indexes optimally
        await conn.execute("ANALYZE registration_projections")
        _logger.debug("Required indexes verified/created and table statistics updated")

    return postgres_pool


@pytest_asyncio.fixture(scope="module")
async def seeded_test_data(
    schema_initialized: asyncpg.Pool,
) -> AsyncGenerator[dict[str, list], None]:
    """Seed test data for query performance verification.

    Creates test records with varied updated_at timestamps and states
    to enable meaningful EXPLAIN ANALYZE testing.

    Args:
        schema_initialized: Pool with schema guaranteed initialized.

    Yields:
        Dictionary containing lists of created entity IDs and metadata.
    """
    pool = schema_initialized
    test_data: dict[str, list] = {
        "entity_ids": [],
        "recent_ids": [],
        "old_ids": [],
        "active_ids": [],
        "pending_ids": [],
    }

    now = datetime.now(UTC)

    async with pool.acquire() as conn:
        # Create 100 test records with varied timestamps and states
        for i in range(100):
            entity_id = uuid4()
            last_event_id = uuid4()

            # Vary the updated_at time: 50 recent (< 1 hour), 50 older (> 24 hours)
            if i < 50:
                updated_at = now - timedelta(minutes=i)
                test_data["recent_ids"].append(entity_id)
            else:
                updated_at = now - timedelta(hours=24 + i)
                test_data["old_ids"].append(entity_id)

            # Vary states: 40 active, 30 pending, 30 awaiting_ack
            if i < 40:
                state = "active"
                test_data["active_ids"].append(entity_id)
            elif i < 70:
                state = "pending_registration"
                test_data["pending_ids"].append(entity_id)
            else:
                state = "awaiting_ack"

            test_data["entity_ids"].append(entity_id)

            await conn.execute(
                """
                INSERT INTO registration_projections (
                    entity_id, domain, current_state, node_type, node_version,
                    last_applied_event_id, last_applied_offset,
                    registered_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (entity_id, domain) DO UPDATE SET
                    current_state = EXCLUDED.current_state,
                    updated_at = EXCLUDED.updated_at,
                    last_applied_event_id = EXCLUDED.last_applied_event_id
                """,
                entity_id,
                "registration",
                state,
                "effect",
                "1.0.0",
                last_event_id,
                i,
                now,
                updated_at,
            )

        # Update statistics after seeding test data
        # This ensures PostgreSQL's query planner has accurate row counts and
        # data distribution info for choosing optimal query plans.
        await conn.execute("ANALYZE registration_projections")

    yield test_data

    # Cleanup: remove test records
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM registration_projections
            WHERE entity_id = ANY($1::uuid[])
            """,
            test_data["entity_ids"],
        )


# =============================================================================
# Query Analysis Utilities
# =============================================================================


class QueryAnalyzerError(Exception):
    """Error raised when query analysis fails."""


def _validate_query_for_explain(query: str) -> None:
    """Validate that a query is safe for EXPLAIN analysis.

    Args:
        query: The SQL query to validate.

    Raises:
        QueryAnalyzerError: If the query is not safe for EXPLAIN analysis.

    Note:
        EXPLAIN supports SELECT, INSERT, UPDATE, DELETE, and VALUES statements.
        We restrict to SELECT for safety in test scenarios.
    """
    # Normalize whitespace and get first keyword
    normalized = query.strip().upper()

    # Only allow SELECT statements for EXPLAIN in tests
    # This prevents accidental mutation via EXPLAIN ANALYZE on INSERT/UPDATE/DELETE
    allowed_prefixes = ("SELECT", "WITH")  # WITH for CTEs that end in SELECT

    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        raise QueryAnalyzerError(
            f"EXPLAIN analysis only supports SELECT queries in test context. "
            f"Query starts with: {normalized[:20]}..."
        )

    # Basic SQL injection check - reject queries with suspicious patterns
    # Note: Parameters are passed separately and handled safely by asyncpg
    suspicious_patterns = [
        ";--",  # Comment injection
        "; DROP",  # Statement injection
        "; DELETE",
        "; INSERT",
        "; UPDATE",
        "; TRUNCATE",
    ]
    for pattern in suspicious_patterns:
        if pattern in normalized:
            raise QueryAnalyzerError(
                f"Query contains suspicious pattern that may indicate SQL injection: "
                f"'{pattern}'"
            )


class QueryAnalyzer:
    """Utility class for analyzing query plans with EXPLAIN ANALYZE.

    Provides methods to execute EXPLAIN ANALYZE and parse the results
    to verify index usage and query efficiency.

    SQL Injection Safety:
        Query parameters are passed separately to asyncpg which uses PostgreSQL's
        prepared statement protocol. This means:

        1. The query string is sent to PostgreSQL as a parameterized query template
        2. Parameter values are sent separately and bound by the database engine
        3. Parameter values CANNOT modify the query structure (injection-proof)

        The EXPLAIN prefix is safe because:
        - _validate_query_for_explain() only allows SELECT/WITH queries
        - The original query is validated BEFORE adding the EXPLAIN prefix
        - f-string interpolation only adds the fixed "EXPLAIN (...) " prefix
        - User input NEVER goes into the f-string - only into asyncpg params

    Attributes:
        pool: Database connection pool.

    Example:
        >>> analyzer = QueryAnalyzer(pool)
        >>> result = await analyzer.explain_analyze(
        ...     "SELECT * FROM registration_projections WHERE updated_at > $1",
        ...     datetime.now(UTC) - timedelta(hours=1)
        ... )
        >>> assert result.uses_index("idx_registration_updated_at")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize query analyzer.

        Args:
            pool: Database connection pool.
        """
        self.pool = pool

    async def explain_analyze(
        self,
        query: str,
        *args: object,
        force_index_scan: bool = False,
    ) -> ExplainResult:
        """Execute EXPLAIN ANALYZE on a query.

        Args:
            query: SQL query to analyze (must be a SELECT statement).
            *args: Query parameters (safely handled by asyncpg).
            force_index_scan: If True, disable sequential scans to force index usage.
                This is useful for verifying that indexes exist and are applicable,
                regardless of whether the optimizer would normally choose them for
                small datasets. PostgreSQL's optimizer correctly prefers seq scans
                for small tables (~100 rows) since they're faster than index scans.

        Returns:
            ExplainResult with parsed plan information.

        Raises:
            QueryAnalyzerError: If query validation fails or EXPLAIN output is malformed.
        """
        # Validate query is safe for EXPLAIN
        _validate_query_for_explain(query)

        # Build EXPLAIN query - the original query's parameters are passed
        # to asyncpg which handles them safely via prepared statements
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        async with self.pool.acquire() as conn:
            if force_index_scan:
                # Disable sequential scans to force index usage for plan verification.
                # We use an explicit transaction so SET LOCAL affects the EXPLAIN query.
                async with conn.transaction():
                    await conn.execute("SET LOCAL enable_seqscan = off")
                    result = await conn.fetchval(explain_query, *args)
            else:
                result = await conn.fetchval(explain_query, *args)

        return ExplainResult(result)

    async def explain_only(
        self,
        query: str,
        *args: object,
        force_index_scan: bool = False,
    ) -> ExplainResult:
        """Execute EXPLAIN (without ANALYZE) on a query.

        Useful for checking query plans without actually running the query.

        Args:
            query: SQL query to analyze (must be a SELECT statement).
            *args: Query parameters (safely handled by asyncpg).
            force_index_scan: If True, disable sequential scans to force index usage.
                This is useful for verifying that indexes exist and are applicable,
                regardless of whether the optimizer would normally choose them for
                small datasets. PostgreSQL's optimizer correctly prefers seq scans
                for small tables (~100 rows) since they're faster than index scans.

        Returns:
            ExplainResult with parsed plan information.

        Raises:
            QueryAnalyzerError: If query validation fails or EXPLAIN output is malformed.
        """
        # Validate query is safe for EXPLAIN
        _validate_query_for_explain(query)

        # Build EXPLAIN query - parameters handled safely by asyncpg
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"

        async with self.pool.acquire() as conn:
            if force_index_scan:
                # Disable sequential scans to force index usage for plan verification.
                # We use an explicit transaction so SET LOCAL affects the EXPLAIN query.
                async with conn.transaction():
                    await conn.execute("SET LOCAL enable_seqscan = off")
                    result = await conn.fetchval(explain_query, *args)
            else:
                result = await conn.fetchval(explain_query, *args)

        return ExplainResult(result)


class ExplainResult:
    """Parsed result from EXPLAIN ANALYZE.

    Provides methods to query the explain plan for index usage,
    scan types, and performance metrics.

    Attributes:
        raw_plan: Raw JSON plan from PostgreSQL.
        plan: Parsed plan dictionary.
    """

    def __init__(self, raw_plan: list[dict[str, object]] | str | None) -> None:
        """Initialize with raw EXPLAIN JSON output.

        Args:
            raw_plan: JSON output from EXPLAIN (FORMAT JSON). Can be:
                - A list of dicts (already parsed by asyncpg)
                - A JSON string (needs parsing)
                - None (error case)

        Raises:
            QueryAnalyzerError: If the raw_plan is malformed or missing expected structure.
        """
        # Validate raw_plan is not None
        if raw_plan is None:
            raise QueryAnalyzerError(
                "EXPLAIN returned None - query may have failed or returned no plan"
            )

        # Defensive handling for string input:
        # asyncpg typically returns JSON columns as already-parsed Python objects,
        # but certain configurations (older asyncpg versions, custom type codecs,
        # or connection poolers) may return raw JSON strings. This defensive check
        # ensures compatibility across different asyncpg deployment scenarios.
        if isinstance(raw_plan, str):
            try:
                raw_plan = json.loads(raw_plan)
            except json.JSONDecodeError as e:
                raise QueryAnalyzerError(
                    f"EXPLAIN output is not valid JSON: {e}"
                ) from e

        # Validate raw_plan is a list
        if not isinstance(raw_plan, list):
            raise QueryAnalyzerError(
                f"EXPLAIN output should be a list, got {type(raw_plan).__name__}"
            )

        # Validate list is not empty
        if not raw_plan:
            raise QueryAnalyzerError("EXPLAIN output is empty - no plan returned")

        # Validate first element is a dict
        if not isinstance(raw_plan[0], dict):
            raise QueryAnalyzerError(
                f"EXPLAIN plan entry should be a dict, got {type(raw_plan[0]).__name__}"
            )

        self.raw_plan = raw_plan
        self.plan = raw_plan[0]

    def _find_nodes(
        self,
        node: dict[str, object] | None = None,
        node_type: str | None = None,
    ) -> list[dict[str, object]]:
        """Recursively find all nodes in the plan tree.

        Args:
            node: Starting node (defaults to root Plan).
            node_type: Optional filter by Node Type.

        Returns:
            List of matching plan nodes.

        Note:
            This method handles malformed plan structures defensively:
            - Missing "Plan" key returns empty list
            - Non-dict nodes are skipped
            - Missing "Plans" key is treated as no children
            - Non-list "Plans" values are skipped
        """
        if node is None:
            plan_node = self.plan.get("Plan")
            if not isinstance(plan_node, dict):
                _logger.warning(
                    "EXPLAIN plan missing 'Plan' key or has unexpected type: %s",
                    type(plan_node).__name__ if plan_node is not None else "None",
                )
                return []
            node = plan_node

        results = []

        # Defensive: ensure node is a dict before accessing
        if not isinstance(node, dict):
            _logger.warning(
                "Unexpected node type in plan tree: %s",
                type(node).__name__,
            )
            return results

        current_type = node.get("Node Type", "")
        if node_type is None or current_type == node_type:
            results.append(node)

        # Recurse into child nodes - defensive handling for malformed Plans
        plans = node.get("Plans", [])
        if not isinstance(plans, list):
            _logger.warning(
                "EXPLAIN plan 'Plans' has unexpected type: %s",
                type(plans).__name__,
            )
            return results

        for child in plans:
            if isinstance(child, dict):
                results.extend(self._find_nodes(child, node_type))

        return results

    def uses_index(self, index_name: str) -> bool:
        """Check if the query plan uses a specific index.

        Args:
            index_name: Name of the index to check for.

        Returns:
            True if the index is used in the query plan.
        """
        for node in self._find_nodes():
            if node.get("Index Name") == index_name:
                return True
        return False

    def uses_any_index(self) -> bool:
        """Check if the query plan uses any index scan.

        Returns:
            True if any index scan node is present in the plan.
        """
        index_node_types = {
            "Index Scan",
            "Index Only Scan",
            "Bitmap Index Scan",
            "Bitmap Heap Scan",
        }
        for node in self._find_nodes():
            if node.get("Node Type") in index_node_types:
                return True
        return False

    def uses_seq_scan(self) -> bool:
        """Check if the query plan uses a sequential scan.

        Returns:
            True if a Seq Scan node is present in the plan.
        """
        return len(self._find_nodes(node_type="Seq Scan")) > 0

    def get_execution_time_ms(self) -> float | None:
        """Get total execution time from EXPLAIN ANALYZE.

        Returns:
            Execution time in milliseconds, or None if not available.
        """
        return self.plan.get("Execution Time")

    def get_planning_time_ms(self) -> float | None:
        """Get planning time from EXPLAIN ANALYZE.

        Returns:
            Planning time in milliseconds, or None if not available.
        """
        return self.plan.get("Planning Time")

    def get_total_cost(self) -> float:
        """Get total estimated cost from the plan.

        Returns:
            Total cost estimate from the root plan node.
        """
        plan_node = self.plan.get("Plan", {})
        cost = plan_node.get("Total Cost", 0.0)
        return float(cost) if cost is not None else 0.0

    def get_actual_rows(self) -> int:
        """Get actual rows returned from EXPLAIN ANALYZE.

        Returns:
            Number of rows actually returned by the root plan node.
        """
        plan_node = self.plan.get("Plan", {})
        rows = plan_node.get("Actual Rows", 0)
        return int(rows) if rows is not None else 0

    def get_index_names(self) -> list[str]:
        """Get all index names used in the query plan.

        Returns:
            List of index names referenced in the plan.
        """
        indexes = []
        for node in self._find_nodes():
            index_name = node.get("Index Name")
            if index_name:
                indexes.append(index_name)
        return indexes

    def __str__(self) -> str:
        """Format explain result for display."""
        return json.dumps(self.raw_plan, indent=2)


@pytest.fixture
def query_analyzer(
    seeded_test_data: dict[str, list],
    schema_initialized: asyncpg.Pool,
) -> QueryAnalyzer:
    """Create QueryAnalyzer for running EXPLAIN ANALYZE tests.

    Args:
        seeded_test_data: Ordering dependency - ensures test data is seeded
            before QueryAnalyzer is created. Value not used directly.
        schema_initialized: Pool with schema initialized.

    Returns:
        QueryAnalyzer instance for test use.
    """
    # Note: seeded_test_data is an ordering dependency, not used directly.
    # It ensures test data exists before running EXPLAIN ANALYZE queries.
    _ = seeded_test_data  # Mark as intentionally unused (ordering dependency)
    return QueryAnalyzer(schema_initialized)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ExplainResult",
    "POSTGRES_AVAILABLE",
    "QueryAnalyzer",
    "QueryAnalyzerError",
    "postgres_pool",
    "query_analyzer",
    "schema_initialized",
    "seeded_test_data",
]
