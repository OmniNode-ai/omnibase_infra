# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for GIN index configuration by ProjectionReaderRegistration methods.

This test suite verifies GIN indexes are properly configured for capability-based
queries executed through the ProjectionReaderRegistration Python methods:

1. get_by_capability_tags_all - Uses @> (contains all) operator
2. get_by_capability_tags_any - Uses && (overlaps/any) operator
3. get_by_intent_type - Uses @> operator on intent_types array
4. get_by_protocol - Uses @> operator on protocols array

The tests verify:
- Index existence in pg_indexes (schema correctness)
- Query correctness with actual SQL patterns from ProjectionReaderRegistration
- Proper use of array operators (@>, &&)

IMPORTANT - PostgreSQL Query Planner Behavior:
    PostgreSQL's cost-based optimizer intelligently chooses execution plans
    based on table size and statistics. For small tables (< ~1000 rows),
    sequential scans are often FASTER than index scans due to:
    - Reduced I/O overhead (fewer page lookups)
    - Better cache locality (sequential access pattern)
    - Lower startup cost (no index traversal)

    Therefore, these tests verify that indexes EXIST and CAN be used,
    not that they WILL be used for test data volumes. At production scale
    (thousands+ rows), PostgreSQL will automatically prefer the GIN indexes.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - PR #118: Add capability fields and GIN indexes

Design Notes:
    - Tests require testcontainers with PostgreSQL
    - Index existence verified via pg_indexes system catalog
    - Query correctness verified via actual query execution
    - Tests verify the exact SQL patterns used by ProjectionReaderRegistration
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.projectors import ProjectorRegistration


# Test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# =============================================================================
# Helper Functions
# =============================================================================


def make_projection_with_capabilities(
    *,
    capability_tags: list[str] | None = None,
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    contract_type: str = "effect",
    offset: int = 100,
) -> ModelRegistrationProjection:
    """Create a test projection with capability fields.

    Args:
        capability_tags: Array of capability tags for discovery
        intent_types: Array of intent types this node handles
        protocols: Array of protocols this node implements
        contract_type: Node contract type (effect, compute, reducer, orchestrator)
        offset: Kafka offset for sequencing

    Returns:
        ModelRegistrationProjection configured for testing
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain="registration",
        current_state=EnumRegistrationState.ACTIVE,
        node_type=EnumNodeKind.EFFECT,
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True),
        # Capability fields (OMN-1134)
        contract_type=contract_type,
        intent_types=intent_types or [],
        protocols=protocols or [],
        capability_tags=capability_tags or [],
        contract_version="1.0.0",
        # Standard fields
        last_applied_event_id=uuid4(),
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


def make_sequence(offset: int) -> ModelSequenceInfo:
    """Create sequence info for testing.

    Args:
        offset: Kafka offset (also used as sequence)

    Returns:
        ModelSequenceInfo configured for testing
    """
    return ModelSequenceInfo(
        sequence=offset,
        partition="0",
        offset=offset,
    )


async def get_query_plan(
    pool: asyncpg.Pool,
    query: str,
    *args: object,
) -> list[str]:
    """Execute EXPLAIN ANALYZE and return the query plan lines.

    Args:
        pool: asyncpg connection pool
        query: SQL query to analyze
        *args: Query parameters

    Returns:
        List of query plan lines from EXPLAIN ANALYZE output
    """
    explain_query = f"EXPLAIN ANALYZE {query}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(explain_query, *args)
    return [row[0] for row in rows]


def plan_uses_gin_index(plan_lines: list[str], index_name: str | None = None) -> bool:
    """Check if the query plan uses a GIN index scan.

    Args:
        plan_lines: Lines from EXPLAIN ANALYZE output
        index_name: Optional specific index name to check for

    Returns:
        True if plan uses GIN-related index scan (Bitmap or regular), False otherwise
    """
    plan_text = "\n".join(plan_lines)

    # GIN indexes typically appear as Bitmap Index Scan or Bitmap Heap Scan
    # because GIN produces multiple row pointers that need to be sorted
    gin_indicators = [
        "Bitmap Index Scan",
        "Bitmap Heap Scan",
        "Index Scan",
    ]

    has_index_scan = any(indicator in plan_text for indicator in gin_indicators)

    if index_name and has_index_scan:
        # Also verify the specific index is used
        return index_name in plan_text

    return has_index_scan


def plan_uses_seq_scan(plan_lines: list[str]) -> bool:
    """Check if the query plan uses a sequential scan on the main table.

    Args:
        plan_lines: Lines from EXPLAIN ANALYZE output

    Returns:
        True if plan uses sequential scan on registration_projections, False otherwise
    """
    plan_text = "\n".join(plan_lines)
    return "Seq Scan on registration_projections" in plan_text


async def gin_index_exists(
    pool: asyncpg.Pool,
    index_name: str,
    table_name: str = "registration_projections",
) -> bool:
    """Check if a GIN index exists on the specified table.

    This verifies that the index is properly created in the schema,
    regardless of whether the query planner chooses to use it for
    small tables.

    Args:
        pool: asyncpg connection pool
        index_name: Name of the index to check
        table_name: Table the index should be on

    Returns:
        True if the GIN index exists, False otherwise
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = $1
              AND indexname = $2
            """,
            table_name,
            index_name,
        )

    if row is None:
        return False

    # Verify it's actually a GIN index
    return "using gin" in row["indexdef"].lower()


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
async def populated_db_for_gin_tests(
    projector: ProjectorRegistration,
    pg_pool: asyncpg.Pool,
) -> asyncpg.Pool:
    """Populate database with test data for GIN index verification tests.

    Inserts 100+ projections with various capability combinations to
    provide enough data for PostgreSQL to prefer GIN index scans over
    sequential scans.

    The data distribution is designed to have:
    - Multiple rows matching each query pattern
    - Diverse capability combinations to test selectivity

    Returns:
        The pg_pool fixture for use in tests.
    """
    # Capability tag combinations for testing
    capability_tag_options = [
        ["postgres.storage"],
        ["kafka.consumer"],
        ["http.client"],
        ["redis.cache"],
        ["postgres.storage", "kafka.consumer"],
        ["http.client", "redis.cache"],
        ["postgres.storage", "kafka.consumer", "http.client"],
        ["postgres.storage", "redis.cache"],
        [],
    ]

    # Intent type combinations for testing
    intent_type_options = [
        ["postgres.upsert"],
        ["postgres.query"],
        ["kafka.publish"],
        ["http.request"],
        ["postgres.upsert", "postgres.query"],
        ["kafka.publish", "kafka.consume"],
        ["postgres.upsert", "kafka.publish"],
        [],
    ]

    # Protocol combinations for testing
    protocol_options = [
        ["ProtocolDatabaseAdapter"],
        ["ProtocolEventPublisher"],
        ["ProtocolHttpClient"],
        ["ProtocolCacheAdapter"],
        ["ProtocolDatabaseAdapter", "ProtocolEventPublisher"],
        ["ProtocolHttpClient", "ProtocolCacheAdapter"],
        [],
    ]

    # Insert 100 projections with diverse combinations
    for i in range(100):
        projection = make_projection_with_capabilities(
            capability_tags=capability_tag_options[i % len(capability_tag_options)],
            intent_types=intent_type_options[i % len(intent_type_options)],
            protocols=protocol_options[i % len(protocol_options)],
            offset=5000 + i,
        )
        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=make_sequence(5000 + i),
        )

    # Run ANALYZE to update table statistics for optimal query planning
    async with pg_pool.acquire() as conn:
        await conn.execute("ANALYZE registration_projections")

    return pg_pool


# =============================================================================
# GIN Index Usage Tests for ProjectionReaderRegistration Methods
# =============================================================================


class TestProjectionReaderGinIndexUsage:
    """Integration tests for GIN index configuration and query correctness.

    These tests verify the SQL patterns used by ProjectionReaderRegistration
    methods work correctly and that GIN indexes exist for efficient queries
    at production scale.

    Each test:
    1. Verifies the GIN index EXISTS in pg_indexes (schema correctness)
    2. Executes the exact SQL pattern the method generates
    3. Confirms query returns correct results

    IMPORTANT: Tests verify index existence, NOT that indexes are used.
    PostgreSQL's cost-based optimizer intelligently chooses sequential
    scans for small tables (~100 rows) because they're actually faster.
    At production scale (thousands+ rows), the GIN indexes will be used
    automatically by the query planner.
    """

    async def test_get_by_capability_tags_all_gin_index_configured(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_all GIN index exists and query works.

        Tests the @> (contains all) operator on capability_tags array.
        This is the exact SQL pattern used by:
            ProjectionReaderRegistration.get_by_capability_tags_all()
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_capability_tags")
        assert index_exists, (
            "GIN index idx_registration_capability_tags does not exist. "
            "Ensure migration 003_capability_fields.sql has been applied."
        )

        # Verify query executes correctly with exact SQL pattern
        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND capability_tags @> $2::text[]
            ORDER BY updated_at DESC
            LIMIT $3
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["postgres.storage", "kafka.consumer"],
                100,
            )

        assert len(rows) > 0, (
            "Expected at least one row with both postgres.storage AND kafka.consumer"
        )

        # Verify results contain BOTH tags
        for row in rows:
            tags = row["capability_tags"]
            assert "postgres.storage" in tags, f"Missing postgres.storage. Has: {tags}"
            assert "kafka.consumer" in tags, f"Missing kafka.consumer. Has: {tags}"

    async def test_get_by_capability_tags_any_gin_index_configured(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_any GIN index exists and query works.

        Tests the && (overlaps/any) operator on capability_tags array.
        This is the exact SQL pattern used by:
            ProjectionReaderRegistration.get_by_capability_tags_any()

        IMPORTANT: This test verifies the && operator is supported by the
        GIN index, which is different from the @> operator.
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_capability_tags")
        assert index_exists, (
            "GIN index idx_registration_capability_tags does not exist."
        )

        # Verify query executes correctly with exact SQL pattern
        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND capability_tags && $2::text[]
            ORDER BY updated_at DESC
            LIMIT $3
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["postgres.storage", "redis.cache"],
                100,
            )

        assert len(rows) > 0, (
            "Expected at least one row with postgres.storage OR redis.cache"
        )

        # Verify results contain at least ONE of the tags
        for row in rows:
            tags = row["capability_tags"]
            has_postgres = "postgres.storage" in tags
            has_redis = "redis.cache" in tags
            assert has_postgres or has_redis, (
                f"Result missing both expected tags. Has: {tags}"
            )

    async def test_get_by_intent_type_gin_index_configured(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_intent_type GIN index exists and query works.

        Tests the @> operator on intent_types array.
        This is the exact SQL pattern used by:
            ProjectionReaderRegistration.get_by_intent_type()
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_intent_types")
        assert index_exists, (
            "GIN index idx_registration_intent_types does not exist. "
            "Ensure migration 003_capability_fields.sql has been applied."
        )

        # Verify query executes correctly
        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND intent_types @> $2::text[]
            ORDER BY updated_at DESC
            LIMIT $3
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["postgres.upsert"],
                100,
            )

        assert len(rows) > 0, "Expected at least one row with postgres.upsert intent"

        # Verify results contain expected intent
        for row in rows:
            assert "postgres.upsert" in row["intent_types"], (
                f"Result missing expected intent. Has: {row['intent_types']}"
            )

    async def test_get_by_protocol_gin_index_configured(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_protocol GIN index exists and query works.

        Tests the @> operator on protocols array.
        This is the exact SQL pattern used by:
            ProjectionReaderRegistration.get_by_protocol()
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_protocols")
        assert index_exists, (
            "GIN index idx_registration_protocols does not exist. "
            "Ensure migration 003_capability_fields.sql has been applied."
        )

        # Verify query executes correctly
        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND protocols @> $2::text[]
            ORDER BY updated_at DESC
            LIMIT $3
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["ProtocolDatabaseAdapter"],
                100,
            )

        assert len(rows) > 0, (
            "Expected at least one row with ProtocolDatabaseAdapter protocol"
        )

        # Verify results contain expected protocol
        for row in rows:
            assert "ProtocolDatabaseAdapter" in row["protocols"], (
                f"Result missing expected protocol. Has: {row['protocols']}"
            )


# =============================================================================
# GIN Index Usage with State Filter Tests
# =============================================================================


class TestProjectionReaderGinIndexWithStateFilter:
    """Tests verifying combined array and state filter queries work correctly.

    These tests verify that capability queries combined with state filtering
    execute correctly and return expected results. GIN indexes exist for the
    array columns and will be used at production scale.
    """

    async def test_capability_tags_all_with_state_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_all with state filter works correctly.

        Tests the combined query pattern:
            capability_tags @> $2::text[] AND current_state = $3
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_capability_tags")
        assert index_exists, "GIN index idx_registration_capability_tags required"

        # Verify combined query executes correctly
        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND capability_tags @> $2::text[]
              AND current_state = $3
            ORDER BY updated_at DESC
            LIMIT $4
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["postgres.storage"],
                "active",
                100,
            )

        assert len(rows) > 0, "Expected results with postgres.storage AND active state"

        # Verify results match both conditions
        for row in rows:
            assert "postgres.storage" in row["capability_tags"]
            assert row["current_state"] == "active"

    async def test_capability_tags_any_with_state_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_any with state filter works correctly.

        Tests the combined query pattern:
            capability_tags && $2::text[] AND current_state = $3
        """
        pool = populated_db_for_gin_tests

        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND capability_tags && $2::text[]
              AND current_state = $3
            ORDER BY updated_at DESC
            LIMIT $4
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["postgres.storage", "kafka.consumer"],
                "active",
                100,
            )

        assert len(rows) > 0, (
            "Expected results with (postgres.storage OR kafka.consumer) AND active"
        )

        # Verify results match conditions
        for row in rows:
            tags = row["capability_tags"]
            has_either = "postgres.storage" in tags or "kafka.consumer" in tags
            assert has_either, f"Result missing expected tags. Has: {tags}"
            assert row["current_state"] == "active"

    async def test_intent_type_with_state_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_intent_type with state filter works correctly."""
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_intent_types")
        assert index_exists, "GIN index idx_registration_intent_types required"

        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND intent_types @> $2::text[]
              AND current_state = $3
            ORDER BY updated_at DESC
            LIMIT $4
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["kafka.publish"],
                "active",
                100,
            )

        assert len(rows) > 0, "Expected results with kafka.publish AND active state"

        for row in rows:
            assert "kafka.publish" in row["intent_types"]
            assert row["current_state"] == "active"

    async def test_protocol_with_state_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify get_by_protocol with state filter works correctly."""
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_protocols")
        assert index_exists, "GIN index idx_registration_protocols required"

        query = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND protocols @> $2::text[]
              AND current_state = $3
            ORDER BY updated_at DESC
            LIMIT $4
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                "registration",
                ["ProtocolEventPublisher"],
                "active",
                100,
            )

        assert len(rows) > 0, (
            "Expected results with ProtocolEventPublisher AND active state"
        )

        for row in rows:
            assert "ProtocolEventPublisher" in row["protocols"]
            assert row["current_state"] == "active"


# =============================================================================
# Edge Case Tests for GIN Index Behavior
# =============================================================================


class TestGinIndexEdgeCases:
    """Edge case tests for array query behavior.

    These tests verify query correctness in edge cases that might
    affect results or query behavior.
    """

    async def test_single_tag_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify single-element array queries work correctly.

        Tests that @> operator with a single tag returns correct results.
        """
        pool = populated_db_for_gin_tests

        # Verify index exists
        index_exists = await gin_index_exists(pool, "idx_registration_capability_tags")
        assert index_exists, "GIN index idx_registration_capability_tags required"

        query = """
            SELECT * FROM registration_projections
            WHERE capability_tags @> $1::text[]
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, ["postgres.storage"])

        assert len(rows) > 0, "Expected results with postgres.storage tag"

        for row in rows:
            assert "postgres.storage" in row["capability_tags"]

    async def test_empty_result_query_returns_empty_list(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify queries for non-existent values return empty results.

        Tests that querying for a tag that doesn't exist returns an
        empty result set without error.
        """
        pool = populated_db_for_gin_tests

        query = """
            SELECT * FROM registration_projections
            WHERE capability_tags @> $1::text[]
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, ["nonexistent.capability.tag.12345"])

        # Should return empty list, not error
        assert len(rows) == 0, f"Expected empty result, got {len(rows)} rows"

    async def test_multiple_tags_any_operator_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify multi-element && (any) query returns correct results.

        The && operator should return rows with ANY of the specified tags.
        """
        pool = populated_db_for_gin_tests

        query = """
            SELECT * FROM registration_projections
            WHERE capability_tags && $1::text[]
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                ["postgres.storage", "kafka.consumer", "http.client", "redis.cache"],
            )

        assert len(rows) > 0, "Expected results with any of the specified tags"

        # Verify each result has at least one of the specified tags
        target_tags = {
            "postgres.storage",
            "kafka.consumer",
            "http.client",
            "redis.cache",
        }
        for row in rows:
            row_tags = set(row["capability_tags"])
            assert row_tags & target_tags, (
                f"Result has none of the target tags. Has: {row_tags}"
            )

    async def test_combined_gin_indexes_query_correctness(
        self,
        populated_db_for_gin_tests: asyncpg.Pool,
    ) -> None:
        """Verify query combining multiple GIN-indexed columns works correctly.

        When querying both capability_tags AND intent_types, results
        should match BOTH conditions.
        """
        pool = populated_db_for_gin_tests

        # Verify both indexes exist
        cap_index_exists = await gin_index_exists(
            pool, "idx_registration_capability_tags"
        )
        intent_index_exists = await gin_index_exists(
            pool, "idx_registration_intent_types"
        )
        assert cap_index_exists, "GIN index idx_registration_capability_tags required"
        assert intent_index_exists, "GIN index idx_registration_intent_types required"

        query = """
            SELECT * FROM registration_projections
            WHERE capability_tags @> $1::text[]
              AND intent_types @> $2::text[]
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                ["postgres.storage"],
                ["postgres.upsert"],
            )

        # May have results depending on data distribution
        # The important thing is the query executes without error
        # and returns only rows matching BOTH conditions
        for row in rows:
            assert "postgres.storage" in row["capability_tags"], (
                f"Missing postgres.storage. Has: {row['capability_tags']}"
            )
            assert "postgres.upsert" in row["intent_types"], (
                f"Missing postgres.upsert. Has: {row['intent_types']}"
            )
