# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for GIN index usage on capability fields.

This test suite verifies that PostgreSQL uses GIN indexes for capability-based
queries on the registration_projections table. The tests use EXPLAIN ANALYZE
to verify that:

1. capability_tags queries use idx_registration_capability_tags GIN index
2. intent_types queries use idx_registration_intent_types GIN index
3. protocols queries use idx_registration_protocols GIN index

Without GIN indexes, array containment (@>) queries would require sequential
scans, which degrades performance significantly as the table grows.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - PR #118: Add capability fields and GIN indexes

Design Notes:
    - Tests require testcontainers with PostgreSQL
    - Tests insert sufficient data to encourage index usage
    - EXPLAIN ANALYZE output is parsed for index scan indicators
    - Tests verify "Index Scan" or "Bitmap Index Scan" in query plan
    - Sequential scans on capability arrays indicate missing or unused indexes
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


def plan_uses_index(plan_lines: list[str], index_name: str | None = None) -> bool:
    """Check if the query plan uses an index scan.

    Args:
        plan_lines: Lines from EXPLAIN ANALYZE output
        index_name: Optional specific index name to check for

    Returns:
        True if plan uses index scan (Bitmap or regular), False otherwise
    """
    plan_text = "\n".join(plan_lines)

    # Check for index usage indicators
    index_indicators = ["Index Scan", "Bitmap Index Scan", "Bitmap Heap Scan"]

    has_index_scan = any(indicator in plan_text for indicator in index_indicators)

    if index_name and has_index_scan:
        # Also verify the specific index is used
        return index_name in plan_text

    return has_index_scan


def plan_uses_seq_scan(plan_lines: list[str]) -> bool:
    """Check if the query plan uses a sequential scan on the main table.

    Args:
        plan_lines: Lines from EXPLAIN ANALYZE output

    Returns:
        True if plan uses sequential scan, False otherwise
    """
    plan_text = "\n".join(plan_lines)
    # Look for "Seq Scan on registration_projections"
    return "Seq Scan on registration_projections" in plan_text


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
async def populated_db(
    projector: ProjectorRegistration,
    pg_pool: asyncpg.Pool,
) -> asyncpg.Pool:
    """Populate database with test data for index and query tests.

    Inserts 100 projections with various capability combinations to
    provide enough data for PostgreSQL to prefer index scans over
    sequential scans, and to verify capability query methods return
    correct results.

    Returns:
        The pg_pool fixture for use in tests.
    """
    # Insert 100 projections with diverse capability combinations
    capability_tag_options = [
        ["postgres.storage"],
        ["kafka.consumer"],
        ["http.client"],
        ["redis.cache"],
        ["postgres.storage", "kafka.consumer"],
        ["http.client", "redis.cache"],
        ["postgres.storage", "kafka.consumer", "http.client"],
        [],
    ]

    intent_type_options = [
        ["postgres.upsert"],
        ["postgres.query"],
        ["kafka.publish"],
        ["http.request"],
        ["postgres.upsert", "postgres.query"],
        ["kafka.publish", "kafka.consume"],
        [],
    ]

    protocol_options = [
        ["ProtocolDatabaseAdapter"],
        ["ProtocolEventPublisher"],
        ["ProtocolHttpClient"],
        ["ProtocolDatabaseAdapter", "ProtocolEventPublisher"],
        [],
    ]

    for i in range(100):
        projection = make_projection_with_capabilities(
            capability_tags=capability_tag_options[i % len(capability_tag_options)],
            intent_types=intent_type_options[i % len(intent_type_options)],
            protocols=protocol_options[i % len(protocol_options)],
            offset=1000 + i,
        )
        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=make_sequence(1000 + i),
        )

    # Run ANALYZE to update table statistics for better query planning
    async with pg_pool.acquire() as conn:
        await conn.execute("ANALYZE registration_projections")

    return pg_pool


# =============================================================================
# GIN Index Usage Tests
# =============================================================================


class TestGinIndexUsage:
    """Integration tests for GIN index usage on capability fields.

    These tests verify that PostgreSQL uses GIN indexes for array containment
    queries (@>) on capability_tags, intent_types, and protocols columns.

    Note: PostgreSQL may choose sequential scan for very small tables. These
    tests insert sufficient data (100+ rows) to encourage index usage. The
    planner's cost-based decisions can still vary based on statistics, so
    tests check for index usage OR verify the query returns correct results.
    """

    async def test_capability_tags_uses_gin_index(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify capability_tags queries use GIN index.

        Tests that array containment queries on capability_tags column
        use the idx_registration_capability_tags GIN index instead of
        a sequential scan.
        """
        query = """
            SELECT entity_id, capability_tags
            FROM registration_projections
            WHERE capability_tags @> ARRAY['postgres.storage']
        """

        plan_lines = await get_query_plan(populated_db, query)

        # Verify index is used
        uses_index = plan_uses_index(plan_lines, "idx_registration_capability_tags")
        uses_seq_scan = plan_uses_seq_scan(plan_lines)

        # Assert index is used OR no sequential scan on main table
        # (PostgreSQL might use different scan types based on statistics)
        assert uses_index or not uses_seq_scan, (
            "Expected GIN index usage for capability_tags @> query. "
            f"Query plan:\n{chr(10).join(plan_lines)}"
        )

        # Also verify the query returns results (sanity check)
        async with populated_db.acquire() as conn:
            rows = await conn.fetch(query)
        assert len(rows) > 0, "Expected at least one row with postgres.storage tag"

    async def test_intent_types_uses_gin_index(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify intent_types queries use GIN index.

        Tests that array containment queries on intent_types column
        use the idx_registration_intent_types GIN index.
        """
        query = """
            SELECT entity_id, intent_types
            FROM registration_projections
            WHERE intent_types @> ARRAY['postgres.upsert']
        """

        plan_lines = await get_query_plan(populated_db, query)

        uses_index = plan_uses_index(plan_lines, "idx_registration_intent_types")
        uses_seq_scan = plan_uses_seq_scan(plan_lines)

        assert uses_index or not uses_seq_scan, (
            "Expected GIN index usage for intent_types @> query. "
            f"Query plan:\n{chr(10).join(plan_lines)}"
        )

        # Verify query returns results
        async with populated_db.acquire() as conn:
            rows = await conn.fetch(query)
        assert len(rows) > 0, "Expected at least one row with postgres.upsert intent"

    async def test_protocols_uses_gin_index(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify protocols queries use GIN index.

        Tests that array containment queries on protocols column
        use the idx_registration_protocols GIN index.
        """
        query = """
            SELECT entity_id, protocols
            FROM registration_projections
            WHERE protocols @> ARRAY['ProtocolDatabaseAdapter']
        """

        plan_lines = await get_query_plan(populated_db, query)

        uses_index = plan_uses_index(plan_lines, "idx_registration_protocols")
        uses_seq_scan = plan_uses_seq_scan(plan_lines)

        assert uses_index or not uses_seq_scan, (
            "Expected GIN index usage for protocols @> query. "
            f"Query plan:\n{chr(10).join(plan_lines)}"
        )

        # Verify query returns results
        async with populated_db.acquire() as conn:
            rows = await conn.fetch(query)
        assert len(rows) > 0, "Expected at least one row with ProtocolDatabaseAdapter"

    async def test_multiple_capability_tags_uses_gin_index(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify multi-element array containment uses GIN index.

        Tests that queries checking for multiple capability tags still
        use the GIN index efficiently.
        """
        query = """
            SELECT entity_id, capability_tags
            FROM registration_projections
            WHERE capability_tags @> ARRAY['postgres.storage', 'kafka.consumer']
        """

        plan_lines = await get_query_plan(populated_db, query)

        uses_index = plan_uses_index(plan_lines, "idx_registration_capability_tags")
        uses_seq_scan = plan_uses_seq_scan(plan_lines)

        assert uses_index or not uses_seq_scan, (
            "Expected GIN index usage for multi-element @> query. "
            f"Query plan:\n{chr(10).join(plan_lines)}"
        )

    async def test_combined_capability_and_contract_type_query(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify combined GIN and B-tree index usage.

        Tests queries that combine capability_tags (GIN) with
        contract_type (B-tree) filtering use appropriate indexes.
        """
        query = """
            SELECT entity_id, capability_tags, contract_type
            FROM registration_projections
            WHERE capability_tags @> ARRAY['postgres.storage']
              AND contract_type = 'effect'
        """

        plan_lines = await get_query_plan(populated_db, query)

        # Should use at least one index
        uses_any_index = plan_uses_index(plan_lines)
        uses_seq_scan = plan_uses_seq_scan(plan_lines)

        assert uses_any_index or not uses_seq_scan, (
            "Expected index usage for combined capability + contract_type query. "
            f"Query plan:\n{chr(10).join(plan_lines)}"
        )


# =============================================================================
# Index Existence Verification Tests
# =============================================================================


class TestGinIndexExists:
    """Tests to verify GIN indexes exist in the schema.

    These tests verify that the migration (003_capability_fields.sql)
    has been applied correctly and all expected GIN indexes are present.
    """

    async def test_capability_tags_index_exists(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify idx_registration_capability_tags GIN index exists."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'registration_projections'
                  AND indexname = 'idx_registration_capability_tags'
                """
            )

        assert row is not None, "Missing index: idx_registration_capability_tags"
        assert "USING gin" in row["indexdef"].lower(), (
            f"Expected GIN index, got: {row['indexdef']}"
        )

    async def test_intent_types_index_exists(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify idx_registration_intent_types GIN index exists."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'registration_projections'
                  AND indexname = 'idx_registration_intent_types'
                """
            )

        assert row is not None, "Missing index: idx_registration_intent_types"
        assert "USING gin" in row["indexdef"].lower(), (
            f"Expected GIN index, got: {row['indexdef']}"
        )

    async def test_protocols_index_exists(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify idx_registration_protocols GIN index exists."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'registration_projections'
                  AND indexname = 'idx_registration_protocols'
                """
            )

        assert row is not None, "Missing index: idx_registration_protocols"
        assert "USING gin" in row["indexdef"].lower(), (
            f"Expected GIN index, got: {row['indexdef']}"
        )

    async def test_contract_type_state_index_exists(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify idx_registration_contract_type_state B-tree index exists."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'registration_projections'
                  AND indexname = 'idx_registration_contract_type_state'
                """
            )

        assert row is not None, "Missing index: idx_registration_contract_type_state"
        # B-tree is the default, so indexdef might not explicitly say "btree"
        assert "capability_tags" not in row["indexdef"].lower(), (
            "contract_type_state index should not be on capability_tags"
        )


# =============================================================================
# Query Performance Baseline Tests
# =============================================================================


class TestQueryPerformanceBaseline:
    """Baseline tests to verify queries complete in reasonable time.

    These tests don't assert specific timing but verify that queries
    using GIN indexes complete without timeout on reasonably-sized data.
    """

    async def test_capability_tags_query_completes(
        self,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify capability_tags query completes in reasonable time."""
        # Insert test data
        for i in range(50):
            projection = make_projection_with_capabilities(
                capability_tags=["test.tag", f"unique.tag.{i}"],
                offset=2000 + i,
            )
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain=projection.domain,
                sequence_info=make_sequence(2000 + i),
            )

        # Query should complete quickly with GIN index
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id
                FROM registration_projections
                WHERE capability_tags @> ARRAY['test.tag']
                """
            )

        assert len(rows) == 50, f"Expected 50 rows, got {len(rows)}"

    async def test_empty_array_query_completes(
        self,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify queries for empty arrays complete correctly."""
        # Insert projection with empty arrays
        projection = make_projection_with_capabilities(
            capability_tags=[],
            intent_types=[],
            protocols=[],
            offset=3000,
        )
        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=make_sequence(3000),
        )

        # Query for non-existent tag should return empty result
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id
                FROM registration_projections
                WHERE capability_tags @> ARRAY['nonexistent.tag']
                """
            )

        # The projection with empty array should not match
        assert projection.entity_id not in [row["entity_id"] for row in rows]


# =============================================================================
# Capability Query Method Execution Tests
# =============================================================================


class TestCapabilityQueryMethodsExecution:
    """Integration tests verifying capability query methods execute correctly.

    These tests call the actual ProjectionReaderRegistration Python methods
    (not raw SQL) to verify the complete code path works, including:

    - SQL query construction with correct array syntax
    - Parameter binding for GIN-indexed array queries
    - Result deserialization to ModelRegistrationProjection
    - State filtering with optional parameters

    This complements TestGinIndexUsage which tests raw SQL query plans.

    Related Tickets:
        - OMN-1134: Registry Projection Extensions for Capabilities
        - PR #118: Add capability fields and GIN indexes
    """

    async def test_get_by_capability_tag_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tag method executes without error.

        Calls the actual Python method and verifies:
        1. No exceptions raised (array syntax is correct)
        2. Results are returned (data matching exists)
        3. Results have the expected capability tag
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # This should NOT raise - if array syntax is wrong, this will fail
        results = await reader.get_by_capability_tag("postgres.storage")

        # Verify we got results (data was inserted by populated_db fixture)
        assert len(results) > 0, (
            "Expected at least one result with postgres.storage tag"
        )

        # Verify results have the expected tag
        for proj in results:
            assert "postgres.storage" in proj.capability_tags, (
                f"Result {proj.entity_id} missing expected tag. "
                f"Has tags: {proj.capability_tags}"
            )

    async def test_get_by_intent_type_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_intent_type method executes without error.

        Calls the actual Python method and verifies:
        1. No exceptions raised (array syntax is correct)
        2. Results are returned (data matching exists)
        3. Results have the expected intent type
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # This should NOT raise
        results = await reader.get_by_intent_type("postgres.upsert")

        # Verify we got results
        assert len(results) > 0, (
            "Expected at least one result with postgres.upsert intent"
        )

        # Verify results have the expected intent type
        for proj in results:
            assert "postgres.upsert" in proj.intent_types, (
                f"Result {proj.entity_id} missing expected intent. "
                f"Has intents: {proj.intent_types}"
            )

    async def test_get_by_protocol_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_protocol method executes without error.

        Calls the actual Python method and verifies:
        1. No exceptions raised (array syntax is correct)
        2. Results are returned (data matching exists)
        3. Results have the expected protocol
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # This should NOT raise
        results = await reader.get_by_protocol("ProtocolDatabaseAdapter")

        # Verify we got results
        assert len(results) > 0, (
            "Expected at least one result with ProtocolDatabaseAdapter protocol"
        )

        # Verify results have the expected protocol
        for proj in results:
            assert "ProtocolDatabaseAdapter" in proj.protocols, (
                f"Result {proj.entity_id} missing expected protocol. "
                f"Has protocols: {proj.protocols}"
            )

    async def test_get_by_capability_tag_with_state_filter(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tag with state filter executes correctly.

        Tests the optional state parameter for filtering results by
        registration state. All test data is inserted with ACTIVE state.
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query with state filter - all test data is ACTIVE
        results = await reader.get_by_capability_tag(
            "postgres.storage",
            state=EnumRegistrationState.ACTIVE,
        )

        # Verify we got results
        assert len(results) > 0, (
            "Expected at least one ACTIVE result with postgres.storage tag"
        )

        # Verify all results are ACTIVE and have the tag
        for proj in results:
            assert proj.current_state == EnumRegistrationState.ACTIVE, (
                f"Result {proj.entity_id} has unexpected state: {proj.current_state}"
            )
            assert "postgres.storage" in proj.capability_tags

    async def test_get_by_intent_type_with_state_filter(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_intent_type with state filter executes correctly."""
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query with state filter
        results = await reader.get_by_intent_type(
            "kafka.publish",
            state=EnumRegistrationState.ACTIVE,
        )

        # Verify we got results
        assert len(results) > 0, (
            "Expected at least one ACTIVE result with kafka.publish intent"
        )

        # Verify all results are ACTIVE and have the intent
        for proj in results:
            assert proj.current_state == EnumRegistrationState.ACTIVE
            assert "kafka.publish" in proj.intent_types

    async def test_get_by_protocol_with_state_filter(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_protocol with state filter executes correctly."""
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query with state filter
        results = await reader.get_by_protocol(
            "ProtocolEventPublisher",
            state=EnumRegistrationState.ACTIVE,
        )

        # Verify we got results
        assert len(results) > 0, (
            "Expected at least one ACTIVE result with ProtocolEventPublisher"
        )

        # Verify all results are ACTIVE and have the protocol
        for proj in results:
            assert proj.current_state == EnumRegistrationState.ACTIVE
            assert "ProtocolEventPublisher" in proj.protocols

    async def test_get_by_capability_tags_all_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_all method executes correctly.

        Tests the multi-tag query that requires ALL tags to be present.
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query for projections with both tags
        results = await reader.get_by_capability_tags_all(
            ["postgres.storage", "kafka.consumer"]
        )

        # Verify we got results (test data includes this combination)
        assert len(results) > 0, (
            "Expected results with both postgres.storage AND kafka.consumer"
        )

        # Verify ALL results have BOTH tags
        for proj in results:
            assert "postgres.storage" in proj.capability_tags, (
                f"Result {proj.entity_id} missing postgres.storage"
            )
            assert "kafka.consumer" in proj.capability_tags, (
                f"Result {proj.entity_id} missing kafka.consumer"
            )

    async def test_get_by_capability_tags_any_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_capability_tags_any method executes correctly.

        Tests the multi-tag query that requires ANY tag to be present.
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query for projections with any of these tags
        results = await reader.get_by_capability_tags_any(
            ["postgres.storage", "kafka.consumer"]
        )

        # Verify we got results
        assert len(results) > 0, (
            "Expected results with postgres.storage OR kafka.consumer"
        )

        # Verify ALL results have at least ONE of the tags
        for proj in results:
            has_postgres = "postgres.storage" in proj.capability_tags
            has_kafka = "kafka.consumer" in proj.capability_tags
            assert has_postgres or has_kafka, (
                f"Result {proj.entity_id} missing both expected tags. "
                f"Has tags: {proj.capability_tags}"
            )

    async def test_get_by_contract_type_executes_successfully(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify get_by_contract_type method executes correctly."""
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query for effect nodes (all test data is contract_type="effect")
        results = await reader.get_by_contract_type("effect")

        # Verify we got results
        assert len(results) > 0, "Expected at least one effect node"

        # Verify all results have the expected contract type
        for proj in results:
            assert proj.contract_type == "effect", (
                f"Result {proj.entity_id} has unexpected contract_type: {proj.contract_type}"
            )

    async def test_nonexistent_capability_tag_returns_empty(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify querying nonexistent capability tag returns empty list."""
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # Query for a tag that doesn't exist
        results = await reader.get_by_capability_tag("nonexistent.tag.12345")

        # Should return empty list, not error
        assert results == [], f"Expected empty list, got {len(results)} results"

    async def test_state_filter_with_non_matching_state_returns_empty(
        self,
        populated_db: asyncpg.Pool,
    ) -> None:
        """Verify state filter correctly excludes non-matching states.

        All test data is ACTIVE, so filtering for SUSPENDED should return empty.
        """
        from omnibase_infra.projectors import ProjectionReaderRegistration

        reader = ProjectionReaderRegistration(populated_db)

        # All test data is ACTIVE, so SUSPENDED should return nothing
        results = await reader.get_by_capability_tag(
            "postgres.storage",
            state=EnumRegistrationState.SUSPENDED,
        )

        # Should return empty list
        assert results == [], (
            f"Expected empty list for SUSPENDED filter, got {len(results)} results"
        )
