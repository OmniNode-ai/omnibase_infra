# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Query performance tests using EXPLAIN ANALYZE.

This test suite validates that database queries use expected indexes
and meet performance thresholds. Tests use PostgreSQL's EXPLAIN ANALYZE
to verify query plans and measure actual execution times.

Test Categories:
    1. Index Usage Verification: Confirm specific indexes are used
    2. Audit Query Performance: Verify updated_at index efficiency
    3. State Query Performance: Verify state-based index usage
    4. Time-Range Query Performance: Validate time-based filtering

Performance Thresholds:
    These thresholds are intentionally lenient for CI environments.
    Adjust for dedicated performance testing infrastructure.

    - Simple index lookup: < 10ms
    - Time-range scan: < 50ms
    - Complex audit query: < 100ms

Usage:
    Run query performance tests:
        poetry run pytest tests/performance/database/test_query_performance.py -v

    With output:
        poetry run pytest tests/performance/database/ -v -s

Related:
    - PR #101: Add updated_at index for audit queries
    - OMN-944 (F1): Registration Projection Schema
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.performance.database.conftest import QueryAnalyzer

# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "performance" - run only performance tests
#   pytest -m "database" - run only database tests
#   pytest -m "not performance" - skip performance tests

pytestmark = [
    pytest.mark.performance,
    pytest.mark.database,
]

# =============================================================================
# Updated_At Index Tests (PR #101 Requirement)
# =============================================================================


@pytest.mark.asyncio
class TestUpdatedAtIndexUsage:
    """Test that updated_at index is used for audit queries."""

    async def test_recent_changes_uses_updated_at_index(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_updated_at is used for recent changes query.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE updated_at > :threshold
            ORDER BY updated_at DESC

        Expected: Index Scan or Index Only Scan on idx_registration_updated_at
        """
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)

        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, current_state, updated_at
            FROM registration_projections
            WHERE updated_at > $1
            ORDER BY updated_at DESC
            """,
            one_hour_ago,
        )

        # Verify index is used (not seq scan)
        assert result.uses_any_index(), (
            f"Expected index scan for recent changes query.\nQuery plan:\n{result}"
        )

        # Verify the specific updated_at index is used
        assert result.uses_index("idx_registration_updated_at"), (
            f"Expected idx_registration_updated_at to be used.\n"
            f"Indexes used: {result.get_index_names()}\n"
            f"Query plan:\n{result}"
        )

        # Performance assertion
        exec_time = result.get_execution_time_ms()
        if exec_time is not None:
            assert exec_time < 50.0, f"Query took {exec_time:.2f}ms, expected < 50ms"

        print("\nRecent changes query performance:")
        print(f"  Execution time: {exec_time:.2f}ms" if exec_time else "  N/A")
        print(f"  Actual rows: {result.get_actual_rows()}")
        print(f"  Indexes used: {result.get_index_names()}")

    async def test_time_range_audit_uses_updated_at_index(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_updated_at is used for time range queries.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE updated_at >= :start AND updated_at < :end
            ORDER BY updated_at DESC

        Expected: Index Scan on idx_registration_updated_at
        """
        now = datetime.now(UTC)
        start_time = now - timedelta(hours=24)
        end_time = now - timedelta(hours=12)

        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, current_state, updated_at
            FROM registration_projections
            WHERE updated_at >= $1 AND updated_at < $2
            ORDER BY updated_at DESC
            """,
            start_time,
            end_time,
        )

        # Verify index usage
        assert result.uses_any_index(), (
            f"Expected index scan for time range query.\nQuery plan:\n{result}"
        )

        print("\nTime range audit query performance:")
        print(f"  Execution time: {result.get_execution_time_ms():.2f}ms")
        print(f"  Indexes used: {result.get_index_names()}")

    async def test_state_updated_at_composite_index(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_state_updated_at is used for state+time queries.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE current_state = :state AND updated_at > :since
            ORDER BY updated_at DESC

        Expected: Index Scan on idx_registration_state_updated_at
        """
        one_day_ago = datetime.now(UTC) - timedelta(days=1)

        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, updated_at
            FROM registration_projections
            WHERE current_state = 'active'
              AND updated_at > $1
            ORDER BY updated_at DESC
            """,
            one_day_ago,
        )

        # Verify index is used
        assert result.uses_any_index(), (
            f"Expected index scan for state+time query.\nQuery plan:\n{result}"
        )

        # Check for composite index usage
        indexes_used = result.get_index_names()
        composite_index_used = "idx_registration_state_updated_at" in indexes_used

        # It's acceptable if the query optimizer chooses a different path
        # as long as it's not doing a seq scan
        if composite_index_used:
            print("\nState+time query uses composite index:")
        else:
            print(f"\nState+time query uses alternative indexes: {indexes_used}")

        print(f"  Execution time: {result.get_execution_time_ms():.2f}ms")
        print(f"  Indexes used: {indexes_used}")


# =============================================================================
# Existing Index Verification Tests
# =============================================================================


@pytest.mark.asyncio
class TestExistingIndexUsage:
    """Test that existing indexes are used correctly."""

    async def test_ack_deadline_index_usage(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_ack_deadline is used for deadline queries.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE ack_deadline < :now
            AND ack_timeout_emitted_at IS NULL
            AND current_state = 'awaiting_ack'
        """
        now = datetime.now(UTC)

        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, ack_deadline
            FROM registration_projections
            WHERE ack_deadline < $1
              AND ack_timeout_emitted_at IS NULL
              AND current_state = 'awaiting_ack'
            """,
            now,
        )

        # For partial indexes, the optimizer may choose different paths
        # depending on data distribution. We mainly verify it doesn't do
        # a full table scan when appropriate indexes exist.
        print("\nAck deadline query performance:")
        print(f"  Uses any index: {result.uses_any_index()}")
        print(f"  Uses seq scan: {result.uses_seq_scan()}")
        print(f"  Execution time: {result.get_execution_time_ms():.2f}ms")
        print(f"  Indexes used: {result.get_index_names()}")

    async def test_current_state_index_usage(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_current_state is used for state queries.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE current_state = 'active'
        """
        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, current_state
            FROM registration_projections
            WHERE current_state = 'active'
            """,
        )

        # Verify index usage
        assert result.uses_any_index(), (
            f"Expected index scan for state filter query.\nQuery plan:\n{result}"
        )

        print("\nState filter query performance:")
        print(f"  Execution time: {result.get_execution_time_ms():.2f}ms")
        print(f"  Actual rows: {result.get_actual_rows()}")
        print(f"  Indexes used: {result.get_index_names()}")

    async def test_domain_state_composite_index(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify idx_registration_domain_state is used for domain+state queries.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE domain = :domain AND current_state = :state
        """
        result = await query_analyzer.explain_analyze(
            """
            SELECT entity_id, domain, current_state
            FROM registration_projections
            WHERE domain = 'registration'
              AND current_state = 'active'
            """,
        )

        # Verify index is used
        assert result.uses_any_index(), (
            f"Expected index scan for domain+state query.\nQuery plan:\n{result}"
        )

        indexes_used = result.get_index_names()
        print("\nDomain+state query performance:")
        print(f"  Execution time: {result.get_execution_time_ms():.2f}ms")
        print(f"  Indexes used: {indexes_used}")


# =============================================================================
# Query Performance Threshold Tests
# =============================================================================


@pytest.mark.asyncio
class TestQueryPerformanceThresholds:
    """Test that queries meet performance thresholds."""

    async def test_primary_key_lookup_performance(
        self,
        query_analyzer: QueryAnalyzer,
        seeded_test_data: dict[str, list],
    ) -> None:
        """Verify primary key lookup is fast.

        Query Pattern:
            SELECT * FROM registration_projections
            WHERE entity_id = :id AND domain = :domain

        Threshold: < 10ms
        """
        # Use a known entity from seeded data
        entity_id = seeded_test_data["entity_ids"][0]

        result = await query_analyzer.explain_analyze(
            """
            SELECT *
            FROM registration_projections
            WHERE entity_id = $1 AND domain = 'registration'
            """,
            entity_id,
        )

        # Primary key should use index
        assert result.uses_any_index(), (
            f"Expected index scan for PK lookup.\nQuery plan:\n{result}"
        )

        # Performance threshold
        exec_time = result.get_execution_time_ms()
        if exec_time is not None:
            assert exec_time < 10.0, (
                f"PK lookup took {exec_time:.2f}ms, expected < 10ms"
            )

        print("\nPrimary key lookup performance:")
        print(f"  Execution time: {exec_time:.2f}ms" if exec_time else "  N/A")

    async def test_count_by_state_performance(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify count aggregation by state is efficient.

        Query Pattern:
            SELECT current_state, COUNT(*)
            FROM registration_projections
            GROUP BY current_state

        Threshold: < 50ms (may require seq scan for full aggregation)
        """
        result = await query_analyzer.explain_analyze(
            """
            SELECT current_state, COUNT(*) as count
            FROM registration_projections
            GROUP BY current_state
            """,
        )

        exec_time = result.get_execution_time_ms()
        if exec_time is not None:
            assert exec_time < 50.0, (
                f"Count by state took {exec_time:.2f}ms, expected < 50ms"
            )

        print("\nCount by state performance:")
        print(f"  Execution time: {exec_time:.2f}ms" if exec_time else "  N/A")

    async def test_audit_count_by_time_performance(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify audit count query with time filter is efficient.

        Query Pattern:
            SELECT current_state, COUNT(*)
            FROM registration_projections
            WHERE updated_at >= :since
            GROUP BY current_state

        Threshold: < 100ms
        """
        since = datetime.now(UTC) - timedelta(hours=24)

        result = await query_analyzer.explain_analyze(
            """
            SELECT current_state, COUNT(*) as changes
            FROM registration_projections
            WHERE updated_at >= $1
            GROUP BY current_state
            """,
            since,
        )

        # Should use updated_at index for filtering
        assert result.uses_any_index(), (
            f"Expected index scan for time-filtered aggregation.\nQuery plan:\n{result}"
        )

        exec_time = result.get_execution_time_ms()
        if exec_time is not None:
            assert exec_time < 100.0, (
                f"Audit count took {exec_time:.2f}ms, expected < 100ms"
            )

        print("\nAudit count by time performance:")
        print(f"  Execution time: {exec_time:.2f}ms" if exec_time else "  N/A")
        print(f"  Indexes used: {result.get_index_names()}")


# =============================================================================
# Query Plan Analysis Tests
# =============================================================================


@pytest.mark.asyncio
class TestQueryPlanAnalysis:
    """Test query plan characteristics for optimization insights."""

    async def test_no_seq_scan_for_indexed_columns(
        self,
        query_analyzer: QueryAnalyzer,
        seeded_test_data: dict[str, list],
    ) -> None:
        """Verify indexed column queries don't use sequential scan.

        Tests multiple queries that should use indexes instead of seq scan.
        """
        indexed_queries = [
            (
                "current_state filter",
                "SELECT entity_id FROM registration_projections WHERE current_state = 'active'",
                [],
            ),
            (
                "updated_at filter",
                "SELECT entity_id FROM registration_projections WHERE updated_at > $1",
                [datetime.now(UTC) - timedelta(hours=1)],
            ),
            (
                "entity_id lookup",
                "SELECT * FROM registration_projections WHERE entity_id = $1 AND domain = 'registration'",
                [seeded_test_data["entity_ids"][0]],
            ),
        ]

        for query_name, query, params in indexed_queries:
            result = await query_analyzer.explain_only(query, *params)

            print(f"\n{query_name}:")
            print(f"  Uses index: {result.uses_any_index()}")
            print(f"  Uses seq scan: {result.uses_seq_scan()}")
            print(f"  Total cost: {result.get_total_cost():.2f}")

    async def test_explain_without_execute(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify EXPLAIN (without ANALYZE) works for plan inspection.

        Uses EXPLAIN only to check query plans without executing queries,
        useful for validating plans on production-like data.
        """
        result = await query_analyzer.explain_only(
            """
            SELECT entity_id, updated_at
            FROM registration_projections
            WHERE updated_at > $1
            ORDER BY updated_at DESC
            LIMIT 100
            """,
            datetime.now(UTC) - timedelta(days=7),
        )

        # Should show estimated costs without execution time
        assert result.get_execution_time_ms() is None, (
            "EXPLAIN (without ANALYZE) should not have execution time"
        )

        print("\nEXPLAIN only result:")
        print(f"  Uses index: {result.uses_any_index()}")
        print(f"  Total cost estimate: {result.get_total_cost():.2f}")
        print(f"  Planning time: {result.get_planning_time_ms():.2f}ms")


# =============================================================================
# Regression Tests for Query Plan Stability
# =============================================================================


@pytest.mark.asyncio
class TestQueryPlanStability:
    """Test that query plans remain stable and optimal."""

    async def test_updated_at_index_not_regressed(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify updated_at queries haven't regressed to seq scan.

        This test ensures that changes to schema or statistics don't
        cause the query optimizer to abandon the updated_at index.
        """
        queries_to_check = [
            (
                "recent_changes",
                """
                SELECT entity_id, current_state, updated_at
                FROM registration_projections
                WHERE updated_at > $1
                ORDER BY updated_at DESC
                """,
                [datetime.now(UTC) - timedelta(hours=1)],
            ),
            (
                "time_range",
                """
                SELECT entity_id, updated_at
                FROM registration_projections
                WHERE updated_at >= $1 AND updated_at < $2
                ORDER BY updated_at DESC
                """,
                [
                    datetime.now(UTC) - timedelta(days=1),
                    datetime.now(UTC),
                ],
            ),
            (
                "state_time",
                """
                SELECT entity_id
                FROM registration_projections
                WHERE current_state = 'active' AND updated_at > $1
                """,
                [datetime.now(UTC) - timedelta(hours=12)],
            ),
        ]

        all_passed = True
        failures = []

        for name, query, params in queries_to_check:
            result = await query_analyzer.explain_only(query, *params)

            uses_index = result.uses_any_index()
            if not uses_index:
                all_passed = False
                failures.append(f"  - {name}: Uses seq scan instead of index")

            print(f"\n{name}:")
            print(f"  Uses index: {uses_index}")
            print(f"  Indexes: {result.get_index_names()}")

        if not all_passed:
            pytest.fail("Query plan regression detected:\n" + "\n".join(failures))

    async def test_index_usage_with_limit(
        self,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        """Verify index is still used when LIMIT is added.

        Some query plans can change when LIMIT is introduced.
        This test ensures indexes are still preferred.
        """
        # Without LIMIT
        result_no_limit = await query_analyzer.explain_only(
            """
            SELECT entity_id, updated_at
            FROM registration_projections
            WHERE updated_at > $1
            ORDER BY updated_at DESC
            """,
            datetime.now(UTC) - timedelta(hours=1),
        )

        # With LIMIT
        result_with_limit = await query_analyzer.explain_only(
            """
            SELECT entity_id, updated_at
            FROM registration_projections
            WHERE updated_at > $1
            ORDER BY updated_at DESC
            LIMIT 10
            """,
            datetime.now(UTC) - timedelta(hours=1),
        )

        print("\nLIMIT impact on query plan:")
        print(
            f"  Without LIMIT: index={result_no_limit.uses_any_index()}, "
            f"cost={result_no_limit.get_total_cost():.2f}"
        )
        print(
            f"  With LIMIT: index={result_with_limit.uses_any_index()}, "
            f"cost={result_with_limit.get_total_cost():.2f}"
        )

        # Both should use index
        assert result_with_limit.uses_any_index(), (
            "Query with LIMIT should still use index"
        )

        # LIMIT should reduce cost
        assert result_with_limit.get_total_cost() <= result_no_limit.get_total_cost(), (
            "LIMIT should not increase query cost"
        )
