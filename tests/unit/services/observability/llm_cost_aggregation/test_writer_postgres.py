# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for WriterLlmCostAggregationPostgres.

Tests:
    - Aggregation row building from events
    - Event deduplication (in-memory cache)
    - Aggregation key generation for all dimensions
    - Rolling window generation (24h, 7d, 30d)
    - Usage source resolution
    - Safe type conversion helpers
    - Estimated coverage percentage calculation
    - Circuit breaker success/failure recording
    - Empty event list handling

All tests mock asyncpg - no real PostgreSQL required.

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres import (
    WriterLlmCostAggregationPostgres,
    _build_aggregation_rows,
    _resolve_usage_source,
    _safe_decimal,
    _safe_int,
    _safe_jsonb,
    _safe_uuid,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with proper async context manager support.

    The key challenge is that asyncpg uses two levels of async context managers:
      1. ``async with pool.acquire() as conn:`` -- pool.acquire() returns an ACM
      2. ``async with conn.transaction():``      -- conn.transaction() is a
         *synchronous* call that returns an async context manager

    We use MagicMock for the synchronous call (conn.transaction) and configure
    its return value with __aenter__/__aexit__ to act as an async CM.
    """
    pool = MagicMock()
    conn = AsyncMock()

    # conn.transaction() is a synchronous call that returns an async CM.
    # Use MagicMock so calling it does NOT produce a coroutine, then give
    # the return value async-CM dunder methods.
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)

    conn.execute = AsyncMock()

    # pool.acquire() returns an async context manager yielding conn
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    return pool


@pytest.fixture
def writer(mock_pool: MagicMock) -> WriterLlmCostAggregationPostgres:
    """Create a writer instance with mock pool."""
    return WriterLlmCostAggregationPostgres(
        pool=mock_pool,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=60.0,
    )


@pytest.fixture
def sample_event() -> dict[str, object]:
    """Create a sample LLM call completed event."""
    return {
        "model_id": "gpt-4o",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "estimated_cost_usd": 0.005,
        "latency_ms": 1200,
        "usage_is_estimated": False,
        "input_hash": f"sha256-{uuid4().hex}",
        "timestamp_iso": "2026-02-16T10:00:00Z",
        "reporting_source": "handler-llm-openai-compatible",
        "session_id": "session-abc-123",
        "extensions": {
            "repo": "omniarchon",
            "pattern_id": "code-review-v1",
        },
        "usage_normalized": {
            "source": "api",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "usage_is_estimated": False,
        },
    }


# =============================================================================
# Tests: _build_aggregation_rows
# =============================================================================


class TestBuildAggregationRows:
    """Tests for _build_aggregation_rows helper."""

    @pytest.mark.unit
    def test_builds_rows_for_all_dimensions_and_windows(
        self, sample_event: dict[str, object]
    ) -> None:
        """Each event produces rows for session, model, repo, pattern x 3 windows."""
        rows = _build_aggregation_rows([sample_event])

        # 4 dimensions x 3 windows = 12 rows
        assert len(rows) == 12

        # Verify all windows present
        windows = {row["window"] for row in rows}
        assert windows == {"24h", "7d", "30d"}

        # Verify all key prefixes present
        keys = {row["aggregation_key"].split(":")[0] for row in rows}
        assert keys == {"session", "model", "repo", "pattern"}

    @pytest.mark.unit
    def test_builds_rows_without_extensions(self) -> None:
        """Events without extensions produce only session and model rows."""
        event = {
            "model_id": "claude-opus-4-20250514",
            "total_tokens": 200,
            "session_id": "sess-1",
        }
        rows = _build_aggregation_rows([event])

        # 2 dimensions (session, model) x 3 windows = 6 rows
        assert len(rows) == 6

        keys = {row["aggregation_key"].split(":")[0] for row in rows}
        assert keys == {"session", "model"}

    @pytest.mark.unit
    def test_builds_rows_without_session(self) -> None:
        """Events without session_id produce only model rows."""
        event = {"model_id": "gpt-4o"}
        rows = _build_aggregation_rows([event])

        # 1 dimension (model) x 3 windows = 3 rows
        assert len(rows) == 3

    @pytest.mark.unit
    def test_empty_events_list(self) -> None:
        """Empty input produces no rows."""
        assert _build_aggregation_rows([]) == []

    @pytest.mark.unit
    def test_aggregation_key_format(self, sample_event: dict[str, object]) -> None:
        """Aggregation keys follow prefix:value format."""
        rows = _build_aggregation_rows([sample_event])

        for row in rows:
            key = row["aggregation_key"]
            assert ":" in key, f"Key {key} missing colon separator"
            prefix, value = key.split(":", 1)
            assert prefix in ("session", "model", "repo", "pattern")
            assert len(value) > 0

    @pytest.mark.unit
    def test_cost_and_tokens_propagation(self) -> None:
        """Cost and token values are propagated correctly to all rows."""
        event = {
            "model_id": "test-model",
            "estimated_cost_usd": 0.01,
            "total_tokens": 500,
            "usage_is_estimated": True,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["total_cost_usd"] == Decimal("0.01")
            assert row["total_tokens"] == 500
            assert row["call_count"] == 1
            assert row["estimated_coverage_pct"] == Decimal("100.00")

    @pytest.mark.unit
    def test_estimated_coverage_for_api_source(self) -> None:
        """API-sourced events get 0% estimated coverage."""
        event = {
            "model_id": "test-model",
            "usage_is_estimated": False,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["estimated_coverage_pct"] == Decimal("0.00")

    @pytest.mark.unit
    def test_aggregation_key_truncation(self) -> None:
        """Long keys are truncated to 512 characters."""
        event = {
            "model_id": "x" * 600,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert len(row["aggregation_key"]) <= 512

    @pytest.mark.unit
    def test_multiple_events_produce_multiple_rows(self) -> None:
        """Multiple events each produce their own set of rows."""
        events = [
            {"model_id": "model-a", "session_id": "s1"},
            {"model_id": "model-b", "session_id": "s2"},
        ]
        rows = _build_aggregation_rows(events)

        # Each event: 2 dimensions x 3 windows = 6 rows, total = 12
        assert len(rows) == 12

    @pytest.mark.unit
    def test_null_cost_defaults_to_zero(self) -> None:
        """None cost is treated as zero in aggregation."""
        event = {
            "model_id": "test-model",
            "estimated_cost_usd": None,
        }
        rows = _build_aggregation_rows([event])
        for row in rows:
            assert row["total_cost_usd"] == Decimal("0")

    @pytest.mark.unit
    def test_missing_optional_fields_produce_defaults(self) -> None:
        """Missing optional fields (cost, tokens, estimated) use safe defaults."""
        event = {"model_id": "bare-minimum-model"}
        rows = _build_aggregation_rows([event])

        assert len(rows) == 3  # 1 dimension (model) x 3 windows
        for row in rows:
            assert row["total_cost_usd"] == Decimal("0")
            assert row["total_tokens"] == 0
            assert row["call_count"] == 1
            # usage_is_estimated defaults to False -> 0% coverage
            assert row["estimated_coverage_pct"] == Decimal("0.00")


# =============================================================================
# Tests: Event Deduplication
# =============================================================================


class TestDeduplication:
    """Tests for event deduplication in the writer."""

    @pytest.mark.unit
    def test_duplicate_detection(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Duplicate event IDs are detected."""
        assert writer._is_duplicate("event-1") is False
        assert writer._is_duplicate("event-1") is True

    @pytest.mark.unit
    def test_different_events_not_duplicate(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Different event IDs are not duplicates."""
        assert writer._is_duplicate("event-1") is False
        assert writer._is_duplicate("event-2") is False

    @pytest.mark.unit
    def test_cache_eviction(self, writer: WriterLlmCostAggregationPostgres) -> None:
        """Cache evicts oldest entries when exceeding max size."""
        # Temporarily lower the max size for testing
        import omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres as mod

        original_max = mod._MAX_DEDUP_CACHE_SIZE
        mod._MAX_DEDUP_CACHE_SIZE = 5

        try:
            for i in range(10):
                writer._is_duplicate(f"event-{i}")

            # Oldest entries should be evicted
            assert len(writer._dedup_cache) == 5

            # First entries should be evicted (not found)
            assert writer._is_duplicate("event-0") is False  # Re-added after eviction
        finally:
            mod._MAX_DEDUP_CACHE_SIZE = original_max


# =============================================================================
# Tests: Usage Source Resolution
# =============================================================================


class TestResolveUsageSource:
    """Tests for _resolve_usage_source helper."""

    @pytest.mark.unit
    def test_api_source_from_normalized(self) -> None:
        """Resolves 'API' from usage_normalized.source."""
        event = {"usage_normalized": {"source": "api"}}
        assert _resolve_usage_source(event) == "API"

    @pytest.mark.unit
    def test_estimated_source_from_normalized(self) -> None:
        """Resolves 'ESTIMATED' from usage_normalized.source."""
        event = {"usage_normalized": {"source": "estimated"}}
        assert _resolve_usage_source(event) == "ESTIMATED"

    @pytest.mark.unit
    def test_missing_source_from_normalized(self) -> None:
        """Resolves 'MISSING' from usage_normalized.source."""
        event = {"usage_normalized": {"source": "missing"}}
        assert _resolve_usage_source(event) == "MISSING"

    @pytest.mark.unit
    def test_estimated_from_flag(self) -> None:
        """Falls back to usage_is_estimated flag."""
        event = {"usage_is_estimated": True}
        assert _resolve_usage_source(event) == "ESTIMATED"

    @pytest.mark.unit
    def test_api_from_token_presence(self) -> None:
        """Falls back to token presence for API detection."""
        event = {"total_tokens": 100, "usage_is_estimated": False}
        assert _resolve_usage_source(event) == "API"

    @pytest.mark.unit
    def test_missing_default(self) -> None:
        """Returns MISSING when no data available."""
        assert _resolve_usage_source({}) == "MISSING"


# =============================================================================
# Tests: Safe Type Conversions
# =============================================================================


class TestSafeConversions:
    """Tests for safe type conversion helpers."""

    @pytest.mark.unit
    def test_safe_int_from_int(self) -> None:
        assert _safe_int(42) == 42

    @pytest.mark.unit
    def test_safe_int_from_float(self) -> None:
        assert _safe_int(42.7) == 42

    @pytest.mark.unit
    def test_safe_int_from_string(self) -> None:
        assert _safe_int("42") == 42

    @pytest.mark.unit
    def test_safe_int_none(self) -> None:
        assert _safe_int(None) is None

    @pytest.mark.unit
    def test_safe_int_bool(self) -> None:
        assert _safe_int(True) is None

    @pytest.mark.unit
    def test_safe_int_invalid_string(self) -> None:
        assert _safe_int("abc") is None

    @pytest.mark.unit
    def test_safe_decimal_from_float(self) -> None:
        result = _safe_decimal(0.005)
        assert isinstance(result, Decimal)
        assert result == Decimal("0.005")

    @pytest.mark.unit
    def test_safe_decimal_from_string(self) -> None:
        assert _safe_decimal("1.23") == Decimal("1.23")

    @pytest.mark.unit
    def test_safe_decimal_none(self) -> None:
        assert _safe_decimal(None) is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_nan_string(self) -> None:
        assert _safe_decimal("NaN") is None
        assert _safe_decimal("nan") is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_infinity_string(self) -> None:
        assert _safe_decimal("Infinity") is None
        assert _safe_decimal("-Infinity") is None
        assert _safe_decimal("inf") is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_nan_decimal(self) -> None:
        assert _safe_decimal(Decimal("NaN")) is None
        assert _safe_decimal(Decimal("sNaN")) is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_infinity_decimal(self) -> None:
        assert _safe_decimal(Decimal("Infinity")) is None
        assert _safe_decimal(Decimal("-Infinity")) is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_nan_float(self) -> None:
        assert _safe_decimal(float("nan")) is None

    @pytest.mark.unit
    def test_safe_decimal_rejects_infinity_float(self) -> None:
        assert _safe_decimal(float("inf")) is None
        assert _safe_decimal(float("-inf")) is None

    @pytest.mark.unit
    def test_safe_uuid_from_string(self) -> None:
        uid = uuid4()
        assert _safe_uuid(str(uid)) == uid

    @pytest.mark.unit
    def test_safe_uuid_none(self) -> None:
        assert _safe_uuid(None) is None

    @pytest.mark.unit
    def test_safe_uuid_invalid(self) -> None:
        assert _safe_uuid("not-a-uuid") is None

    @pytest.mark.unit
    def test_safe_jsonb_from_dict(self) -> None:
        result = _safe_jsonb({"key": "value"})
        assert result is not None
        assert '"key"' in result

    @pytest.mark.unit
    def test_safe_jsonb_from_string(self) -> None:
        assert _safe_jsonb('{"a": 1}') == '{"a": 1}'

    @pytest.mark.unit
    def test_safe_jsonb_none(self) -> None:
        assert _safe_jsonb(None) is None


# =============================================================================
# Tests: Writer Methods
# =============================================================================


class TestWriterCallMetrics:
    """Tests for write_call_metrics method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_empty_batch(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Empty batch returns 0."""
        result = await writer.write_call_metrics([])
        assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_deduplicates_events(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
        sample_event: dict[str, object],
    ) -> None:
        """Events with same input_hash are deduplicated."""
        event1 = {**sample_event, "input_hash": "sha256-same"}
        event2 = {**sample_event, "input_hash": "sha256-same"}

        result = await writer.write_call_metrics([event1, event2])

        # Only one should be written (the first)
        assert result == 1

        # Verify execute was called for the INSERT (plus the SET LOCAL statement)
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        # SET LOCAL + 1 INSERT = 2 execute calls
        assert conn.execute.call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_call_metrics_handles_empty_after_dedup(
        self,
        writer: WriterLlmCostAggregationPostgres,
        sample_event: dict[str, object],
    ) -> None:
        """Returns 0 when all events are filtered as duplicates."""
        event = {**sample_event, "input_hash": "sha256-already-seen"}

        # Pre-populate dedup cache
        writer._is_duplicate("sha256-already-seen")

        result = await writer.write_call_metrics([event])
        assert result == 0


class TestWriterCostAggregates:
    """Tests for write_cost_aggregates method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_empty_batch(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Empty batch returns 0."""
        result = await writer.write_cost_aggregates([])
        assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_cost_aggregates_success_resets_circuit_breaker(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Successful write calls _reset_circuit_breaker, keeping state closed."""
        events = [{"model_id": "test-model", "input_hash": "unique-hash-agg-1"}]

        result = await writer.write_cost_aggregates(events)

        # 1 model dimension x 3 windows = 3 rows upserted
        assert result == 3

        # Circuit breaker should still be closed (0 failures)
        state = writer.get_circuit_breaker_state()
        assert state["state"] == "closed"
        assert state["failures"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_cost_aggregates_failure_records_circuit_failure(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Database errors call _record_circuit_failure, incrementing failure count."""
        # Make pool.acquire raise to simulate DB connection failure
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=ConnectionError("DB down")
        )

        events = [{"model_id": "test-model", "input_hash": "unique-hash-agg-2"}]

        with pytest.raises(ConnectionError, match="DB down"):
            await writer.write_cost_aggregates(events)

        # Circuit breaker should have recorded the failure
        state = writer.get_circuit_breaker_state()
        assert state["failures"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_call_metrics_failure_records_circuit_failure(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Database errors in write_call_metrics increment circuit breaker failures."""
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=ConnectionError("DB unreachable")
        )

        events = [{"model_id": "test-model", "input_hash": "unique-hash-cm-1"}]

        with pytest.raises(ConnectionError, match="DB unreachable"):
            await writer.write_call_metrics(events)

        state = writer.get_circuit_breaker_state()
        assert state["failures"] == 1


class TestCircuitBreakerState:
    """Tests for circuit breaker state reporting."""

    @pytest.mark.unit
    def test_get_circuit_breaker_state(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Circuit breaker state is reported correctly."""
        state = writer.get_circuit_breaker_state()
        assert "state" in state
        assert "failures" in state
        assert state["state"] == "closed"
        assert state["failures"] == 0
        assert state["threshold"] == 5
        assert state["reset_timeout_seconds"] == 60.0
