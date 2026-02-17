# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Edge case and validation tests for WriterLlmCostAggregationPostgres.

Supplements test_writer_postgres.py with focused coverage on:
1. DB CHECK constraint alignment via safe conversion helpers
2. NULL vs zero cost distinction in aggregation rows
3. Aggregate consistency when invalid data is attempted
4. Pre-aggregation correctness with edge-case inputs
5. Safe type conversion boundary conditions

Cost data is financial data -- silent coercion or zero-filling would
corrupt cost analytics.  These tests verify that the writer's helper
functions enforce the same semantic distinctions as the PostgreSQL
CHECK constraints.

Related Tickets:
    - OMN-2295: LLM cost tracking: input validation and edge case tests
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres import (
    WriterLlmCostAggregationPostgres,
    _build_aggregation_rows,
    _derive_stable_dedup_key,
    _has_empty_dedup_fields,
    _pre_aggregate_rows,
    _safe_decimal,
    _safe_int,
    _safe_int_or_zero,
    _safe_numeric_or_none,
    _sanitize_dimension_value,
    _truncate_input_hash,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with proper async context manager support."""
    pool = MagicMock()
    conn = AsyncMock()

    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)
    conn.execute = AsyncMock()

    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    return pool


@pytest.fixture
def writer(mock_pool: MagicMock) -> WriterLlmCostAggregationPostgres:
    """Create a writer instance with mock pool."""
    return WriterLlmCostAggregationPostgres(pool=mock_pool)


# =============================================================================
# Tests: _safe_decimal -- DB CHECK constraint alignment
# =============================================================================


class TestSafeDecimalEdgeCases:
    """_safe_decimal must align with PostgreSQL NUMERIC(12,6) CHECK constraints.

    The DB has CHECK (estimated_cost_usd IS NULL OR estimated_cost_usd >= 0).
    _safe_decimal is the gateway that converts event data before insertion.
    """

    @pytest.mark.unit
    def test_negative_decimal_passes_through(self) -> None:
        """_safe_decimal does NOT reject negatives -- the DB CHECK does.

        _safe_decimal is a type converter, not a business validator. The
        PostgreSQL CHECK constraint handles non-negative enforcement.
        """
        result = _safe_decimal(-0.01)
        assert result == Decimal("-0.01")

    @pytest.mark.unit
    def test_zero_decimal_accepted(self) -> None:
        """Zero cost is a valid value (free tier)."""
        result = _safe_decimal(0)
        assert result == Decimal("0")

    @pytest.mark.unit
    def test_zero_decimal_from_string(self) -> None:
        """'0' converts to Decimal zero."""
        result = _safe_decimal("0")
        assert result == Decimal("0")

    @pytest.mark.unit
    def test_zero_decimal_from_float(self) -> None:
        """0.0 converts to Decimal zero."""
        result = _safe_decimal(0.0)
        assert result == Decimal("0.0")

    @pytest.mark.unit
    def test_none_returns_none(self) -> None:
        """None maps to SQL NULL."""
        assert _safe_decimal(None) is None

    @pytest.mark.unit
    def test_very_small_positive(self) -> None:
        """Very small positive decimal is preserved."""
        result = _safe_decimal("0.000001")
        assert result == Decimal("0.000001")

    @pytest.mark.unit
    def test_very_large_positive(self) -> None:
        """Large positive decimal is preserved."""
        result = _safe_decimal("999999.999999")
        assert result == Decimal("999999.999999")

    @pytest.mark.unit
    def test_nan_string_returns_none(self) -> None:
        """NaN string returns None to prevent aggregation corruption."""
        assert _safe_decimal("NaN") is None

    @pytest.mark.unit
    def test_nan_float_returns_none(self) -> None:
        """NaN float returns None."""
        assert _safe_decimal(float("nan")) is None

    @pytest.mark.unit
    def test_infinity_returns_none(self) -> None:
        """Infinity returns None to prevent aggregation corruption."""
        assert _safe_decimal(float("inf")) is None
        assert _safe_decimal(float("-inf")) is None
        assert _safe_decimal("Infinity") is None

    @pytest.mark.unit
    def test_non_numeric_string_returns_none(self) -> None:
        """Non-numeric strings return None."""
        assert _safe_decimal("abc") is None
        assert _safe_decimal("$1.50") is None
        assert _safe_decimal("") is None

    @pytest.mark.unit
    def test_bool_input_returns_none(self) -> None:
        """Boolean inputs return None (str(True)='True' is not a valid Decimal)."""
        assert _safe_decimal(True) is None
        assert _safe_decimal(False) is None


# =============================================================================
# Tests: _safe_int -- Token field boundary conditions
# =============================================================================


class TestSafeIntEdgeCases:
    """_safe_int converts event data to integers for token columns.

    The DB has CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0).
    Like _safe_decimal, _safe_int is a type converter, not a business validator.
    """

    @pytest.mark.unit
    def test_negative_int_passes_through(self) -> None:
        """Negative int is not rejected by _safe_int (DB CHECK handles it)."""
        result = _safe_int(-5)
        assert result == -5

    @pytest.mark.unit
    def test_zero_preserved(self) -> None:
        """Zero is a valid token count."""
        assert _safe_int(0) == 0

    @pytest.mark.unit
    def test_bool_true_returns_none(self) -> None:
        """bool(True) returns None to prevent int(True)=1 confusion."""
        assert _safe_int(True) is None

    @pytest.mark.unit
    def test_bool_false_returns_none(self) -> None:
        """bool(False) returns None to prevent int(False)=0 confusion."""
        assert _safe_int(False) is None

    @pytest.mark.unit
    def test_nan_float_returns_none(self) -> None:
        """NaN float returns None."""
        assert _safe_int(float("nan")) is None

    @pytest.mark.unit
    def test_inf_float_returns_none(self) -> None:
        """Infinity float returns None."""
        assert _safe_int(float("inf")) is None
        assert _safe_int(float("-inf")) is None

    @pytest.mark.unit
    def test_float_truncates(self) -> None:
        """Float is truncated to int (not rounded)."""
        assert _safe_int(42.9) == 42
        assert _safe_int(42.1) == 42

    @pytest.mark.unit
    def test_large_int(self) -> None:
        """Very large integers are preserved."""
        assert _safe_int(2**31 - 1) == 2**31 - 1

    @pytest.mark.unit
    def test_numeric_string(self) -> None:
        """Numeric string converts."""
        assert _safe_int("42") == 42

    @pytest.mark.unit
    def test_non_numeric_string_returns_none(self) -> None:
        """Non-numeric string returns None."""
        assert _safe_int("abc") is None
        assert _safe_int("") is None


# =============================================================================
# Tests: _safe_int_or_zero
# =============================================================================


class TestSafeIntOrZeroEdgeCases:
    """_safe_int_or_zero correctly distinguishes legitimate 0 from None."""

    @pytest.mark.unit
    def test_legitimate_zero_preserved(self) -> None:
        """int(0) is preserved, not coerced via falsy 'or 0' path."""
        assert _safe_int_or_zero(0) == 0

    @pytest.mark.unit
    def test_none_becomes_zero(self) -> None:
        """None maps to 0 for aggregation safety."""
        assert _safe_int_or_zero(None) == 0

    @pytest.mark.unit
    def test_invalid_string_becomes_zero(self) -> None:
        """Invalid string converts to 0 via None -> 0 path."""
        assert _safe_int_or_zero("abc") == 0

    @pytest.mark.unit
    def test_valid_string_converts(self) -> None:
        """Valid numeric string converts to int."""
        assert _safe_int_or_zero("42") == 42


# =============================================================================
# Tests: _safe_numeric_or_none -- Latency column edge cases
# =============================================================================


class TestSafeNumericOrNoneEdgeCases:
    """_safe_numeric_or_none for the NUMERIC(10,2) latency_ms column."""

    @pytest.mark.unit
    def test_none_returns_none(self) -> None:
        """None maps to SQL NULL (no latency data)."""
        assert _safe_numeric_or_none(None) is None

    @pytest.mark.unit
    def test_zero_preserved(self) -> None:
        """Zero latency is valid (not coerced to None)."""
        assert _safe_numeric_or_none(0) == 0.0

    @pytest.mark.unit
    def test_sub_millisecond_rounded(self) -> None:
        """Sub-millisecond precision is rounded to 2 decimal places."""
        result = _safe_numeric_or_none(1.2345)
        assert result == 1.23

    @pytest.mark.unit
    def test_bool_returns_none(self) -> None:
        """Booleans return None to prevent confusion."""
        assert _safe_numeric_or_none(True) is None
        assert _safe_numeric_or_none(False) is None

    @pytest.mark.unit
    def test_nan_returns_none(self) -> None:
        """NaN returns None."""
        assert _safe_numeric_or_none(float("nan")) is None

    @pytest.mark.unit
    def test_inf_returns_none(self) -> None:
        """Infinity returns None."""
        assert _safe_numeric_or_none(float("inf")) is None

    @pytest.mark.unit
    def test_negative_float(self) -> None:
        """Negative float passes through (_safe_numeric does not validate sign)."""
        result = _safe_numeric_or_none(-1.5)
        assert result == -1.5

    @pytest.mark.unit
    def test_string_numeric(self) -> None:
        """Numeric string converts to float."""
        result = _safe_numeric_or_none("42.567")
        assert result == 42.57

    @pytest.mark.unit
    def test_non_numeric_string_returns_none(self) -> None:
        """Non-numeric string returns None."""
        assert _safe_numeric_or_none("abc") is None


# =============================================================================
# Tests: NULL cost distinction in aggregation rows
# =============================================================================


class TestAggregationRowsCostNullVsZero:
    """Verify aggregation rows correctly handle NULL vs zero cost.

    _build_aggregation_rows converts None cost to Decimal("0") for
    the aggregation sum. This is intentional: aggregate totals must be
    numeric for SQL SUM operations. The llm_call_metrics table preserves
    NULL cost for individual records; aggregates always have a numeric
    total_cost_usd.
    """

    @pytest.mark.unit
    def test_null_cost_event_produces_zero_aggregate_cost(self) -> None:
        """Event with None cost produces aggregation row with Decimal(0) cost.

        NULL cost means 'unknown' at the record level, but in aggregation
        the cost contribution is zero (not counted, not assumed).
        """
        event: dict[str, object] = {
            "model_id": "unknown-model",
            "estimated_cost_usd": None,
            "total_tokens": 100,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["total_cost_usd"] == Decimal("0")

    @pytest.mark.unit
    def test_zero_cost_event_produces_zero_aggregate_cost(self) -> None:
        """Event with 0.0 cost produces aggregation row with Decimal(0) cost."""
        event: dict[str, object] = {
            "model_id": "free-model",
            "estimated_cost_usd": 0.0,
            "total_tokens": 100,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["total_cost_usd"] == Decimal("0.0")

    @pytest.mark.unit
    def test_positive_cost_event_preserved(self) -> None:
        """Event with positive cost produces correct aggregation cost."""
        event: dict[str, object] = {
            "model_id": "gpt-4o",
            "estimated_cost_usd": 0.005,
            "total_tokens": 150,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["total_cost_usd"] == Decimal("0.005")

    @pytest.mark.unit
    def test_null_tokens_event_produces_zero_aggregate_tokens(self) -> None:
        """Event with None tokens produces aggregation row with 0 tokens."""
        event: dict[str, object] = {
            "model_id": "model",
            "total_tokens": None,
        }
        rows = _build_aggregation_rows([event])

        for row in rows:
            assert row["total_tokens"] == 0

    @pytest.mark.unit
    def test_mixed_cost_events_aggregate_correctly(self) -> None:
        """Events with mixed NULL/zero/positive costs aggregate correctly.

        NULL costs contribute 0 to the sum, positive costs are summed.
        Pre-aggregation merges rows with the same (key, window) pair.
        """
        events: list[dict[str, object]] = [
            {"model_id": "same-model", "estimated_cost_usd": None, "total_tokens": 100},
            {"model_id": "same-model", "estimated_cost_usd": 0.0, "total_tokens": 200},
            {"model_id": "same-model", "estimated_cost_usd": 0.01, "total_tokens": 300},
        ]
        raw_rows = _build_aggregation_rows(events)
        aggregated = _pre_aggregate_rows(raw_rows)

        # All three events share model_id -> one key per window -> 3 aggregated rows
        model_rows = [
            r for r in aggregated if str(r["aggregation_key"]).startswith("model:")
        ]
        assert len(model_rows) == 3  # one per window

        for row in model_rows:
            # NULL(0) + 0.0 + 0.01 = 0.01
            assert row["total_cost_usd"] == Decimal("0") + Decimal("0.0") + Decimal(
                "0.01"
            )
            assert row["total_tokens"] == 600
            assert row["call_count"] == 3


# =============================================================================
# Tests: Pre-aggregation edge cases
# =============================================================================


class TestPreAggregateRowsEdgeCases:
    """Edge cases for _pre_aggregate_rows."""

    @pytest.mark.unit
    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert _pre_aggregate_rows([]) == []

    @pytest.mark.unit
    def test_single_row_passthrough(self) -> None:
        """Single row passes through unchanged."""
        row = {
            "aggregation_key": "model:gpt-4o",
            "window": "24h",
            "total_cost_usd": Decimal("0.01"),
            "total_tokens": 100,
            "call_count": 1,
            "estimated_coverage_pct": Decimal("0.00"),
        }
        result = _pre_aggregate_rows([row])
        assert len(result) == 1
        assert result[0]["total_cost_usd"] == Decimal("0.01")

    @pytest.mark.unit
    def test_duplicate_keys_merged(self) -> None:
        """Rows with same (aggregation_key, window) are merged."""
        rows = [
            {
                "aggregation_key": "model:gpt-4o",
                "window": "24h",
                "total_cost_usd": Decimal("0.01"),
                "total_tokens": 100,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("0.00"),
            },
            {
                "aggregation_key": "model:gpt-4o",
                "window": "24h",
                "total_cost_usd": Decimal("0.02"),
                "total_tokens": 200,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("100.00"),
            },
        ]
        result = _pre_aggregate_rows(rows)
        assert len(result) == 1
        assert result[0]["total_cost_usd"] == Decimal("0.03")
        assert result[0]["total_tokens"] == 300
        assert result[0]["call_count"] == 2
        # Weighted average: (0 * 1 + 100 * 1) / 2 = 50
        assert result[0]["estimated_coverage_pct"] == Decimal("50.00")

    @pytest.mark.unit
    def test_different_windows_not_merged(self) -> None:
        """Rows with same key but different windows stay separate."""
        rows = [
            {
                "aggregation_key": "model:gpt-4o",
                "window": "24h",
                "total_cost_usd": Decimal("0.01"),
                "total_tokens": 100,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("0.00"),
            },
            {
                "aggregation_key": "model:gpt-4o",
                "window": "7d",
                "total_cost_usd": Decimal("0.02"),
                "total_tokens": 200,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("0.00"),
            },
        ]
        result = _pre_aggregate_rows(rows)
        assert len(result) == 2

    @pytest.mark.unit
    def test_zero_cost_rows_merge_correctly(self) -> None:
        """Multiple zero-cost rows merge to zero total."""
        rows = [
            {
                "aggregation_key": "model:free",
                "window": "24h",
                "total_cost_usd": Decimal("0"),
                "total_tokens": 10,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("0.00"),
            },
            {
                "aggregation_key": "model:free",
                "window": "24h",
                "total_cost_usd": Decimal("0"),
                "total_tokens": 20,
                "call_count": 1,
                "estimated_coverage_pct": Decimal("0.00"),
            },
        ]
        result = _pre_aggregate_rows(rows)
        assert len(result) == 1
        assert result[0]["total_cost_usd"] == Decimal("0")
        assert result[0]["total_tokens"] == 30


# =============================================================================
# Tests: Dimension value sanitization
# =============================================================================


class TestSanitizeDimensionValue:
    """_sanitize_dimension_value prevents aggregation key format corruption."""

    @pytest.mark.unit
    def test_colon_replaced_with_underscore(self) -> None:
        """Colons are replaced to avoid prefix:value ambiguity."""
        result = _sanitize_dimension_value("model:gpt-4o")
        assert ":" not in result

    @pytest.mark.unit
    def test_control_chars_removed(self) -> None:
        """Control characters are stripped."""
        result = _sanitize_dimension_value("model\x00name\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    @pytest.mark.unit
    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped."""
        result = _sanitize_dimension_value("  model-name  ")
        assert result == "model-name"

    @pytest.mark.unit
    def test_normal_string_unchanged(self) -> None:
        """Normal strings pass through unchanged."""
        result = _sanitize_dimension_value("gpt-4o")
        assert result == "gpt-4o"


# =============================================================================
# Tests: Input hash truncation
# =============================================================================


class TestTruncateInputHash:
    """_truncate_input_hash enforces VARCHAR(71) column limit."""

    @pytest.mark.unit
    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None for SQL NULL."""
        assert _truncate_input_hash("") is None

    @pytest.mark.unit
    def test_standard_sha256_preserved(self) -> None:
        """Standard sha256-<64hex> (71 chars total) is preserved."""
        sha = "sha256-" + "a" * 64
        assert len(sha) == 71
        result = _truncate_input_hash(sha)
        assert result == sha

    @pytest.mark.unit
    def test_long_hash_truncated(self) -> None:
        """Hashes longer than 71 chars are truncated."""
        long_hash = "sha256-" + "a" * 100
        result = _truncate_input_hash(long_hash)
        assert result is not None
        assert len(result) <= 71


# =============================================================================
# Tests: Dedup key derivation edge cases
# =============================================================================


class TestDedupKeyEdgeCases:
    """_derive_stable_dedup_key handles empty/missing fields."""

    @pytest.mark.unit
    def test_input_hash_used_when_long_enough(self) -> None:
        """input_hash >= 8 chars is used directly as dedup key."""
        event: dict[str, object] = {"input_hash": "sha256-abcdef0123456789"}
        key = _derive_stable_dedup_key(event)
        assert key == "sha256-abcdef0123456789"

    @pytest.mark.unit
    def test_short_input_hash_falls_through(self) -> None:
        """input_hash < 8 chars falls through to composite key."""
        event: dict[str, object] = {"input_hash": "short", "model_id": "test"}
        key = _derive_stable_dedup_key(event)
        # Should be a SHA-256 hash (64 hex chars), not "short"
        assert key != "short"
        assert len(key) == 64  # SHA-256 hex digest

    @pytest.mark.unit
    def test_empty_event_produces_deterministic_key(self) -> None:
        """Completely empty event produces a deterministic composite key."""
        key1 = _derive_stable_dedup_key({})
        key2 = _derive_stable_dedup_key({})
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest

    @pytest.mark.unit
    def test_has_empty_dedup_fields_true_for_empty(self) -> None:
        """Empty event has all dedup fields empty."""
        assert _has_empty_dedup_fields({}) is True

    @pytest.mark.unit
    def test_has_empty_dedup_fields_false_with_input_hash(self) -> None:
        """Event with long input_hash has reliable dedup."""
        event: dict[str, object] = {"input_hash": "sha256-abcdef0123456789"}
        assert _has_empty_dedup_fields(event) is False

    @pytest.mark.unit
    def test_has_empty_dedup_fields_false_with_model_id(self) -> None:
        """Event with model_id has some dedup data."""
        event: dict[str, object] = {"model_id": "gpt-4o"}
        assert _has_empty_dedup_fields(event) is False


# =============================================================================
# Tests: Writer initialization validation
# =============================================================================


class TestWriterInitializationValidation:
    """WriterLlmCostAggregationPostgres constructor validation."""

    @pytest.mark.unit
    def test_zero_query_timeout_rejected(self) -> None:
        """query_timeout=0 is rejected."""
        pool = MagicMock()
        with pytest.raises(ProtocolConfigurationError, match="query_timeout"):
            WriterLlmCostAggregationPostgres(pool=pool, query_timeout=0)

    @pytest.mark.unit
    def test_negative_query_timeout_rejected(self) -> None:
        """Negative query_timeout is rejected."""
        pool = MagicMock()
        with pytest.raises(ProtocolConfigurationError, match="query_timeout"):
            WriterLlmCostAggregationPostgres(pool=pool, query_timeout=-1.0)

    @pytest.mark.unit
    def test_nan_query_timeout_rejected(self) -> None:
        """NaN query_timeout is rejected."""
        pool = MagicMock()
        with pytest.raises(ProtocolConfigurationError, match="query_timeout"):
            WriterLlmCostAggregationPostgres(pool=pool, query_timeout=float("nan"))

    @pytest.mark.unit
    def test_inf_query_timeout_rejected(self) -> None:
        """Infinity query_timeout is rejected."""
        pool = MagicMock()
        with pytest.raises(ProtocolConfigurationError, match="query_timeout"):
            WriterLlmCostAggregationPostgres(pool=pool, query_timeout=float("inf"))

    @pytest.mark.unit
    def test_valid_query_timeout_accepted(self) -> None:
        """Positive finite query_timeout is accepted."""
        pool = MagicMock()
        writer = WriterLlmCostAggregationPostgres(pool=pool, query_timeout=5.0)
        assert writer._query_timeout == 5.0

    @pytest.mark.unit
    def test_statement_timeout_ms_bounded(self) -> None:
        """_statement_timeout_ms is bounded to [1, 600_000]."""
        pool = MagicMock()
        writer = WriterLlmCostAggregationPostgres(pool=pool, query_timeout=0.0001)
        assert writer._statement_timeout_ms() >= 1

        writer2 = WriterLlmCostAggregationPostgres(pool=pool, query_timeout=700.0)
        assert writer2._statement_timeout_ms() <= 600_000


# =============================================================================
# Tests: Writer batch handling with edge-case events
# =============================================================================


class TestWriterBatchEdgeCases:
    """Batch write edge cases with NULL/zero/invalid data."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_call_metrics_empty_returns_zero(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Empty batch returns 0 without touching the database."""
        result = await writer.write_call_metrics([])
        assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_cost_aggregates_empty_returns_zero(
        self, writer: WriterLlmCostAggregationPostgres
    ) -> None:
        """Empty aggregation batch returns 0."""
        result = await writer.write_cost_aggregates([])
        assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_call_metrics_with_null_cost(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Event with null cost is written (NULL preserved for the column)."""
        event: dict[str, object] = {
            "model_id": "unknown-model",
            "estimated_cost_usd": None,
            "total_tokens": 100,
            "input_hash": "sha256-" + "b" * 64,
        }
        result = await writer.write_call_metrics([event])
        assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_call_metrics_with_zero_cost(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Event with zero cost is written (0.0 preserved, not coerced to NULL)."""
        event = {
            "model_id": "free-model",
            "estimated_cost_usd": 0.0,
            "total_tokens": 100,
            "input_hash": "sha256-" + "c" * 64,
        }
        result = await writer.write_call_metrics([event])
        assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_cost_aggregates_with_null_cost_event(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Aggregation of event with null cost uses 0 for the sum."""
        event: dict[str, object] = {
            "model_id": "test-model",
            "estimated_cost_usd": None,
            "total_tokens": 100,
            "input_hash": "sha256-" + "d" * 64,
        }
        result = await writer.write_cost_aggregates([event])
        # 1 model dimension x 3 windows = 3 aggregate rows
        assert result == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_dedup_within_single_call(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Duplicate events within a single batch are deduplicated."""
        hash_val = "sha256-" + "e" * 64
        events: list[dict[str, object]] = [
            {"model_id": "m1", "input_hash": hash_val},
            {"model_id": "m1", "input_hash": hash_val},
            {"model_id": "m1", "input_hash": hash_val},
        ]
        result = await writer.write_call_metrics(events)
        assert result == 1  # Only one unique event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_dedup_across_calls(
        self,
        writer: WriterLlmCostAggregationPostgres,
        mock_pool: MagicMock,
    ) -> None:
        """Events already seen in prior calls are deduplicated."""
        hash_val = "sha256-" + "f" * 64
        events: list[dict[str, object]] = [{"model_id": "m1", "input_hash": hash_val}]

        # First write succeeds
        result1 = await writer.write_call_metrics(events)
        assert result1 == 1

        # Second write is deduplicated
        result2 = await writer.write_call_metrics(events)
        assert result2 == 0
