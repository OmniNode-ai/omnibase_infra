# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ReaderInjectionEffectivenessPostgres.

Tests query methods with mocked asyncpg pool.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelInjectionEffectivenessQuery,
    ModelInjectionEffectivenessRow,
    ModelLatencyBreakdownRow,
    ModelPatternHitRateRow,
)
from omnibase_infra.services.observability.injection_effectiveness.reader_postgres import (
    ReaderInjectionEffectivenessPostgres,
)

from .conftest import (
    make_effectiveness_row,
    make_latency_row,
    make_pattern_hit_rate_row,
)


class TestQueryBySessionId:
    """Tests for query_by_session_id()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_pool: MagicMock) -> None:
        """Returns None when session_id does not exist."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_by_session_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_row_when_found(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Returns ModelInjectionEffectivenessRow when found."""
        row_data = make_effectiveness_row(
            session_id=sample_session_id,
            correlation_id=sample_correlation_id,
        )
        mock_pool._test_conn.fetchrow = AsyncMock(return_value=row_data)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_by_session_id(
            sample_session_id, correlation_id=sample_correlation_id
        )

        assert result is not None
        assert isinstance(result, ModelInjectionEffectivenessRow)
        assert result.session_id == sample_session_id
        assert result.correlation_id == sample_correlation_id
        assert result.utilization_score == 0.75
        assert result.cohort == "treatment"

    @pytest.mark.asyncio
    async def test_sets_statement_timeout(
        self, mock_pool: MagicMock, sample_session_id
    ) -> None:
        """Verifies statement_timeout is set on connection."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool, query_timeout=15.0)
        await reader.query_by_session_id(sample_session_id)

        mock_pool._test_conn.execute.assert_any_call(
            "SET LOCAL statement_timeout = $1", "15000"
        )


class TestQuery:
    """Tests for query()."""

    @pytest.mark.asyncio
    async def test_empty_result(self, mock_pool: MagicMock) -> None:
        """Returns empty result for no matching rows."""
        mock_pool._test_conn.fetchval = AsyncMock(return_value=0)
        mock_pool._test_conn.fetch = AsyncMock(return_value=[])

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        query = ModelInjectionEffectivenessQuery()
        result = await reader.query(query)

        assert result.total_count == 0
        assert result.rows == ()
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_returns_paginated_results(self, mock_pool: MagicMock) -> None:
        """Returns paginated results with correct metadata."""
        rows = [make_effectiveness_row() for _ in range(3)]
        mock_pool._test_conn.fetchval = AsyncMock(return_value=10)
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        query = ModelInjectionEffectivenessQuery(limit=3, offset=0)
        result = await reader.query(query)

        assert result.total_count == 10
        assert len(result.rows) == 3
        assert result.has_more is True
        assert result.query == query

    @pytest.mark.asyncio
    async def test_has_more_false_at_end(self, mock_pool: MagicMock) -> None:
        """has_more is False when no more results exist."""
        rows = [make_effectiveness_row() for _ in range(2)]
        mock_pool._test_conn.fetchval = AsyncMock(return_value=2)
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        query = ModelInjectionEffectivenessQuery(limit=10, offset=0)
        result = await reader.query(query)

        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_filters_by_cohort(self, mock_pool: MagicMock) -> None:
        """Passes cohort filter to SQL query."""
        mock_pool._test_conn.fetchval = AsyncMock(return_value=0)
        mock_pool._test_conn.fetch = AsyncMock(return_value=[])

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        query = ModelInjectionEffectivenessQuery(cohort="control")
        await reader.query(query)

        # Verify the count query was called with cohort parameter
        count_call_args = mock_pool._test_conn.fetchval.call_args
        assert "control" in count_call_args.args

    @pytest.mark.asyncio
    async def test_rejects_offset_above_max(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when offset > 1000000."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        query = ModelInjectionEffectivenessQuery.model_construct(
            limit=100,
            offset=1000001,
            session_id=None,
            correlation_id=None,
            cohort=None,
            utilization_method=None,
            start_time=None,
            end_time=None,
        )
        with pytest.raises(ValueError, match="offset must be between 0 and 1000000"):
            await reader.query(query)

    @pytest.mark.asyncio
    async def test_rows_are_typed_models(self, mock_pool: MagicMock) -> None:
        """Result rows are ModelInjectionEffectivenessRow instances."""
        rows = [make_effectiveness_row()]
        mock_pool._test_conn.fetchval = AsyncMock(return_value=1)
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query(ModelInjectionEffectivenessQuery())

        assert isinstance(result.rows[0], ModelInjectionEffectivenessRow)


class TestQueryLatencyBreakdowns:
    """Tests for query_latency_breakdowns()."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, mock_pool: MagicMock) -> None:
        """Returns empty list when no breakdowns exist."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_latency_breakdowns(uuid4())
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_typed_rows(
        self, mock_pool: MagicMock, sample_session_id
    ) -> None:
        """Returns ModelLatencyBreakdownRow instances."""
        rows = [
            make_latency_row(session_id=sample_session_id),
            make_latency_row(session_id=sample_session_id),
        ]
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_latency_breakdowns(sample_session_id)

        assert len(result) == 2
        assert all(isinstance(r, ModelLatencyBreakdownRow) for r in result)
        assert all(r.session_id == sample_session_id for r in result)

    @pytest.mark.asyncio
    async def test_rejects_limit_below_one(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when limit < 1."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="limit must be between 1 and 10000"):
            await reader.query_latency_breakdowns(uuid4(), limit=0)

    @pytest.mark.asyncio
    async def test_rejects_limit_above_max(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when limit > 10000."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="limit must be between 1 and 10000"):
            await reader.query_latency_breakdowns(uuid4(), limit=10001)

    @pytest.mark.asyncio
    async def test_rejects_negative_offset(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when offset < 0."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            await reader.query_latency_breakdowns(uuid4(), offset=-1)


class TestQueryPatternHitRates:
    """Tests for query_pattern_hit_rates()."""

    @pytest.mark.asyncio
    async def test_returns_all_patterns(self, mock_pool: MagicMock) -> None:
        """Returns all patterns when no filter specified."""
        rows = [make_pattern_hit_rate_row(), make_pattern_hit_rate_row()]
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_pattern_hit_rates()

        assert len(result) == 2
        assert all(isinstance(r, ModelPatternHitRateRow) for r in result)

    @pytest.mark.asyncio
    async def test_filters_by_pattern_id(self, mock_pool: MagicMock) -> None:
        """Passes pattern_id filter to query."""
        pid = uuid4()
        rows = [make_pattern_hit_rate_row(pattern_id=pid)]
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_pattern_hit_rates(pattern_id=pid)

        assert len(result) == 1
        assert result[0].pattern_id == pid

    @pytest.mark.asyncio
    async def test_confident_only_flag(self, mock_pool: MagicMock) -> None:
        """Verifies confident_only adds confidence IS NOT NULL filter."""
        mock_pool._test_conn.fetch = AsyncMock(return_value=[])

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        await reader.query_pattern_hit_rates(confident_only=True)

        # Verify fetch was called with the SQL containing confidence filter
        fetch_call = mock_pool._test_conn.fetch.call_args
        sql = fetch_call.args[0]
        assert "confidence IS NOT NULL" in sql

    @pytest.mark.asyncio
    async def test_returns_confidence_for_sufficient_samples(
        self, mock_pool: MagicMock
    ) -> None:
        """Patterns with sample_count >= 20 have non-null confidence."""
        rows = [make_pattern_hit_rate_row(sample_count=25)]
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_pattern_hit_rates()

        assert result[0].confidence is not None

    @pytest.mark.asyncio
    async def test_returns_null_confidence_for_insufficient_samples(
        self, mock_pool: MagicMock
    ) -> None:
        """Patterns with sample_count < 20 have null confidence."""
        rows = [make_pattern_hit_rate_row(sample_count=5)]
        mock_pool._test_conn.fetch = AsyncMock(return_value=rows)

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        result = await reader.query_pattern_hit_rates()

        assert result[0].confidence is None

    @pytest.mark.asyncio
    async def test_rejects_limit_below_one(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when limit < 1."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="limit must be between 1 and 10000"):
            await reader.query_pattern_hit_rates(limit=0)

    @pytest.mark.asyncio
    async def test_rejects_limit_above_max(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when limit > 10000."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="limit must be between 1 and 10000"):
            await reader.query_pattern_hit_rates(limit=10001)

    @pytest.mark.asyncio
    async def test_rejects_negative_offset(self, mock_pool: MagicMock) -> None:
        """Raises ValueError when offset < 0."""
        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            await reader.query_pattern_hit_rates(offset=-1)


class TestReaderProtocolCompliance:
    """Tests that the reader satisfies ProtocolInjectionEffectivenessReader."""

    def test_satisfies_protocol(self, mock_pool: MagicMock) -> None:
        """ReaderInjectionEffectivenessPostgres satisfies the protocol."""
        from omnibase_infra.services.observability.injection_effectiveness.protocol_reader import (
            ProtocolInjectionEffectivenessReader,
        )

        reader = ReaderInjectionEffectivenessPostgres(mock_pool)
        assert isinstance(reader, ProtocolInjectionEffectivenessReader)
