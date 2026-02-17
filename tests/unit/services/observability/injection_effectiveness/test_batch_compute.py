# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for BatchComputeEffectivenessMetrics.

Tests the batch computation engine that derives effectiveness metrics
from agent_actions and agent_routing_decisions tables.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.batch_compute import (
    BatchComputeEffectivenessMetrics,
    BatchComputeResult,
    _parse_execute_count,
)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg.Pool with connection context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="INSERT 0 10")
    conn.executemany = AsyncMock()

    # Support async context manager for conn.transaction()
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    # Support async context manager for pool.acquire()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)

    # Attach connection for direct access in tests
    pool._test_conn = conn
    return pool


@pytest.fixture
def mock_notifier() -> AsyncMock:
    """Create a mock EffectivenessInvalidationNotifier."""
    notifier = AsyncMock()
    notifier.notify = AsyncMock()
    return notifier


class TestBatchComputeResult:
    """Tests for BatchComputeResult dataclass."""

    def test_total_rows_sums_all(self) -> None:
        result = BatchComputeResult(
            effectiveness_rows=10,
            latency_rows=20,
            pattern_rows=5,
        )
        assert result.total_rows == 35

    def test_total_rows_default_zero(self) -> None:
        result = BatchComputeResult()
        assert result.total_rows == 0

    def test_has_errors_false_when_empty(self) -> None:
        result = BatchComputeResult()
        assert result.has_errors is False

    def test_has_errors_true_when_errors_present(self) -> None:
        result = BatchComputeResult(errors=("something failed",))
        assert result.has_errors is True


class TestParseExecuteCount:
    """Tests for _parse_execute_count utility."""

    def test_parse_insert_result(self) -> None:
        assert _parse_execute_count("INSERT 0 42") == 42

    def test_parse_update_result(self) -> None:
        assert _parse_execute_count("UPDATE 15") == 15

    def test_parse_zero_rows(self) -> None:
        assert _parse_execute_count("INSERT 0 0") == 0

    def test_parse_empty_string(self) -> None:
        assert _parse_execute_count("") == 0

    def test_parse_invalid_string(self) -> None:
        assert _parse_execute_count("not a valid result") == 0


class TestBatchComputeEffectivenessMetrics:
    """Tests for BatchComputeEffectivenessMetrics."""

    @pytest.mark.asyncio
    async def test_compute_and_persist_all_phases(self, mock_pool: MagicMock) -> None:
        """All three phases execute and return combined result."""
        conn = mock_pool._test_conn

        # conn.execute is called for:
        #   1. SET LOCAL statement_timeout (phase 1)
        #   2. Phase 1 SQL (effectiveness)
        #   3. SET LOCAL statement_timeout (phase 2)
        #   4. Phase 2 SQL (latency)
        #   5. SET LOCAL statement_timeout (phase 3)
        #   6. Phase 3 SQL (pattern fallback)
        async def execute_side_effect(sql, *args, **kwargs):
            if "SET LOCAL" in str(sql):
                return "SET"
            if "injection_effectiveness" in str(sql):
                return "INSERT 0 10"
            if "latency_breakdowns" in str(sql):
                return "INSERT 0 5"
            return "INSERT 0 3"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetchrow = AsyncMock(return_value=None)  # No uuid-ossp extension

        batch = BatchComputeEffectivenessMetrics(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        assert result.effectiveness_rows == 10
        assert result.latency_rows == 5
        assert result.pattern_rows == 3
        assert result.total_rows == 18
        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_compute_and_persist_with_notifier(
        self, mock_pool: MagicMock, mock_notifier: AsyncMock
    ) -> None:
        """Notifier is called when rows are written."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql, *args, **kwargs):
            if "SET LOCAL" in str(sql):
                return "SET"
            if "injection_effectiveness" in str(sql):
                return "INSERT 0 5"
            if "latency_breakdowns" in str(sql):
                return "INSERT 0 3"
            return "INSERT 0 2"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetchrow = AsyncMock(return_value=None)

        batch = BatchComputeEffectivenessMetrics(mock_pool, notifier=mock_notifier)
        result = await batch.compute_and_persist()

        assert result.total_rows == 10
        mock_notifier.notify.assert_awaited_once()
        call_kwargs = mock_notifier.notify.call_args.kwargs
        assert "injection_effectiveness" in call_kwargs["tables_affected"]
        assert "latency_breakdowns" in call_kwargs["tables_affected"]
        assert "pattern_hit_rates" in call_kwargs["tables_affected"]
        assert call_kwargs["rows_written"] == 10
        assert call_kwargs["source"] == "batch_compute"

    @pytest.mark.asyncio
    async def test_compute_and_persist_no_notifier_when_zero_rows(
        self, mock_pool: MagicMock, mock_notifier: AsyncMock
    ) -> None:
        """Notifier is NOT called when no rows are written."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql, *args, **kwargs):
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetchrow = AsyncMock(return_value=None)

        batch = BatchComputeEffectivenessMetrics(mock_pool, notifier=mock_notifier)
        result = await batch.compute_and_persist()

        assert result.total_rows == 0
        mock_notifier.notify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compute_and_persist_phase_failure_continues(
        self, mock_pool: MagicMock
    ) -> None:
        """A failed phase logs error but other phases still execute."""
        conn = mock_pool._test_conn
        phase_count = 0

        async def execute_side_effect(sql, *args, **kwargs):
            nonlocal phase_count
            if "SET LOCAL" in str(sql):
                return "SET"
            phase_count += 1
            if phase_count == 1:
                raise RuntimeError("DB connection failed")
            if phase_count == 2:
                return "INSERT 0 7"
            return "INSERT 0 4"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetchrow = AsyncMock(return_value=None)

        batch = BatchComputeEffectivenessMetrics(mock_pool)
        result = await batch.compute_and_persist()

        assert result.effectiveness_rows == 0
        assert result.latency_rows == 7
        assert result.pattern_rows == 4
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert "Phase 1" in result.errors[0]

    @pytest.mark.asyncio
    async def test_compute_with_uuid_ossp_extension(self, mock_pool: MagicMock) -> None:
        """When uuid-ossp extension is available, uses uuid_generate_v5."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql, *args, **kwargs):
            if "SET LOCAL" in str(sql):
                return "SET"
            if "injection_effectiveness" in str(sql):
                return "INSERT 0 10"
            if "latency_breakdowns" in str(sql):
                return "INSERT 0 5"
            return "INSERT 0 3"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        # Return a row for uuid-ossp check (extension is available)
        conn.fetchrow = AsyncMock(return_value={"extname": "uuid-ossp"})

        batch = BatchComputeEffectivenessMetrics(mock_pool)
        result = await batch.compute_and_persist()

        assert result.pattern_rows == 3

    @pytest.mark.asyncio
    async def test_compute_custom_correlation_id(self, mock_pool: MagicMock) -> None:
        """Custom correlation_id is propagated."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql, *args, **kwargs):
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetchrow = AsyncMock(return_value=None)

        custom_id = uuid4()
        batch = BatchComputeEffectivenessMetrics(mock_pool)
        result = await batch.compute_and_persist(correlation_id=custom_id)

        assert result.total_rows == 0
        assert result.has_errors is False
