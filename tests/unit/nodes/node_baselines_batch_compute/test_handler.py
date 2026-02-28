# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerBaselinesBatchCompute.

Ports all existing ServiceBatchComputeBaselines tests plus adds
handler-specific tests per plan (D5, D6 policies).

Ticket: OMN-3044
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_baselines_batch_compute.handlers.handler_baselines_batch_compute import (
    HandlerBaselinesBatchCompute,
)
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_command import (
    ModelBaselinesBatchComputeCommand,
)
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_output import (
    ModelBaselinesBatchComputeOutput,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg.Pool with connection context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="INSERT 0 5")
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
def mock_publisher() -> AsyncMock:
    """Create a mock publisher callable."""
    pub = AsyncMock(return_value=True)
    return pub


@pytest.fixture
def mock_publisher_raises() -> AsyncMock:
    """Create a mock publisher that raises on call."""
    pub = AsyncMock(side_effect=RuntimeError("Kafka unavailable"))
    return pub


class TestHandlerBaselinesBatchCompute:
    """Tests for HandlerBaselinesBatchCompute — ported from service tests."""

    @pytest.mark.asyncio
    async def test_handle_all_phases(self, mock_pool: MagicMock) -> None:
        """All three phases execute and return combined result."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            if "INSERT INTO baselines_comparisons" in str(sql):
                return "INSERT 0 7"
            if "INSERT INTO baselines_trend" in str(sql):
                return "INSERT 0 14"
            if "INSERT INTO baselines_breakdown" in str(sql):
                return "INSERT 0 3"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.comparisons_rows == 7
        assert out.result.trend_rows == 14
        assert out.result.breakdown_rows == 3
        assert out.result.total_rows == 24
        assert out.result.has_errors is False
        assert out.result.completed_at >= out.result.started_at

    @pytest.mark.asyncio
    async def test_handle_phase_failure_is_isolated(self, mock_pool: MagicMock) -> None:
        """Phase 1 failure does not prevent phases 2 and 3 from running."""
        conn = mock_pool._test_conn
        call_count = 0

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            nonlocal call_count
            if "SET LOCAL" in str(sql):
                return "SET"
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Phase 1 DB failure")
            if "INSERT INTO baselines_trend" in str(sql):
                return "INSERT 0 8"
            if "INSERT INTO baselines_breakdown" in str(sql):
                return "INSERT 0 4"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.comparisons_rows == 0
        assert out.result.trend_rows == 8
        assert out.result.breakdown_rows == 4
        assert out.result.has_errors is True
        assert len(out.result.errors) == 1
        assert "Phase 1 (baselines_comparisons) failed" in out.result.errors[0]

    @pytest.mark.asyncio
    async def test_handle_all_phases_fail(self, mock_pool: MagicMock) -> None:
        """All three phases failing captures three error messages."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            raise RuntimeError("DB unavailable")

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.total_rows == 0
        assert out.result.has_errors is True
        assert len(out.result.errors) == 3

    @pytest.mark.asyncio
    async def test_handle_accepts_required_correlation_id(
        self, mock_pool: MagicMock
    ) -> None:
        """D1: Required correlation_id is accepted and used."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        cid = uuid4()
        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=cid)
        )

        assert out.result.has_errors is False

    @pytest.mark.asyncio
    async def test_handle_timestamps_set(self, mock_pool: MagicMock) -> None:
        """Result has started_at before completed_at."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 1"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.started_at <= out.result.completed_at

    @pytest.mark.asyncio
    async def test_zero_rows_written_on_empty_source(
        self, mock_pool: MagicMock
    ) -> None:
        """Zero rows written when source tables are empty (INSERT 0 0)."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=100)
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.total_rows == 0
        assert out.result.has_errors is False

    @pytest.mark.asyncio
    async def test_breakdown_batch_size_warning_logged(
        self, mock_pool: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged when breakdown returns exactly batch_size rows."""
        conn = mock_pool._test_conn
        batch_size = 5

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            if "INSERT INTO baselines_breakdown" in str(sql):
                return f"INSERT 0 {batch_size}"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        import logging

        with caplog.at_level(logging.WARNING):
            handler = HandlerBaselinesBatchCompute(mock_pool, batch_size=batch_size)
            out = await handler.handle(
                ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
            )

        assert out.result.breakdown_rows == batch_size
        assert any(
            "batch_size" in record.message and "breakdown" in record.message.lower()
            for record in caplog.records
        )


class TestHandlerSnapshotPolicies:
    """Tests for D5/D6 snapshot emit policies."""

    @pytest.mark.asyncio
    async def test_snapshot_emitted_false_when_publisher_raises(
        self, mock_pool: MagicMock, mock_publisher_raises: AsyncMock
    ) -> None:
        """D5/D6: snapshot_emitted=False when publisher raises; error recorded."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            if "INSERT INTO baselines_comparisons" in str(sql):
                return "INSERT 0 5"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetch = AsyncMock(return_value=[])

        handler = HandlerBaselinesBatchCompute(
            mock_pool, publisher=mock_publisher_raises, batch_size=100
        )
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.snapshot_emitted is False
        # Snapshot emit failure is recorded as an error
        assert len(out.result.errors) > 0

    @pytest.mark.asyncio
    async def test_emit_snapshot_calls_publisher_with_correct_topic(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        """D6: publisher is called with correct topic and required fields."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            if "INSERT INTO baselines_comparisons" in str(sql):
                return "INSERT 0 2"
            if "INSERT INTO baselines_trend" in str(sql):
                return "INSERT 0 4"
            if "INSERT INTO baselines_breakdown" in str(sql):
                return "INSERT 0 1"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)
        conn.fetch = AsyncMock(return_value=[])

        handler = HandlerBaselinesBatchCompute(
            mock_pool, publisher=mock_publisher, batch_size=100
        )
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.snapshot_emitted is True
        mock_publisher.assert_called_once()
        call_kw = mock_publisher.call_args.kwargs
        assert call_kw["topic"] == "onex.evt.omnibase-infra.baselines-computed.v1"
        assert "correlation_id" in call_kw
        assert "event_type" in call_kw
        # Payload must be JSON-serializable
        json.dumps(call_kw["payload"])  # must not raise

    @pytest.mark.asyncio
    async def test_no_emit_when_all_phases_zero_rows(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        """D5: no emit when all phases return zero rows."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(
            mock_pool, publisher=mock_publisher, batch_size=100
        )
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.snapshot_emitted is False
        mock_publisher.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_emit_when_all_phases_fail(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        """D5: no emit when all phases fail and total_rows == 0."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            raise RuntimeError("DB unavailable")

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(
            mock_pool, publisher=mock_publisher, batch_size=100
        )
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.result.total_rows == 0
        assert out.snapshot_emitted is False
        mock_publisher.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_emit_when_no_publisher(self, mock_pool: MagicMock) -> None:
        """snapshot_emitted=False when publisher=None (even with rows written)."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 5"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        handler = HandlerBaselinesBatchCompute(
            mock_pool, publisher=None, batch_size=100
        )
        out = await handler.handle(
            ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        )

        assert out.snapshot_emitted is False
        assert out.result.total_rows > 0  # rows were written, just no publisher

    def test_handler_type_properties(self, mock_pool: MagicMock) -> None:
        """Handler exposes correct type and category."""
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        handler = HandlerBaselinesBatchCompute(mock_pool)
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
