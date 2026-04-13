# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for HandlerBaselineCapture (OMN-7484).

Validates the baseline capture handler contract:
  - Returns ModelBaselineCaptureOutput with correct fields
  - D3: no snapshot emitted when measurements_captured == 0
  - D3: snapshot emitted when measurements_captured > 0
  - D2: lookback_hours capped at 168

Tests use a mock asyncpg pool (no live DB required in CI). Tests that require
a live PostgreSQL instance are gated on POSTGRES_INTEGRATION_TESTS=1.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_baseline_capture.handlers.handler_baseline_capture import (
    HandlerBaselineCapture,
)
from omnibase_infra.nodes.node_baseline_capture.models.model_baseline_capture_command import (
    ModelBaselineCaptureCommand,
)
from omnibase_infra.nodes.node_baseline_capture.models.model_baseline_capture_output import (
    ModelBaselineCaptureOutput,
)

POSTGRES_AVAILABLE = os.getenv("POSTGRES_INTEGRATION_TESTS") == "1"


def _make_mock_pool(
    total_count: int = 0,
    agent_rows: list[dict] | None = None,
) -> MagicMock:
    """Build a mock asyncpg pool that returns configurable query results."""
    rows = agent_rows or []

    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[_make_agent_row(r) for r in rows])
    mock_conn.fetchrow = AsyncMock(return_value={"total": total_count})

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_pool


def _make_agent_row(data: dict) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    return row


class TestHandlerBaselineCaptureContract:
    """Verify the handler satisfies its DoD contract (mock pool — CI safe)."""

    @pytest.mark.anyio
    async def test_returns_model_baseline_capture_output(self) -> None:
        pool = _make_mock_pool(total_count=0)
        handler = HandlerBaselineCapture(pool=pool)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())

        result = await handler.handle(cmd)

        assert isinstance(result, ModelBaselineCaptureOutput)

    @pytest.mark.anyio
    async def test_d3_no_snapshot_when_no_measurements(self) -> None:
        """D3: snapshot_emitted must be False when measurements_captured == 0."""
        publisher = AsyncMock(return_value=True)
        pool = _make_mock_pool(total_count=0)
        handler = HandlerBaselineCapture(pool=pool, publisher=publisher)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())

        result = await handler.handle(cmd)

        assert result.measurements_captured == 0
        assert result.snapshot_emitted is False
        publisher.assert_not_called()

    @pytest.mark.anyio
    async def test_d3_snapshot_emitted_when_measurements_present(self) -> None:
        """D3: snapshot_emitted must be True when measurements_captured > 0."""
        now = datetime.now(UTC)
        agent_rows = [
            {
                "pattern_id": uuid4(),
                "pattern_label": "test-agent",
                "sample_count": 5,
                "avg_latency_ms": 120.0,
                "total_tokens": 1000,
                "avg_tokens": 200.0,
                "success_count": 5,
                "treatment_count": 5,
                "control_count": 0,
                "computed_at": now,
                "created_at": now,
                "updated_at": now,
            }
        ]
        publisher = AsyncMock(return_value=True)
        pool = _make_mock_pool(total_count=5, agent_rows=agent_rows)
        handler = HandlerBaselineCapture(pool=pool, publisher=publisher)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())

        result = await handler.handle(cmd)

        assert result.measurements_captured == 5
        assert result.snapshot_emitted is True
        publisher.assert_called_once()

    @pytest.mark.anyio
    async def test_d2_lookback_capped_at_168_hours(self) -> None:
        """D2: lookback_hours values above 168 must be clamped to 168."""
        pool = _make_mock_pool(total_count=0)
        handler = HandlerBaselineCapture(pool=pool)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4(), lookback_hours=999)

        # Patch datetime.now to capture the `since` value passed to the query
        captured_since: list[datetime] = []
        original_fetchrow = pool.acquire.return_value.__aenter__.return_value.fetchrow

        async def capture_fetchrow(sql: str, since: datetime) -> dict:
            captured_since.append(since)
            return {"total": 0}

        pool.acquire.return_value.__aenter__.return_value.fetchrow = capture_fetchrow

        await handler.handle(cmd)

        assert captured_since, "fetchrow should have been called"
        from datetime import timedelta

        expected_max_window = timedelta(hours=168)
        actual_window = datetime.now(UTC) - captured_since[0]
        assert actual_window <= expected_max_window + timedelta(seconds=5)

    @pytest.mark.anyio
    async def test_no_publisher_no_emit(self) -> None:
        """When no publisher is provided, snapshot_emitted is always False."""
        pool = _make_mock_pool(total_count=10)
        handler = HandlerBaselineCapture(pool=pool, publisher=None)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())

        result = await handler.handle(cmd)

        assert result.snapshot_emitted is False

    @pytest.mark.anyio
    async def test_db_error_captured_in_errors_tuple(self) -> None:
        """DB errors must be captured in result.errors, not raised."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=RuntimeError("db connection lost"))
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("db connection lost"))

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        handler = HandlerBaselineCapture(pool=mock_pool)
        cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())

        result = await handler.handle(cmd)

        assert result.measurements_captured == 0
        assert result.snapshot_emitted is False
        assert len(result.errors) > 0


__all__: list[str] = []
