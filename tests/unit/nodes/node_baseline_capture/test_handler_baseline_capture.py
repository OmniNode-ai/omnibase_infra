# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerBaselineCapture.

Covers:
    - No-op when zero agent_actions rows (D3: no empty snapshot)
    - Emits snapshot when measurements exist
    - Publisher not called when publisher=None
    - Publisher failure is non-fatal (errors tuple populated, snapshot_emitted=False)
    - lookback_hours capped at 168

Ticket: OMN-7484
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_baseline_capture.handlers.handler_baseline_capture import (
    HandlerBaselineCapture,
)
from omnibase_infra.nodes.node_baseline_capture.models.model_baseline_capture_command import (
    ModelBaselineCaptureCommand,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pool(total_count: int = 0, agent_rows: list[dict] | None = None) -> MagicMock:
    """Build a mock asyncpg.Pool.

    total_count: returned by COUNT(*) query.
    agent_rows: list of dicts representing per-agent aggregation rows.
    """
    agent_rows = agent_rows or []

    pool = MagicMock()
    conn = AsyncMock()

    now = datetime.now(UTC)

    # Build asyncpg-like Record mocks
    def _make_record(d: dict) -> MagicMock:
        rec = MagicMock()
        rec.__getitem__ = MagicMock(side_effect=lambda k: d[k])
        return rec

    agent_records = [
        _make_record(
            {
                "pattern_id": uuid4(),
                "pattern_label": row["agent_name"],
                "sample_count": row["sample_count"],
                "success_count": row.get("success_count", row["sample_count"]),
                "treatment_count": row["sample_count"],
                "control_count": 0,
                "computed_at": now,
                "created_at": now,
                "updated_at": now,
            }
        )
        for row in agent_rows
    ]

    total_record = MagicMock()
    total_record.__getitem__ = MagicMock(
        side_effect=lambda k: total_count if k == "total" else None
    )

    conn.fetch = AsyncMock(return_value=agent_records)
    conn.fetchrow = AsyncMock(return_value=total_record)

    # async context manager for set_statement_timeout (execute)
    conn.execute = AsyncMock(return_value=None)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool._test_conn = conn
    return pool


@pytest.fixture
def mock_publisher() -> AsyncMock:
    pub = AsyncMock(return_value=True)
    return pub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_emit_when_zero_rows() -> None:
    """D3: publisher must NOT be called when agent_actions has zero rows."""
    pool = _make_pool(total_count=0, agent_rows=[])
    publisher = AsyncMock(return_value=True)
    handler = HandlerBaselineCapture(pool=pool, publisher=publisher)

    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())
    result = await handler.handle(cmd)

    assert result.measurements_captured == 0
    assert result.snapshot_emitted is False
    publisher.assert_not_called()


@pytest.mark.asyncio
async def test_emits_snapshot_when_measurements_exist(
    mock_publisher: AsyncMock,
) -> None:
    """Snapshot emitted when agent_actions has rows."""
    pool = _make_pool(
        total_count=42,
        agent_rows=[{"agent_name": "agent-alpha", "sample_count": 42}],
    )
    handler = HandlerBaselineCapture(pool=pool, publisher=mock_publisher)

    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())
    result = await handler.handle(cmd)

    assert result.measurements_captured == 42
    assert result.snapshot_emitted is True
    assert result.errors == ()
    mock_publisher.assert_called_once()

    # Verify published payload contains snapshot_id and breakdown
    call_kwargs = mock_publisher.call_args.kwargs
    assert call_kwargs["event_type"] == "baselines.computed"
    assert call_kwargs["topic"] == "onex.evt.omnibase-infra.baselines-computed.v1"
    payload = call_kwargs["payload"]
    assert "snapshot_id" in payload
    assert isinstance(payload["breakdown"], list)
    assert len(payload["breakdown"]) == 1
    assert payload["comparisons"] == []
    assert payload["trend"] == []


@pytest.mark.asyncio
async def test_no_publisher_no_emit() -> None:
    """When publisher=None, snapshot_emitted is False even with data."""
    pool = _make_pool(
        total_count=10,
        agent_rows=[{"agent_name": "agent-beta", "sample_count": 10}],
    )
    handler = HandlerBaselineCapture(pool=pool, publisher=None)

    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())
    result = await handler.handle(cmd)

    assert result.measurements_captured == 10
    assert result.snapshot_emitted is False
    assert result.errors == ()


@pytest.mark.asyncio
async def test_publisher_failure_is_non_fatal() -> None:
    """Publisher exception must not propagate — errors tuple populated, snapshot_emitted=False."""
    pool = _make_pool(
        total_count=5,
        agent_rows=[{"agent_name": "agent-gamma", "sample_count": 5}],
    )
    failing_publisher = AsyncMock(side_effect=RuntimeError("kafka unavailable"))
    handler = HandlerBaselineCapture(pool=pool, publisher=failing_publisher)

    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())
    result = await handler.handle(cmd)

    assert result.snapshot_emitted is False
    assert len(result.errors) == 1
    assert "Snapshot emit failed" in result.errors[0]


@pytest.mark.asyncio
async def test_lookback_capped_at_168_hours() -> None:
    """lookback_hours > 168 is capped; query still executes normally."""
    pool = _make_pool(total_count=0, agent_rows=[])
    publisher = AsyncMock(return_value=True)
    handler = HandlerBaselineCapture(pool=pool, publisher=publisher)

    # 999 hours should be silently capped to 168
    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4(), lookback_hours=999)
    result = await handler.handle(cmd)

    assert result.measurements_captured == 0
    assert result.snapshot_emitted is False


@pytest.mark.asyncio
async def test_db_read_failure_recorded_in_errors() -> None:
    """If pool.acquire() raises, error is captured and snapshot_emitted=False."""
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)

    publisher = AsyncMock(return_value=True)
    handler = HandlerBaselineCapture(pool=pool, publisher=publisher)

    cmd = ModelBaselineCaptureCommand(correlation_id=uuid4())
    result = await handler.handle(cmd)

    assert result.snapshot_emitted is False
    assert len(result.errors) == 1
    publisher.assert_not_called()
