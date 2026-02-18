# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceBatchComputeBaselines.

Tests the batch computation engine that derives treatment/control
baselines comparisons from agent_routing_decisions and agent_actions.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.emit_daemon.topics import TOPIC_BASELINES_COMPUTED
from omnibase_infra.services.observability.baselines.models.model_baselines_snapshot_event import (
    ModelBaselinesSnapshotEvent,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)
from omnibase_infra.services.observability.baselines.service_batch_compute_baselines import (
    ServiceBatchComputeBaselines,
    parse_execute_count,
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


class TestModelBatchComputeBaselinesResult:
    """Tests for ModelBatchComputeBaselinesResult Pydantic model."""

    def test_total_rows_sums_all(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(
            comparisons_rows=10,
            trend_rows=20,
            breakdown_rows=5,
            completed_at=now,
        )
        assert result.total_rows == 35

    def test_total_rows_default_zero(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(completed_at=now)
        assert result.total_rows == 0

    def test_has_errors_false_when_empty(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(completed_at=now)
        assert result.has_errors is False

    def test_has_errors_true_when_errors_present(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(
            errors=("phase 1 failed",), completed_at=now
        )
        assert result.has_errors is True

    def test_default_started_at_is_utc(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(completed_at=now)
        # started_at should be a recent timestamp
        delta = abs((now - result.started_at).total_seconds())
        assert delta < 5.0

    def test_all_rows_and_errors(self) -> None:
        now = datetime.now(UTC)
        result = ModelBatchComputeBaselinesResult(
            comparisons_rows=3,
            trend_rows=14,
            breakdown_rows=7,
            errors=("phase 2 failed",),
            completed_at=now,
        )
        assert result.total_rows == 24
        assert result.has_errors is True


class TestParseExecuteCount:
    """Tests for parse_execute_count utility."""

    def test_parse_insert_result(self) -> None:
        assert parse_execute_count("INSERT 0 42") == 42

    def test_parse_update_result(self) -> None:
        assert parse_execute_count("UPDATE 15") == 15

    def test_parse_zero_rows(self) -> None:
        assert parse_execute_count("INSERT 0 0") == 0

    def test_parse_empty_string(self) -> None:
        assert parse_execute_count("") == 0

    def test_parse_invalid_string(self) -> None:
        assert parse_execute_count("not a valid result") == 0

    def test_parse_none_returns_zero(self) -> None:
        assert parse_execute_count(None) == 0  # type: ignore[arg-type]

    def test_parse_non_string_returns_zero(self) -> None:
        assert parse_execute_count(123) == 0  # type: ignore[arg-type]


class TestServiceBatchComputeBaselines:
    """Tests for ServiceBatchComputeBaselines."""

    @pytest.mark.asyncio
    async def test_compute_and_persist_all_phases(self, mock_pool: MagicMock) -> None:
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

        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        assert result.comparisons_rows == 7
        assert result.trend_rows == 14
        assert result.breakdown_rows == 3
        assert result.total_rows == 24
        assert result.has_errors is False
        assert result.completed_at >= result.started_at

    @pytest.mark.asyncio
    async def test_compute_and_persist_phase_failure_is_isolated(
        self, mock_pool: MagicMock
    ) -> None:
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

        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        # Phase 1 failed, phases 2 and 3 succeeded
        assert result.comparisons_rows == 0
        assert result.trend_rows == 8
        assert result.breakdown_rows == 4
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert "Phase 1 (baselines_comparisons) failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_compute_and_persist_all_phases_fail(
        self, mock_pool: MagicMock
    ) -> None:
        """All three phases failing captures three error messages."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            raise RuntimeError("DB unavailable")

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        assert result.total_rows == 0
        assert result.has_errors is True
        assert len(result.errors) == 3

    @pytest.mark.asyncio
    async def test_compute_and_persist_accepts_correlation_id(
        self, mock_pool: MagicMock
    ) -> None:
        """Custom correlation_id is accepted without error."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 0"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        cid = uuid4()
        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist(correlation_id=cid)

        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_compute_and_persist_timestamps_set(
        self, mock_pool: MagicMock
    ) -> None:
        """Result has started_at before completed_at."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            return "INSERT 0 1"

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        assert result.started_at <= result.completed_at

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
            batch = ServiceBatchComputeBaselines(mock_pool, batch_size=batch_size)
            result = await batch.compute_and_persist()

        assert result.breakdown_rows == batch_size
        assert any(
            "batch_size" in record.message and "breakdown" in record.message.lower()
            for record in caplog.records
        )

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

        batch = ServiceBatchComputeBaselines(mock_pool, batch_size=100)
        result = await batch.compute_and_persist()

        assert result.total_rows == 0
        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_event_bus_publish_envelope_called_with_snapshot(
        self, mock_pool: MagicMock
    ) -> None:
        """publish_envelope is called on the event_bus with a ModelBaselinesSnapshotEvent payload."""
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
        # _emit_snapshot reads back rows via conn.fetch; return empty lists
        conn.fetch = AsyncMock(return_value=[])

        mock_event_bus = AsyncMock()
        mock_event_bus.publish_envelope = AsyncMock()

        batch = ServiceBatchComputeBaselines(
            mock_pool, batch_size=100, event_bus=mock_event_bus
        )
        await batch.compute_and_persist()

        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args
        envelope = call_args.args[0]
        assert isinstance(envelope, ModelEventEnvelope)
        # The payload is a dict (model_dump result injected with metadata);
        # deserialise to verify it carries a valid ModelBaselinesSnapshotEvent shape.
        raw_payload: dict[str, object] = envelope.payload
        assert "snapshot_id" in raw_payload
        assert "computed_at_utc" in raw_payload
        assert "comparisons" in raw_payload
        assert "trend" in raw_payload
        assert "breakdown" in raw_payload
        # Verify the topic argument passed to publish_envelope.
        topic = call_args.args[1]
        assert topic == TOPIC_BASELINES_COMPUTED

    @pytest.mark.asyncio
    async def test_emit_snapshot_skipped_when_all_phases_fail(
        self, mock_pool: MagicMock
    ) -> None:
        """publish_envelope is NOT called when all phases fail and total_rows == 0."""
        conn = mock_pool._test_conn

        async def execute_side_effect(sql: str, *args: object, **kwargs: object) -> str:
            if "SET LOCAL" in str(sql):
                return "SET"
            raise RuntimeError("DB unavailable")

        conn.execute = AsyncMock(side_effect=execute_side_effect)

        mock_event_bus = AsyncMock()
        mock_event_bus.publish_envelope = AsyncMock()

        batch = ServiceBatchComputeBaselines(
            mock_pool, batch_size=100, event_bus=mock_event_bus
        )
        result = await batch.compute_and_persist()

        assert result.total_rows == 0
        assert result.has_errors is True
        mock_event_bus.publish_envelope.assert_not_called()
