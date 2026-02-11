# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for WriterAgentActionsPostgres.

This module tests:
    - Idempotency behavior (ON CONFLICT DO NOTHING / DO UPDATE)
    - Batch write operations (empty, single, multiple items)
    - Circuit breaker state and error handling
    - JSON serialization for JSONB columns

All tests mock asyncpg pool - no real database required.

Related Tickets:
    - OMN-1743: Migrate agent_actions_consumer to omnibase_infra
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.services.observability.agent_actions.models import (
    ModelAgentAction,
    ModelAgentStatusEvent,
    ModelDetectionFailure,
    ModelExecutionLog,
    ModelPerformanceMetric,
    ModelRoutingDecision,
    ModelTransformationEvent,
)
from omnibase_infra.services.observability.agent_actions.writer_postgres import (
    WriterAgentActionsPostgres,
)

if TYPE_CHECKING:
    import asyncpg


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.executemany = AsyncMock()

    # Make pool.acquire() return an async context manager
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    return pool


@pytest.fixture
def mock_conn(mock_pool: MagicMock) -> AsyncMock:
    """Get the mock connection from the pool."""
    conn: AsyncMock = mock_pool.acquire.return_value.__aenter__.return_value
    return conn


@pytest.fixture
def writer(mock_pool: MagicMock) -> WriterAgentActionsPostgres:
    """Create a writer with mocked pool."""
    return WriterAgentActionsPostgres(
        pool=mock_pool,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=30.0,
    )


@pytest.fixture
def sample_agent_action() -> ModelAgentAction:
    """Create a sample agent action model."""
    return ModelAgentAction(
        id=uuid4(),
        correlation_id=uuid4(),
        agent_name="test-agent",
        action_type="tool_call",
        action_name="Read",
        created_at=datetime.now(UTC),
        status="completed",
        duration_ms=150,
        metadata={"file": "/test/path.py"},
    )


@pytest.fixture
def sample_routing_decision() -> ModelRoutingDecision:
    """Create a sample routing decision model."""
    return ModelRoutingDecision(
        id=uuid4(),
        correlation_id=uuid4(),
        selected_agent="api-architect",
        confidence_score=0.95,
        created_at=datetime.now(UTC),
        alternatives=("testing", "debug"),
    )


@pytest.fixture
def sample_transformation_event() -> ModelTransformationEvent:
    """Create a sample transformation event model."""
    return ModelTransformationEvent(
        id=uuid4(),
        correlation_id=uuid4(),
        source_agent="polymorphic-agent",
        target_agent="api-architect",
        created_at=datetime.now(UTC),
        trigger="Domain pattern match",
    )


@pytest.fixture
def sample_performance_metric() -> ModelPerformanceMetric:
    """Create a sample performance metric model."""
    return ModelPerformanceMetric(
        id=uuid4(),
        metric_name="routing_latency_ms",
        metric_value=45.2,
        created_at=datetime.now(UTC),
        labels={"operation": "route"},
    )


@pytest.fixture
def sample_detection_failure() -> ModelDetectionFailure:
    """Create a sample detection failure model."""
    return ModelDetectionFailure(
        correlation_id=uuid4(),
        failure_reason="No matching pattern",
        created_at=datetime.now(UTC),
        attempted_patterns=("code-review", "testing"),
    )


@pytest.fixture
def sample_execution_log() -> ModelExecutionLog:
    """Create a sample execution log model."""
    now = datetime.now(UTC)
    return ModelExecutionLog(
        execution_id=uuid4(),
        correlation_id=uuid4(),
        agent_name="testing",
        status="completed",
        created_at=now,
        updated_at=now,
        duration_ms=5000,
        exit_code=0,
    )


# =============================================================================
# Batch Write Tests - Empty List
# =============================================================================


class TestBatchWriteEmpty:
    """Test batch write methods with empty list input."""

    @pytest.mark.asyncio
    async def test_write_agent_actions_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty list should return 0 without database call."""
        result = await writer.write_agent_actions([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_routing_decisions_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty routing decisions should return 0."""
        result = await writer.write_routing_decisions([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_transformation_events_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty transformation events should return 0."""
        result = await writer.write_transformation_events([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_performance_metrics_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty performance metrics should return 0."""
        result = await writer.write_performance_metrics([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_detection_failures_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty detection failures should return 0."""
        result = await writer.write_detection_failures([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_execution_logs_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty execution logs should return 0."""
        result = await writer.write_execution_logs([])

        assert result == 0
        mock_conn.executemany.assert_not_called()


# =============================================================================
# Batch Write Tests - Single Item
# =============================================================================


class TestBatchWriteSingleItem:
    """Test batch write methods with single item."""

    @pytest.mark.asyncio
    async def test_write_agent_actions_single_item(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Writing single agent action should call executemany with 1 item."""
        result = await writer.write_agent_actions([sample_agent_action])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        # Verify SQL contains ON CONFLICT DO NOTHING
        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (id) DO NOTHING" in sql
        assert "agent_actions" in sql

    @pytest.mark.asyncio
    async def test_write_routing_decisions_single_item(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_routing_decision: ModelRoutingDecision,
    ) -> None:
        """Writing single routing decision should work correctly."""
        result = await writer.write_routing_decisions([sample_routing_decision])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (id) DO NOTHING" in sql
        assert "agent_routing_decisions" in sql

    @pytest.mark.asyncio
    async def test_write_execution_logs_single_item_uses_do_update(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_execution_log: ModelExecutionLog,
    ) -> None:
        """Execution logs should use ON CONFLICT DO UPDATE for lifecycle tracking."""
        result = await writer.write_execution_logs([sample_execution_log])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (execution_id) DO UPDATE" in sql
        assert "agent_execution_logs" in sql


# =============================================================================
# Batch Write Tests - Multiple Items
# =============================================================================


class TestBatchWriteMultipleItems:
    """Test batch write methods with multiple items."""

    @pytest.mark.asyncio
    async def test_write_agent_actions_multiple_items(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing multiple agent actions should process all items."""
        actions = [
            ModelAgentAction(
                id=uuid4(),
                correlation_id=uuid4(),
                agent_name=f"agent-{i}",
                action_type="tool_call",
                action_name="Read",
                created_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        result = await writer.write_agent_actions(actions)

        assert result == 5
        mock_conn.executemany.assert_called_once()

        # Verify 5 items in the batch
        call_args = mock_conn.executemany.call_args
        batch_data = list(call_args[0][1])
        assert len(batch_data) == 5

    @pytest.mark.asyncio
    async def test_write_routing_decisions_multiple_items(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing multiple routing decisions should process all items."""
        decisions = [
            ModelRoutingDecision(
                id=uuid4(),
                correlation_id=uuid4(),
                selected_agent=f"agent-{i}",
                confidence_score=0.8 + (i * 0.01),
                created_at=datetime.now(UTC),
            )
            for i in range(10)
        ]

        result = await writer.write_routing_decisions(decisions)

        assert result == 10


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestIdempotency:
    """Test idempotency behavior of write methods."""

    @pytest.mark.asyncio
    async def test_agent_actions_sql_has_on_conflict_do_nothing(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Agent actions should use ON CONFLICT (id) DO NOTHING."""
        await writer.write_agent_actions([sample_agent_action])

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_detection_failures_sql_has_on_conflict_do_nothing(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_detection_failure: ModelDetectionFailure,
    ) -> None:
        """Detection failures should use ON CONFLICT (correlation_id) DO NOTHING."""
        await writer.write_detection_failures([sample_detection_failure])

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        # Detection failures use correlation_id as the idempotency key
        assert "ON CONFLICT (correlation_id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_execution_logs_sql_has_on_conflict_do_update(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_execution_log: ModelExecutionLog,
    ) -> None:
        """Execution logs should use ON CONFLICT DO UPDATE for lifecycle tracking."""
        await writer.write_execution_logs([sample_execution_log])

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (execution_id) DO UPDATE" in sql
        # Verify update fields
        assert "status = EXCLUDED.status" in sql
        assert "completed_at = EXCLUDED.completed_at" in sql
        assert "duration_ms = EXCLUDED.duration_ms" in sql


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakerState:
    """Test circuit breaker state accessibility."""

    def test_circuit_breaker_state_initially_closed(
        self,
        writer: WriterAgentActionsPostgres,
    ) -> None:
        """Circuit breaker should start in closed state."""
        state = writer.get_circuit_breaker_state()

        assert state["state"] == "closed"
        assert state["failures"] == 0

    def test_circuit_breaker_state_returns_dict(
        self,
        writer: WriterAgentActionsPostgres,
    ) -> None:
        """Circuit breaker state should return expected fields."""
        state = writer.get_circuit_breaker_state()

        assert "state" in state
        assert "failures" in state
        assert "threshold" in state
        assert "reset_timeout_seconds" in state


class TestCircuitBreakerErrorHandling:
    """Test circuit breaker error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_raises_infra_connection_error(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Connection errors should raise InfraConnectionError."""
        import asyncpg

        mock_conn.executemany.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await writer.write_agent_actions([sample_agent_action])

    @pytest.mark.asyncio
    async def test_timeout_error_raises_infra_timeout_error(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Query timeout should raise InfraTimeoutError."""
        import asyncpg

        mock_conn.executemany.side_effect = asyncpg.QueryCanceledError(
            "canceling statement due to statement timeout"
        )

        with pytest.raises(InfraTimeoutError):
            await writer.write_agent_actions([sample_agent_action])

    @pytest.mark.asyncio
    async def test_repeated_failures_open_circuit(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Repeated failures should open the circuit breaker."""
        import asyncpg

        # Create writer with low threshold for testing
        writer = WriterAgentActionsPostgres(
            pool=mock_pool,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_timeout=30.0,
        )

        mock_conn.executemany.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )

        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(InfraConnectionError):
                await writer.write_agent_actions([action])

        # Circuit should be open now - next call should raise InfraUnavailableError
        with pytest.raises(InfraUnavailableError):
            await writer.write_agent_actions([action])


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJSONSerialization:
    """Test JSON serialization for JSONB columns."""

    def test_serialize_json_with_dict(self) -> None:
        """_serialize_json should convert dict to JSON string."""
        result = WriterAgentActionsPostgres._serialize_json({"key": "value"})
        assert result == '{"key": "value"}'

    def test_serialize_json_with_none(self) -> None:
        """_serialize_json should return None for None input."""
        result = WriterAgentActionsPostgres._serialize_json(None)
        assert result is None

    def test_serialize_json_with_nested_dict(self) -> None:
        """_serialize_json should handle nested structures."""
        data = {"outer": {"inner": {"deep": "value"}}}
        result = WriterAgentActionsPostgres._serialize_json(data)
        assert result is not None
        assert '"outer"' in result
        assert '"inner"' in result
        assert '"deep"' in result

    def test_serialize_list_with_strings(self) -> None:
        """_serialize_list should convert list to JSON string."""
        result = WriterAgentActionsPostgres._serialize_list(["a", "b", "c"])
        assert result == '["a", "b", "c"]'

    def test_serialize_list_with_none(self) -> None:
        """_serialize_list should return None for None input."""
        result = WriterAgentActionsPostgres._serialize_list(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_metadata_serialized_in_write_call(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Metadata dict should be serialized to JSON string in write call."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata={"tool": "Read", "path": "/test.py"},
        )

        await writer.write_agent_actions([action])

        call_args = mock_conn.executemany.call_args
        batch_data = list(call_args[0][1])
        # metadata is at index 10 in the INSERT tuple
        # (id, correlation_id, agent_name, action_type, action_name,
        #  created_at, status, duration_ms, result, error_message, metadata,
        #  project_path, project_name, working_directory)
        metadata_value = batch_data[0][10]
        assert isinstance(metadata_value, str)
        assert '"tool"' in metadata_value
        assert '"Read"' in metadata_value


# =============================================================================
# Correlation ID Tests
# =============================================================================


class TestCorrelationId:
    """Test correlation ID handling."""

    @pytest.mark.asyncio
    async def test_provided_correlation_id_used(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Provided correlation_id should be used in the operation."""
        cid = uuid4()
        await writer.write_agent_actions([sample_agent_action], correlation_id=cid)

        # Writer should complete successfully with the provided correlation_id
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_correlation_id_generates_new(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_action: ModelAgentAction,
    ) -> None:
        """Missing correlation_id should be auto-generated."""
        await writer.write_agent_actions([sample_agent_action])

        # Writer should complete successfully
        mock_conn.executemany.assert_called_once()


# =============================================================================
# All Write Methods Coverage
# =============================================================================


class TestAllWriteMethods:
    """Ensure all write methods are callable and work correctly."""

    @pytest.mark.asyncio
    async def test_write_transformation_events(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_transformation_event: ModelTransformationEvent,
    ) -> None:
        """Transformation events should write successfully."""
        result = await writer.write_transformation_events([sample_transformation_event])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "agent_transformation_events" in sql
        assert "ON CONFLICT (id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_write_performance_metrics(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_performance_metric: ModelPerformanceMetric,
    ) -> None:
        """Performance metrics should write successfully."""
        result = await writer.write_performance_metrics([sample_performance_metric])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "router_performance_metrics" in sql
        assert "ON CONFLICT (id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_write_detection_failures(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_detection_failure: ModelDetectionFailure,
    ) -> None:
        """Detection failures should write successfully."""
        result = await writer.write_detection_failures([sample_detection_failure])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "agent_detection_failures" in sql
        assert "ON CONFLICT (correlation_id) DO NOTHING" in sql


# =============================================================================
# Agent Status Events Writer Tests (OMN-1849)
# =============================================================================


@pytest.fixture
def sample_agent_status_event() -> ModelAgentStatusEvent:
    """Create a sample agent status event model."""
    return ModelAgentStatusEvent(
        id=uuid4(),
        correlation_id=uuid4(),
        agent_name="test-agent",
        session_id="session-abc123",
        state="working",
        status_schema_version=1,
        message="Processing code review",
        progress=0.5,
        current_phase="analysis",
        current_task="Checking imports",
        created_at=datetime.now(UTC),
        metadata={"file": "/test/path.py"},
    )


class TestWriteAgentStatusEvents:
    """Test write_agent_status_events method."""

    @pytest.mark.asyncio
    async def test_write_agent_status_events_empty_returns_zero(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing empty list should return 0 without database call."""
        result = await writer.write_agent_status_events([])

        assert result == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_agent_status_events_single_item(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_status_event: ModelAgentStatusEvent,
    ) -> None:
        """Writing single agent status event should call executemany with 1 item."""
        result = await writer.write_agent_status_events([sample_agent_status_event])

        assert result == 1
        mock_conn.executemany.assert_called_once()

        # Verify SQL contains ON CONFLICT DO NOTHING
        call_args = mock_conn.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT (id) DO NOTHING" in sql
        assert "agent_status_events" in sql

    @pytest.mark.asyncio
    async def test_write_agent_status_events_multiple_items(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
    ) -> None:
        """Writing multiple agent status events should process all items."""
        events = [
            ModelAgentStatusEvent(
                id=uuid4(),
                correlation_id=uuid4(),
                agent_name=f"agent-{i}",
                session_id="session-123",
                state="working",
                message=f"Processing task {i}",
                created_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        result = await writer.write_agent_status_events(events)

        assert result == 5
        mock_conn.executemany.assert_called_once()

        # Verify 5 items in the batch
        call_args = mock_conn.executemany.call_args
        batch_data = list(call_args[0][1])
        assert len(batch_data) == 5

    @pytest.mark.asyncio
    async def test_write_agent_status_events_metadata_serialized(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_status_event: ModelAgentStatusEvent,
    ) -> None:
        """Metadata dict should be serialized to JSON string in write call."""
        await writer.write_agent_status_events([sample_agent_status_event])

        call_args = mock_conn.executemany.call_args
        batch_data = list(call_args[0][1])
        # metadata is the last field in the INSERT tuple
        metadata_value = batch_data[0][-1]
        assert isinstance(metadata_value, str)
        assert '"file"' in metadata_value

    @pytest.mark.asyncio
    async def test_write_agent_status_events_connection_error(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_status_event: ModelAgentStatusEvent,
    ) -> None:
        """Connection errors should raise InfraConnectionError."""
        import asyncpg

        mock_conn.executemany.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await writer.write_agent_status_events([sample_agent_status_event])

    @pytest.mark.asyncio
    async def test_write_agent_status_events_timeout_error(
        self,
        writer: WriterAgentActionsPostgres,
        mock_conn: AsyncMock,
        sample_agent_status_event: ModelAgentStatusEvent,
    ) -> None:
        """Query timeout should raise InfraTimeoutError."""
        import asyncpg

        mock_conn.executemany.side_effect = asyncpg.QueryCanceledError(
            "canceling statement due to statement timeout"
        )

        with pytest.raises(InfraTimeoutError):
            await writer.write_agent_status_events([sample_agent_status_event])


__all__ = [
    "TestBatchWriteEmpty",
    "TestBatchWriteSingleItem",
    "TestBatchWriteMultipleItems",
    "TestIdempotency",
    "TestCircuitBreakerState",
    "TestCircuitBreakerErrorHandling",
    "TestJSONSerialization",
    "TestCorrelationId",
    "TestAllWriteMethods",
    "TestWriteAgentStatusEvents",
]
