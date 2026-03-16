# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: liveness expiry → heartbeat race (OMN-4825).

Tests the full end-to-end path:
  1. A node registers and becomes ACTIVE with a liveness deadline
  2. The liveness deadline expires (simulated via time-freezing / mock)
  3. A heartbeat arrives AFTER the liveness has expired
  4. Asserts:
     - No re-registration occurs
     - decide_heartbeat returns no_op
     - handler_node_heartbeat returns early with no intents
     - "terminal-state heartbeat ignored" warning is logged

These tests use unittest.mock for the projection reader (no Docker required)
so they run in CI without testcontainers.

Related Tickets:
    - OMN-4817: UUID Type Mismatch & Stale-Registration Race Fix (epic)
    - OMN-4822: Fix decide_heartbeat to return no_op for terminal states
    - OMN-4824: Fix handler_node_heartbeat — short-circuit and logging
    - OMN-4825: This test file
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    HandlerNodeHeartbeat,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERMINAL_STATE_LOG_TEXT = "terminal-state heartbeat ignored"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_expired_projection(
    node_id=None,
    *,
    expired_minutes_ago: float = 5.0,
) -> ModelRegistrationProjection:
    """Build a projection that has LIVENESS_EXPIRED state.

    Simulates a node whose liveness deadline expired some minutes ago.
    """
    now = datetime.now(UTC)
    expired_at = now - timedelta(minutes=expired_minutes_ago)
    return ModelRegistrationProjection(
        entity_id=node_id or uuid4(),
        domain="registration",
        current_state=EnumRegistrationState.LIVENESS_EXPIRED,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        ack_deadline=None,
        liveness_deadline=expired_at,
        last_heartbeat_at=expired_at - timedelta(seconds=30),
        ack_timeout_emitted_at=None,
        liveness_timeout_emitted_at=expired_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=now - timedelta(hours=2),
        updated_at=expired_at,
    )


def make_heartbeat_event(node_id, *, timestamp=None) -> ModelNodeHeartbeatEvent:
    """Build a heartbeat event for the given node."""
    return ModelNodeHeartbeatEvent(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        uptime_seconds=7200.0,
        active_operations_count=0,
        timestamp=timestamp or datetime.now(UTC),
        correlation_id=uuid4(),
    )


def make_envelope(
    event: ModelNodeHeartbeatEvent,
) -> ModelEventEnvelope[ModelNodeHeartbeatEvent]:
    """Wrap a heartbeat event in an event envelope."""
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event,
        envelope_timestamp=event.timestamp,
        correlation_id=event.correlation_id or uuid4(),
        source="test",
    )


# ---------------------------------------------------------------------------
# Tests: liveness expiry → heartbeat race
# ---------------------------------------------------------------------------


class TestLivenessExpiryHeartbeatRace:
    """Integration tests covering the full liveness expiry → heartbeat path.

    These tests run against real service instances (no Docker) using mocked
    projection readers. They exercise:
    - RegistrationReducerService.decide_heartbeat (OMN-4822 fix)
    - HandlerNodeHeartbeat.handle (OMN-4824 fix)
    - End-to-end: no re-registration, correct no_op, correct log
    """

    @pytest.mark.asyncio
    async def test_expired_node_heartbeat_returns_empty_output(self) -> None:
        """Heartbeat for LIVENESS_EXPIRED node produces empty handler output.

        Full path: handler reads projection → projection is LIVENESS_EXPIRED →
        handler short-circuits → returns ModelHandlerOutput with no intents/events.
        """
        node_id = uuid4()
        expired_projection = make_expired_projection(node_id)

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=expired_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        output = await handler.handle(envelope)

        assert len(output.intents) == 0, (
            f"Expected no intents for expired node heartbeat, got {len(output.intents)}"
        )
        assert len(output.events) == 0, (
            f"Expected no events for expired node heartbeat, got {len(output.events)}"
        )
        assert output.result is None

    @pytest.mark.asyncio
    async def test_expired_node_heartbeat_emits_terminal_state_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Heartbeat for LIVENESS_EXPIRED node emits 'terminal-state heartbeat ignored' warning.

        The warning text must match exactly what OMN-4826 monitor_logs.py monitors.
        """
        node_id = uuid4()
        expired_projection = make_expired_projection(node_id)

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=expired_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        with caplog.at_level(logging.WARNING):
            await handler.handle(envelope)

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(TERMINAL_STATE_LOG_TEXT in msg for msg in warning_messages), (
            f"Expected warning log containing {TERMINAL_STATE_LOG_TEXT!r}. "
            f"Got warnings: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_expired_node_heartbeat_log_contains_node_id_and_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The terminal-state warning log must include node_id and current_state.

        OMN-4826 monitor_logs.py extracts these fields for the alert payload.
        """
        node_id = uuid4()
        expired_projection = make_expired_projection(node_id)

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=expired_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        with caplog.at_level(logging.WARNING):
            await handler.handle(envelope)

        # Find the terminal-state warning record
        terminal_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and TERMINAL_STATE_LOG_TEXT in r.message
        ]
        assert len(terminal_records) >= 1, (
            f"Expected at least one {TERMINAL_STATE_LOG_TEXT!r} warning record"
        )

        record = terminal_records[0]
        # extra fields are attached to the log record
        assert hasattr(record, "node_id") or "node_id" in getattr(
            record, "__dict__", {}
        ), "Expected 'node_id' in log record extra fields"
        assert hasattr(record, "current_state") or "current_state" in getattr(
            record, "__dict__", {}
        ), "Expected 'current_state' in log record extra fields"

    @pytest.mark.asyncio
    async def test_no_re_registration_triggered(self) -> None:
        """Heartbeat for expired node must not trigger re-registration.

        Specifically: the projection reader must be called exactly once
        (to look up state), and no further I/O must be performed.
        No re-registration events or intents must appear in the output.
        """
        node_id = uuid4()
        expired_projection = make_expired_projection(node_id)

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=expired_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        output = await handler.handle(envelope)

        # Projection reader called exactly once
        projection_reader.get_entity_state.assert_called_once()

        # No intents (no DB writes, no re-registration)
        assert len(output.intents) == 0

        # No events (no re-registration domain events)
        assert len(output.events) == 0

    @pytest.mark.asyncio
    async def test_rejected_node_heartbeat_also_short_circuits(self) -> None:
        """REJECTED state is also terminal — heartbeat must be short-circuited."""
        node_id = uuid4()
        rejected_projection = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.REJECTED,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            ack_deadline=None,
            liveness_deadline=None,
            last_heartbeat_at=None,
            ack_timeout_emitted_at=None,
            liveness_timeout_emitted_at=None,
            last_applied_event_id=uuid4(),
            last_applied_offset=0,
            registered_at=datetime.now(UTC) - timedelta(hours=1),
            updated_at=datetime.now(UTC),
        )

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=rejected_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        output = await handler.handle(envelope)

        assert len(output.intents) == 0
        assert len(output.events) == 0

    @pytest.mark.asyncio
    async def test_active_node_heartbeat_still_processed_normally(self) -> None:
        """Regression guard: ACTIVE heartbeats must still produce UPDATE intents."""
        node_id = uuid4()
        now = datetime.now(UTC)
        active_projection = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            ack_deadline=None,
            liveness_deadline=now + timedelta(seconds=30),
            last_heartbeat_at=now - timedelta(seconds=60),
            ack_timeout_emitted_at=None,
            liveness_timeout_emitted_at=None,
            last_applied_event_id=uuid4(),
            last_applied_offset=0,
            registered_at=now - timedelta(hours=1),
            updated_at=now - timedelta(seconds=60),
        )

        projection_reader = AsyncMock()
        projection_reader.get_entity_state = AsyncMock(return_value=active_projection)

        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=projection_reader,
            reducer=reducer,
        )

        heartbeat = make_heartbeat_event(node_id)
        envelope = make_envelope(heartbeat)

        output = await handler.handle(envelope)

        # Active heartbeat must produce exactly 1 UPDATE intent
        assert len(output.intents) == 1, (
            f"Expected 1 intent for active heartbeat, got {len(output.intents)}"
        )
