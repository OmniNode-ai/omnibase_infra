# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistrationReducerService -- reducer-driven, pure function, no I/O.

Validates all four reducer-driven decide_* methods:
    - decide_introspection: New-node and re-registration decisions
    - decide_ack: Acknowledgment processing
    - decide_heartbeat: Liveness deadline extension
    - decide_timeout: Ack timeout and liveness expiry detection

All tests exercise the reducer service directly (no mocked handlers or event bus).
The service is stateless and pure-functional, so tests instantiate it directly.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
from omnibase_infra.models.registration.events.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_received import (
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_timed_out import (
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_decision import (
    ModelReducerDecision,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
    ModelRegistrationAckUpdate,
    ModelRegistrationHeartbeatUpdate,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)

# ---------------------------------------------------------------------------
# Fixed test time for deterministic testing
# ---------------------------------------------------------------------------
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

# Default configuration matching RegistrationReducerService defaults
DEFAULT_ACK_TIMEOUT = 30.0
DEFAULT_LIVENESS_INTERVAL = 60
DEFAULT_LIVENESS_WINDOW = 90.0


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_projection(
    state: EnumRegistrationState,
    *,
    entity_id: UUID | None = None,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    last_heartbeat_at: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Build a real ModelRegistrationProjection for testing."""
    eid = entity_id or uuid4()
    return ModelRegistrationProjection(
        entity_id=eid,
        domain="registration",
        current_state=state,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        last_heartbeat_at=last_heartbeat_at,
        ack_timeout_emitted_at=ack_timeout_emitted_at,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
    )


def make_introspection_event(
    *,
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> ModelNodeIntrospectionEvent:
    """Build a real ModelNodeIntrospectionEvent for testing."""
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type=EnumNodeKind.EFFECT,
        correlation_id=correlation_id or uuid4(),
        timestamp=TEST_NOW,
    )


def make_ack_command(
    *,
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> ModelNodeRegistrationAcked:
    """Build a real ModelNodeRegistrationAcked command for testing."""
    return ModelNodeRegistrationAcked(
        node_id=node_id or uuid4(),
        correlation_id=correlation_id or uuid4(),
        timestamp=TEST_NOW,
    )


# ---------------------------------------------------------------------------
# decide_introspection tests
# ---------------------------------------------------------------------------


class TestDecideIntrospectionNewAndRetriable:
    """Tests for reducer-driven decide_introspection: emit path (new node + retriable states)."""

    def test_new_node_emits_registration(self) -> None:
        """projection=None (new node) -> action='emit', 2 events, postgres upsert intent."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()
        correlation_id = uuid4()

        decision = service.decide_introspection(
            projection=None,
            event=event,
            correlation_id=correlation_id,
            now=TEST_NOW,
        )

        assert isinstance(decision, ModelReducerDecision)
        assert decision.action == "emit"
        assert len(decision.events) == 2
        assert isinstance(decision.events[0], ModelNodeRegistrationInitiated)
        assert isinstance(decision.events[1], ModelNodeRegistrationAccepted)
        # Must have at least the postgres upsert intent
        assert len(decision.intents) >= 1
        postgres_intents = [
            i
            for i in decision.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        assert len(postgres_intents) == 1

    def test_liveness_expired_emits_registration(self) -> None:
        """projection with state=LIVENESS_EXPIRED -> action='emit'."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.LIVENESS_EXPIRED)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 2

    def test_rejected_emits_registration(self) -> None:
        """projection with state=REJECTED -> action='emit'."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.REJECTED)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 2

    def test_ack_timed_out_emits_registration(self) -> None:
        """projection with state=ACK_TIMED_OUT -> action='emit'."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.ACK_TIMED_OUT)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 2


class TestDecideIntrospectionBlockingStates:
    """Tests for reducer-driven decide_introspection: no-op path (blocking states)."""

    def test_pending_registration_blocks(self) -> None:
        """state=PENDING_REGISTRATION -> action='no_op'."""
        service = RegistrationReducerService()
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.PENDING_REGISTRATION)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert len(decision.events) == 0
        assert len(decision.intents) == 0

    def test_accepted_blocks(self) -> None:
        """state=ACCEPTED -> action='no_op'."""
        service = RegistrationReducerService()
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.ACCEPTED)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"

    def test_awaiting_ack_blocks(self) -> None:
        """state=AWAITING_ACK -> action='no_op'."""
        service = RegistrationReducerService()
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.AWAITING_ACK)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"

    def test_ack_received_blocks(self) -> None:
        """state=ACK_RECEIVED -> action='no_op'."""
        service = RegistrationReducerService()
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.ACK_RECEIVED)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"

    def test_active_blocks(self) -> None:
        """state=ACTIVE -> action='no_op'."""
        service = RegistrationReducerService()
        event = make_introspection_event()
        projection = make_projection(EnumRegistrationState.ACTIVE)

        decision = service.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"


class TestDecideIntrospectionEventFields:
    """Tests for reducer-driven decide_introspection: event field correctness."""

    def test_emits_accepted_event(self) -> None:
        """Verify ModelNodeRegistrationAccepted has correct ack_deadline."""
        ack_timeout = 45.0
        service = RegistrationReducerService(
            ack_timeout_seconds=ack_timeout,
            consul_enabled=False,
        )
        event = make_introspection_event()

        decision = service.decide_introspection(
            projection=None,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        accepted_events = [
            e for e in decision.events if isinstance(e, ModelNodeRegistrationAccepted)
        ]
        assert len(accepted_events) == 1
        accepted = accepted_events[0]
        expected_deadline = TEST_NOW + timedelta(seconds=ack_timeout)
        assert accepted.ack_deadline == expected_deadline
        assert accepted.emitted_at == TEST_NOW

    def test_initial_state_is_awaiting_ack(self) -> None:
        """Verify projection record in postgres intent has current_state='awaiting_ack'."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()

        decision = service.decide_introspection(
            projection=None,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.new_state == EnumRegistrationState.AWAITING_ACK

        postgres_intents = [
            i
            for i in decision.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        assert len(postgres_intents) == 1
        record = postgres_intents[0].payload.record.model_dump()
        assert record["current_state"] == EnumRegistrationState.AWAITING_ACK.value


class TestDecideIntrospectionConsulToggle:
    """Tests for reducer-driven decide_introspection: consul intent toggle."""

    def test_consul_intent_when_enabled(self) -> None:
        """consul_enabled=True -> 2 intents (postgres + consul)."""
        service = RegistrationReducerService(consul_enabled=True)
        event = make_introspection_event()

        decision = service.decide_introspection(
            projection=None,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert len(decision.intents) == 2

        consul_intents = [
            i
            for i in decision.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        ]
        assert len(consul_intents) == 1

    def test_no_consul_intent_when_disabled(self) -> None:
        """consul_enabled=False -> 1 intent (postgres only)."""
        service = RegistrationReducerService(consul_enabled=False)
        event = make_introspection_event()

        decision = service.decide_introspection(
            projection=None,
            event=event,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert len(decision.intents) == 1

        consul_intents = [
            i
            for i in decision.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        ]
        assert len(consul_intents) == 0


# ---------------------------------------------------------------------------
# decide_ack tests
# ---------------------------------------------------------------------------


class TestDecideAckEmitPath:
    """Tests for reducer-driven decide_ack: emit path (valid ack states)."""

    def test_awaiting_ack_emits_activation(self) -> None:
        """state=AWAITING_ACK -> action='emit', events include AckReceived + BecameActive."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)
        correlation_id = uuid4()

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=correlation_id,
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert decision.new_state == EnumRegistrationState.ACTIVE
        assert len(decision.events) == 2
        assert isinstance(decision.events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(decision.events[1], ModelNodeBecameActive)

    def test_accepted_emits_activation(self) -> None:
        """state=ACCEPTED -> action='emit'."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACCEPTED,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "emit"
        assert decision.new_state == EnumRegistrationState.ACTIVE
        assert len(decision.events) == 2
        assert isinstance(decision.events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(decision.events[1], ModelNodeBecameActive)

    def test_ack_emits_update_intent(self) -> None:
        """Verify intents include postgres.update_registration with current_state=ACTIVE."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert len(decision.intents) == 1
        intent = decision.intents[0]
        assert isinstance(intent.payload, ModelPayloadPostgresUpdateRegistration)
        assert intent.payload.intent_type == "postgres.update_registration"
        assert intent.payload.entity_id == node_id
        assert (
            intent.payload.updates.current_state == EnumRegistrationState.ACTIVE.value
        )


class TestDecideAckNoOpPath:
    """Tests for reducer-driven decide_ack: no-op path (invalid states)."""

    def test_none_projection_noop(self) -> None:
        """projection=None -> action='no_op'."""
        service = RegistrationReducerService()
        command = make_ack_command()

        decision = service.decide_ack(
            projection=None,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert "Unknown node" in decision.reason

    def test_active_duplicate_noop(self) -> None:
        """state=ACTIVE -> action='no_op' (duplicate ack)."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACTIVE,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert "Duplicate ack" in decision.reason

    def test_pending_too_early_noop(self) -> None:
        """state=PENDING_REGISTRATION -> action='no_op' (ack too early)."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.PENDING_REGISTRATION,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert "too early" in decision.reason

    def test_timed_out_too_late_noop(self) -> None:
        """state=ACK_TIMED_OUT -> action='no_op' (ack too late)."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACK_TIMED_OUT,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert "too late" in decision.reason

    def test_terminal_noop(self) -> None:
        """state=REJECTED (terminal) -> action='no_op'."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.REJECTED,
            entity_id=node_id,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert "Terminal" in decision.reason


# ---------------------------------------------------------------------------
# decide_heartbeat tests
# ---------------------------------------------------------------------------


class TestDecideHeartbeat:
    """Tests for reducer-driven decide_heartbeat: liveness deadline extension."""

    def test_heartbeat_emits_update_intent(self) -> None:
        """projection exists -> action='emit', 1 intent with last_heartbeat_at + liveness_deadline."""
        service = RegistrationReducerService(liveness_window_seconds=90.0)
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACTIVE,
            entity_id=node_id,
        )
        heartbeat_ts = TEST_NOW
        correlation_id = uuid4()

        ctx = ModelReducerContext(correlation_id=correlation_id, now=TEST_NOW)
        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=heartbeat_ts,
            ctx=ctx,
        )

        assert decision.action == "emit"
        assert len(decision.intents) == 1
        intent = decision.intents[0]
        assert isinstance(intent.payload, ModelPayloadPostgresUpdateRegistration)
        assert intent.payload.updates.last_heartbeat_at == heartbeat_ts
        assert isinstance(intent.payload.updates, ModelRegistrationHeartbeatUpdate)

    def test_heartbeat_no_events(self) -> None:
        """Verify events tuple is empty (heartbeat produces no events)."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACTIVE,
            entity_id=node_id,
        )

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW)
        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 0

    def test_heartbeat_unknown_node_noop(self) -> None:
        """projection=None -> action='no_op'."""
        service = RegistrationReducerService()

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW)
        decision = service.decide_heartbeat(
            projection=None,
            node_id=uuid4(),
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "no_op"
        assert "Unknown node" in decision.reason

    def test_heartbeat_liveness_deadline_calculation(self) -> None:
        """Verify liveness_deadline = heartbeat_timestamp + liveness_window_seconds."""
        liveness_window = 120.0
        service = RegistrationReducerService(liveness_window_seconds=liveness_window)
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACTIVE,
            entity_id=node_id,
        )
        heartbeat_ts = TEST_NOW

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW)
        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=heartbeat_ts,
            ctx=ctx,
        )

        assert decision.action == "emit"
        expected_deadline = heartbeat_ts + timedelta(seconds=liveness_window)
        actual_deadline = decision.intents[0].payload.updates.liveness_deadline
        assert actual_deadline == expected_deadline


# ---------------------------------------------------------------------------
# decide_timeout tests
# ---------------------------------------------------------------------------


class TestDecideTimeout:
    """Tests for reducer-driven decide_timeout: ack timeout and liveness expiry detection."""

    def test_ack_timeout_emits_events(self) -> None:
        """overdue_ack projections -> ModelNodeRegistrationAckTimedOut events."""
        service = RegistrationReducerService()
        overdue_ack = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
        )
        tick_id = uuid4()

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=tick_id)
        decision = service.decide_timeout(
            overdue_ack_projections=[overdue_ack],
            overdue_liveness_projections=[],
            ctx=ctx,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 1
        assert isinstance(decision.events[0], ModelNodeRegistrationAckTimedOut)
        assert decision.events[0].entity_id == overdue_ack.entity_id
        assert decision.events[0].deadline_at == overdue_ack.ack_deadline
        assert decision.events[0].causation_id == tick_id

    def test_liveness_timeout_emits_events(self) -> None:
        """overdue_liveness projections -> ModelNodeLivenessExpired events."""
        service = RegistrationReducerService()
        last_hb = TEST_NOW - timedelta(minutes=10)
        overdue_liveness = make_projection(
            EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),
            last_heartbeat_at=last_hb,
        )
        tick_id = uuid4()

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=tick_id)
        decision = service.decide_timeout(
            overdue_ack_projections=[],
            overdue_liveness_projections=[overdue_liveness],
            ctx=ctx,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 1
        assert isinstance(decision.events[0], ModelNodeLivenessExpired)
        assert decision.events[0].entity_id == overdue_liveness.entity_id
        assert decision.events[0].last_heartbeat_at == last_hb

    def test_mixed_timeouts(self) -> None:
        """both overdue_ack and overdue_liveness -> combined events."""
        service = RegistrationReducerService()
        overdue_ack = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
        )
        overdue_liveness = make_projection(
            EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),
            last_heartbeat_at=TEST_NOW - timedelta(minutes=10),
        )

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=uuid4())
        decision = service.decide_timeout(
            overdue_ack_projections=[overdue_ack],
            overdue_liveness_projections=[overdue_liveness],
            ctx=ctx,
        )

        assert decision.action == "emit"
        assert len(decision.events) == 2
        event_types = {type(e) for e in decision.events}
        assert ModelNodeRegistrationAckTimedOut in event_types
        assert ModelNodeLivenessExpired in event_types

    def test_no_timeouts_noop(self) -> None:
        """empty lists -> action='no_op'."""
        service = RegistrationReducerService()

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=uuid4())
        decision = service.decide_timeout(
            overdue_ack_projections=[],
            overdue_liveness_projections=[],
            ctx=ctx,
        )

        assert decision.action == "no_op"
        assert "No timeouts" in decision.reason

    def test_already_emitted_skipped(self) -> None:
        """projection with ack_timeout_emitted_at set -> skipped (needs_ack_timeout_event returns False)."""
        service = RegistrationReducerService()
        # This projection has already had its ack timeout emitted
        overdue_ack = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
            ack_timeout_emitted_at=TEST_NOW - timedelta(minutes=3),
        )

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=uuid4())
        decision = service.decide_timeout(
            overdue_ack_projections=[overdue_ack],
            overdue_liveness_projections=[],
            ctx=ctx,
        )

        # Should be no_op because needs_ack_timeout_event returns False
        assert decision.action == "no_op"

    def test_liveness_already_emitted_skipped(self) -> None:
        """projection with liveness_timeout_emitted_at set -> skipped."""
        service = RegistrationReducerService()
        overdue_liveness = make_projection(
            EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),
            last_heartbeat_at=TEST_NOW - timedelta(minutes=10),
            liveness_timeout_emitted_at=TEST_NOW - timedelta(minutes=1),
        )

        ctx = ModelReducerContext(correlation_id=uuid4(), now=TEST_NOW, tick_id=uuid4())
        decision = service.decide_timeout(
            overdue_ack_projections=[],
            overdue_liveness_projections=[overdue_liveness],
            ctx=ctx,
        )

        assert decision.action == "no_op"


# ---------------------------------------------------------------------------
# _sanitize_tool_name tests
# ---------------------------------------------------------------------------


class TestSanitizeToolName:
    """Tests for _sanitize_tool_name helper used in Consul tag generation."""

    def test_all_special_chars_produces_unnamed(self) -> None:
        """A name consisting entirely of special characters falls back to 'unnamed'."""
        assert RegistrationReducerService._sanitize_tool_name("!!!") == "unnamed"
        assert RegistrationReducerService._sanitize_tool_name("@#$%") == "unnamed"
        assert RegistrationReducerService._sanitize_tool_name("---") == "unnamed"

    def test_two_different_all_special_char_names_both_produce_unnamed(self) -> None:
        """Two nodes whose tool names are entirely special characters both map to
        'unnamed', demonstrating they would produce the same 'mcp-tool:unnamed'
        Consul tag and therefore silently collide.

        This is a documented limitation (see _sanitize_tool_name docstring):
        when multiple nodes have tool names that all consist entirely of special
        characters, the fallback 'unnamed' value is identical for all of them.
        """
        result_a = RegistrationReducerService._sanitize_tool_name("!!!@@@")
        result_b = RegistrationReducerService._sanitize_tool_name("$$$^^^")

        # Both different-but-all-special inputs produce the same fallback
        assert result_a == "unnamed"
        assert result_b == "unnamed"
        # Demonstrates that the Consul tags would collide: "mcp-tool:unnamed"
        assert result_a == result_b
