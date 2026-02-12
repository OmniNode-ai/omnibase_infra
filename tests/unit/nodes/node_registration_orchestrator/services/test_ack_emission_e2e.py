# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit-level E2E test for the complete ACK emission flow.

Validates the full registration ACK pipeline using the pure-function
RegistrationReducerService (no real Kafka or infrastructure needed):

    1. Build a projection in ACCEPTED/AWAITING_ACK state
    2. Call decide_ack() with a ModelNodeRegistrationAcked command
    3. Verify action="emit" with new_state=ACTIVE
    4. Verify ModelNodeRegistrationAckReceived event is emitted
    5. Verify ModelNodeBecameActive event is emitted
    6. Verify postgres.update_registration intent transitions to ACTIVE

This mirrors the real pipeline where:
    - Node's _ack_listener_loop receives accepted event
    - Node emits ModelNodeRegistrationAcked command
    - Orchestrator's HandlerNodeRegistrationAcked processes the command
    - Reducer decides ack -> emits events + intents

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_received import (
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_decision import (
    ModelReducerDecision,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)

# ---------------------------------------------------------------------------
# Fixed test time for deterministic testing
# ---------------------------------------------------------------------------
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

# Default configuration matching RegistrationReducerService defaults
DEFAULT_ACK_TIMEOUT = 30.0
DEFAULT_LIVENESS_INTERVAL = 60


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_projection(
    state: EnumRegistrationState,
    *,
    entity_id: UUID | None = None,
    capabilities: ModelNodeCapabilities | None = None,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
) -> ModelRegistrationProjection:
    """Build a real ModelRegistrationProjection for testing."""
    eid = entity_id or uuid4()
    return ModelRegistrationProjection(
        entity_id=eid,
        domain="registration",
        current_state=state,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=capabilities or ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
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
# E2E ACK emission tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAckEmissionE2EFromAccepted:
    """E2E test: ACCEPTED state -> decide_ack -> ACTIVE with full event/intent set.

    Simulates the complete pipeline starting from an ACCEPTED projection
    (both Consul + Postgres confirmed), validating every emitted artifact.
    """

    def test_accepted_to_active_full_pipeline(self) -> None:
        """Full pipeline: ACCEPTED -> decide_ack -> ACTIVE with all artifacts."""
        service = RegistrationReducerService()
        node_id = uuid4()
        correlation_id = uuid4()
        caps = ModelNodeCapabilities(postgres=True, read=True)

        projection = make_projection(
            EnumRegistrationState.ACCEPTED,
            entity_id=node_id,
            capabilities=caps,
            ack_deadline=TEST_NOW + timedelta(seconds=30),
        )
        command = make_ack_command(
            node_id=node_id,
            correlation_id=correlation_id,
        )

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=correlation_id,
            now=TEST_NOW,
        )

        # Step 1: Verify decision structure
        assert isinstance(decision, ModelReducerDecision)
        assert decision.action == "emit"
        assert decision.new_state == EnumRegistrationState.ACTIVE
        assert "ACTIVE" in decision.reason

        # Step 2: Verify emitted events
        assert len(decision.events) == 2

        ack_received = decision.events[0]
        assert isinstance(ack_received, ModelNodeRegistrationAckReceived)
        assert ack_received.node_id == node_id
        assert ack_received.entity_id == node_id
        assert ack_received.correlation_id == correlation_id
        assert ack_received.causation_id == command.command_id
        assert ack_received.emitted_at == TEST_NOW
        # Liveness deadline should be now + liveness_interval_seconds (default 60)
        expected_liveness = TEST_NOW + timedelta(seconds=DEFAULT_LIVENESS_INTERVAL)
        assert ack_received.liveness_deadline == expected_liveness

        became_active = decision.events[1]
        assert isinstance(became_active, ModelNodeBecameActive)
        assert became_active.node_id == node_id
        assert became_active.entity_id == node_id
        assert became_active.correlation_id == correlation_id
        assert became_active.causation_id == command.command_id
        assert became_active.emitted_at == TEST_NOW
        assert became_active.capabilities == caps

        # Step 3: Verify postgres.update_registration intent
        assert len(decision.intents) == 1
        intent = decision.intents[0]
        assert isinstance(intent.payload, ModelPayloadPostgresUpdateRegistration)
        assert intent.payload.intent_type == "postgres.update_registration"
        assert intent.payload.entity_id == node_id
        assert intent.payload.correlation_id == correlation_id
        assert intent.payload.domain == "registration"
        assert (
            intent.payload.updates["current_state"]
            == EnumRegistrationState.ACTIVE.value
        )
        assert intent.payload.updates["liveness_deadline"] == expected_liveness
        assert intent.payload.updates["updated_at"] == TEST_NOW

        # Step 4: Verify intent target URI follows convention
        assert intent.target == f"postgres://node_registrations/{node_id}"

    def test_accepted_to_active_preserves_capabilities(self) -> None:
        """Verify that BecameActive event carries the projection's capabilities."""
        service = RegistrationReducerService()
        node_id = uuid4()
        caps = ModelNodeCapabilities(postgres=True, write=True, database=True)

        projection = make_projection(
            EnumRegistrationState.ACCEPTED,
            entity_id=node_id,
            capabilities=caps,
        )
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        active_event = [
            e for e in decision.events if isinstance(e, ModelNodeBecameActive)
        ]
        assert len(active_event) == 1
        assert active_event[0].capabilities.postgres is True
        assert active_event[0].capabilities.write is True
        assert active_event[0].capabilities.database is True

    def test_accepted_to_active_custom_liveness_interval(self) -> None:
        """Custom liveness_interval_seconds is respected in events and intents."""
        custom_liveness = 120
        service = RegistrationReducerService(liveness_interval_seconds=custom_liveness)
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

        ack_received = decision.events[0]
        assert isinstance(ack_received, ModelNodeRegistrationAckReceived)
        expected = TEST_NOW + timedelta(seconds=custom_liveness)
        assert ack_received.liveness_deadline == expected

        # Intent should also reflect the custom liveness deadline
        intent = decision.intents[0]
        assert isinstance(intent.payload, ModelPayloadPostgresUpdateRegistration)
        assert intent.payload.updates["liveness_deadline"] == expected


@pytest.mark.unit
class TestAckEmissionE2EFromAwaitingAck:
    """E2E test: AWAITING_ACK state -> decide_ack -> ACTIVE.

    Same pipeline as ACCEPTED but from the AWAITING_ACK state, which is the
    standard happy-path state when the orchestrator has sent the accepted
    event but the node has not yet acknowledged.
    """

    def test_awaiting_ack_to_active_full_pipeline(self) -> None:
        """Full pipeline: AWAITING_ACK -> decide_ack -> ACTIVE."""
        service = RegistrationReducerService()
        node_id = uuid4()
        correlation_id = uuid4()

        projection = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            entity_id=node_id,
            ack_deadline=TEST_NOW + timedelta(seconds=30),
        )
        command = make_ack_command(
            node_id=node_id,
            correlation_id=correlation_id,
        )

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=correlation_id,
            now=TEST_NOW,
        )

        # Verify full pipeline output
        assert decision.action == "emit"
        assert decision.new_state == EnumRegistrationState.ACTIVE
        assert len(decision.events) == 2
        assert len(decision.intents) == 1

        # Verify event types in order
        assert isinstance(decision.events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(decision.events[1], ModelNodeBecameActive)

        # Verify intent type
        assert isinstance(
            decision.intents[0].payload, ModelPayloadPostgresUpdateRegistration
        )

        # Verify all events reference the correct node_id
        for event in decision.events:
            assert hasattr(event, "node_id")
            assert event.node_id == node_id  # type: ignore[union-attr]

        # Verify all events reference the correct correlation_id
        for event in decision.events:
            assert hasattr(event, "correlation_id")
            assert event.correlation_id == correlation_id  # type: ignore[union-attr]

    def test_awaiting_ack_causation_chain(self) -> None:
        """Verify both events use command_id as causation_id for traceability."""
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

        for event in decision.events:
            assert hasattr(event, "causation_id")
            assert event.causation_id == command.command_id  # type: ignore[union-attr]


@pytest.mark.unit
class TestAckEmissionE2ENoOpPaths:
    """E2E test: verify no-op paths produce zero events and zero intents.

    These states should NOT produce events or intents when an ACK is received,
    ensuring the pipeline is safe against spurious or duplicate ACKs.
    """

    @pytest.mark.parametrize(
        ("state", "expected_reason_fragment"),
        [
            (EnumRegistrationState.ACTIVE, "Duplicate"),
            (EnumRegistrationState.ACK_RECEIVED, "Duplicate"),
            (EnumRegistrationState.PENDING_REGISTRATION, "too early"),
            (EnumRegistrationState.ACK_TIMED_OUT, "too late"),
            (EnumRegistrationState.REJECTED, "Terminal"),
        ],
    )
    def test_noop_states_produce_no_artifacts(
        self,
        state: EnumRegistrationState,
        expected_reason_fragment: str,
    ) -> None:
        """No-op states produce action='no_op' with zero events and intents."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(state, entity_id=node_id)
        command = make_ack_command(node_id=node_id)

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert len(decision.events) == 0
        assert len(decision.intents) == 0
        assert expected_reason_fragment in decision.reason

    def test_none_projection_produces_no_artifacts(self) -> None:
        """projection=None (unknown node) -> no_op with zero artifacts."""
        service = RegistrationReducerService()
        command = make_ack_command()

        decision = service.decide_ack(
            projection=None,
            command=command,
            correlation_id=uuid4(),
            now=TEST_NOW,
        )

        assert decision.action == "no_op"
        assert len(decision.events) == 0
        assert len(decision.intents) == 0
        assert "Unknown" in decision.reason


@pytest.mark.unit
class TestAckEmissionE2EEventFieldIntegrity:
    """Validates field-level integrity of emitted events and intents.

    Ensures that all UUIDs, timestamps, and domain values are
    consistently threaded through the complete decision output.
    """

    def test_all_uuids_are_consistent(self) -> None:
        """All events and intents reference the same node_id and correlation_id."""
        service = RegistrationReducerService()
        node_id = uuid4()
        correlation_id = uuid4()

        projection = make_projection(
            EnumRegistrationState.AWAITING_ACK,
            entity_id=node_id,
        )
        command = make_ack_command(
            node_id=node_id,
            correlation_id=correlation_id,
        )

        decision = service.decide_ack(
            projection=projection,
            command=command,
            correlation_id=correlation_id,
            now=TEST_NOW,
        )

        # Events: node_id + correlation_id consistency
        for event in decision.events:
            assert event.node_id == node_id  # type: ignore[union-attr]
            assert event.correlation_id == correlation_id  # type: ignore[union-attr]
            assert event.entity_id == node_id  # type: ignore[union-attr]

        # Intent payload: entity_id + correlation_id consistency
        for intent in decision.intents:
            payload = intent.payload
            assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
            assert payload.entity_id == node_id
            assert payload.correlation_id == correlation_id

    def test_timestamps_are_injected_not_generated(self) -> None:
        """emitted_at fields match the injected TEST_NOW, not wall clock."""
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

        for event in decision.events:
            assert hasattr(event, "emitted_at")
            assert event.emitted_at == TEST_NOW  # type: ignore[union-attr]

    def test_decision_model_is_frozen(self) -> None:
        """ModelReducerDecision is immutable (frozen=True)."""
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

        with pytest.raises(Exception):  # ValidationError for frozen model
            decision.action = "no_op"  # type: ignore[misc]
