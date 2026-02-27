# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceResolutionEventPublisher.

Tests cover:
    - Publisher correctly serializes event data to JSON
    - Publisher uses the correct topic name
    - Publisher works with in-memory event bus (unit test)
    - Publisher handles serialization of all field types (UUID, datetime, nested models)
    - Publisher is fire-and-forget (failures logged, not raised)
    - Publisher returns True on success, False on failure
    - publish_dict validates input and delegates to publish

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
    - TOPIC_RESOLUTION_DECIDED: onex.evt.platform.resolution-decided.v1
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.topic_constants import TOPIC_RESOLUTION_DECIDED
from omnibase_infra.services.resolution.model_resolution_event_local import (
    ModelResolutionEventLocal,
    ModelResolutionProofLocal,
    ModelTierAttemptLocal,
)
from omnibase_infra.services.resolution.service_resolution_event_publisher import (
    ServiceResolutionEventPublisher,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORRELATION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_EVENT_ID = UUID("11111111-2222-3333-4444-555555555555")
_ROUTE_PLAN_ID = UUID("66666666-7777-8888-9999-aaaaaaaaaaaa")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_bus() -> EventBusInmemory:
    """Create an EventBusInmemory for testing."""
    return EventBusInmemory(environment="test", group="test-resolution")


def _make_simple_event(
    *,
    success: bool = True,
    capability: str = "database.relational",
) -> ModelResolutionEventLocal:
    """Create a simple resolution event for testing."""
    return ModelResolutionEventLocal(
        event_id=_EVENT_ID,
        dependency_capability=capability,
        success=success,
        registry_snapshot_hash="blake3-abc123",
        policy_bundle_hash="sha256-def456",
        trust_graph_hash="sha256-ghi789",
    )


def _make_full_event() -> ModelResolutionEventLocal:
    """Create a resolution event with all fields populated."""
    tier_1 = ModelTierAttemptLocal(
        tier="local_exact",
        attempted_at=datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
        candidates_found=3,
        candidates_after_trust_filter=2,
        failure_code="no_match",
        failure_reason="No exact match found at local tier",
        duration_ms=12.5,
    )
    tier_2 = ModelTierAttemptLocal(
        tier="org_trusted",
        attempted_at=datetime(2026, 2, 26, 10, 0, 1, tzinfo=UTC),
        candidates_found=5,
        candidates_after_trust_filter=3,
        duration_ms=25.0,
    )
    proof_1 = ModelResolutionProofLocal(
        proof_type="node_identity",
        verified=True,
        verification_notes="Ed25519 signature verified",
        verified_at=datetime(2026, 2, 26, 10, 0, 1, 500000, tzinfo=UTC),
    )
    proof_2 = ModelResolutionProofLocal(
        proof_type="capability_attested",
        verified=True,
        verification_notes="Token valid and not expired",
        verified_at=datetime(2026, 2, 26, 10, 0, 1, 600000, tzinfo=UTC),
    )
    return ModelResolutionEventLocal(
        event_id=_EVENT_ID,
        timestamp=datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
        dependency_capability="database.relational",
        registry_snapshot_hash="blake3-registry-hash-abc",
        policy_bundle_hash="sha256-policy-hash-def",
        trust_graph_hash="sha256-trust-graph-hash-ghi",
        route_plan_id=_ROUTE_PLAN_ID,
        tier_progression=(tier_1, tier_2),
        proofs_attempted=(proof_1, proof_2),
        success=True,
        failure_code=None,
        failure_reason=None,
    )


def _make_failed_event() -> ModelResolutionEventLocal:
    """Create a failed resolution event."""
    tier_attempt = ModelTierAttemptLocal(
        tier="local_exact",
        attempted_at=datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
        candidates_found=0,
        candidates_after_trust_filter=0,
        failure_code="tier_exhausted",
        failure_reason="No candidates at any tier",
        duration_ms=5.0,
    )
    return ModelResolutionEventLocal(
        event_id=_EVENT_ID,
        dependency_capability="messaging.pubsub",
        registry_snapshot_hash="blake3-empty",
        policy_bundle_hash="sha256-empty",
        trust_graph_hash="sha256-empty",
        tier_progression=(tier_attempt,),
        success=False,
        failure_code="tier_exhausted",
        failure_reason="All tiers exhausted without finding a match",
    )


# ---------------------------------------------------------------------------
# Topic name tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTopicName:
    """Tests that the publisher uses the correct topic name."""

    @pytest.mark.asyncio
    async def test_default_topic_is_resolution_decided(self) -> None:
        """Publisher defaults to TOPIC_RESOLUTION_DECIDED."""
        bus = _make_bus()
        publisher = ServiceResolutionEventPublisher(bus)
        assert publisher.topic == TOPIC_RESOLUTION_DECIDED

    @pytest.mark.asyncio
    async def test_custom_topic_override(self) -> None:
        """Publisher accepts custom topic override."""
        bus = _make_bus()
        publisher = ServiceResolutionEventPublisher(
            bus, topic="custom.resolution.topic"
        )
        assert publisher.topic == "custom.resolution.topic"

    @pytest.mark.asyncio
    async def test_publishes_to_correct_topic(self) -> None:
        """Published message lands on TOPIC_RESOLUTION_DECIDED topic."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_simple_event()
        result = await publisher.publish(event, correlation_id=_CORRELATION_ID)

        assert result is True
        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        assert len(history) == 1
        assert history[0].topic == TOPIC_RESOLUTION_DECIDED

        await bus.close()

    @pytest.mark.asyncio
    async def test_topic_constant_value(self) -> None:
        """TOPIC_RESOLUTION_DECIDED has the expected value."""
        assert TOPIC_RESOLUTION_DECIDED == "onex.evt.platform.resolution-decided.v1"


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerialization:
    """Tests that events are correctly serialized to JSON."""

    @pytest.mark.asyncio
    async def test_simple_event_serializable(self) -> None:
        """Simple event is published as valid JSON."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_simple_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        assert payload["dependency_capability"] == "database.relational"
        assert payload["success"] is True
        assert payload["registry_snapshot_hash"] == "blake3-abc123"

        await bus.close()

    @pytest.mark.asyncio
    async def test_uuid_fields_serialized_as_strings(self) -> None:
        """UUID fields are serialized as strings in the JSON payload."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_full_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        # event_id should be a string UUID
        assert payload["event_id"] == str(_EVENT_ID)
        # route_plan_id should be a string UUID
        assert payload["route_plan_id"] == str(_ROUTE_PLAN_ID)
        # Verify they're valid UUIDs
        UUID(payload["event_id"])
        UUID(payload["route_plan_id"])

        await bus.close()

    @pytest.mark.asyncio
    async def test_datetime_fields_serialized_as_iso(self) -> None:
        """Datetime fields are serialized as ISO 8601 strings."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_full_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        # timestamp should be a valid ISO string
        ts = payload["timestamp"]
        assert isinstance(ts, str)
        # Should parse without error
        datetime.fromisoformat(ts)

        await bus.close()

    @pytest.mark.asyncio
    async def test_nested_tier_progression_serialized(self) -> None:
        """Tier progression nested models are fully serialized."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_full_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        tiers = payload["tier_progression"]
        assert len(tiers) == 2

        assert tiers[0]["tier"] == "local_exact"
        assert tiers[0]["candidates_found"] == 3
        assert tiers[0]["failure_code"] == "no_match"
        assert tiers[0]["duration_ms"] == 12.5

        assert tiers[1]["tier"] == "org_trusted"
        assert tiers[1]["candidates_found"] == 5
        assert tiers[1]["failure_code"] is None

        await bus.close()

    @pytest.mark.asyncio
    async def test_nested_proofs_serialized(self) -> None:
        """Proof attempt nested models are fully serialized."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_full_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        proofs = payload["proofs_attempted"]
        assert len(proofs) == 2

        assert proofs[0]["proof_type"] == "node_identity"
        assert proofs[0]["verified"] is True
        assert "Ed25519" in proofs[0]["verification_notes"]

        assert proofs[1]["proof_type"] == "capability_attested"
        assert proofs[1]["verified"] is True

        await bus.close()

    @pytest.mark.asyncio
    async def test_failed_event_serialized(self) -> None:
        """Failed resolution event is correctly serialized."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_failed_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        assert payload["success"] is False
        assert payload["failure_code"] == "tier_exhausted"
        assert "exhausted" in payload["failure_reason"]
        assert payload["route_plan_id"] is None

        await bus.close()

    @pytest.mark.asyncio
    async def test_empty_tier_progression_serialized(self) -> None:
        """Event with no tier progression serializes correctly."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_simple_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        payload = json.loads(history[0].value.decode("utf-8"))

        assert payload["tier_progression"] == []
        assert payload["proofs_attempted"] == []

        await bus.close()

    @pytest.mark.asyncio
    async def test_message_key_is_event_id(self) -> None:
        """Published message key is the event_id encoded as UTF-8 bytes."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_simple_event()
        await publisher.publish(event)

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        assert history[0].key == str(_EVENT_ID).encode("utf-8")

        await bus.close()


# ---------------------------------------------------------------------------
# In-memory event bus integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInmemoryBusIntegration:
    """Tests that the publisher works correctly with EventBusInmemory."""

    @pytest.mark.asyncio
    async def test_publish_returns_true_on_success(self) -> None:
        """publish() returns True when event is published successfully."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        result = await publisher.publish(_make_simple_event())

        assert result is True
        await bus.close()

    @pytest.mark.asyncio
    async def test_multiple_events_published(self) -> None:
        """Multiple events can be published sequentially."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event_1 = ModelResolutionEventLocal(
            dependency_capability="database.relational",
            success=True,
        )
        event_2 = ModelResolutionEventLocal(
            dependency_capability="messaging.pubsub",
            success=False,
            failure_code="no_match",
        )

        result_1 = await publisher.publish(event_1)
        result_2 = await publisher.publish(event_2)

        assert result_1 is True
        assert result_2 is True

        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        assert len(history) == 2

        payload_1 = json.loads(history[0].value.decode("utf-8"))
        payload_2 = json.loads(history[1].value.decode("utf-8"))
        assert payload_1["dependency_capability"] == "database.relational"
        assert payload_2["dependency_capability"] == "messaging.pubsub"

        await bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_receives_published_event(self) -> None:
        """A subscriber on the topic receives the published event."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        received: list[bytes] = []

        async def handler(msg: object) -> None:
            received.append(msg.value)  # type: ignore[attr-defined]

        from omnibase_infra.models import ModelNodeIdentity

        identity = ModelNodeIdentity(
            env="test",
            service="test-service",
            node_name="resolution-consumer",
            version="v1",
        )
        unsub = await bus.subscribe(TOPIC_RESOLUTION_DECIDED, identity, handler)

        event = _make_simple_event()
        await publisher.publish(event)

        assert len(received) == 1
        payload = json.loads(received[0].decode("utf-8"))
        assert payload["dependency_capability"] == "database.relational"

        await unsub()
        await bus.close()


# ---------------------------------------------------------------------------
# Resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResilience:
    """Tests that publish failures are caught and never raised."""

    @pytest.mark.asyncio
    async def test_returns_false_when_bus_not_started(self) -> None:
        """publish() returns False when bus is not started."""
        bus = _make_bus()
        # Do NOT start the bus
        publisher = ServiceResolutionEventPublisher(bus)

        event = _make_simple_event()
        result = await publisher.publish(event)

        # Should return False (bus raises InfraUnavailableError)
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_dict_returns_false_on_invalid_data(self) -> None:
        """publish_dict() returns False when event data is invalid."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        # Missing required field 'dependency_capability'
        result = await publisher.publish_dict({"success": True})

        assert result is False
        await bus.close()


# ---------------------------------------------------------------------------
# publish_dict tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublishDict:
    """Tests for the publish_dict convenience method."""

    @pytest.mark.asyncio
    async def test_publish_dict_success(self) -> None:
        """publish_dict() validates and publishes valid event data."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event_data = {
            "event_id": str(_EVENT_ID),
            "dependency_capability": "cache.distributed",
            "success": True,
            "registry_snapshot_hash": "blake3-hash",
            "policy_bundle_hash": "sha256-hash",
            "trust_graph_hash": "sha256-hash",
        }
        result = await publisher.publish_dict(event_data)

        assert result is True
        history = await bus.get_event_history(topic=TOPIC_RESOLUTION_DECIDED)
        assert len(history) == 1

        payload = json.loads(history[0].value.decode("utf-8"))
        assert payload["dependency_capability"] == "cache.distributed"

        await bus.close()

    @pytest.mark.asyncio
    async def test_publish_dict_with_correlation_id(self) -> None:
        """publish_dict() passes correlation_id through to publish."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event_data = {
            "dependency_capability": "storage.object",
            "success": True,
        }
        result = await publisher.publish_dict(
            event_data, correlation_id=_CORRELATION_ID
        )

        assert result is True
        await bus.close()

    @pytest.mark.asyncio
    async def test_publish_dict_rejects_extra_fields(self) -> None:
        """publish_dict() rejects dicts with extra fields (extra='forbid')."""
        bus = _make_bus()
        await bus.start()
        publisher = ServiceResolutionEventPublisher(bus)

        event_data = {
            "dependency_capability": "storage.object",
            "success": True,
            "unknown_field": "should_fail",
        }
        result = await publisher.publish_dict(event_data)

        assert result is False
        await bus.close()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResolutionEventLocal:
    """Tests for the local resolution event model."""

    def test_frozen_model(self) -> None:
        """Model instances are immutable (frozen=True)."""
        event = _make_simple_event()
        with pytest.raises(Exception):
            event.success = False  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Model rejects extra fields (extra='forbid')."""
        with pytest.raises(Exception):
            ModelResolutionEventLocal(
                dependency_capability="test",
                success=True,
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_default_event_id_generated(self) -> None:
        """event_id is auto-generated when not provided."""
        event = ModelResolutionEventLocal(
            dependency_capability="test",
            success=True,
        )
        assert isinstance(event.event_id, UUID)

    def test_default_timestamp_generated(self) -> None:
        """timestamp is auto-generated when not provided."""
        event = ModelResolutionEventLocal(
            dependency_capability="test",
            success=True,
        )
        assert isinstance(event.timestamp, datetime)

    def test_to_publishable_dict_json_safe(self) -> None:
        """to_publishable_dict() returns JSON-serializable dict."""
        event = _make_full_event()
        d = event.to_publishable_dict()
        # Must not raise
        json.dumps(d)

    def test_to_publishable_dict_contains_all_fields(self) -> None:
        """to_publishable_dict() includes all model fields."""
        event = _make_full_event()
        d = event.to_publishable_dict()
        expected_keys = {
            "event_id",
            "timestamp",
            "dependency_capability",
            "registry_snapshot_hash",
            "policy_bundle_hash",
            "trust_graph_hash",
            "route_plan_id",
            "tier_progression",
            "proofs_attempted",
            "success",
            "failure_code",
            "failure_reason",
        }
        assert set(d.keys()) == expected_keys

    def test_round_trip_serialization(self) -> None:
        """Model survives JSON round-trip (serialize -> deserialize)."""
        event = _make_full_event()
        d = event.to_publishable_dict()
        json_str = json.dumps(d)
        restored = ModelResolutionEventLocal.model_validate_json(json_str)

        assert restored.event_id == event.event_id
        assert restored.dependency_capability == event.dependency_capability
        assert restored.success == event.success
        assert len(restored.tier_progression) == 2
        assert len(restored.proofs_attempted) == 2


@pytest.mark.unit
class TestModelTierAttemptLocal:
    """Tests for the tier attempt local model."""

    def test_frozen(self) -> None:
        """Tier attempt model is immutable."""
        attempt = ModelTierAttemptLocal(
            tier="local_exact",
            attempted_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(Exception):
            attempt.tier = "org_trusted"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Tier attempt survives JSON round-trip."""
        attempt = ModelTierAttemptLocal(
            tier="org_trusted",
            attempted_at=datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
            candidates_found=5,
            candidates_after_trust_filter=3,
            failure_code="policy_denied",
            failure_reason="Classification gate blocked",
            duration_ms=42.5,
        )
        d = attempt.model_dump(mode="json")
        restored = ModelTierAttemptLocal.model_validate(d)
        assert restored.tier == "org_trusted"
        assert restored.candidates_found == 5
        assert restored.duration_ms == 42.5


@pytest.mark.unit
class TestModelResolutionProofLocal:
    """Tests for the resolution proof local model."""

    def test_frozen(self) -> None:
        """Proof model is immutable."""
        proof = ModelResolutionProofLocal(
            proof_type="node_identity",
            verified=True,
        )
        with pytest.raises(Exception):
            proof.verified = False  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Proof model survives JSON round-trip."""
        proof = ModelResolutionProofLocal(
            proof_type="capability_attested",
            verified=True,
            verification_notes="Valid token",
            verified_at=datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
        )
        d = proof.model_dump(mode="json")
        restored = ModelResolutionProofLocal.model_validate(d)
        assert restored.proof_type == "capability_attested"
        assert restored.verified is True
        assert restored.verified_at is not None
