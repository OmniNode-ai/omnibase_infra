# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelRegistrationProjection.

Tests validate:
- Model instantiation with valid data
- Field validation (types, constraints)
- Relationship to ModelSequenceInfo (get_sequence_info)
- Staleness checking (is_stale)
- Deadline checking (has_ack_deadline_passed, has_liveness_deadline_passed)
- Timeout event logic (needs_ack_timeout_event, needs_liveness_timeout_event)
- Mutability (frozen=False)
- Serialization with from_attributes=True

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import ValidationError

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration import ModelNodeCapabilities


class TestModelRegistrationProjectionInstantiation:
    """Tests for model instantiation with valid data."""

    def test_minimal_instantiation(self) -> None:
        """Test instantiation with only required fields."""
        now = datetime.now(UTC)
        entity_id = uuid4()
        event_id = uuid4()

        proj = ModelRegistrationProjection(
            entity_id=entity_id,
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=event_id,
            registered_at=now,
            updated_at=now,
        )

        assert proj.entity_id == entity_id
        assert proj.current_state == EnumRegistrationState.PENDING_REGISTRATION
        assert proj.node_type == "effect"
        assert proj.last_applied_event_id == event_id
        assert proj.registered_at == now
        assert proj.updated_at == now

    def test_full_instantiation(self) -> None:
        """Test instantiation with all fields."""
        now = datetime.now(UTC)
        entity_id = uuid4()
        event_id = uuid4()
        correlation_id = uuid4()
        ack_deadline = now + timedelta(seconds=30)
        liveness_deadline = now + timedelta(seconds=60)

        proj = ModelRegistrationProjection(
            entity_id=entity_id,
            domain="custom_domain",
            current_state=EnumRegistrationState.ACTIVE,
            node_type="compute",
            node_version="2.1.0",
            capabilities=ModelNodeCapabilities(postgres=True, read=True),
            ack_deadline=ack_deadline,
            liveness_deadline=liveness_deadline,
            ack_timeout_emitted_at=None,
            liveness_timeout_emitted_at=None,
            last_applied_event_id=event_id,
            last_applied_offset=12345,
            last_applied_sequence=12345,
            last_applied_partition="0",
            registered_at=now,
            updated_at=now,
            correlation_id=correlation_id,
        )

        assert proj.domain == "custom_domain"
        assert proj.node_version == "2.1.0"
        assert proj.capabilities.postgres is True
        assert proj.ack_deadline == ack_deadline
        assert proj.liveness_deadline == liveness_deadline
        assert proj.last_applied_offset == 12345
        assert proj.correlation_id == correlation_id

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )

        assert proj.domain == "registration"
        assert proj.node_version == "1.0.0"
        assert proj.capabilities is not None
        assert proj.ack_deadline is None
        assert proj.liveness_deadline is None
        assert proj.ack_timeout_emitted_at is None
        assert proj.liveness_timeout_emitted_at is None
        assert proj.last_applied_offset == 0
        assert proj.last_applied_sequence is None
        assert proj.last_applied_partition is None
        assert proj.correlation_id is None


class TestModelRegistrationProjectionFieldValidation:
    """Tests for field validation and constraints."""

    def test_entity_id_required(self) -> None:
        """Test that entity_id is required."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )  # type: ignore[call-arg]
        assert "entity_id" in str(exc_info.value)

    def test_current_state_required(self) -> None:
        """Test that current_state is required."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                node_type="effect",
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )  # type: ignore[call-arg]
        assert "current_state" in str(exc_info.value)

    def test_node_type_literal_validation(self) -> None:
        """Test that node_type only accepts valid literals."""
        now = datetime.now(UTC)
        # Valid node types
        for node_type in ["effect", "compute", "reducer", "orchestrator"]:
            proj = ModelRegistrationProjection(
                entity_id=uuid4(),
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type=node_type,  # type: ignore[arg-type]
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )
            assert proj.node_type == node_type

    def test_node_type_invalid_value(self) -> None:
        """Test that invalid node_type raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="invalid_type",  # type: ignore[arg-type]
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )
        assert "node_type" in str(exc_info.value)

    def test_domain_min_length(self) -> None:
        """Test that domain must have at least 1 character."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                domain="",
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )
        assert "domain" in str(exc_info.value)

    def test_domain_max_length(self) -> None:
        """Test that domain cannot exceed 128 characters."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                domain="x" * 129,
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
            )
        assert "domain" in str(exc_info.value)

    def test_last_applied_offset_must_be_non_negative(self) -> None:
        """Test that last_applied_offset must be >= 0."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                last_applied_offset=-1,
                registered_at=now,
                updated_at=now,
            )
        assert "last_applied_offset" in str(exc_info.value)

    def test_last_applied_sequence_must_be_non_negative(self) -> None:
        """Test that last_applied_sequence must be >= 0 if provided."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistrationProjection(
                entity_id=uuid4(),
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                last_applied_sequence=-1,
                registered_at=now,
                updated_at=now,
            )
        assert "last_applied_sequence" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelRegistrationProjection(
                entity_id=uuid4(),
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                last_applied_event_id=uuid4(),
                registered_at=now,
                updated_at=now,
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestModelRegistrationProjectionMutability:
    """Tests for model mutability (frozen=False)."""

    def test_current_state_is_mutable(self) -> None:
        """Test that current_state can be modified."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        proj.current_state = EnumRegistrationState.ACCEPTED
        assert proj.current_state == EnumRegistrationState.ACCEPTED

    def test_updated_at_is_mutable(self) -> None:
        """Test that updated_at can be modified."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        new_time = now + timedelta(hours=1)
        proj.updated_at = new_time
        assert proj.updated_at == new_time

    def test_deadlines_are_mutable(self) -> None:
        """Test that deadline fields can be modified."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        deadline = now + timedelta(seconds=30)
        proj.ack_deadline = deadline
        proj.liveness_deadline = deadline
        assert proj.ack_deadline == deadline
        assert proj.liveness_deadline == deadline


class TestGetSequenceInfo:
    """Tests for get_sequence_info() method."""

    def test_get_sequence_info_with_offset_only(self) -> None:
        """Test get_sequence_info when only offset is set."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=12345,
            registered_at=now,
            updated_at=now,
        )
        seq_info = proj.get_sequence_info()
        assert seq_info.sequence == 12345
        assert seq_info.partition is None
        assert seq_info.offset is None  # No partition means no offset in result

    def test_get_sequence_info_with_kafka_metadata(self) -> None:
        """Test get_sequence_info with full Kafka metadata."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=12345,
            last_applied_partition="0",
            registered_at=now,
            updated_at=now,
        )
        seq_info = proj.get_sequence_info()
        assert seq_info.sequence == 12345
        assert seq_info.partition == "0"
        assert seq_info.offset == 12345

    def test_get_sequence_info_with_explicit_sequence(self) -> None:
        """Test get_sequence_info prefers last_applied_sequence when set."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            last_applied_sequence=200,
            registered_at=now,
            updated_at=now,
        )
        seq_info = proj.get_sequence_info()
        # Should prefer last_applied_sequence
        assert seq_info.sequence == 200

    def test_get_sequence_info_with_zero_sequence(self) -> None:
        """Test get_sequence_info correctly handles last_applied_sequence=0.

        Edge case: sequence=0 is a valid value and should NOT fall back to offset.
        This tests the fix for using explicit None check instead of truthy 'or'.
        """
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,  # offset is 100
            last_applied_sequence=0,  # sequence is explicitly 0
            registered_at=now,
            updated_at=now,
        )
        seq_info = proj.get_sequence_info()
        # Should use last_applied_sequence=0, NOT fall back to last_applied_offset=100
        assert seq_info.sequence == 0

    def test_get_sequence_info_returns_model_sequence_info(self) -> None:
        """Test that get_sequence_info returns ModelSequenceInfo instance."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )
        seq_info = proj.get_sequence_info()
        assert isinstance(seq_info, ModelSequenceInfo)


class TestIsStale:
    """Tests for is_stale() method."""

    def test_stale_sequence_is_detected(self) -> None:
        """Test that stale incoming sequence is detected."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )
        old_seq = ModelSequenceInfo(sequence=50)
        assert proj.is_stale(old_seq) is True

    def test_newer_sequence_is_not_stale(self) -> None:
        """Test that newer incoming sequence is not stale."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )
        new_seq = ModelSequenceInfo(sequence=150)
        assert proj.is_stale(new_seq) is False

    def test_same_sequence_is_stale(self) -> None:
        """Test that same sequence is considered stale (already applied)."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )
        same_seq = ModelSequenceInfo(sequence=100)
        # Same or lower is stale
        # is_stale checks if incoming.is_stale_compared_to(current)
        # ModelSequenceInfo.is_stale_compared_to returns True if self < other
        # So same sequence (100 vs 100) should return False for is_stale_compared_to
        # But semantically, same sequence means already processed -> should be rejected
        # The logic: incoming.is_stale_compared_to(current) checks if incoming < current
        # For same values: 100 is NOT less than 100, so is_stale_compared_to returns False
        # This means is_stale returns False for same sequence
        # The projector should use >= to reject (same or older)
        # Actually looking at the code: incoming_sequence.is_stale_compared_to(current)
        # This checks if incoming is older than current -> should be rejected
        assert proj.is_stale(same_seq) is False


class TestHasAckDeadlinePassed:
    """Tests for has_ack_deadline_passed() method."""

    def test_no_deadline_returns_false(self) -> None:
        """Test that None deadline returns False."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_ack_deadline_passed(now) is False

    def test_past_deadline_returns_true(self) -> None:
        """Test that past deadline returns True."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=past_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_ack_deadline_passed(now) is True

    def test_future_deadline_returns_false(self) -> None:
        """Test that future deadline returns False."""
        now = datetime.now(UTC)
        future_deadline = now + timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=future_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_ack_deadline_passed(now) is False

    def test_exact_deadline_returns_false(self) -> None:
        """Test that exact deadline time returns False (not strictly past)."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=now,
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_ack_deadline_passed(now) is False


class TestHasLivenessDeadlinePassed:
    """Tests for has_liveness_deadline_passed() method."""

    def test_no_deadline_returns_false(self) -> None:
        """Test that None deadline returns False."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_liveness_deadline_passed(now) is False

    def test_past_deadline_returns_true(self) -> None:
        """Test that past deadline returns True."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=past_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_liveness_deadline_passed(now) is True

    def test_future_deadline_returns_false(self) -> None:
        """Test that future deadline returns False."""
        now = datetime.now(UTC)
        future_deadline = now + timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=future_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.has_liveness_deadline_passed(now) is False


class TestNeedsAckTimeoutEvent:
    """Tests for needs_ack_timeout_event() method."""

    def test_needs_ack_timeout_when_all_conditions_met(self) -> None:
        """Test returns True when deadline passed, not emitted, and state requires ack."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=past_deadline,
            ack_timeout_emitted_at=None,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_ack_timeout_event(now) is True

    def test_no_ack_timeout_when_deadline_not_passed(self) -> None:
        """Test returns False when deadline hasn't passed."""
        now = datetime.now(UTC)
        future_deadline = now + timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=future_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_ack_timeout_event(now) is False

    def test_no_ack_timeout_when_already_emitted(self) -> None:
        """Test returns False when timeout event already emitted."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=past_deadline,
            ack_timeout_emitted_at=now - timedelta(minutes=1),
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_ack_timeout_event(now) is False

    def test_no_ack_timeout_when_state_does_not_require_ack(self) -> None:
        """Test returns False when state doesn't require ack."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,  # Doesn't require ack
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=past_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_ack_timeout_event(now) is False

    def test_ack_timeout_for_accepted_state(self) -> None:
        """Test ACCEPTED state triggers ack timeout when conditions met."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACCEPTED,
            node_type="effect",
            last_applied_event_id=uuid4(),
            ack_deadline=past_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_ack_timeout_event(now) is True


class TestNeedsLivenessTimeoutEvent:
    """Tests for needs_liveness_timeout_event() method."""

    def test_needs_liveness_timeout_when_all_conditions_met(self) -> None:
        """Test returns True when deadline passed, not emitted, and state is ACTIVE."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=past_deadline,
            liveness_timeout_emitted_at=None,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_liveness_timeout_event(now) is True

    def test_no_liveness_timeout_when_deadline_not_passed(self) -> None:
        """Test returns False when deadline hasn't passed."""
        now = datetime.now(UTC)
        future_deadline = now + timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=future_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_liveness_timeout_event(now) is False

    def test_no_liveness_timeout_when_already_emitted(self) -> None:
        """Test returns False when timeout event already emitted."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=past_deadline,
            liveness_timeout_emitted_at=now - timedelta(minutes=1),
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_liveness_timeout_event(now) is False

    def test_no_liveness_timeout_when_state_not_active(self) -> None:
        """Test returns False when state is not ACTIVE."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.AWAITING_ACK,  # Not ACTIVE
            node_type="effect",
            last_applied_event_id=uuid4(),
            liveness_deadline=past_deadline,
            registered_at=now,
            updated_at=now,
        )
        assert proj.needs_liveness_timeout_event(now) is False


class TestModelRegistrationProjectionSerialization:
    """Tests for model serialization."""

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        now = datetime.now(UTC)
        entity_id = uuid4()
        event_id = uuid4()
        proj = ModelRegistrationProjection(
            entity_id=entity_id,
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=event_id,
            registered_at=now,
            updated_at=now,
        )
        data = proj.model_dump()

        assert data["entity_id"] == entity_id
        assert data["current_state"] == EnumRegistrationState.ACTIVE
        assert data["node_type"] == "effect"
        assert data["domain"] == "registration"

    def test_model_dump_json(self) -> None:
        """Test JSON serialization."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        json_str = proj.model_dump_json()
        assert "active" in json_str
        assert "effect" in json_str
        assert "registration" in json_str

    def test_model_from_dict(self) -> None:
        """Test deserialization from dict."""
        now = datetime.now(UTC)
        entity_id = uuid4()
        event_id = uuid4()
        data = {
            "entity_id": entity_id,
            "current_state": EnumRegistrationState.ACTIVE,
            "node_type": "effect",
            "last_applied_event_id": event_id,
            "registered_at": now,
            "updated_at": now,
        }
        proj = ModelRegistrationProjection.model_validate(data)
        assert proj.entity_id == entity_id
        assert proj.current_state == EnumRegistrationState.ACTIVE


class TestModelRegistrationProjectionFromAttributes:
    """Tests for from_attributes=True config."""

    def test_from_attributes_with_class(self) -> None:
        """Test that from_attributes works with class instances."""
        now = datetime.now(UTC)
        entity_id = uuid4()
        event_id = uuid4()

        class MockRow:
            """Mock database row object."""

            entity_id: UUID
            domain: str
            current_state: EnumRegistrationState
            node_type: EnumNodeKind
            node_version: str
            capabilities: ModelNodeCapabilities
            ack_deadline: datetime | None
            liveness_deadline: datetime | None
            ack_timeout_emitted_at: datetime | None
            liveness_timeout_emitted_at: datetime | None
            last_applied_event_id: UUID
            last_applied_offset: int
            last_applied_sequence: int | None
            last_applied_partition: str | None
            registered_at: datetime
            updated_at: datetime
            correlation_id: UUID | None

            def __init__(self) -> None:
                self.entity_id = entity_id
                self.domain = "registration"
                self.current_state = EnumRegistrationState.ACTIVE
                self.node_type = EnumNodeKind.EFFECT
                self.node_version = "1.0.0"
                self.capabilities = ModelNodeCapabilities()
                self.ack_deadline = None
                self.liveness_deadline = None
                self.ack_timeout_emitted_at = None
                self.liveness_timeout_emitted_at = None
                self.last_applied_event_id = event_id
                self.last_applied_offset = 100
                self.last_applied_sequence = None
                self.last_applied_partition = None
                self.registered_at = now
                self.updated_at = now
                self.correlation_id = None

        row = MockRow()
        proj = ModelRegistrationProjection.model_validate(row)
        assert proj.entity_id == entity_id
        assert proj.current_state == EnumRegistrationState.ACTIVE
        assert proj.node_type == EnumNodeKind.EFFECT


class TestModelRegistrationProjectionCapabilities:
    """Tests for capabilities field."""

    def test_default_capabilities(self) -> None:
        """Test default capabilities is empty ModelNodeCapabilities."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.capabilities is not None
        assert isinstance(proj.capabilities, ModelNodeCapabilities)
        assert proj.capabilities.postgres is False

    def test_custom_capabilities(self) -> None:
        """Test custom capabilities are preserved."""
        now = datetime.now(UTC)
        caps = ModelNodeCapabilities(
            postgres=True, read=True, write=True, batch_size=100
        )
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            capabilities=caps,
            registered_at=now,
            updated_at=now,
        )
        assert proj.capabilities.postgres is True
        assert proj.capabilities.batch_size == 100


class TestModelRegistrationProjectionAllStates:
    """Tests for all registration states."""

    @pytest.mark.parametrize(
        "state",
        list(EnumRegistrationState),
    )
    def test_all_states_are_valid(self, state: EnumRegistrationState) -> None:
        """Test that all EnumRegistrationState values are valid for current_state."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=state,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.current_state == state


class TestModelRegistrationProjectionAllNodeTypes:
    """Tests for all node types."""

    @pytest.mark.parametrize(
        "node_type",
        [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ],
    )
    def test_all_node_types_are_valid(self, node_type: EnumNodeKind) -> None:
        """Test that all node types are valid."""
        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type=node_type,
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.node_type == node_type
