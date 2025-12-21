# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EnumRegistrationState.

Tests validate:
- All 8 FSM state values exist and have correct string representations
- Enum value uniqueness (enforced by @unique decorator)
- Helper methods: is_terminal(), is_active(), requires_ack(), requires_liveness()
- Retry capability: can_retry()
- State transition validation: can_transition_to()
- Human-readable descriptions: get_description()

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumRegistrationState


class TestEnumRegistrationStateValues:
    """Tests for enum value existence and string representations."""

    def test_all_states_exist(self) -> None:
        """Verify all 8 FSM states are defined."""
        expected_states = {
            "PENDING_REGISTRATION",
            "ACCEPTED",
            "AWAITING_ACK",
            "REJECTED",
            "ACK_TIMED_OUT",
            "ACK_RECEIVED",
            "ACTIVE",
            "LIVENESS_EXPIRED",
        }
        actual_states = {state.name for state in EnumRegistrationState}
        assert actual_states == expected_states

    def test_state_count(self) -> None:
        """Verify exactly 8 states exist."""
        assert len(EnumRegistrationState) == 8

    def test_string_values_are_snake_case(self) -> None:
        """Verify all state values use snake_case format."""
        expected_values = {
            "pending_registration",
            "accepted",
            "awaiting_ack",
            "rejected",
            "ack_timed_out",
            "ack_received",
            "active",
            "liveness_expired",
        }
        actual_values = {state.value for state in EnumRegistrationState}
        assert actual_values == expected_values

    def test_str_returns_value(self) -> None:
        """Verify __str__ returns the enum value for serialization."""
        assert str(EnumRegistrationState.PENDING_REGISTRATION) == "pending_registration"
        assert str(EnumRegistrationState.ACCEPTED) == "accepted"
        assert str(EnumRegistrationState.AWAITING_ACK) == "awaiting_ack"
        assert str(EnumRegistrationState.REJECTED) == "rejected"
        assert str(EnumRegistrationState.ACK_TIMED_OUT) == "ack_timed_out"
        assert str(EnumRegistrationState.ACK_RECEIVED) == "ack_received"
        assert str(EnumRegistrationState.ACTIVE) == "active"
        assert str(EnumRegistrationState.LIVENESS_EXPIRED) == "liveness_expired"

    def test_value_matches_str(self) -> None:
        """Verify value property equals str() for all states."""
        for state in EnumRegistrationState:
            assert state.value == str(state)

    def test_enum_is_str_enum(self) -> None:
        """Verify enum inherits from StrEnum for proper serialization."""
        # StrEnum values should be strings
        for state in EnumRegistrationState:
            assert isinstance(state.value, str)
            # StrEnum allows direct comparison with strings
            assert state == state.value


class TestEnumRegistrationStateUniqueness:
    """Tests for enum value uniqueness (enforced by @unique decorator)."""

    def test_all_values_unique(self) -> None:
        """Verify all enum values are unique."""
        values = [state.value for state in EnumRegistrationState]
        assert len(values) == len(set(values))

    def test_all_names_unique(self) -> None:
        """Verify all enum names are unique."""
        names = [state.name for state in EnumRegistrationState]
        assert len(names) == len(set(names))


class TestIsTerminal:
    """Tests for EnumRegistrationState.is_terminal() method."""

    def test_rejected_is_terminal(self) -> None:
        """REJECTED is a terminal state."""
        assert EnumRegistrationState.REJECTED.is_terminal() is True

    def test_liveness_expired_is_terminal(self) -> None:
        """LIVENESS_EXPIRED is a terminal state."""
        assert EnumRegistrationState.LIVENESS_EXPIRED.is_terminal() is True

    def test_pending_registration_is_not_terminal(self) -> None:
        """PENDING_REGISTRATION is not a terminal state."""
        assert EnumRegistrationState.PENDING_REGISTRATION.is_terminal() is False

    def test_accepted_is_not_terminal(self) -> None:
        """ACCEPTED is not a terminal state."""
        assert EnumRegistrationState.ACCEPTED.is_terminal() is False

    def test_awaiting_ack_is_not_terminal(self) -> None:
        """AWAITING_ACK is not a terminal state."""
        assert EnumRegistrationState.AWAITING_ACK.is_terminal() is False

    def test_ack_timed_out_is_not_terminal(self) -> None:
        """ACK_TIMED_OUT is not terminal (retriable)."""
        assert EnumRegistrationState.ACK_TIMED_OUT.is_terminal() is False

    def test_ack_received_is_not_terminal(self) -> None:
        """ACK_RECEIVED is not a terminal state."""
        assert EnumRegistrationState.ACK_RECEIVED.is_terminal() is False

    def test_active_is_not_terminal(self) -> None:
        """ACTIVE is not a terminal state (can transition to LIVENESS_EXPIRED)."""
        assert EnumRegistrationState.ACTIVE.is_terminal() is False

    def test_only_two_terminal_states(self) -> None:
        """Verify exactly REJECTED and LIVENESS_EXPIRED are terminal."""
        terminal_states = [s for s in EnumRegistrationState if s.is_terminal()]
        assert set(terminal_states) == {
            EnumRegistrationState.REJECTED,
            EnumRegistrationState.LIVENESS_EXPIRED,
        }


class TestIsActive:
    """Tests for EnumRegistrationState.is_active() method."""

    def test_active_is_active(self) -> None:
        """ACTIVE state returns True for is_active()."""
        assert EnumRegistrationState.ACTIVE.is_active() is True

    def test_pending_registration_is_not_active(self) -> None:
        """PENDING_REGISTRATION is not active."""
        assert EnumRegistrationState.PENDING_REGISTRATION.is_active() is False

    def test_accepted_is_not_active(self) -> None:
        """ACCEPTED is not active."""
        assert EnumRegistrationState.ACCEPTED.is_active() is False

    def test_awaiting_ack_is_not_active(self) -> None:
        """AWAITING_ACK is not active."""
        assert EnumRegistrationState.AWAITING_ACK.is_active() is False

    def test_rejected_is_not_active(self) -> None:
        """REJECTED is not active."""
        assert EnumRegistrationState.REJECTED.is_active() is False

    def test_ack_timed_out_is_not_active(self) -> None:
        """ACK_TIMED_OUT is not active."""
        assert EnumRegistrationState.ACK_TIMED_OUT.is_active() is False

    def test_ack_received_is_not_active(self) -> None:
        """ACK_RECEIVED is not active."""
        assert EnumRegistrationState.ACK_RECEIVED.is_active() is False

    def test_liveness_expired_is_not_active(self) -> None:
        """LIVENESS_EXPIRED is not active."""
        assert EnumRegistrationState.LIVENESS_EXPIRED.is_active() is False

    def test_only_active_state_is_active(self) -> None:
        """Verify only ACTIVE returns True for is_active()."""
        active_states = [s for s in EnumRegistrationState if s.is_active()]
        assert active_states == [EnumRegistrationState.ACTIVE]


class TestRequiresAck:
    """Tests for EnumRegistrationState.requires_ack() method."""

    def test_accepted_requires_ack(self) -> None:
        """ACCEPTED requires acknowledgment."""
        assert EnumRegistrationState.ACCEPTED.requires_ack() is True

    def test_awaiting_ack_requires_ack(self) -> None:
        """AWAITING_ACK requires acknowledgment."""
        assert EnumRegistrationState.AWAITING_ACK.requires_ack() is True

    def test_pending_registration_does_not_require_ack(self) -> None:
        """PENDING_REGISTRATION does not require ack."""
        assert EnumRegistrationState.PENDING_REGISTRATION.requires_ack() is False

    def test_rejected_does_not_require_ack(self) -> None:
        """REJECTED does not require ack."""
        assert EnumRegistrationState.REJECTED.requires_ack() is False

    def test_ack_timed_out_does_not_require_ack(self) -> None:
        """ACK_TIMED_OUT does not require ack (timeout already occurred)."""
        assert EnumRegistrationState.ACK_TIMED_OUT.requires_ack() is False

    def test_ack_received_does_not_require_ack(self) -> None:
        """ACK_RECEIVED does not require ack (already acknowledged)."""
        assert EnumRegistrationState.ACK_RECEIVED.requires_ack() is False

    def test_active_does_not_require_ack(self) -> None:
        """ACTIVE does not require ack."""
        assert EnumRegistrationState.ACTIVE.requires_ack() is False

    def test_liveness_expired_does_not_require_ack(self) -> None:
        """LIVENESS_EXPIRED does not require ack."""
        assert EnumRegistrationState.LIVENESS_EXPIRED.requires_ack() is False

    def test_exactly_two_states_require_ack(self) -> None:
        """Verify exactly ACCEPTED and AWAITING_ACK require ack."""
        ack_required_states = [s for s in EnumRegistrationState if s.requires_ack()]
        assert set(ack_required_states) == {
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
        }


class TestRequiresLiveness:
    """Tests for EnumRegistrationState.requires_liveness() method."""

    def test_active_requires_liveness(self) -> None:
        """ACTIVE requires liveness monitoring."""
        assert EnumRegistrationState.ACTIVE.requires_liveness() is True

    def test_pending_registration_does_not_require_liveness(self) -> None:
        """PENDING_REGISTRATION does not require liveness monitoring."""
        assert EnumRegistrationState.PENDING_REGISTRATION.requires_liveness() is False

    def test_accepted_does_not_require_liveness(self) -> None:
        """ACCEPTED does not require liveness monitoring."""
        assert EnumRegistrationState.ACCEPTED.requires_liveness() is False

    def test_awaiting_ack_does_not_require_liveness(self) -> None:
        """AWAITING_ACK does not require liveness monitoring."""
        assert EnumRegistrationState.AWAITING_ACK.requires_liveness() is False

    def test_rejected_does_not_require_liveness(self) -> None:
        """REJECTED does not require liveness monitoring."""
        assert EnumRegistrationState.REJECTED.requires_liveness() is False

    def test_ack_timed_out_does_not_require_liveness(self) -> None:
        """ACK_TIMED_OUT does not require liveness monitoring."""
        assert EnumRegistrationState.ACK_TIMED_OUT.requires_liveness() is False

    def test_ack_received_does_not_require_liveness(self) -> None:
        """ACK_RECEIVED does not require liveness monitoring."""
        assert EnumRegistrationState.ACK_RECEIVED.requires_liveness() is False

    def test_liveness_expired_does_not_require_liveness(self) -> None:
        """LIVENESS_EXPIRED does not require liveness monitoring (already expired)."""
        assert EnumRegistrationState.LIVENESS_EXPIRED.requires_liveness() is False

    def test_only_active_requires_liveness(self) -> None:
        """Verify only ACTIVE requires liveness monitoring."""
        liveness_states = [s for s in EnumRegistrationState if s.requires_liveness()]
        assert liveness_states == [EnumRegistrationState.ACTIVE]


class TestCanRetry:
    """Tests for EnumRegistrationState.can_retry() method."""

    def test_ack_timed_out_can_retry(self) -> None:
        """ACK_TIMED_OUT allows retry."""
        assert EnumRegistrationState.ACK_TIMED_OUT.can_retry() is True

    def test_pending_registration_cannot_retry(self) -> None:
        """PENDING_REGISTRATION cannot retry (already initial state)."""
        assert EnumRegistrationState.PENDING_REGISTRATION.can_retry() is False

    def test_accepted_cannot_retry(self) -> None:
        """ACCEPTED cannot retry."""
        assert EnumRegistrationState.ACCEPTED.can_retry() is False

    def test_awaiting_ack_cannot_retry(self) -> None:
        """AWAITING_ACK cannot retry."""
        assert EnumRegistrationState.AWAITING_ACK.can_retry() is False

    def test_rejected_cannot_retry(self) -> None:
        """REJECTED cannot retry (terminal)."""
        assert EnumRegistrationState.REJECTED.can_retry() is False

    def test_ack_received_cannot_retry(self) -> None:
        """ACK_RECEIVED cannot retry."""
        assert EnumRegistrationState.ACK_RECEIVED.can_retry() is False

    def test_active_cannot_retry(self) -> None:
        """ACTIVE cannot retry."""
        assert EnumRegistrationState.ACTIVE.can_retry() is False

    def test_liveness_expired_cannot_retry(self) -> None:
        """LIVENESS_EXPIRED cannot retry (terminal)."""
        assert EnumRegistrationState.LIVENESS_EXPIRED.can_retry() is False

    def test_only_ack_timed_out_can_retry(self) -> None:
        """Verify only ACK_TIMED_OUT can retry."""
        retry_states = [s for s in EnumRegistrationState if s.can_retry()]
        assert retry_states == [EnumRegistrationState.ACK_TIMED_OUT]


class TestCanTransitionTo:
    """Tests for EnumRegistrationState.can_transition_to() method."""

    def test_pending_to_accepted_valid(self) -> None:
        """PENDING_REGISTRATION -> ACCEPTED is valid."""
        assert EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            EnumRegistrationState.ACCEPTED
        )

    def test_pending_to_rejected_valid(self) -> None:
        """PENDING_REGISTRATION -> REJECTED is valid."""
        assert EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            EnumRegistrationState.REJECTED
        )

    def test_pending_to_active_invalid(self) -> None:
        """PENDING_REGISTRATION -> ACTIVE is invalid (skips steps)."""
        assert not EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            EnumRegistrationState.ACTIVE
        )

    def test_accepted_to_awaiting_ack_valid(self) -> None:
        """ACCEPTED -> AWAITING_ACK is valid."""
        assert EnumRegistrationState.ACCEPTED.can_transition_to(
            EnumRegistrationState.AWAITING_ACK
        )

    def test_accepted_to_active_invalid(self) -> None:
        """ACCEPTED -> ACTIVE is invalid (must go through AWAITING_ACK)."""
        assert not EnumRegistrationState.ACCEPTED.can_transition_to(
            EnumRegistrationState.ACTIVE
        )

    def test_awaiting_ack_to_ack_received_valid(self) -> None:
        """AWAITING_ACK -> ACK_RECEIVED is valid."""
        assert EnumRegistrationState.AWAITING_ACK.can_transition_to(
            EnumRegistrationState.ACK_RECEIVED
        )

    def test_awaiting_ack_to_ack_timed_out_valid(self) -> None:
        """AWAITING_ACK -> ACK_TIMED_OUT is valid."""
        assert EnumRegistrationState.AWAITING_ACK.can_transition_to(
            EnumRegistrationState.ACK_TIMED_OUT
        )

    def test_awaiting_ack_to_active_invalid(self) -> None:
        """AWAITING_ACK -> ACTIVE is invalid (must go through ACK_RECEIVED)."""
        assert not EnumRegistrationState.AWAITING_ACK.can_transition_to(
            EnumRegistrationState.ACTIVE
        )

    def test_ack_received_to_active_valid(self) -> None:
        """ACK_RECEIVED -> ACTIVE is valid."""
        assert EnumRegistrationState.ACK_RECEIVED.can_transition_to(
            EnumRegistrationState.ACTIVE
        )

    def test_active_to_liveness_expired_valid(self) -> None:
        """ACTIVE -> LIVENESS_EXPIRED is valid."""
        assert EnumRegistrationState.ACTIVE.can_transition_to(
            EnumRegistrationState.LIVENESS_EXPIRED
        )

    def test_active_to_pending_invalid(self) -> None:
        """ACTIVE -> PENDING_REGISTRATION is invalid."""
        assert not EnumRegistrationState.ACTIVE.can_transition_to(
            EnumRegistrationState.PENDING_REGISTRATION
        )

    def test_ack_timed_out_to_pending_valid(self) -> None:
        """ACK_TIMED_OUT -> PENDING_REGISTRATION is valid (retry)."""
        assert EnumRegistrationState.ACK_TIMED_OUT.can_transition_to(
            EnumRegistrationState.PENDING_REGISTRATION
        )

    def test_rejected_has_no_valid_transitions(self) -> None:
        """REJECTED (terminal) has no valid transitions."""
        for target in EnumRegistrationState:
            assert not EnumRegistrationState.REJECTED.can_transition_to(target)

    def test_liveness_expired_has_no_valid_transitions(self) -> None:
        """LIVENESS_EXPIRED (terminal) has no valid transitions."""
        for target in EnumRegistrationState:
            assert not EnumRegistrationState.LIVENESS_EXPIRED.can_transition_to(target)

    def test_self_transition_is_invalid(self) -> None:
        """Self-transitions are not valid for any state."""
        for state in EnumRegistrationState:
            assert not state.can_transition_to(state)


class TestGetDescription:
    """Tests for EnumRegistrationState.get_description() method."""

    def test_pending_registration_description(self) -> None:
        """PENDING_REGISTRATION has appropriate description."""
        desc = EnumRegistrationState.get_description(
            EnumRegistrationState.PENDING_REGISTRATION
        )
        assert "initial" in desc.lower() or "registration" in desc.lower()

    def test_accepted_description(self) -> None:
        """ACCEPTED has appropriate description."""
        desc = EnumRegistrationState.get_description(EnumRegistrationState.ACCEPTED)
        assert "accepted" in desc.lower() or "acknowledgment" in desc.lower()

    def test_awaiting_ack_description(self) -> None:
        """AWAITING_ACK has appropriate description."""
        desc = EnumRegistrationState.get_description(EnumRegistrationState.AWAITING_ACK)
        assert "waiting" in desc.lower() or "acknowledge" in desc.lower()

    def test_rejected_description(self) -> None:
        """REJECTED has appropriate description."""
        desc = EnumRegistrationState.get_description(EnumRegistrationState.REJECTED)
        assert "rejected" in desc.lower()

    def test_ack_timed_out_description(self) -> None:
        """ACK_TIMED_OUT has appropriate description."""
        desc = EnumRegistrationState.get_description(
            EnumRegistrationState.ACK_TIMED_OUT
        )
        assert "deadline" in desc.lower() or "timeout" in desc.lower()

    def test_ack_received_description(self) -> None:
        """ACK_RECEIVED has appropriate description."""
        desc = EnumRegistrationState.get_description(EnumRegistrationState.ACK_RECEIVED)
        assert "acknowledged" in desc.lower() or "received" in desc.lower()

    def test_active_description(self) -> None:
        """ACTIVE has appropriate description."""
        desc = EnumRegistrationState.get_description(EnumRegistrationState.ACTIVE)
        assert "active" in desc.lower() or "healthy" in desc.lower()

    def test_liveness_expired_description(self) -> None:
        """LIVENESS_EXPIRED has appropriate description."""
        desc = EnumRegistrationState.get_description(
            EnumRegistrationState.LIVENESS_EXPIRED
        )
        assert "liveness" in desc.lower() or "dead" in desc.lower()

    def test_all_states_have_descriptions(self) -> None:
        """Verify all states have non-empty descriptions."""
        for state in EnumRegistrationState:
            desc = EnumRegistrationState.get_description(state)
            assert desc
            assert len(desc) > 0


class TestWorkflowIntegration:
    """Integration tests for typical workflow state transitions."""

    def test_happy_path_workflow(self) -> None:
        """Test typical successful registration workflow."""
        # PENDING_REGISTRATION -> ACCEPTED
        assert EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            EnumRegistrationState.ACCEPTED
        )
        # ACCEPTED -> AWAITING_ACK
        assert EnumRegistrationState.ACCEPTED.can_transition_to(
            EnumRegistrationState.AWAITING_ACK
        )
        # AWAITING_ACK -> ACK_RECEIVED
        assert EnumRegistrationState.AWAITING_ACK.can_transition_to(
            EnumRegistrationState.ACK_RECEIVED
        )
        # ACK_RECEIVED -> ACTIVE
        assert EnumRegistrationState.ACK_RECEIVED.can_transition_to(
            EnumRegistrationState.ACTIVE
        )
        # ACTIVE -> LIVENESS_EXPIRED (end of lifecycle)
        assert EnumRegistrationState.ACTIVE.can_transition_to(
            EnumRegistrationState.LIVENESS_EXPIRED
        )

    def test_rejection_workflow(self) -> None:
        """Test registration rejection workflow."""
        # PENDING_REGISTRATION -> REJECTED
        assert EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            EnumRegistrationState.REJECTED
        )
        # REJECTED is terminal
        assert EnumRegistrationState.REJECTED.is_terminal()

    def test_retry_workflow(self) -> None:
        """Test ack timeout and retry workflow."""
        # AWAITING_ACK -> ACK_TIMED_OUT
        assert EnumRegistrationState.AWAITING_ACK.can_transition_to(
            EnumRegistrationState.ACK_TIMED_OUT
        )
        # ACK_TIMED_OUT can retry
        assert EnumRegistrationState.ACK_TIMED_OUT.can_retry()
        # ACK_TIMED_OUT -> PENDING_REGISTRATION (retry)
        assert EnumRegistrationState.ACK_TIMED_OUT.can_transition_to(
            EnumRegistrationState.PENDING_REGISTRATION
        )

    def test_ack_monitoring_states(self) -> None:
        """Verify states that require ack monitoring have valid timeout transitions."""
        for state in EnumRegistrationState:
            if state.requires_ack():
                # States requiring ack should transition through the workflow
                assert state in {
                    EnumRegistrationState.ACCEPTED,
                    EnumRegistrationState.AWAITING_ACK,
                }

    def test_liveness_monitoring_state(self) -> None:
        """Verify active state requiring liveness has liveness_expired transition."""
        assert EnumRegistrationState.ACTIVE.requires_liveness()
        assert EnumRegistrationState.ACTIVE.can_transition_to(
            EnumRegistrationState.LIVENESS_EXPIRED
        )
