# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Registration State Enumeration.

Defines FSM state values for the ONEX two-way node registration workflow.
Used to track the registration lifecycle from initial introspection through
active operation and eventual expiry.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from enum import StrEnum, unique


@unique
class EnumRegistrationState(StrEnum):
    """
    FSM state values for the node registration workflow.

    Represents the operational state of a node's registration at any moment.
    The registration transitions through these states during its lifecycle:

    Typical Flow:
        PENDING_REGISTRATION -> ACCEPTED -> AWAITING_ACK -> ACK_RECEIVED -> ACTIVE
        ACTIVE -> LIVENESS_EXPIRED (terminal)

    Alternative Flows:
        PENDING_REGISTRATION -> REJECTED (terminal)
        AWAITING_ACK -> ACK_TIMED_OUT -> PENDING_REGISTRATION (retry)

    Values:
        PENDING_REGISTRATION: Initial state after NodeRegistrationInitiated
        ACCEPTED: Registration accepted by orchestrator
        AWAITING_ACK: Waiting for node to acknowledge registration
        REJECTED: Registration rejected by orchestrator (terminal)
        ACK_TIMED_OUT: Ack deadline passed (retriable)
        ACK_RECEIVED: Node acknowledged registration
        ACTIVE: Node is fully active and healthy
        LIVENESS_EXPIRED: Liveness check failed (terminal)

    Example:
        >>> state = EnumRegistrationState.ACTIVE
        >>> state.is_active()
        True
        >>> EnumRegistrationState.REJECTED.is_terminal()
        True
        >>> str(EnumRegistrationState.PENDING_REGISTRATION)
        'pending_registration'
    """

    PENDING_REGISTRATION = "pending_registration"
    """Initial state after NodeRegistrationInitiated event."""

    ACCEPTED = "accepted"
    """Registration accepted by orchestrator, awaiting node acknowledgment."""

    AWAITING_ACK = "awaiting_ack"
    """Waiting for node to acknowledge its registration."""

    REJECTED = "rejected"
    """Registration rejected by orchestrator (terminal state)."""

    ACK_TIMED_OUT = "ack_timed_out"
    """Acknowledgment deadline passed without response (retriable)."""

    ACK_RECEIVED = "ack_received"
    """Node acknowledged its registration, transitioning to active."""

    ACTIVE = "active"
    """Node is fully active and healthy."""

    LIVENESS_EXPIRED = "liveness_expired"
    """Liveness check failed - node is considered dead (terminal)."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def is_terminal(self) -> bool:
        """
        Check if this is a terminal state (no further transitions expected).

        Terminal states represent registration endpoints where the workflow
        cannot continue without external intervention (re-registration).

        Returns:
            True if the state is terminal, False otherwise

        Example:
            >>> EnumRegistrationState.REJECTED.is_terminal()
            True
            >>> EnumRegistrationState.ACTIVE.is_terminal()
            False
        """
        return self in {
            EnumRegistrationState.REJECTED,
            EnumRegistrationState.LIVENESS_EXPIRED,
        }

    def is_active(self) -> bool:
        """
        Check if node is considered operationally active.

        Active state means the node is fully registered and healthy,
        ready to receive work assignments.

        Returns:
            True if the node is active, False otherwise

        Example:
            >>> EnumRegistrationState.ACTIVE.is_active()
            True
            >>> EnumRegistrationState.AWAITING_ACK.is_active()
            False
        """
        return self == EnumRegistrationState.ACTIVE

    def requires_ack(self) -> bool:
        """
        Check if state is waiting for node acknowledgment.

        These states have an ack_deadline that must be monitored
        for timeout handling (per C2 durable timeout).

        Returns:
            True if waiting for ack, False otherwise

        Example:
            >>> EnumRegistrationState.AWAITING_ACK.requires_ack()
            True
            >>> EnumRegistrationState.ACTIVE.requires_ack()
            False
        """
        return self in {
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
        }

    def requires_liveness(self) -> bool:
        """
        Check if state requires liveness monitoring.

        Active nodes must maintain liveness through heartbeats.
        The liveness_deadline must be monitored for expiry.

        Returns:
            True if liveness monitoring required, False otherwise

        Example:
            >>> EnumRegistrationState.ACTIVE.requires_liveness()
            True
            >>> EnumRegistrationState.PENDING_REGISTRATION.requires_liveness()
            False
        """
        return self == EnumRegistrationState.ACTIVE

    def can_retry(self) -> bool:
        """
        Check if this state allows retry of registration.

        Some non-terminal states allow the registration process
        to be retried by transitioning back to PENDING_REGISTRATION.

        Returns:
            True if retry is allowed, False otherwise

        Example:
            >>> EnumRegistrationState.ACK_TIMED_OUT.can_retry()
            True
            >>> EnumRegistrationState.REJECTED.can_retry()
            False
        """
        return self == EnumRegistrationState.ACK_TIMED_OUT

    def can_transition_to(self, target: "EnumRegistrationState") -> bool:
        """
        Check if transition to target state is valid per FSM contract.

        Validates that the requested state transition follows the
        defined workflow rules.

        Args:
            target: The target state to transition to

        Returns:
            True if the transition is valid, False otherwise

        Example:
            >>> EnumRegistrationState.PENDING_REGISTRATION.can_transition_to(
            ...     EnumRegistrationState.ACCEPTED
            ... )
            True
            >>> EnumRegistrationState.ACTIVE.can_transition_to(
            ...     EnumRegistrationState.PENDING_REGISTRATION
            ... )
            False
        """
        valid_transitions: dict[EnumRegistrationState, set[EnumRegistrationState]] = {
            EnumRegistrationState.PENDING_REGISTRATION: {
                EnumRegistrationState.ACCEPTED,
                EnumRegistrationState.REJECTED,
            },
            EnumRegistrationState.ACCEPTED: {
                EnumRegistrationState.AWAITING_ACK,
            },
            EnumRegistrationState.AWAITING_ACK: {
                EnumRegistrationState.ACK_RECEIVED,
                EnumRegistrationState.ACK_TIMED_OUT,
            },
            EnumRegistrationState.ACK_RECEIVED: {
                EnumRegistrationState.ACTIVE,
            },
            EnumRegistrationState.ACTIVE: {
                EnumRegistrationState.LIVENESS_EXPIRED,
            },
            EnumRegistrationState.ACK_TIMED_OUT: {
                EnumRegistrationState.PENDING_REGISTRATION,  # Retry
            },
            # Terminal states have no valid transitions
            EnumRegistrationState.REJECTED: set(),
            EnumRegistrationState.LIVENESS_EXPIRED: set(),
        }
        return target in valid_transitions.get(self, set())

    @classmethod
    def get_description(cls, state: "EnumRegistrationState") -> str:
        """
        Get a human-readable description of the registration state.

        Args:
            state: The registration state to describe

        Returns:
            A human-readable description of the state

        Example:
            >>> EnumRegistrationState.get_description(EnumRegistrationState.ACTIVE)
            'Node is fully active and healthy'
        """
        descriptions = {
            cls.PENDING_REGISTRATION: "Initial state after registration initiated",
            cls.ACCEPTED: "Registration accepted, awaiting node acknowledgment",
            cls.AWAITING_ACK: "Waiting for node to acknowledge registration",
            cls.REJECTED: "Registration rejected by orchestrator",
            cls.ACK_TIMED_OUT: "Acknowledgment deadline passed without response",
            cls.ACK_RECEIVED: "Node acknowledged registration",
            cls.ACTIVE: "Node is fully active and healthy",
            cls.LIVENESS_EXPIRED: "Liveness check failed - node is considered dead",
        }
        return descriptions.get(state, "Unknown registration state")


__all__: list[str] = ["EnumRegistrationState"]
