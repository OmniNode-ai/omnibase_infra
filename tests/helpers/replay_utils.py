# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Replay testing utilities for OMN-955.

This module provides shared utilities for replay testing scenarios including:
- Output comparison helpers for determinism testing
- Event sequence log models for replay verification
- Event factory for deterministic event creation
- Ordering violation detection utilities

These utilities are extracted from replay tests to reduce duplication and
provide a consistent interface for replay testing across the test suite.

Example usage:
    >>> from tests.helpers.replay_utils import compare_outputs, EventFactory
    >>>
    >>> # Compare two reducer outputs
    >>> are_equal, differences = compare_outputs(output1, output2)
    >>>
    >>> # Create deterministic events
    >>> factory = EventFactory()
    >>> events = factory.create_event_sequence(count=5)

Related Tickets:
    - OMN-955: Event Replay Verification
    - OMN-914: Reducer Purity Enforcement Gates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_core.nodes import ModelReducerOutput


# =============================================================================
# Type Definitions
# =============================================================================

NodeType = Literal["effect", "compute", "reducer", "orchestrator"]


# =============================================================================
# Output Comparison Helpers
# =============================================================================


def compare_outputs(
    output1: ModelReducerOutput,
    output2: ModelReducerOutput,
) -> tuple[bool, list[str]]:
    """Compare two reducer outputs for equality.

    Performs a deep comparison of reducer outputs to verify determinism.
    Compares result state, intent count, intent types, and targets.

    Args:
        output1: First output to compare.
        output2: Second output to compare.

    Returns:
        Tuple of (are_equal, list of differences). If are_equal is True,
        the differences list will be empty.

    Example:
        >>> output1 = reducer.reduce(state, event)
        >>> output2 = reducer.reduce(state, event)
        >>> are_equal, differences = compare_outputs(output1, output2)
        >>> assert are_equal, f"Outputs differ: {differences}"
    """
    differences: list[str] = []

    # Compare result state
    if output1.result.status != output2.result.status:
        differences.append(
            f"Status mismatch: {output1.result.status} != {output2.result.status}"
        )

    if output1.result.node_id != output2.result.node_id:
        differences.append(
            f"Node ID mismatch: {output1.result.node_id} != {output2.result.node_id}"
        )

    if output1.result.consul_confirmed != output2.result.consul_confirmed:
        differences.append(
            f"Consul confirmed mismatch: "
            f"{output1.result.consul_confirmed} != {output2.result.consul_confirmed}"
        )

    if output1.result.postgres_confirmed != output2.result.postgres_confirmed:
        differences.append(
            f"Postgres confirmed mismatch: "
            f"{output1.result.postgres_confirmed} != {output2.result.postgres_confirmed}"
        )

    # Compare intents
    if len(output1.intents) != len(output2.intents):
        differences.append(
            f"Intent count mismatch: {len(output1.intents)} != {len(output2.intents)}"
        )
    else:
        for i, (intent1, intent2) in enumerate(
            zip(output1.intents, output2.intents, strict=True)
        ):
            if intent1.intent_type != intent2.intent_type:
                differences.append(
                    f"Intent {i} type mismatch: "
                    f"{intent1.intent_type} != {intent2.intent_type}"
                )
            if intent1.target != intent2.target:
                differences.append(
                    f"Intent {i} target mismatch: {intent1.target} != {intent2.target}"
                )
            if intent1.payload.get("correlation_id") != intent2.payload.get(
                "correlation_id"
            ):
                differences.append(f"Intent {i} correlation_id mismatch")

    return len(differences) == 0, differences


# =============================================================================
# Ordering Violation Detection
# =============================================================================


@dataclass
class OrderingViolation:
    """An ordering violation in an event sequence.

    Capture and report violations in event ordering, such as out-of-order
    timestamps or sequence number gaps.

    Attributes:
        position: Index in the sequence where violation occurred.
        event_timestamp: Timestamp of the violating event.
        previous_timestamp: Timestamp of the previous event.
        violation_type: Type of ordering violation (e.g., "timestamp_reorder",
            "timestamp_duplicate", "sequence_mismatch").
    """

    position: int
    event_timestamp: datetime
    previous_timestamp: datetime
    violation_type: str


def detect_timestamp_order_violations(
    events: list[ModelNodeIntrospectionEvent],
) -> list[OrderingViolation]:
    """Detect timestamp ordering violations in an event sequence.

    Checks for events that arrive with timestamps earlier than or equal
    to their predecessors, which indicates ordering issues.

    Args:
        events: List of events to check.

    Returns:
        List of OrderingViolation instances for each violation found.
        Empty list if events are in proper chronological order.

    Example:
        >>> events = factory.create_event_sequence(count=5)
        >>> violations = detect_timestamp_order_violations(events)
        >>> assert len(violations) == 0, "Events should be in order"
    """
    violations: list[OrderingViolation] = []

    for i in range(1, len(events)):
        current = events[i]
        previous = events[i - 1]

        if current.timestamp < previous.timestamp:
            violations.append(
                OrderingViolation(
                    position=i,
                    event_timestamp=current.timestamp,
                    previous_timestamp=previous.timestamp,
                    violation_type="timestamp_reorder",
                )
            )
        elif current.timestamp == previous.timestamp:
            violations.append(
                OrderingViolation(
                    position=i,
                    event_timestamp=current.timestamp,
                    previous_timestamp=previous.timestamp,
                    violation_type="timestamp_duplicate",
                )
            )

    return violations


# =============================================================================
# Event Sequence Models
# =============================================================================


@dataclass(frozen=True)
class EventSequenceEntry:
    """A single entry in an event sequence log.

    Captures an event and its expected outcome for replay verification.
    Frozen to ensure immutability of recorded entries.

    Attributes:
        event: The introspection event.
        expected_status: Expected state status after processing.
        expected_intent_count: Expected number of intents emitted.
        sequence_number: Position in the sequence (1-indexed).
    """

    event: ModelNodeIntrospectionEvent
    expected_status: str
    expected_intent_count: int
    sequence_number: int


@dataclass
class EventSequenceLog:
    """A log of events for replay testing.

    Provides methods for capturing, serializing, and replaying event sequences.
    Supports serialization to/from dictionary format for storage.

    Attributes:
        entries: List of sequence entries in processing order.
        initial_state: The state before any events were processed.

    Example:
        >>> log = EventSequenceLog()
        >>> log.append(event, expected_status="pending", expected_intent_count=2)
        >>> data = log.to_dict()
        >>> restored = EventSequenceLog.from_dict(data)
    """

    entries: list[EventSequenceEntry] = field(default_factory=list)
    initial_state: ModelRegistrationState = field(
        default_factory=ModelRegistrationState
    )

    def append(
        self,
        event: ModelNodeIntrospectionEvent,
        expected_status: str,
        expected_intent_count: int,
    ) -> None:
        """Append an event to the sequence log.

        Automatically assigns the next sequence number.

        Args:
            event: The introspection event to append.
            expected_status: Expected state status after processing.
            expected_intent_count: Expected number of intents emitted.
        """
        entry = EventSequenceEntry(
            event=event,
            expected_status=expected_status,
            expected_intent_count=expected_intent_count,
            sequence_number=len(self.entries) + 1,
        )
        self.entries.append(entry)

    def to_dict(self) -> dict[str, object]:
        """Serialize the log to a dictionary for storage/transport.

        Returns:
            Dictionary representation of the event sequence log.
        """
        return {
            "initial_state": self.initial_state.model_dump(mode="json"),
            "entries": [
                {
                    "event": entry.event.model_dump(mode="json"),
                    "expected_status": entry.expected_status,
                    "expected_intent_count": entry.expected_intent_count,
                    "sequence_number": entry.sequence_number,
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EventSequenceLog:
        """Deserialize a log from a dictionary.

        Args:
            data: Dictionary representation of the event sequence log.

        Returns:
            Reconstructed EventSequenceLog instance.
        """
        initial_state = ModelRegistrationState.model_validate(data["initial_state"])
        log = cls(initial_state=initial_state)

        for entry_data in data["entries"]:
            event = ModelNodeIntrospectionEvent.model_validate(entry_data["event"])
            log.append(
                event=event,
                expected_status=entry_data["expected_status"],
                expected_intent_count=entry_data["expected_intent_count"],
            )

        return log

    def __len__(self) -> int:
        """Return the number of entries in the log."""
        return len(self.entries)


def detect_sequence_number_violations(
    log: EventSequenceLog,
) -> list[OrderingViolation]:
    """Detect sequence number ordering violations in an event log.

    Checks that sequence numbers in the log are consecutive starting from 1.
    Any gaps or out-of-order numbers are reported as violations.

    Args:
        log: Event sequence log to check.

    Returns:
        List of OrderingViolation instances for each violation found.
        Empty list if sequence numbers are valid.
    """
    violations: list[OrderingViolation] = []

    for i, entry in enumerate(log.entries):
        expected_seq = i + 1
        if entry.sequence_number != expected_seq:
            violations.append(
                OrderingViolation(
                    position=i,
                    event_timestamp=entry.event.timestamp,
                    previous_timestamp=(
                        log.entries[i - 1].event.timestamp
                        if i > 0
                        else entry.event.timestamp
                    ),
                    violation_type=f"sequence_mismatch (expected {expected_seq}, got {entry.sequence_number})",
                )
            )

    return violations


# =============================================================================
# Event Factory
# =============================================================================


@dataclass
class EventFactory:
    """Factory for creating deterministic introspection events.

    Uses deterministic generators for reproducible test data.
    Provides methods for creating single events or sequences of events.

    Attributes:
        seed: Seed for deterministic UUID generation (default: 100).
        id_gen: Deterministic UUID generator (initialized from seed).
        clock: Deterministic timestamp generator.

    Example:
        >>> factory = EventFactory()  # Uses default seed=100
        >>> factory = EventFactory(seed=42)  # Custom seed
        >>> event = factory.create_event(node_type="effect")
        >>> events = factory.create_event_sequence(count=5)
    """

    seed: int = 100
    id_gen: DeterministicIdGenerator = field(init=False)
    clock: DeterministicClock = field(default_factory=DeterministicClock)

    def __post_init__(self) -> None:
        """Initialize id_gen with the configured seed."""
        self.id_gen = DeterministicIdGenerator(seed=self.seed)

    def create_event(
        self,
        node_type: NodeType = "effect",
        node_id: UUID | None = None,
        correlation_id: UUID | None = None,
        node_version: str = "1.0.0",
        endpoints: dict[str, str] | None = None,
        advance_time_seconds: int = 0,
    ) -> ModelNodeIntrospectionEvent:
        """Create a deterministic introspection event.

        Args:
            node_type: ONEX node type (effect, compute, reducer, orchestrator).
            node_id: Optional fixed node ID (generates if not provided).
            correlation_id: Optional fixed correlation ID (generates if not provided).
            node_version: Semantic version string.
            endpoints: Optional endpoints dict.
            advance_time_seconds: Seconds to advance clock before creating event.

        Returns:
            A deterministic ModelNodeIntrospectionEvent.
        """
        if advance_time_seconds > 0:
            self.clock.advance(advance_time_seconds)

        return ModelNodeIntrospectionEvent(
            node_id=node_id or self.id_gen.next_uuid(),
            node_type=node_type,
            node_version=node_version,
            correlation_id=correlation_id or self.id_gen.next_uuid(),
            timestamp=self.clock.now(),
            endpoints=endpoints or {},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

    def create_event_sequence(
        self,
        count: int,
        node_type: NodeType = "effect",
        time_between_events: int = 60,
    ) -> list[ModelNodeIntrospectionEvent]:
        """Create a sequence of deterministic events.

        Args:
            count: Number of events to create.
            node_type: ONEX node type for all events.
            time_between_events: Seconds between events.

        Returns:
            List of deterministic events in chronological order.
        """
        events: list[ModelNodeIntrospectionEvent] = []
        for i in range(count):
            advance = time_between_events if i > 0 else 0
            events.append(
                self.create_event(
                    node_type=node_type,
                    advance_time_seconds=advance,
                )
            )
        return events

    def reset(self) -> None:
        """Reset the factory's generators to initial state.

        Resets id_gen with the configured seed and clock to initial timestamp.
        Useful for resetting between test cases to ensure reproducibility.
        """
        self.id_gen.reset(seed=self.seed)
        self.clock.reset()


# =============================================================================
# State Helpers
# =============================================================================


def create_introspection_event(
    node_id: UUID,
    correlation_id: UUID,
    timestamp: datetime,
    node_type: str = "effect",
    node_version: str = "1.0.0",
    endpoints: dict[str, str] | None = None,
) -> ModelNodeIntrospectionEvent:
    """Create an introspection event with controlled parameters.

    Provide a convenience function for creating events with explicit parameters.
    For deterministic testing, use EventFactory instead.

    Args:
        node_id: UUID of the node being registered.
        correlation_id: Correlation ID for the event (used as event_id).
        timestamp: Event timestamp.
        node_type: Node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        endpoints: Optional endpoints dictionary.

    Returns:
        Configured ModelNodeIntrospectionEvent instance.
    """
    return ModelNodeIntrospectionEvent(
        node_id=node_id,
        node_type=node_type,
        node_version=node_version,
        correlation_id=correlation_id,
        timestamp=timestamp,
        endpoints=endpoints or {"health": "http://localhost:8080/health"},
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        metadata=ModelNodeMetadata(environment="test"),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type definitions
    "NodeType",
    # Output comparison
    "compare_outputs",
    # Ordering violation detection
    "OrderingViolation",
    "detect_timestamp_order_violations",
    "detect_sequence_number_violations",
    # Event sequence models
    "EventSequenceEntry",
    "EventSequenceLog",
    # Event factory
    "EventFactory",
    # State helpers
    "create_introspection_event",
]
