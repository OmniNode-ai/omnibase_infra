# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Replay-specific test fixtures for OMN-955.

This module provides fixtures for event replay verification testing, including:
- Deterministic ID and timestamp generators
- Event sequence builders
- State factory functions
- Replay orchestration helpers

The fixtures ensure reproducible test behavior by using deterministic
generators instead of random UUIDs and real timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

import pytest

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

# =============================================================================
# Type Definitions
# =============================================================================

NodeType = Literal["effect", "compute", "reducer", "orchestrator"]


# =============================================================================
# Event Sequence Models
# =============================================================================


@dataclass(frozen=True)
class EventSequenceEntry:
    """A single entry in an event sequence log.

    Captures an event and its expected outcome for replay verification.

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

    Attributes:
        entries: List of sequence entries in processing order.
        initial_state: The state before any events were processed.
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

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> EventSequenceLog:
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


# =============================================================================
# Event Factory
# =============================================================================


@dataclass
class EventFactory:
    """Factory for creating deterministic introspection events.

    Uses deterministic generators for reproducible test data.

    Attributes:
        id_gen: Deterministic UUID generator.
        clock: Deterministic timestamp generator.
    """

    id_gen: DeterministicIdGenerator = field(
        default_factory=lambda: DeterministicIdGenerator(seed=100)
    )
    clock: DeterministicClock = field(default_factory=DeterministicClock)

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
            node_type: ONEX node type.
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
        """Reset the factory's generators to initial state."""
        self.id_gen.reset(seed=100)
        self.clock.reset()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reducer() -> RegistrationReducer:
    """Create a RegistrationReducer instance for replay testing.

    Returns:
        A new RegistrationReducer instance.
    """
    return RegistrationReducer()


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle state for replay testing.

    Returns:
        A new ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


@pytest.fixture
def id_generator() -> DeterministicIdGenerator:
    """Create a deterministic ID generator.

    Returns:
        A DeterministicIdGenerator with seed=100.
    """
    return DeterministicIdGenerator(seed=100)


@pytest.fixture
def clock() -> DeterministicClock:
    """Create a deterministic clock.

    Returns:
        A DeterministicClock starting at 2024-01-01 00:00:00 UTC.
    """
    return DeterministicClock()


@pytest.fixture
def event_factory() -> EventFactory:
    """Create an event factory for deterministic event creation.

    Returns:
        An EventFactory with deterministic generators.
    """
    return EventFactory()


@pytest.fixture
def event_sequence_log() -> EventSequenceLog:
    """Create an empty event sequence log.

    Returns:
        An empty EventSequenceLog.
    """
    return EventSequenceLog()


@pytest.fixture
def fixed_node_id() -> UUID:
    """Provide a fixed node ID for deterministic testing.

    Returns:
        A fixed UUID for node identification.
    """
    return UUID("12345678-1234-1234-1234-123456789abc")


@pytest.fixture
def fixed_correlation_id() -> UUID:
    """Provide a fixed correlation ID for deterministic testing.

    Returns:
        A fixed UUID for correlation tracking.
    """
    return UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")


@pytest.fixture
def fixed_timestamp() -> datetime:
    """Provide a fixed timestamp for deterministic testing.

    Returns:
        A fixed datetime (2025-01-01 12:00:00 UTC).
    """
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def complete_registration_sequence(
    event_factory: EventFactory,
) -> list[ModelNodeIntrospectionEvent]:
    """Create a complete registration workflow event sequence.

    This sequence represents a typical registration workflow:
    1. Initial introspection event
    2. Follow-up events for other nodes

    Args:
        event_factory: Factory for creating events.

    Returns:
        List of events representing a complete registration workflow.
    """
    return event_factory.create_event_sequence(count=5, time_between_events=60)


@pytest.fixture
def multi_node_type_sequence(
    event_factory: EventFactory,
) -> list[tuple[NodeType, ModelNodeIntrospectionEvent]]:
    """Create events for all node types.

    Args:
        event_factory: Factory for creating events.

    Returns:
        List of (node_type, event) tuples for all four node types.
    """
    node_types: list[NodeType] = ["effect", "compute", "reducer", "orchestrator"]
    result: list[tuple[NodeType, ModelNodeIntrospectionEvent]] = []

    for node_type in node_types:
        event = event_factory.create_event(
            node_type=node_type,
            advance_time_seconds=30,
        )
        result.append((node_type, event))

    return result
