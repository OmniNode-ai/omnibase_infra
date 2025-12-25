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

Note:
    The core models (EventFactory, EventSequenceLog, EventSequenceEntry) are
    defined in tests.helpers.replay_utils and re-exported here for backwards
    compatibility with test imports from conftest.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

# Import and re-export models from helpers for backwards compatibility
from tests.helpers.replay_utils import (
    EventFactory,
    EventSequenceEntry,
    EventSequenceLog,
    NodeType,
)

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
