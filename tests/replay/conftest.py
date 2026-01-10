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

# =============================================================================
# Cross-Module Fixture Imports
# =============================================================================
# These imports bring in shared test utilities from tests/helpers/:
#
# From tests/helpers/deterministic.py:
#   - DeterministicClock: Controllable clock for reproducible timestamps
#     Starts at 2025-01-01 00:00:00 UTC by default.
#   - DeterministicIdGenerator: Deterministic UUID generator for reproducible IDs
#     Uses a counter-based approach for predictable UUID values.
#
# From tests/helpers/replay_utils.py:
#   - EventFactory: Factory for creating deterministic introspection events
#     Combines DeterministicClock and DeterministicIdGenerator for reproducible
#     event sequences suitable for snapshot testing and replay verification.
#   - EventSequenceLog: Tracks event processing order and results
#     Used for verifying replay behavior matches original execution.
#   - NodeType: Type alias for ONEX node types ("effect", "compute", etc.)
#
# These utilities ensure replay tests produce deterministic, reproducible results.
# =============================================================================
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator
from tests.helpers.replay_utils import (
    EventFactory,
    EventSequenceLog,
    NodeType,
)

# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Dynamically add replay marker to all tests in the replay directory.

    This hook runs after test collection and adds the 'replay' marker to any
    test whose file path contains 'tests/replay'. This is necessary because
    pytestmark defined in conftest.py does NOT automatically apply to tests
    in other files within the same directory.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    Usage:
        Run only replay tests: pytest -m replay
        Exclude replay tests: pytest -m "not replay"
    """
    replay_marker = pytest.mark.replay

    for item in items:
        # Check if the test file is in the replay directory
        if "tests/replay" in str(item.fspath):
            # Only add marker if not already present
            if not any(marker.name == "replay" for marker in item.iter_markers()):
                item.add_marker(replay_marker)


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
