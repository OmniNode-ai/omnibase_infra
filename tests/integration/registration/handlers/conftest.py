# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for registration handler integration tests.

This module provides fixtures for testing registration handlers against real
PostgreSQL using testcontainers. Reuses projector fixtures and adds handler-
specific fixtures for heartbeat processing tests.

Fixture Hierarchy:
    Session-scoped:
        - docker_available (from projectors)
        - postgres_container (from projectors)

    Function-scoped:
        - pg_pool (from projectors)
        - projector (from projectors)
        - reader (from projectors)
        - heartbeat_handler: HandlerNodeHeartbeat instance

Usage:
    The fixtures handle:
    1. Container lifecycle management (via projector fixtures)
    2. Handler initialization with projector and reader
    3. Test isolation through schema reset
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
)

# =============================================================================
# Cross-Module Fixture Imports
# =============================================================================
# These fixtures are imported from tests/integration/projectors/conftest.py
# to provide shared PostgreSQL testcontainer infrastructure for handler tests.
#
# Why imported:
#   - Reuses expensive PostgreSQL container setup (session-scoped)
#   - Ensures consistent schema initialization across projector and handler tests
#   - Provides test isolation via TRUNCATE in pg_pool fixture teardown
#
# Imported fixtures:
#   - DOCKER_AVAILABLE: Module constant - True if Docker is running
#   - SCHEMA_FILE: Path to registration_projections schema SQL
#   - docker_available: Session fixture - Docker availability check
#   - event_loop_policy: Session fixture - asyncio event loop policy
#   - postgres_container: Session fixture - PostgreSQL testcontainer (expensive)
#   - pg_pool: Function fixture - Fresh asyncpg pool per test (isolated)
#   - projector: Function fixture - ProjectorRegistration instance
#   - reader: Function fixture - ProjectionReaderRegistration instance
#
# These are re-exported in __all__ for pytest discovery.
# =============================================================================
from tests.integration.projectors.conftest import (
    DOCKER_AVAILABLE,
    SCHEMA_FILE,
    docker_available,
    event_loop_policy,
    pg_pool,
    postgres_container,
    projector,
    reader,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )

# Re-export fixtures for pytest discovery
__all__ = [
    "DOCKER_AVAILABLE",
    "SCHEMA_FILE",
    "docker_available",
    "event_loop_policy",
    "pg_pool",
    "postgres_container",
    "projector",
    "reader",
    "heartbeat_handler",
    "heartbeat_handler_fast_window",
]


@pytest.fixture
def heartbeat_handler(
    reader: ProjectionReaderRegistration,
    projector: ProjectorRegistration,
) -> HandlerNodeHeartbeat:
    """Function-scoped HandlerNodeHeartbeat instance.

    Creates a handler with the default liveness window (DEFAULT_LIVENESS_WINDOW_SECONDS).
    Suitable for most integration tests.

    Args:
        reader: ProjectionReaderRegistration fixture for state lookups.
        projector: ProjectorRegistration fixture for state updates.

    Returns:
        HandlerNodeHeartbeat configured with default liveness window.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )

    return HandlerNodeHeartbeat(
        projection_reader=reader,
        projector=projector,
        liveness_window_seconds=DEFAULT_LIVENESS_WINDOW_SECONDS,
    )


@pytest.fixture
def heartbeat_handler_fast_window(
    reader: ProjectionReaderRegistration,
    projector: ProjectorRegistration,
) -> HandlerNodeHeartbeat:
    """Handler with short liveness window for testing deadline extension.

    Uses a 5-second liveness window to make deadline calculations easier
    to verify in tests without waiting for long timeouts.

    Args:
        reader: ProjectionReaderRegistration fixture.
        projector: ProjectorRegistration fixture.

    Returns:
        HandlerNodeHeartbeat configured with 5-second liveness window.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )

    return HandlerNodeHeartbeat(
        projection_reader=reader,
        projector=projector,
        liveness_window_seconds=5.0,
    )
