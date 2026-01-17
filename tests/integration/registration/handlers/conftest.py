# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for registration handler integration tests.

This module provides fixtures for testing registration handlers against real
PostgreSQL using testcontainers. Reuses projector fixtures and adds handler-
specific fixtures for heartbeat processing tests.

IMPORTANT: Event Loop Scope Configuration (pytest-asyncio 0.25+)
================================================================

This module imports session-scoped async fixtures from ``tests/integration/projectors/conftest.py``.
When using these fixtures, ensure your test module has proper loop scope configuration:

.. code-block:: python

    pytestmark = [pytest.mark.asyncio(loop_scope="session")]

Without this, you may encounter RuntimeError: "Task got Future attached to a different loop"
due to pytest-asyncio 0.25+ defaulting to function-scoped event loops.

See ``tests/integration/projectors/conftest.py`` for detailed documentation.

Fixture Hierarchy:
    Session-scoped:
        - docker_available (from projectors)
        - postgres_container (from projectors)
        - event_loop_policy (from projectors)

    Function-scoped:
        - pg_pool (from projectors)
        - projector (from projectors) - ProjectorShell instance
        - reader (from projectors)
        - heartbeat_handler: HandlerNodeHeartbeat instance

Usage:
    The fixtures handle:
    1. Container lifecycle management (via projector fixtures)
    2. Handler initialization with projector and reader
    3. Test isolation through schema reset

Related Tickets:
    - OMN-1169: ProjectorShell for contract-driven projections
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
#   - projector: Function fixture - ProjectorShell instance (contract-driven)
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
    # TYPE_CHECKING imports: These imports are only used for type annotations.
    # They are NOT imported at runtime, which:
    # 1. Avoids circular import issues (handler modules may reference test utilities)
    # 2. Allows fixtures to declare return types without importing at collection time
    # 3. Enables IDE autocompletion for fixture parameters and return types
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell

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
    projector: ProjectorShell,
) -> HandlerNodeHeartbeat:
    """Function-scoped HandlerNodeHeartbeat instance.

    Creates a handler with the default liveness window (DEFAULT_LIVENESS_WINDOW_SECONDS).
    Suitable for most integration tests.

    Args:
        reader: ProjectionReaderRegistration fixture for state lookups.
        projector: ProjectorShell fixture for state updates (contract-driven).

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
    projector: ProjectorShell,
) -> HandlerNodeHeartbeat:
    """Handler with short liveness window for testing deadline extension.

    Uses a 5-second liveness window to make deadline calculations easier
    to verify in tests without waiting for long timeouts.

    Args:
        reader: ProjectionReaderRegistration fixture.
        projector: ProjectorShell fixture (contract-driven).

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
