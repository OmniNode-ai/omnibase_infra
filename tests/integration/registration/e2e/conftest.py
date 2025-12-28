# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for registration E2E integration tests.

This module provides fixtures for end-to-end testing of the registration
orchestrator against real infrastructure (Kafka, Consul, PostgreSQL).

Infrastructure Requirements:
    Tests require ALL infrastructure services to be available:
    - PostgreSQL: 192.168.86.200:5436 (database: omninode_bridge)
    - Consul: 192.168.86.200:28500
    - Kafka/Redpanda: 192.168.86.200:29092

    Environment variables required:
    - POSTGRES_HOST, POSTGRES_PASSWORD (for PostgreSQL)
    - CONSUL_HOST (for Consul)
    - KAFKA_BOOTSTRAP_SERVERS (for Kafka)

CI/CD Graceful Skip Behavior:
    These tests skip gracefully when infrastructure is unavailable:
    - All tests in this directory require full infrastructure
    - Module-level pytestmark applies skipif to all tests
    - Clear skip messages indicate which infrastructure is missing

Container Wiring Pattern:
    This module uses the declarative orchestrator pattern:
    1. wire_infrastructure_services() - Register PolicyRegistry, etc.
    2. wire_registration_handlers() - Register handlers with projection reader
    3. NodeRegistrationOrchestrator - Declarative workflow orchestrator

Fixture Dependency Graph:
    postgres_pool
        -> wired_container
            -> registration_orchestrator
    real_kafka_event_bus
        -> registration_orchestrator
        -> introspectable_test_node
    real_consul_handler
        -> cleanup_consul_services

Related Tickets:
    - OMN-892: E2E Registration Tests
    - OMN-888: Registration Orchestrator
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from omnibase_core.enums.enum_node_kind import EnumNodeKind

# Load environment configuration with priority:
# 1. .env.docker in this directory (for Docker compose infrastructure)
# 2. .env in project root (for remote infrastructure)
# This allows easy switching between Docker and remote infrastructure
_e2e_dir = Path(__file__).parent
_project_root = _e2e_dir.parent.parent.parent.parent

# Check for Docker-specific env file first (for docker-compose.e2e.yml)
_docker_env_file = _e2e_dir / ".env.docker"
_project_env_file = _project_root / ".env"

if _docker_env_file.exists():
    # Docker infrastructure mode - use localhost ports
    load_dotenv(_docker_env_file, override=True)
elif _project_env_file.exists():
    # Remote infrastructure mode - use project .env
    load_dotenv(_project_env_file)

# Import infrastructure configuration
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from tests.infrastructure_config import (
    DEFAULT_CONSUL_PORT,
    DEFAULT_KAFKA_PORT,
    DEFAULT_POSTGRES_PORT,
    REMOTE_INFRA_HOST,
)

if TYPE_CHECKING:
    import asyncpg
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.handlers import ConsulHandler
    from omnibase_infra.nodes.node_registration_orchestrator import (
        NodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
    )
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )
    from omnibase_infra.services import TimeoutEmitter, TimeoutScanner


# =============================================================================
# Envelope Helper
# =============================================================================


def wrap_event_in_envelope(
    event: ModelNodeIntrospectionEvent,
) -> ModelEventEnvelope[ModelNodeIntrospectionEvent]:
    """Wrap an event in a ModelEventEnvelope for Kafka publishing.

    Events MUST be wrapped in envelopes on the wire. The envelope provides:
    - correlation_id for tracing
    - timestamp for ordering
    - metadata for extensibility

    This helper is shared across all E2E tests to ensure consistent
    envelope formatting.

    Args:
        event: The introspection event to wrap

    Returns:
        ModelEventEnvelope containing the event as payload
    """
    return ModelEventEnvelope(
        payload=event,
        correlation_id=event.correlation_id,
        envelope_timestamp=datetime.now(UTC),
    )


# =============================================================================
# Infrastructure Availability Checks
# =============================================================================

# PostgreSQL availability
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", str(DEFAULT_POSTGRES_PORT)))
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Normalize empty password to None for consistent checking
if POSTGRES_PASSWORD and not POSTGRES_PASSWORD.strip():
    POSTGRES_PASSWORD = None

POSTGRES_AVAILABLE = bool(POSTGRES_HOST and POSTGRES_PASSWORD)

# Kafka availability
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = bool(KAFKA_BOOTSTRAP_SERVERS)

# Consul availability
CONSUL_HOST = os.getenv("CONSUL_HOST")
CONSUL_PORT = int(os.getenv("CONSUL_PORT", str(DEFAULT_CONSUL_PORT)))


def _check_consul_reachable() -> bool:
    """Check if Consul server is reachable via TCP."""
    if not CONSUL_HOST:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            result = sock.connect_ex((CONSUL_HOST, CONSUL_PORT))
            return result == 0
    except (OSError, TimeoutError):
        return False


CONSUL_AVAILABLE = _check_consul_reachable()

# Combined availability check
ALL_INFRA_AVAILABLE = KAFKA_AVAILABLE and CONSUL_AVAILABLE and POSTGRES_AVAILABLE


# =============================================================================
# Module-Level Markers
# =============================================================================
# All tests in this module require full infrastructure availability.
# This pytestmark applies to all test functions in files that import this conftest.

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not ALL_INFRA_AVAILABLE,
        reason=(
            "Full infrastructure required for E2E tests. "
            f"Kafka: {'available' if KAFKA_AVAILABLE else 'MISSING (set KAFKA_BOOTSTRAP_SERVERS)'}. "
            f"Consul: {'available' if CONSUL_AVAILABLE else 'MISSING (set CONSUL_HOST or unreachable)'}. "
            f"PostgreSQL: {'available' if POSTGRES_AVAILABLE else 'MISSING (set POSTGRES_HOST and POSTGRES_PASSWORD)'}."
        ),
    ),
]


# =============================================================================
# Database Fixtures
# =============================================================================

# Path to SQL schema file for registration projections
# Path from tests/integration/registration/e2e/ -> project root -> src/...
SCHEMA_FILE = (
    Path(__file__).parent.parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "schemas"
    / "schema_registration_projection.sql"
)


def _build_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Returns:
        PostgreSQL connection string in standard format.

    Note:
        This function should only be called after verifying
        POSTGRES_PASSWORD is set.
    """
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
    )


@pytest.fixture
async def postgres_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create asyncpg connection pool for real PostgreSQL.

    This fixture creates a connection pool to the real PostgreSQL database
    on the infrastructure server and ensures the registration_projections
    schema is initialized.

    Yields:
        asyncpg.Pool: Connection pool for database operations.

    Note:
        Function scope ensures each test gets a fresh pool, avoiding
        asyncio event loop issues with module-scoped async fixtures.
    """
    import asyncpg

    if not POSTGRES_AVAILABLE:
        pytest.skip(
            "PostgreSQL not available (POSTGRES_HOST or POSTGRES_PASSWORD not set)"
        )

    dsn = _build_postgres_dsn()
    pool = await asyncpg.create_pool(
        dsn,
        min_size=2,
        max_size=10,
        command_timeout=60.0,
    )

    # Ensure registration_projections schema exists
    # The schema SQL is idempotent (uses IF NOT EXISTS throughout)
    if SCHEMA_FILE.exists():
        schema_sql = SCHEMA_FILE.read_text()
        async with pool.acquire() as conn:
            await conn.execute(schema_sql)

    yield pool

    await pool.close()


# =============================================================================
# Container Wiring Fixtures
# =============================================================================


@pytest.fixture
async def wired_container(
    postgres_pool: asyncpg.Pool,
) -> AsyncGenerator[ModelONEXContainer, None]:
    """Container with infrastructure services and registration handlers wired.

    This fixture creates a fully wired ModelONEXContainer with:
    1. Infrastructure services (PolicyRegistry, ProtocolBindingRegistry, RegistryCompute)
    2. Registration handlers (HandlerNodeIntrospected, HandlerRuntimeTick, etc.)
    3. ProjectionReaderRegistration for state queries

    Args:
        postgres_pool: Database connection pool.

    Yields:
        ModelONEXContainer: Fully wired container for dependency injection.
    """
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.container_wiring import (
        wire_infrastructure_services,
        wire_registration_handlers,
    )

    container = ModelONEXContainer()

    # Wire infrastructure services
    await wire_infrastructure_services(container)

    # Wire registration handlers with database pool
    await wire_registration_handlers(container, postgres_pool)

    # Yield container for proper fixture teardown semantics.
    # ModelONEXContainer doesn't have explicit cleanup methods currently,
    # but using yield allows for future cleanup needs and ensures proper
    # pytest async fixture lifecycle management.
    return container


@pytest.fixture
async def projection_reader(
    wired_container: ModelONEXContainer,
) -> ProjectionReaderRegistration:
    """Get ProjectionReaderRegistration from wired container.

    Args:
        wired_container: Container with handlers wired.

    Returns:
        ProjectionReaderRegistration for state queries.
    """
    from omnibase_infra.runtime.container_wiring import (
        get_projection_reader_from_container,
    )

    return await get_projection_reader_from_container(wired_container)


@pytest.fixture
async def handler_node_introspected(
    wired_container: ModelONEXContainer,
) -> HandlerNodeIntrospected:
    """Get HandlerNodeIntrospected from wired container.

    Args:
        wired_container: Container with handlers wired.

    Returns:
        HandlerNodeIntrospected for processing introspection events.
    """
    from omnibase_infra.runtime.container_wiring import (
        get_handler_node_introspected_from_container,
    )

    return await get_handler_node_introspected_from_container(wired_container)


# =============================================================================
# Kafka Event Bus Fixtures
# =============================================================================


@pytest.fixture
async def real_kafka_event_bus() -> AsyncGenerator[KafkaEventBus, None]:
    """Connected KafkaEventBus with proper cleanup.

    This fixture creates a real KafkaEventBus connected to the
    infrastructure server's Kafka/Redpanda cluster.

    Yields:
        KafkaEventBus: Started event bus ready for publish/subscribe.

    Note:
        The event bus is stopped and cleaned up after each test.
    """
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

    if not KAFKA_AVAILABLE:
        pytest.skip("Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)")

    bus = KafkaEventBus(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        environment="e2e-test",
        group="registration-e2e",
        timeout_seconds=30,
        max_retry_attempts=3,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=60.0,
    )

    await bus.start()

    yield bus

    await bus.close()


# =============================================================================
# Consul Handler Fixtures
# =============================================================================


@pytest.fixture
async def real_consul_handler() -> AsyncGenerator[ConsulHandler, None]:
    """Connected ConsulHandler with cleanup.

    This fixture creates a ConsulHandler connected to the real Consul
    server on the infrastructure server.

    Yields:
        ConsulHandler: Initialized handler ready for operations.

    Note:
        The handler is shut down after each test.
    """
    from omnibase_infra.handlers import ConsulHandler

    if not CONSUL_AVAILABLE:
        pytest.skip("Consul not available (CONSUL_HOST not set or unreachable)")

    handler = ConsulHandler()
    await handler.initialize(
        {
            "host": CONSUL_HOST,
            "port": CONSUL_PORT,
            "scheme": "http",
            "timeout_seconds": 30.0,
            "max_concurrent_operations": 5,
            "circuit_breaker_enabled": True,
            "circuit_breaker_failure_threshold": 3,
            "circuit_breaker_reset_timeout_seconds": 30.0,
        }
    )

    yield handler

    await handler.shutdown()


@pytest.fixture
async def cleanup_consul_services(
    real_consul_handler: ConsulHandler,
) -> AsyncGenerator[list[str], None]:
    """Track and cleanup test services from Consul.

    Yields a list where tests can append service IDs they register.
    After the test, all listed services are deregistered.

    Yields:
        List to append service IDs for cleanup.
    """
    services_to_cleanup: list[str] = []

    yield services_to_cleanup

    # Cleanup: deregister all tracked services
    for service_id in services_to_cleanup:
        try:
            envelope = {
                "operation": "consul.deregister_service",
                "payload": {"service_id": service_id},
            }
            await real_consul_handler.execute(envelope)
        except Exception as e:
            logger.warning(
                "Cleanup failed for Consul service %s: %s",
                service_id,
                e,
                exc_info=True,
            )


# =============================================================================
# Projector Fixtures
# =============================================================================


@pytest.fixture
async def real_projector(
    postgres_pool: asyncpg.Pool,
) -> ProjectorRegistration:
    """Create ProjectorRegistration for persisting handler outputs.

    Args:
        postgres_pool: Database connection pool.

    Returns:
        ProjectorRegistration for persisting projections.
    """
    from omnibase_infra.projectors import ProjectorRegistration

    return ProjectorRegistration(postgres_pool)


# =============================================================================
# Timeout Services Fixtures
# =============================================================================


@pytest.fixture
async def timeout_scanner(
    projection_reader: ProjectionReaderRegistration,
) -> TimeoutScanner:
    """Create TimeoutScanner for querying overdue entities.

    Args:
        projection_reader: Reader for querying projections.

    Returns:
        TimeoutScanner for finding overdue registrations.
    """
    from omnibase_infra.services import TimeoutScanner

    return TimeoutScanner(projection_reader)


@pytest.fixture
async def timeout_emitter(
    timeout_scanner: TimeoutScanner,
    real_kafka_event_bus: KafkaEventBus,
    real_projector: ProjectorRegistration,
) -> TimeoutEmitter:
    """Create TimeoutEmitter for emitting timeout events.

    Args:
        timeout_scanner: Scanner for finding overdue entities.
        real_kafka_event_bus: Event bus for publishing events.
        real_projector: Projector for updating markers.

    Returns:
        TimeoutEmitter for processing timeouts.
    """
    from omnibase_infra.services import TimeoutEmitter

    return TimeoutEmitter(
        timeout_query=timeout_scanner,
        event_bus=real_kafka_event_bus,
        projector=real_projector,
    )


# =============================================================================
# Orchestrator Fixtures
# =============================================================================


@pytest.fixture
async def registration_orchestrator(
    wired_container: ModelONEXContainer,
    timeout_scanner: TimeoutScanner,
    timeout_emitter: TimeoutEmitter,
    projection_reader: ProjectionReaderRegistration,
    real_projector: ProjectorRegistration,
) -> NodeRegistrationOrchestrator:
    """Fully wired NodeRegistrationOrchestrator for E2E tests.

    This fixture creates a complete orchestrator with:
    - Container-based dependency injection
    - TimeoutCoordinator for RuntimeTick handling
    - HandlerNodeHeartbeat for liveness tracking

    Args:
        wired_container: Container with handlers wired.
        timeout_scanner: Scanner for timeout queries.
        timeout_emitter: Emitter for timeout events.
        projection_reader: Reader for projection queries.
        real_projector: Projector for persisting state.

    Returns:
        NodeRegistrationOrchestrator: Fully configured orchestrator.
    """
    from omnibase_infra.nodes.node_registration_orchestrator import (
        NodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
        TimeoutCoordinator,
    )

    # Create orchestrator
    orchestrator = NodeRegistrationOrchestrator(wired_container)

    # Wire timeout coordinator
    timeout_coordinator = TimeoutCoordinator(timeout_scanner, timeout_emitter)
    orchestrator.set_timeout_coordinator(timeout_coordinator)

    # Wire heartbeat handler
    heartbeat_handler = HandlerNodeHeartbeat(
        projection_reader=projection_reader,
        projector=real_projector,
        liveness_window_seconds=90.0,
    )
    orchestrator.set_heartbeat_handler(heartbeat_handler)

    return orchestrator


# =============================================================================
# Test Node Fixtures
# =============================================================================


@pytest.fixture
def unique_node_id() -> UUID:
    """Generate a unique node ID for test isolation.

    Returns:
        UUID: Unique identifier for test nodes.
    """
    return uuid4()


@pytest.fixture
def unique_correlation_id() -> UUID:
    """Generate a unique correlation ID for test tracing.

    Returns:
        UUID: Unique correlation ID for test events.
    """
    return uuid4()


@pytest.fixture
async def introspectable_test_node(
    real_kafka_event_bus: KafkaEventBus,
    unique_node_id: UUID,
) -> IntrospectableTestNode:
    """Test node implementing MixinNodeIntrospection for E2E testing.

    This fixture creates a test node that can publish introspection
    events to the real Kafka event bus.

    Args:
        real_kafka_event_bus: Connected Kafka event bus.
        unique_node_id: Unique identifier for this test node.

    Returns:
        IntrospectableTestNode: Test node with introspection capability.
    """
    from omnibase_infra.mixins import MixinNodeIntrospection
    from omnibase_infra.models.discovery import ModelIntrospectionConfig

    class IntrospectableTestNode(MixinNodeIntrospection):
        """Test node for E2E introspection testing."""

        def __init__(
            self,
            node_id: UUID,
            event_bus: KafkaEventBus,
            node_type: EnumNodeKind = EnumNodeKind.EFFECT,
            version: str = "1.0.0",
        ) -> None:
            self._node_id = node_id
            self._node_type_value = node_type
            self._version = version
            self.health_url = f"http://localhost:8080/{node_id}/health"
            self.api_url = f"http://localhost:8080/{node_id}/api"

            # Get topic from environment or use docker-compose default
            # The runtime expects: dev.onex.evt.node-introspection.v1 (from ONEX_INPUT_TOPIC)
            introspection_topic = os.getenv(
                "ONEX_INPUT_TOPIC", "dev.onex.evt.node-introspection.v1"
            )
            config = ModelIntrospectionConfig(
                node_id=node_id,
                node_type=node_type,
                event_bus=event_bus,
                version=version,
                cache_ttl=60.0,
                introspection_topic=introspection_topic,
            )
            self.initialize_introspection(config)

        @property
        def node_id(self) -> UUID:
            return self._node_id

        @property
        def node_type(self) -> EnumNodeKind:
            return self._node_type_value

        @property
        def version(self) -> str:
            return self._version

        async def execute_operation(self, data: dict[str, object]) -> dict[str, object]:
            """Sample operation for capability discovery."""
            return {"result": "processed", "input": data}

        async def handle_request(self, request: object) -> object:
            """Sample handler for capability discovery."""
            return {"status": "handled", "request": request}

    return IntrospectableTestNode(
        node_id=unique_node_id,
        event_bus=real_kafka_event_bus,
    )


# Export the IntrospectableTestNode class for type hints
class IntrospectableTestNode:
    """Type stub for IntrospectableTestNode fixture.

    The actual implementation is defined inside the fixture to have
    access to the event bus. This class is exported for type hints.
    """

    node_id: UUID
    node_type: EnumNodeKind
    version: str


# =============================================================================
# Event Factory Fixtures
# =============================================================================


@pytest.fixture
def introspection_event_factory(
    unique_node_id: UUID,
    unique_correlation_id: UUID,
):
    """Factory for creating ModelNodeIntrospectionEvent instances.

    Returns a callable that creates introspection events with the
    test's unique node and correlation IDs.

    Args:
        unique_node_id: Unique node ID for this test.
        unique_correlation_id: Unique correlation ID for this test.

    Returns:
        Callable that creates ModelNodeIntrospectionEvent instances.
    """
    from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
    from omnibase_infra.models.registration.model_node_capabilities import (
        ModelNodeCapabilities,
    )
    from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata

    def _create_event(
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        node_version: str = "1.0.0",
        endpoints: dict[str, str] | None = None,
        node_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelNodeIntrospectionEvent:
        """Create an introspection event with test-specific IDs."""
        return ModelNodeIntrospectionEvent(
            node_id=node_id or unique_node_id,
            node_type=node_type.value,
            node_version=node_version,
            capabilities=ModelNodeCapabilities(),
            endpoints=endpoints or {"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=correlation_id or unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

    return _create_event


@pytest.fixture
def runtime_tick_factory(
    unique_correlation_id: UUID,
):
    """Factory for creating ModelRuntimeTick instances.

    Returns a callable that creates runtime tick events with
    deterministic timestamps for timeout testing.

    Args:
        unique_correlation_id: Unique correlation ID for this test.

    Returns:
        Callable that creates ModelRuntimeTick instances.
    """
    from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

    sequence = 0

    def _create_tick(
        now: datetime | None = None,
        tick_interval_ms: int = 1000,
        correlation_id: UUID | None = None,
    ) -> ModelRuntimeTick:
        """Create a runtime tick with specified 'now' time."""
        nonlocal sequence
        sequence += 1

        tick_now = now or datetime.now(UTC)
        return ModelRuntimeTick(
            now=tick_now,
            tick_id=uuid4(),
            sequence_number=sequence,
            scheduled_at=tick_now,
            correlation_id=correlation_id or unique_correlation_id,
            scheduler_id="e2e-test-scheduler",
            tick_interval_ms=tick_interval_ms,
        )

    return _create_tick


# =============================================================================
# Deterministic Time Fixtures
# =============================================================================


@pytest.fixture
def deterministic_clock():
    """Create a deterministic clock for time control.

    Returns:
        DeterministicClock: Clock with controllable time.
    """
    from tests.helpers.deterministic import DeterministicClock

    return DeterministicClock()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture
async def cleanup_projections(
    postgres_pool: asyncpg.Pool,
    unique_node_id: UUID,
) -> AsyncGenerator[None, None]:
    """Cleanup test projections after test completion.

    This fixture ensures projection records created during the test
    are removed, preventing test data from polluting the database.

    Args:
        postgres_pool: Database connection pool.
        unique_node_id: Node ID to cleanup.

    Warning:
        PRODUCTION DATABASE SAFETY: This fixture executes DELETE operations
        against the configured database. The cleanup is scoped to a specific
        entity_id (unique_node_id) which should be a test-generated UUID.

        - NEVER run E2E tests against a production database
        - Always verify POSTGRES_HOST points to a test/dev environment
        - Use .env.docker or dedicated test infrastructure
        - Production databases should have network isolation
    """
    yield

    # Cleanup: remove projection records for this test's node
    try:
        async with postgres_pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM registration_projections
                WHERE entity_id = $1
                """,
                unique_node_id,
            )
    except Exception as e:
        logger.warning(
            "Cleanup failed for projection entity_id %s: %s",
            unique_node_id,
            e,
            exc_info=True,
        )


@pytest.fixture
async def cleanup_node_ids(
    postgres_pool: asyncpg.Pool,
) -> AsyncGenerator[list[UUID], None]:
    """Track and cleanup multiple node IDs from projections.

    Yields a list where tests can append node IDs they register.
    After the test, all listed node projections are removed.

    This fixture is useful for tests that create multiple nodes
    dynamically (e.g., concurrent registration tests).

    Args:
        postgres_pool: Database connection pool.

    Yields:
        List to append node IDs for cleanup.

    Warning:
        PRODUCTION DATABASE SAFETY: This fixture executes DELETE operations
        against the configured database. The cleanup is scoped to specific
        entity_ids which should be test-generated UUIDs.

        - NEVER run E2E tests against a production database
        - Always verify POSTGRES_HOST points to a test/dev environment
        - Use .env.docker or dedicated test infrastructure
        - Production databases should have network isolation
    """
    node_ids_to_cleanup: list[UUID] = []

    yield node_ids_to_cleanup

    # Cleanup: remove projection records for all tracked nodes
    if node_ids_to_cleanup:
        try:
            async with postgres_pool.acquire() as conn:
                await conn.execute(
                    """
                    DELETE FROM registration_projections
                    WHERE entity_id = ANY($1::uuid[])
                    """,
                    node_ids_to_cleanup,
                )
        except Exception as e:
            logger.warning(
                "Cleanup failed for %d projection entity_ids: %s",
                len(node_ids_to_cleanup),
                e,
                exc_info=True,
            )


# =============================================================================
# Export Fixtures
# =============================================================================

__all__ = [
    # Availability flags
    "ALL_INFRA_AVAILABLE",
    "CONSUL_AVAILABLE",
    "KAFKA_AVAILABLE",
    "POSTGRES_AVAILABLE",
    # Database fixtures
    "postgres_pool",
    # Container fixtures
    "wired_container",
    "projection_reader",
    "handler_node_introspected",
    # Kafka fixtures
    "real_kafka_event_bus",
    # Consul fixtures
    "real_consul_handler",
    "cleanup_consul_services",
    # Projector fixtures
    "real_projector",
    # Timeout fixtures
    "timeout_scanner",
    "timeout_emitter",
    # Orchestrator fixtures
    "registration_orchestrator",
    # Test node fixtures
    "unique_node_id",
    "unique_correlation_id",
    "introspectable_test_node",
    "IntrospectableTestNode",
    # Event factory fixtures
    "introspection_event_factory",
    "runtime_tick_factory",
    # Time fixtures
    "deterministic_clock",
    # Cleanup fixtures
    "cleanup_projections",
    "cleanup_node_ids",
]
