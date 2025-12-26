# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for registration workflow integration tests.

This module provides fixtures that wire the complete registration workflow:
1. RegistrationReducer for intent computation
2. NodeRegistryEffect for backend registration execution
3. Test doubles for Consul and PostgreSQL backends
4. Call order tracking for orchestration verification

Design Principles:
    - Real components: Uses actual RegistrationReducer and NodeRegistryEffect
    - Test doubles: Backend clients are controllable implementations
    - Call tracking: Enables verification of orchestration order
    - Test isolation: Each test gets fresh instances
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

if TYPE_CHECKING:
    from omnibase_core.nodes import ModelReducerOutput

from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.effects import NodeRegistryEffect
from omnibase_infra.nodes.effects.models import (
    ModelBackendResult,
    ModelEffectIdempotencyConfig,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

# Import deterministic utilities from test helpers
from tests.helpers.deterministic import DeterministicClock

# Import test doubles from sibling package
from tests.integration.registration.effect.test_doubles import (
    StubConsulClient,
    StubPostgresAdapter,
)

# =============================================================================
# Introspectable Test Node for MixinNodeIntrospection Testing
# =============================================================================


class IntrospectableTestNode(MixinNodeIntrospection):
    """Test node that implements MixinNodeIntrospection for testing.

    This node provides a minimal implementation suitable for testing
    the introspection workflow without real infrastructure.

    Attributes:
        node_id: Unique node identifier.
        node_type: Node type classification (EnumNodeKind.EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        version: Node version string.
        health_url: Health endpoint URL.
    """

    def __init__(
        self,
        node_id: UUID | None = None,
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        version: str = "1.0.0",
        event_bus: InMemoryEventBus | None = None,
    ) -> None:
        """Initialize the test node.

        Args:
            node_id: Optional node ID (generated if not provided).
            node_type: Node type classification (EnumNodeKind).
            version: Node version string.
            event_bus: Optional event bus for publishing.
        """
        self._node_id = node_id or uuid4()
        self._node_type_value = node_type
        self._version = version
        self.health_url = f"http://localhost:8080/{self._node_id}/health"
        self.api_url = f"http://localhost:8080/{self._node_id}/api"

        config = ModelIntrospectionConfig(
            node_id=self._node_id,
            node_type=node_type,
            event_bus=event_bus,
            version=version,
            cache_ttl=60.0,  # Short TTL for testing
        )
        self.initialize_introspection(config)

    @property
    def node_id(self) -> UUID:
        """Get node identifier."""
        return self._node_id

    @property
    def node_type(self) -> EnumNodeKind:
        """Get node type."""
        return self._node_type_value

    @property
    def version(self) -> str:
        """Get node version."""
        return self._version

    async def execute_operation(self, data: dict[str, object]) -> dict[str, object]:
        """Sample operation method for capability discovery.

        Args:
            data: Input data.

        Returns:
            Processed output data.
        """
        return {"result": "processed", "input": data}

    async def handle_request(self, request: object) -> object:
        """Sample handler method for capability discovery.

        Args:
            request: Input request.

        Returns:
            Processed response.
        """
        return {"status": "handled", "request": request}


@dataclass
class CallRecord:
    """Record of a component call with timestamp for order tracking.

    Attributes:
        component: Name of the component ("reducer" or "effect").
        method: Method that was called.
        timestamp: When the call occurred.
        correlation_id: Correlation ID for the call.
    """

    component: Literal["reducer", "effect"]
    method: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None


class CallOrderTracker:
    """Tracks call order across components for orchestration verification.

    This tracker enables tests to verify that the reducer is called before
    the effect during orchestrated registration workflows.

    Attributes:
        calls: List of CallRecord instances in order.

    Example:
        >>> tracker = CallOrderTracker()
        >>> tracker.record("reducer", "reduce")
        >>> tracker.record("effect", "register_node")
        >>> assert tracker.get_call_order() == ["reducer", "effect"]
    """

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def record(
        self,
        component: Literal["reducer", "effect"],
        method: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Record a component call.

        Args:
            component: Name of the component.
            method: Method that was called.
            correlation_id: Optional correlation ID.
        """
        self.calls.append(
            CallRecord(
                component=component,
                method=method,
                correlation_id=correlation_id,
            )
        )

    def get_call_order(self) -> list[str]:
        """Get the order of component calls.

        Returns:
            List of component names in call order.
        """
        return [call.component for call in self.calls]

    def get_reducer_calls(self) -> list[CallRecord]:
        """Get all reducer calls.

        Returns:
            List of reducer CallRecord instances.
        """
        return [call for call in self.calls if call.component == "reducer"]

    def get_effect_calls(self) -> list[CallRecord]:
        """Get all effect calls.

        Returns:
            List of effect CallRecord instances.
        """
        return [call for call in self.calls if call.component == "effect"]

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.calls.clear()


class TrackedRegistrationReducer:
    """RegistrationReducer wrapper that tracks calls for orchestration tests.

    This wrapper delegates to the real RegistrationReducer while recording
    calls for verification of orchestration order.

    Attributes:
        reducer: The underlying RegistrationReducer.
        tracker: CallOrderTracker for recording calls.
        reduce_call_count: Number of times reduce was called.
    """

    def __init__(
        self,
        reducer: RegistrationReducer,
        tracker: CallOrderTracker,
    ) -> None:
        self.reducer = reducer
        self.tracker = tracker
        self.reduce_call_count = 0

    def reduce(
        self,
        state: ModelRegistrationState,
        event: ModelNodeIntrospectionEvent,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Delegate to reducer and track the call.

        Args:
            state: Current registration state.
            event: Introspection event to process.

        Returns:
            ModelReducerOutput from the underlying reducer.
        """
        self.reduce_call_count += 1
        self.tracker.record(
            "reducer",
            "reduce",
            correlation_id=event.correlation_id,
        )
        return self.reducer.reduce(state, event)


class TrackedNodeRegistryEffect:
    """NodeRegistryEffect wrapper that tracks calls for orchestration tests.

    This wrapper delegates to the real NodeRegistryEffect while recording
    calls for verification of orchestration order.

    Attributes:
        effect: The underlying NodeRegistryEffect.
        tracker: CallOrderTracker for recording calls.
        register_node_call_count: Number of times register_node was called.
    """

    def __init__(
        self,
        effect: NodeRegistryEffect,
        tracker: CallOrderTracker,
    ) -> None:
        self.effect = effect
        self.tracker = tracker
        self.register_node_call_count = 0

    async def register_node(
        self,
        request: ModelRegistryRequest,
        *,
        skip_consul: bool = False,
        skip_postgres: bool = False,
    ) -> ModelRegistryResponse:
        """Delegate to effect and track the call.

        Args:
            request: Registration request.
            skip_consul: If True, skip Consul registration.
            skip_postgres: If True, skip PostgreSQL registration.

        Returns:
            ModelRegistryResponse from the underlying effect.
        """
        self.register_node_call_count += 1
        self.tracker.record(
            "effect",
            "register_node",
            correlation_id=request.correlation_id,
        )
        return await self.effect.register_node(
            request,
            skip_consul=skip_consul,
            skip_postgres=skip_postgres,
        )

    async def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        """Delegate to effect."""
        return await self.effect.get_completed_backends(correlation_id)

    async def clear_completed_backends(self, correlation_id: UUID) -> None:
        """Delegate to effect."""
        return await self.effect.clear_completed_backends(correlation_id)


@pytest.fixture
def call_tracker() -> CallOrderTracker:
    """Create a fresh CallOrderTracker.

    Returns:
        CallOrderTracker for recording component calls.
    """
    return CallOrderTracker()


@pytest.fixture
def consul_client() -> StubConsulClient:
    """Create a fresh StubConsulClient.

    Returns:
        StubConsulClient with default (success) configuration.
    """
    return StubConsulClient()


@pytest.fixture
def postgres_adapter() -> StubPostgresAdapter:
    """Create a fresh StubPostgresAdapter.

    Returns:
        StubPostgresAdapter with default (success) configuration.
    """
    return StubPostgresAdapter()


@pytest.fixture
def idempotency_store() -> InMemoryEffectIdempotencyStore:
    """Create a fresh InMemoryEffectIdempotencyStore.

    Returns:
        InMemoryEffectIdempotencyStore with default configuration.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=1000,
        cache_ttl_seconds=3600.0,
    )
    return InMemoryEffectIdempotencyStore(config=config)


@pytest.fixture
def registration_reducer() -> RegistrationReducer:
    """Create a fresh RegistrationReducer.

    Returns:
        RegistrationReducer instance.
    """
    return RegistrationReducer()


@pytest.fixture
def tracked_reducer(
    registration_reducer: RegistrationReducer,
    call_tracker: CallOrderTracker,
) -> TrackedRegistrationReducer:
    """Create a tracked RegistrationReducer wrapper.

    Args:
        registration_reducer: The underlying reducer.
        call_tracker: Tracker for recording calls.

    Returns:
        TrackedRegistrationReducer that records calls.
    """
    return TrackedRegistrationReducer(registration_reducer, call_tracker)


@pytest.fixture
def registry_effect(
    consul_client: StubConsulClient,
    postgres_adapter: StubPostgresAdapter,
    idempotency_store: InMemoryEffectIdempotencyStore,
) -> NodeRegistryEffect:
    """Create NodeRegistryEffect with test double backends.

    Args:
        consul_client: Test double Consul client.
        postgres_adapter: Test double PostgreSQL adapter.
        idempotency_store: In-memory idempotency store.

    Returns:
        NodeRegistryEffect configured with test doubles.
    """
    return NodeRegistryEffect(
        consul_client=consul_client,
        postgres_adapter=postgres_adapter,
        idempotency_store=idempotency_store,
    )


@pytest.fixture
def tracked_effect(
    registry_effect: NodeRegistryEffect,
    call_tracker: CallOrderTracker,
) -> TrackedNodeRegistryEffect:
    """Create a tracked NodeRegistryEffect wrapper.

    Args:
        registry_effect: The underlying effect.
        call_tracker: Tracker for recording calls.

    Returns:
        TrackedNodeRegistryEffect that records calls.
    """
    return TrackedNodeRegistryEffect(registry_effect, call_tracker)


@pytest.fixture
def sample_introspection_event() -> ModelNodeIntrospectionEvent:
    """Create a sample introspection event for testing.

    Returns:
        ModelNodeIntrospectionEvent with valid test data.
    """
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT.value,
        node_version="1.0.0",
        correlation_id=uuid4(),
        endpoints={"health": "http://localhost:8080/health"},
    )


@pytest.fixture
def introspection_event_factory() -> Callable[..., ModelNodeIntrospectionEvent]:
    """Factory for creating unique introspection events.

    Returns:
        Callable that creates unique ModelNodeIntrospectionEvent instances.
    """

    def _create_event(
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        node_version: str = "1.0.0",
        correlation_id: UUID | None = None,
        node_id: UUID | None = None,
    ) -> ModelNodeIntrospectionEvent:
        return ModelNodeIntrospectionEvent(
            node_id=node_id or uuid4(),
            node_type=node_type.value,
            node_version=node_version,
            correlation_id=correlation_id or uuid4(),
            endpoints={"health": "http://localhost:8080/health"},
        )

    return _create_event


@pytest.fixture
async def event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Create and start an in-memory event bus.

    Yields:
        InMemoryEventBus instance.
    """
    bus = InMemoryEventBus(environment="test", group="workflow")
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def test_node_factory(
    event_bus: InMemoryEventBus,
) -> Callable[..., IntrospectableTestNode]:
    """Factory for creating IntrospectableTestNode instances.

    Args:
        event_bus: Event bus for publishing introspection events.

    Returns:
        Callable that creates IntrospectableTestNode instances.
    """

    def _create_node(
        node_id: UUID | None = None,
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        version: str = "1.0.0",
    ) -> IntrospectableTestNode:
        return IntrospectableTestNode(
            node_id=node_id,
            node_type=node_type,
            version=version,
            event_bus=event_bus,
        )

    return _create_node


@pytest.fixture
def test_node(event_bus: InMemoryEventBus) -> IntrospectableTestNode:
    """Create an introspectable test node.

    Args:
        event_bus: Event bus for publishing introspection events.

    Returns:
        IntrospectableTestNode configured with event bus.
    """
    return IntrospectableTestNode(event_bus=event_bus)


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle registration state.

    Returns:
        ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


# =============================================================================
# Deterministic Clock Fixture
# =============================================================================
# Note: DeterministicClock class is imported from tests.helpers.deterministic


@pytest.fixture
def deterministic_clock() -> DeterministicClock:
    """Create a deterministic clock for time control.

    Returns:
        Fresh DeterministicClock instance.
    """
    return DeterministicClock()


# =============================================================================
# Deterministic UUID Fixtures
# =============================================================================


@dataclass
class DeterministicUUIDGenerator:
    """Generator for deterministic UUIDs in tests.

    Creates reproducible UUIDs based on a counter, enabling
    stable snapshots and predictable test behavior.

    Attributes:
        _counter: Counter for UUID generation.
        _prefix: Prefix for generated UUIDs.

    Example:
        >>> gen = DeterministicUUIDGenerator()
        >>> uuid1 = gen.next()
        >>> uuid2 = gen.next()
        >>> assert uuid1 != uuid2
        >>> gen.reset()
        >>> uuid1_again = gen.next()
        >>> assert uuid1 == uuid1_again
    """

    _counter: int = field(default=0)
    _prefix: str = field(default="00000000-0000-0000-0000")

    def next(self) -> UUID:
        """Generate next deterministic UUID.

        Returns:
            Next UUID in sequence.
        """
        self._counter += 1
        # Format: 00000000-0000-0000-0000-{12-digit counter}
        counter_str = f"{self._counter:012d}"
        uuid_str = f"{self._prefix}-{counter_str}"
        return UUID(uuid_str)

    def reset(self) -> None:
        """Reset counter to 0."""
        self._counter = 0

    @property
    def count(self) -> int:
        """Get current counter value."""
        return self._counter


@pytest.fixture
def uuid_generator() -> DeterministicUUIDGenerator:
    """Create a deterministic UUID generator.

    Returns:
        Fresh DeterministicUUIDGenerator instance.
    """
    return DeterministicUUIDGenerator()


@pytest.fixture
def correlation_id(uuid_generator: DeterministicUUIDGenerator) -> UUID:
    """Generate a deterministic correlation ID for tests.

    Args:
        uuid_generator: Deterministic UUID generator.

    Returns:
        Deterministic UUID for correlation tracking.
    """
    return uuid_generator.next()


# =============================================================================
# Mock Backend Fixtures with Call Tracking
# =============================================================================


@dataclass
class MockConsulEffect:
    """Mock Consul effect with call tracking and failure injection.

    Wraps StubConsulClient with additional tracking capabilities
    for workflow verification.

    Attributes:
        _stub: Underlying stub client.
        _call_history: List of call details.
        _failure_mode: Current failure mode.

    Example:
        >>> mock = MockConsulEffect()
        >>> await mock.register_service("id-1", "svc", ["tag"])
        >>> assert mock.call_count == 1
        >>> mock.set_failure_mode("connection_refused")
        >>> result = await mock.register_service("id-2", "svc", ["tag"])
        >>> assert result.success is False
    """

    _stub: StubConsulClient = field(default_factory=StubConsulClient)
    _call_history: list[dict[str, object]] = field(default_factory=list)
    _failure_mode: str | None = field(default=None)

    async def register_service(
        self,
        service_id: str,
        service_name: str,
        tags: list[str],
        health_check: dict[str, str] | None = None,
    ) -> ModelBackendResult:
        """Register a service with call tracking.

        Args:
            service_id: Unique identifier for the service.
            service_name: Name of the service.
            tags: List of tags.
            health_check: Optional health check config.

        Returns:
            ModelBackendResult with operation outcome.
        """
        # Record call
        self._call_history.append(
            {
                "method": "register_service",
                "service_id": service_id,
                "service_name": service_name,
                "tags": tags,
                "health_check": health_check,
            }
        )

        # Check for injected failure
        if self._failure_mode is not None:
            return ModelBackendResult(
                success=False,
                error=f"Consul operation failed: {self._failure_mode}",
                error_code="CONSUL_MOCK_FAILURE",
            )

        # Delegate to stub
        return await self._stub.register_service(
            service_id=service_id,
            service_name=service_name,
            tags=tags,
            health_check=health_check,
        )

    def set_failure_mode(self, mode: str | None) -> None:
        """Set or clear failure mode.

        Args:
            mode: Failure mode string or None to clear.
        """
        self._failure_mode = mode

    def clear_failure_mode(self) -> None:
        """Clear any failure mode."""
        self._failure_mode = None

    @property
    def call_count(self) -> int:
        """Get total number of calls."""
        return len(self._call_history)

    @property
    def call_history(self) -> list[dict[str, object]]:
        """Get full call history."""
        return list(self._call_history)

    def reset(self) -> None:
        """Reset all state and history."""
        self._stub.reset()
        self._call_history.clear()
        self._failure_mode = None


@dataclass
class MockPostgresEffect:
    """Mock PostgreSQL effect with call tracking and failure injection.

    Wraps StubPostgresAdapter with additional tracking capabilities
    for workflow verification.

    Attributes:
        _stub: Underlying stub adapter.
        _call_history: List of call details.
        _failure_mode: Current failure mode.
    """

    _stub: StubPostgresAdapter = field(default_factory=StubPostgresAdapter)
    _call_history: list[dict[str, object]] = field(default_factory=list)
    _failure_mode: str | None = field(default=None)

    async def upsert(
        self,
        node_id: UUID,
        node_type: EnumNodeKind,
        node_version: str,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> ModelBackendResult:
        """Upsert registration record with call tracking.

        Args:
            node_id: Unique node identifier.
            node_type: ONEX node type (EnumNodeKind).
            node_version: Semantic version.
            endpoints: Endpoint URLs.
            metadata: Additional metadata.

        Returns:
            ModelBackendResult with operation outcome.
        """
        # Record call
        self._call_history.append(
            {
                "method": "upsert",
                "node_id": node_id,
                "node_type": node_type,
                "node_version": node_version,
                "endpoints": endpoints,
                "metadata": metadata,
            }
        )

        # Check for injected failure
        if self._failure_mode is not None:
            return ModelBackendResult(
                success=False,
                error=f"PostgreSQL operation failed: {self._failure_mode}",
                error_code="POSTGRES_MOCK_FAILURE",
            )

        # Delegate to stub
        return await self._stub.upsert(
            node_id=node_id,
            node_type=node_type,
            node_version=node_version,
            endpoints=endpoints,
            metadata=metadata,
        )

    def set_failure_mode(self, mode: str | None) -> None:
        """Set or clear failure mode.

        Args:
            mode: Failure mode string or None to clear.
        """
        self._failure_mode = mode

    def clear_failure_mode(self) -> None:
        """Clear any failure mode."""
        self._failure_mode = None

    @property
    def call_count(self) -> int:
        """Get total number of calls."""
        return len(self._call_history)

    @property
    def call_history(self) -> list[dict[str, object]]:
        """Get full call history."""
        return list(self._call_history)

    def reset(self) -> None:
        """Reset all state and history."""
        self._stub.reset()
        self._call_history.clear()
        self._failure_mode = None


@pytest.fixture
def mock_consul_effect() -> MockConsulEffect:
    """Create a mock Consul effect with call tracking.

    Returns:
        Fresh MockConsulEffect instance.
    """
    return MockConsulEffect()


@pytest.fixture
def mock_postgres_effect() -> MockPostgresEffect:
    """Create a mock PostgreSQL effect with call tracking.

    Returns:
        Fresh MockPostgresEffect instance.
    """
    return MockPostgresEffect()


@pytest.fixture
def registry_effect_with_mocks(
    mock_consul_effect: MockConsulEffect,
    mock_postgres_effect: MockPostgresEffect,
    idempotency_store: InMemoryEffectIdempotencyStore,
) -> NodeRegistryEffect:
    """Create NodeRegistryEffect with mock backends for call tracking.

    Args:
        mock_consul_effect: Mock Consul effect.
        mock_postgres_effect: Mock PostgreSQL effect.
        idempotency_store: Idempotency store.

    Returns:
        NodeRegistryEffect configured with mocks.
    """
    return NodeRegistryEffect(
        consul_client=mock_consul_effect,  # type: ignore[arg-type]
        postgres_adapter=mock_postgres_effect,  # type: ignore[arg-type]
        idempotency_store=idempotency_store,
    )


# =============================================================================
# Snapshot Normalization Helpers
# =============================================================================


@dataclass
class SnapshotNormalizer:
    """Helper for normalizing test output for stable snapshots.

    Provides methods to replace dynamic values (UUIDs, timestamps)
    with stable placeholders for snapshot testing.

    Attributes:
        _uuid_map: Mapping of seen UUIDs to placeholder IDs.
        _uuid_counter: Counter for UUID placeholders.
    """

    _uuid_map: dict[UUID, str] = field(default_factory=dict)
    _uuid_counter: int = field(default=0)

    def normalize_uuid(self, uuid_val: UUID) -> str:
        """Normalize a UUID to a stable placeholder.

        Args:
            uuid_val: UUID to normalize.

        Returns:
            Stable placeholder string like "<UUID-1>".
        """
        if uuid_val not in self._uuid_map:
            self._uuid_counter += 1
            self._uuid_map[uuid_val] = f"<UUID-{self._uuid_counter}>"
        return self._uuid_map[uuid_val]

    def normalize_timestamp(self, timestamp: datetime) -> str:
        """Normalize a timestamp to a stable placeholder.

        Args:
            timestamp: Timestamp to normalize.

        Returns:
            Stable placeholder string.
        """
        return "<TIMESTAMP>"

    def normalize_state(self, state: ModelRegistrationState) -> dict[str, object]:
        """Normalize a registration state for snapshot comparison.

        Args:
            state: Registration state to normalize.

        Returns:
            Dict with normalized values.
        """
        result: dict[str, object] = {
            "status": state.status,
            "consul_confirmed": state.consul_confirmed,
            "postgres_confirmed": state.postgres_confirmed,
            "failure_reason": state.failure_reason,
        }
        if state.node_id is not None:
            result["node_id"] = self.normalize_uuid(state.node_id)
        else:
            result["node_id"] = None
        if state.last_processed_event_id is not None:
            result["last_processed_event_id"] = self.normalize_uuid(
                state.last_processed_event_id
            )
        else:
            result["last_processed_event_id"] = None
        return result

    def normalize_response(self, response: ModelRegistryResponse) -> dict[str, object]:
        """Normalize a registry response for snapshot comparison.

        Args:
            response: Registry response to normalize.

        Returns:
            Dict with normalized values.
        """
        return {
            "status": response.status,
            "node_id": self.normalize_uuid(response.node_id),
            "correlation_id": self.normalize_uuid(response.correlation_id),
            "consul_success": response.consul_result.success,
            "postgres_success": response.postgres_result.success,
            "error_summary": response.error_summary,
        }

    def reset(self) -> None:
        """Reset all normalization state."""
        self._uuid_map.clear()
        self._uuid_counter = 0


@pytest.fixture
def snapshot_normalizer() -> SnapshotNormalizer:
    """Create a snapshot normalizer for stable assertions.

    Returns:
        Fresh SnapshotNormalizer instance.
    """
    return SnapshotNormalizer()


# =============================================================================
# Workflow Scenario Helpers
# =============================================================================


@dataclass
class WorkflowScenarioContext:
    """Context for workflow scenario execution.

    Provides access to all components and helpers needed for
    end-to-end workflow testing.

    Attributes:
        reducer: Registration reducer instance.
        consul_effect: Mock Consul effect.
        postgres_effect: Mock PostgreSQL effect.
        clock: Deterministic clock.
        uuid_gen: Deterministic UUID generator.
        normalizer: Snapshot normalizer.
        call_tracker: Call order tracker.
    """

    reducer: RegistrationReducer
    consul_effect: MockConsulEffect
    postgres_effect: MockPostgresEffect
    clock: DeterministicClock
    uuid_gen: DeterministicUUIDGenerator
    normalizer: SnapshotNormalizer
    call_tracker: CallOrderTracker

    def reset_all(self) -> None:
        """Reset all components to initial state."""
        self.consul_effect.reset()
        self.postgres_effect.reset()
        self.clock.reset()
        self.uuid_gen.reset()
        self.normalizer.reset()
        self.call_tracker.clear()

    def get_total_effect_calls(self) -> int:
        """Get total number of effect calls across all backends."""
        return self.consul_effect.call_count + self.postgres_effect.call_count


@pytest.fixture
def workflow_context(
    registration_reducer: RegistrationReducer,
    mock_consul_effect: MockConsulEffect,
    mock_postgres_effect: MockPostgresEffect,
    deterministic_clock: DeterministicClock,
    uuid_generator: DeterministicUUIDGenerator,
    snapshot_normalizer: SnapshotNormalizer,
    call_tracker: CallOrderTracker,
) -> WorkflowScenarioContext:
    """Create a complete workflow scenario context.

    Combines all fixtures into a single context object for
    convenient access in tests.

    Args:
        registration_reducer: Reducer fixture.
        mock_consul_effect: Consul effect fixture.
        mock_postgres_effect: PostgreSQL effect fixture.
        deterministic_clock: Clock fixture.
        uuid_generator: UUID generator fixture.
        snapshot_normalizer: Normalizer fixture.
        call_tracker: Call tracker fixture.

    Returns:
        WorkflowScenarioContext with all components.
    """
    return WorkflowScenarioContext(
        reducer=registration_reducer,
        consul_effect=mock_consul_effect,
        postgres_effect=mock_postgres_effect,
        clock=deterministic_clock,
        uuid_gen=uuid_generator,
        normalizer=snapshot_normalizer,
        call_tracker=call_tracker,
    )


# =============================================================================
# Failure Injection Helpers
# =============================================================================


@dataclass
class FailureInjector:
    """Helper for injecting failures into mock backends.

    Provides a convenient API for setting up failure scenarios
    across multiple backends.

    Attributes:
        consul_effect: Mock Consul effect.
        postgres_effect: Mock PostgreSQL effect.
    """

    consul_effect: MockConsulEffect
    postgres_effect: MockPostgresEffect

    def fail_consul(self, reason: str = "connection_refused") -> None:
        """Configure Consul to fail.

        Args:
            reason: Failure reason string.
        """
        self.consul_effect.set_failure_mode(reason)

    def fail_postgres(self, reason: str = "connection_refused") -> None:
        """Configure PostgreSQL to fail.

        Args:
            reason: Failure reason string.
        """
        self.postgres_effect.set_failure_mode(reason)

    def fail_both(self, reason: str = "service_unavailable") -> None:
        """Configure both backends to fail.

        Args:
            reason: Failure reason string.
        """
        self.consul_effect.set_failure_mode(reason)
        self.postgres_effect.set_failure_mode(reason)

    def clear_all_failures(self) -> None:
        """Clear all failure modes."""
        self.consul_effect.clear_failure_mode()
        self.postgres_effect.clear_failure_mode()


@pytest.fixture
def failure_injector(
    mock_consul_effect: MockConsulEffect,
    mock_postgres_effect: MockPostgresEffect,
) -> FailureInjector:
    """Create a failure injector for scenario testing.

    Args:
        mock_consul_effect: Consul effect fixture.
        mock_postgres_effect: PostgreSQL effect fixture.

    Returns:
        FailureInjector configured with the mock backends.
    """
    return FailureInjector(
        consul_effect=mock_consul_effect,
        postgres_effect=mock_postgres_effect,
    )


# =============================================================================
# Factory for deterministic introspection events
# =============================================================================


@pytest.fixture
def deterministic_introspection_event_factory(
    uuid_generator: DeterministicUUIDGenerator,
    deterministic_clock: DeterministicClock,
) -> Callable[..., ModelNodeIntrospectionEvent]:
    """Factory for creating introspection events with deterministic values.

    Returns a callable that generates ModelNodeIntrospectionEvent instances
    with deterministic UUIDs and timestamps.

    Args:
        uuid_generator: Deterministic UUID generator.
        deterministic_clock: Deterministic clock.

    Returns:
        Callable that creates introspection events.
    """
    from omnibase_infra.models.registration.model_node_capabilities import (
        ModelNodeCapabilities,
    )
    from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata

    def _create_event(
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        node_version: str = "1.0.0",
        endpoints: dict[str, str] | None = None,
        capabilities: ModelNodeCapabilities | None = None,
        metadata: ModelNodeMetadata | None = None,
        node_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelNodeIntrospectionEvent:
        """Create an introspection event with deterministic values.

        Args:
            node_type: ONEX node type (EnumNodeKind).
            node_version: Semantic version.
            endpoints: Endpoint URLs.
            capabilities: Node capabilities.
            metadata: Additional metadata.
            node_id: Optional specific node ID.
            correlation_id: Optional specific correlation ID.

        Returns:
            ModelNodeIntrospectionEvent with deterministic values.
        """
        return ModelNodeIntrospectionEvent(
            node_id=node_id or uuid_generator.next(),
            node_type=node_type.value,
            node_version=node_version,
            capabilities=capabilities or ModelNodeCapabilities(),
            endpoints=endpoints or {"health": "http://localhost:8080/health"},
            metadata=metadata or ModelNodeMetadata(),
            correlation_id=correlation_id or uuid_generator.next(),
            timestamp=deterministic_clock.now(),
        )

    return _create_event


@pytest.fixture
def registry_request_factory(
    uuid_generator: DeterministicUUIDGenerator,
    deterministic_clock: DeterministicClock,
) -> Callable[..., ModelRegistryRequest]:
    """Factory for creating registry requests with deterministic values.

    Returns a callable that generates ModelRegistryRequest instances
    with deterministic UUIDs and timestamps.

    Args:
        uuid_generator: Deterministic UUID generator.
        deterministic_clock: Deterministic clock.

    Returns:
        Callable that creates registry requests.
    """

    def _create_request(
        node_type: EnumNodeKind = EnumNodeKind.EFFECT,
        node_version: str = "1.0.0",
        endpoints: dict[str, str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        node_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelRegistryRequest:
        """Create a registry request with deterministic values.

        Args:
            node_type: ONEX node type (EnumNodeKind).
            node_version: Semantic version.
            endpoints: Endpoint URLs.
            tags: Service tags.
            metadata: Additional metadata.
            node_id: Optional specific node ID.
            correlation_id: Optional specific correlation ID.

        Returns:
            ModelRegistryRequest with deterministic values.
        """
        return ModelRegistryRequest(
            node_id=node_id or uuid_generator.next(),
            node_type=node_type,
            node_version=node_version,
            correlation_id=correlation_id or uuid_generator.next(),
            service_name=f"onex-{node_type.value}",
            endpoints=endpoints or {"health": "http://localhost:8080/health"},
            tags=tags or ["onex", node_type.value, "test"],
            metadata=metadata or {"environment": "test"},
            timestamp=deterministic_clock.now(),
        )

    return _create_request


# =============================================================================
# Export all fixtures
# =============================================================================

__all__ = [
    # Clock fixtures
    "DeterministicClock",
    "deterministic_clock",
    # UUID fixtures
    "DeterministicUUIDGenerator",
    "uuid_generator",
    "correlation_id",
    # Call tracking
    "CallRecord",
    "CallOrderTracker",
    "call_tracker",
    # Mock effect fixtures
    "MockConsulEffect",
    "MockPostgresEffect",
    "mock_consul_effect",
    "mock_postgres_effect",
    # Stub fixtures (compatibility)
    "consul_client",
    "postgres_adapter",
    # Tracked wrappers
    "TrackedRegistrationReducer",
    "TrackedNodeRegistryEffect",
    "tracked_reducer",
    "tracked_effect",
    # Component fixtures
    "registration_reducer",
    "initial_state",
    "idempotency_store",
    "registry_effect",
    "registry_effect_with_mocks",
    # Event fixtures
    "sample_introspection_event",
    "introspection_event_factory",
    "deterministic_introspection_event_factory",
    "registry_request_factory",
    # Event bus fixtures
    "event_bus",
    "test_node_factory",
    "test_node",
    "IntrospectableTestNode",
    # Snapshot fixtures
    "SnapshotNormalizer",
    "snapshot_normalizer",
    # Workflow scenario fixtures
    "WorkflowScenarioContext",
    "workflow_context",
    # Failure injection fixtures
    "FailureInjector",
    "failure_injector",
]
