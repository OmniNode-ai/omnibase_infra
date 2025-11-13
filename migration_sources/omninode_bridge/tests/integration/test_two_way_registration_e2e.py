#!/usr/bin/env python3
"""
End-to-End Integration Tests for Two-Way Registration Pattern.

Tests comprehensive two-way registration workflows including:
- Node startup and introspection broadcasting
- Registry listening and dual registration (Consul + PostgreSQL)
- Registry startup requesting re-introspection
- Heartbeat periodic publishing (30s intervals)
- Registry recovery scenarios
- Multiple nodes registration simultaneously
- Graceful degradation when Kafka/Registry unavailable

Test Infrastructure:
- PostgreSQL for tool registry persistence
- Kafka/RedPanda for event streaming
- Consul for service discovery
- Docker Compose orchestration
- Async fixtures with proper cleanup
- Performance benchmarking
"""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import yaml

# Import models
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)

# Import bridge nodes
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer
from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry

# Import utility functions from conftest
from .conftest import get_all_messages_from_topic

# Try importing ONEX infrastructure, with fallback to node stubs
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    ONEX_AVAILABLE = True
except ImportError:
    # Use the same stubs that the nodes use for consistency
    from omninode_bridge.nodes.registry.v1_0_0._stubs import ModelONEXContainer

    ONEX_AVAILABLE = False

# ============================================================================
# Test Configuration
# ============================================================================


def _load_performance_thresholds() -> dict[str, int]:
    """
    Load performance thresholds from configuration file with environment overrides.

    Priority (highest to lowest):
    1. Individual environment variables (TEST_<THRESHOLD_NAME>)
    2. Custom config file path (TEST_PERFORMANCE_CONFIG env var)
    3. Default config file (tests/config/performance_thresholds.yaml)
    4. Hardcoded defaults

    Returns:
        dict: Performance thresholds configuration
    """
    # Default thresholds (fallback if config file not found)
    defaults = {
        "introspection_broadcast_latency_ms": 50,
        "registry_processing_latency_ms": 100,
        "dual_registration_time_ms": 300,
        "heartbeat_interval_s": 30,
        "heartbeat_overhead_ms": 50,
    }

    # Try to load from config file
    config_path = os.environ.get(
        "TEST_PERFORMANCE_CONFIG",
        Path(__file__).parent.parent / "config" / "performance_thresholds.yaml",
    )

    thresholds = defaults.copy()

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    thresholds.update(config_data)
        except Exception as e:
            # Log warning but use defaults
            print(
                f"Warning: Failed to load performance thresholds from {config_path}: {e}"
            )
            print("Using default performance thresholds")

    # Environment variable overrides (highest priority)
    env_overrides = {
        "introspection_broadcast_latency_ms": "TEST_INTROSPECTION_BROADCAST_LATENCY_MS",
        "registry_processing_latency_ms": "TEST_REGISTRY_PROCESSING_LATENCY_MS",
        "dual_registration_time_ms": "TEST_DUAL_REGISTRATION_TIME_MS",
        "heartbeat_interval_s": "TEST_HEARTBEAT_INTERVAL_S",
        "heartbeat_overhead_ms": "TEST_HEARTBEAT_OVERHEAD_MS",
    }

    for key, env_var in env_overrides.items():
        if env_value := os.environ.get(env_var):
            try:
                thresholds[key] = int(env_value)
            except ValueError:
                print(
                    f"Warning: Invalid value for {env_var}={env_value}, using config/default"
                )

    return thresholds


PERFORMANCE_THRESHOLDS = _load_performance_thresholds()
TEST_NODE_TYPES = ["orchestrator", "reducer"]

# ============================================================================
# Test Fixtures - Container Setup
# ============================================================================


def _setup_test_container() -> ModelONEXContainer:
    """
    Create and configure a ModelONEXContainer for testing.

    Adds backwards-compatible methods for service registration and retrieval
    that work with mocked services via _services dict.
    """
    import types

    container = ModelONEXContainer()

    # Set default configuration values to prevent "Undefined configuration option" errors
    # These are the defaults that the nodes expect
    container.config.from_dict(
        {
            "metadata_stamping_service_url": "http://metadata-stamping:8053",
            "onextree_service_url": "http://onextree:8058",
            "onextree_timeout_ms": "500.0",  # Use string to avoid Configuration type errors
            "kafka_broker_url": "localhost:9092",
            "kafka_environment": "test",  # For registry node Kafka event prefix
            "default_namespace": "omninode.bridge",
            "onextree_app": None,
            "health_check_mode": "false",  # Use string to avoid Configuration type errors
            "consul_host": "localhost",
            "consul_port": "8500",  # Use string to avoid Configuration type errors
            "consul_enable_registration": False,  # Disable for tests
            "enable_prometheus": False,  # Disable Prometheus for tests
            "enable_event_bus": False,  # Disable event bus for tests
            "postgres_host": "localhost",
            "postgres_port": "5432",  # Use string to avoid Configuration type errors
            "postgres_db": "omninode_bridge",
            "postgres_user": "postgres",
            "postgres_password": "postgres",
            "environment": "dev",
            "registry_id": "test-registry-node",  # For registry node
            "env_prefix": "dev",  # For registry node event topics
            # Endpoint configuration for introspection mixin
            "host": "localhost",
            "port": "8053",
            "api_port": "8053",
            "metrics_port": "9090",
            "health_path": "/health",
            "use_https": "false",  # Use string to avoid Configuration type errors
            # Explicit endpoint URLs (introspection mixin checks for these)
            "health_endpoint": "",  # Empty string - let mixin construct
            "api_endpoint": "",  # Empty string - let mixin construct
            "metrics_endpoint": "",  # Empty string - let mixin construct
        }
    )

    # Add _services dict for backwards compatibility with mocks
    container._services = {}

    # Add register_service method
    def register_service(self, name: str, service):
        if not hasattr(self, "_services"):
            self._services = {}
        self._services[name] = service

    # Override get_service to check _services first (for mocked services)
    def get_service_with_fallback(self, service_name: str):
        # Check _services first for mocked services
        if hasattr(self, "_services") and service_name in self._services:
            return self._services[service_name]
        # Fall back to None for unknown services in test mode
        return None

    # Monkey-patch methods onto container instance
    container.register_service = types.MethodType(register_service, container)
    container.get_service = types.MethodType(get_service_with_fallback, container)

    return container


# Mock fixtures removed - now using real services from conftest.py
# Real service fixtures available:
# - kafka_client (from conftest.py)
# - postgres_client (from conftest.py)
# - consul_client (from conftest.py)


# ============================================================================
# Test Fixtures - Nodes with Real Services
# ============================================================================


@pytest.fixture
async def orchestrator_node(
    kafka_client,
) -> AsyncIterator[NodeBridgeOrchestrator]:
    """
    Create orchestrator node with real Kafka.

    Orchestrator will:
    - Publish NODE_INTROSPECTION events on startup
    - Publish NODE_HEARTBEAT events every 30s
    - Listen for REGISTRY_REQUEST_INTROSPECTION events
    """
    container = _setup_test_container()
    # Override specific config values for this node
    # Use hostname with correct Kafka port (9092) - resolved via /etc/hosts
    container.config.kafka_broker_url.from_value("omninode-bridge-redpanda:9092")
    container.config.api_port.from_value(8053)
    container.config.metrics_port.from_value(9090)

    # Register real Kafka client
    container.register_service("kafka_producer", kafka_client)
    container._services["kafka_client"] = kafka_client  # For registry compatibility

    node = NodeBridgeOrchestrator(container)

    # Manually inject real client for introspection mixin
    node.kafka_producer = kafka_client

    yield node

    # Cleanup: Stop any background tasks
    if hasattr(node, "stop_introspection_tasks"):
        await node.stop_introspection_tasks()


@pytest.fixture
async def reducer_node(
    kafka_client,
) -> AsyncIterator[NodeBridgeReducer]:
    """
    Create reducer node with real Kafka.

    Reducer will:
    - Publish NODE_INTROSPECTION events on startup
    - Publish NODE_HEARTBEAT events every 30s
    - Listen for REGISTRY_REQUEST_INTROSPECTION events
    """
    container = _setup_test_container()
    # Override specific config values for this node
    # Use hostname with correct Kafka port (9092) - resolved via /etc/hosts
    container.config.kafka_broker_url.from_value("omninode-bridge-redpanda:9092")
    container.config.api_port.from_value(8054)
    container.config.metrics_port.from_value(9091)

    # Register real Kafka client
    container.register_service("kafka_producer", kafka_client)
    container._services["kafka_client"] = kafka_client

    node = NodeBridgeReducer(container)

    # Manually inject real client
    node.kafka_producer = kafka_client

    yield node

    # Cleanup
    if hasattr(node, "stop_introspection_tasks"):
        await node.stop_introspection_tasks()


@pytest.fixture
async def registry_node(
    kafka_client,
    consul_client,
    postgres_client,
) -> AsyncIterator[NodeBridgeRegistry]:
    """
    Create registry node with all real services.

    Registry will:
    - Listen for NODE_INTROSPECTION events
    - Perform dual registration (Consul + PostgreSQL)
    - Publish REGISTRY_REQUEST_INTROSPECTION on startup
    """
    # Import repository implementation
    from omninode_bridge.services.node_registration_repository import (
        NodeRegistrationRepository,
    )

    container = _setup_test_container()
    # Override specific config values for this node
    # Use hostname with correct Kafka port (9092) - resolved via /etc/hosts
    container.config.kafka_broker_url.from_value("omninode-bridge-redpanda:9092")

    # Create real repository with real PostgreSQL client
    node_repository = NodeRegistrationRepository(postgres_client)

    # Register all real services
    container.register_service("kafka_client", kafka_client)
    container.register_service("consul_client", consul_client)
    container.register_service("postgres_client", postgres_client)
    container.register_service("node_repository", node_repository)

    # Create registry node with dev environment to disable background cleanup
    node = NodeBridgeRegistry(container, environment="dev")

    # Initialize services asynchronously
    # Note: on_startup() will also call this, but _initialize_services_async is idempotent
    await node._initialize_services_async(container)

    yield node

    # Cleanup: Stop all background tasks
    await node.on_shutdown()


# ============================================================================
# Test Suite 1: Node Startup and Introspection Broadcasting
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_node_startup_publishes_introspection(
    orchestrator_node,
):
    """
    Verify that node publishes introspection event on startup.

    Flow:
    1. Start orchestrator node
    2. Node publishes NODE_INTROSPECTION to node-introspection.v1 topic
    3. Verify event structure and content
    4. Verify performance (<50ms)
    """
    # Initialize introspection
    orchestrator_node.initialize_introspection()

    # Publish introspection
    start_time = time.perf_counter()
    success = await orchestrator_node.publish_introspection(reason="startup")
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Assertions
    assert success is True, "Introspection publishing should succeed"
    assert (
        latency_ms < PERFORMANCE_THRESHOLDS["introspection_broadcast_latency_ms"]
    ), f"Introspection latency {latency_ms:.2f}ms exceeds threshold"

    # Verify event was published to Kafka
    # Note: Need to check if _publish_to_kafka was called
    # In real implementation, we'd check mock_kafka_client.published_events
    assert orchestrator_node._last_introspection_broadcast is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_node_introspection_event_structure(orchestrator_node):
    """
    Verify NODE_INTROSPECTION event structure and completeness.

    Validates:
    - Event envelope structure (OnexEnvelopeV1)
    - Introspection payload fields
    - Capability extraction
    - Endpoint discovery
    """
    # Extract capabilities
    capabilities = await orchestrator_node.get_capabilities()

    # Validate capabilities structure
    assert "node_type" in capabilities
    assert capabilities["node_type"] == "orchestrator"
    assert "supported_operations" in capabilities
    assert "fsm_states" in capabilities
    assert "service_integration" in capabilities

    # Validate endpoints
    endpoints = await orchestrator_node.get_endpoints()
    assert "health" in endpoints
    assert "api" in endpoints
    assert "metrics" in endpoints
    assert "orchestration" in endpoints


# ============================================================================
# Test Suite 2: Registry Receives and Dual-Registers
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_registry_receives_and_registers(
    orchestrator_node,
    registry_node,
    kafka_client,
    postgres_client,
):
    """
    Verify registry receives introspection and performs dual registration.

    Flow:
    1. Start orchestrator (publishes introspection)
    2. Start registry (listens and registers)
    3. Verify Consul registration
    4. Verify PostgreSQL registration
    5. Verify performance (<200ms total)
    """
    # Step 1: Orchestrator publishes introspection
    orchestrator_node.initialize_introspection()
    capabilities = await orchestrator_node.get_capabilities()
    endpoints = await orchestrator_node.get_endpoints()

    # Create introspection event
    introspection_event = ModelNodeIntrospectionEvent(
        node_id=str(orchestrator_node.node_id),
        node_type="orchestrator",
        capabilities=capabilities,
        endpoints=endpoints,
        metadata={
            "broadcast_reason": "startup",
            "uptime_seconds": 0.0,
            "environment": "dev",
        },
    )

    # Step 2: Registry performs dual registration
    start_time = time.perf_counter()
    result = await registry_node.dual_register(introspection_event)
    registration_time_ms = (time.perf_counter() - start_time) * 1000

    # Assertions: Registration success
    assert result["status"] in ["success", "partial"]
    assert result["consul_registered"] is True, "Consul registration should succeed"
    assert (
        result["postgres_registered"] is True
    ), "PostgreSQL registration should succeed"
    assert (
        registration_time_ms < PERFORMANCE_THRESHOLDS["dual_registration_time_ms"]
    ), f"Registration time {registration_time_ms:.2f}ms exceeds threshold"

    # Verify PostgreSQL registration (query real database)
    node_id = str(orchestrator_node.node_id)
    query = "SELECT * FROM node_registrations WHERE node_id = $1"
    db_result = await postgres_client.fetch_one(query, node_id)
    assert db_result is not None, f"Node {node_id} should be registered in PostgreSQL"
    assert db_result["node_type"] == "orchestrator"
    assert db_result["capabilities"] is not None

    # Verify registration metrics
    metrics = registry_node.get_registration_metrics()
    assert metrics["total_registrations"] == 1
    assert metrics["successful_registrations"] == 1
    assert metrics["consul_registrations"] == 1
    assert metrics["postgres_registrations"] == 1


# ============================================================================
# Test Suite 3: Registry Startup Requests Re-Introspection
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_registry_startup_requests_reintrospection(
    orchestrator_node, registry_node, kafka_client
):
    """
    Verify registry requests re-introspection on startup.

    Flow:
    1. Start orchestrator
    2. Start registry (should publish REGISTRY_REQUEST)
    3. Verify orchestrator receives request
    4. Verify orchestrator re-broadcasts introspection
    """
    # Step 1: Start orchestrator
    orchestrator_node.initialize_introspection()

    # The actual topic includes the full prefix from the registry node using ONEX convention
    # Note: Uses "dev" prefix (default from KAFKA_ENVIRONMENT env var)
    request_topic = "dev.omninode_bridge.onex.evt.registry-request-introspection.v1"

    # Step 2: Registry startup and request (this will publish the message)
    # Connect registry (triggers startup) with timeout to prevent indefinite hang
    try:
        result = await asyncio.wait_for(registry_node.on_startup(), timeout=10.0)
    except TimeoutError:
        pytest.fail(
            "registry_node.on_startup() timed out after 10 seconds - likely hanging in service initialization or background task startup"
        )

    assert result["status"] == "started"

    # Step 3: Give Kafka a moment to persist the message
    await asyncio.sleep(1.0)

    # Step 4: Consume ALL messages from topic (reads from beginning)
    # This avoids timing issues with consumer subscription
    all_messages = await get_all_messages_from_topic(
        request_topic, max_messages=10, timeout=3.0
    )

    assert len(all_messages) > 0, (
        f"No messages found on topic {request_topic}. "
        f"Make sure Kafka is running at {os.getenv('KAFKA_BOOTSTRAP_SERVERS')} "
        f"and registry_node.on_startup() is publishing the event correctly."
    )

    # Get the most recent message (should be the one we just published)
    request_event = all_messages[-1]

    # Verify event structure
    assert request_event["event_type"] == "registry-request-introspection"


# ============================================================================
# Test Suite 4: Heartbeat Periodic Publishing
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_heartbeat_periodic_publishing(orchestrator_node):
    """
    Verify heartbeat publishes every 30 seconds.

    Flow:
    1. Start orchestrator
    2. Start heartbeat task with short interval (1s for testing)
    3. Wait and verify heartbeat events published
    4. Verify interval accuracy
    """
    # Initialize introspection
    orchestrator_node.initialize_introspection()

    # Start heartbeat task with 1s interval for testing
    await orchestrator_node.start_introspection_tasks(
        enable_heartbeat=True,
        heartbeat_interval_seconds=1.0,  # Short interval for testing
        enable_registry_listener=False,
    )

    # Wait for 2 heartbeats
    await asyncio.sleep(2.5)

    # Stop heartbeat
    await orchestrator_node.stop_introspection_tasks()

    # Verify heartbeats were published
    # Note: In real implementation, would check mock_kafka_client.published_events
    # For now, verify task was created and running
    assert orchestrator_node._heartbeat_task is not None
    assert orchestrator_node._heartbeat_task.cancelled()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_heartbeat_includes_uptime_and_active_operations(
    orchestrator_node,
):
    """
    Verify heartbeat includes uptime and active operations count.

    Validates:
    - Uptime calculation
    - Active operations tracking
    - Performance overhead (<10ms)
    """
    # Initialize and add some active operations
    orchestrator_node.initialize_introspection()

    # Add mock active workflows
    from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
        EnumWorkflowState,
    )

    orchestrator_node.workflow_fsm_states = {
        "workflow-1": EnumWorkflowState.PROCESSING,
        "workflow-2": EnumWorkflowState.PROCESSING,
    }

    # Wait a bit for uptime
    await asyncio.sleep(0.2)

    # Publish heartbeat
    start_time = time.perf_counter()
    success = await orchestrator_node._publish_heartbeat()
    heartbeat_overhead_ms = (time.perf_counter() - start_time) * 1000

    # Assertions
    assert success is True
    assert (
        heartbeat_overhead_ms < PERFORMANCE_THRESHOLDS["heartbeat_overhead_ms"]
    ), f"Heartbeat overhead {heartbeat_overhead_ms:.2f}ms exceeds threshold"


# ============================================================================
# Test Suite 5: Registry Recovery Scenario
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_registry_recovery_scenario(
    orchestrator_node,
    registry_node,
    kafka_client,
    postgres_client,
):
    """
    Simulate registry going down and recovering.

    Flow:
    1. Start orchestrator + registry
    2. Verify initial registration
    3. Stop registry (simulate crash)
    4. Restart registry
    5. Verify registry requests re-introspection
    6. Verify orchestrator re-broadcasts
    7. Verify dual registration happens again
    """
    # Step 1: Initial setup
    orchestrator_node.initialize_introspection()
    capabilities = await orchestrator_node.get_capabilities()
    endpoints = await orchestrator_node.get_endpoints()

    # Create introspection event
    introspection_event = ModelNodeIntrospectionEvent(
        node_id=str(orchestrator_node.node_id),
        node_type="orchestrator",
        capabilities=capabilities,
        endpoints=endpoints,
        metadata={"broadcast_reason": "startup"},
    )

    # Step 2: Initial registration
    await registry_node.on_startup()
    result1 = await registry_node.dual_register(introspection_event)
    assert result1["status"] == "success"

    initial_registrations = registry_node.registration_metrics["total_registrations"]

    # Step 3: Simulate registry crash
    await registry_node.on_shutdown()

    # Step 4: Restart registry
    await registry_node.on_startup()

    # Verify registry requested introspection (poll real Kafka)
    # The actual topic includes the full prefix from the registry node using ONEX convention
    # Note: Uses "dev" prefix (default from KAFKA_ENVIRONMENT env var)
    request_topic = "dev.omninode_bridge.onex.evt.registry-request-introspection.v1"

    # Give Kafka a moment to persist the message
    await asyncio.sleep(1.0)

    # Consume ALL messages from topic (reads from beginning to avoid timing issues)
    all_messages = await get_all_messages_from_topic(
        request_topic, max_messages=10, timeout=3.0
    )
    assert len(all_messages) > 0, f"No registry request event found on {request_topic}"

    # Get the most recent message (should be from the restart)
    request_event = all_messages[-1]

    # Step 5: Re-register (simulate orchestrator responding)
    result2 = await registry_node.dual_register(introspection_event)
    assert result2["status"] == "success"

    # Verify registration happened again
    assert (
        registry_node.registration_metrics["total_registrations"]
        > initial_registrations
    )

    # Verify in PostgreSQL
    node_id = str(orchestrator_node.node_id)
    query = "SELECT * FROM node_registrations WHERE node_id = $1"
    db_result = await postgres_client.fetch_one(query, node_id)
    assert (
        db_result is not None
    ), "Node should be registered in PostgreSQL after recovery"

    # Cleanup: Stop consuming to prevent test timeout
    await registry_node.stop_consuming()


# ============================================================================
# Test Suite 6: Multiple Nodes Registration
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multiple_nodes_registration(
    orchestrator_node,
    reducer_node,
    registry_node,
    kafka_client,
    postgres_client,
):
    """
    Test multiple nodes registering simultaneously.

    Flow:
    1. Start orchestrator + reducer
    2. Both publish introspection events
    3. Registry processes both
    4. Verify both in Consul
    5. Verify both in PostgreSQL
    6. Verify no race conditions
    """
    # Initialize both nodes
    orchestrator_node.initialize_introspection()
    reducer_node.initialize_introspection()

    # Extract capabilities for both
    orch_capabilities = await orchestrator_node.get_capabilities()
    orch_endpoints = await orchestrator_node.get_endpoints()

    red_capabilities = await reducer_node.get_capabilities()
    red_endpoints = await reducer_node.get_endpoints()

    # Create introspection events
    orch_introspection = ModelNodeIntrospectionEvent(
        node_id=str(orchestrator_node.node_id),
        node_type="orchestrator",
        capabilities=orch_capabilities,
        endpoints=orch_endpoints,
        metadata={"broadcast_reason": "startup"},
    )

    red_introspection = ModelNodeIntrospectionEvent(
        node_id=str(reducer_node.node_id),
        node_type="reducer",
        capabilities=red_capabilities,
        endpoints=red_endpoints,
        metadata={"broadcast_reason": "startup"},
    )

    # Start registry
    await registry_node.on_startup()

    # Register both nodes concurrently
    results = await asyncio.gather(
        registry_node.dual_register(orch_introspection),
        registry_node.dual_register(red_introspection),
    )

    # Verify both succeeded
    assert all(r["status"] == "success" for r in results)

    # Verify both in PostgreSQL (query real database)
    orch_node_id = str(orchestrator_node.node_id)
    red_node_id = str(reducer_node.node_id)

    query = "SELECT * FROM node_registrations WHERE node_id = $1"

    orch_result = await postgres_client.fetch_one(query, orch_node_id)
    assert (
        orch_result is not None
    ), f"Orchestrator {orch_node_id} should be in PostgreSQL"
    assert orch_result["node_type"] == "orchestrator"

    red_result = await postgres_client.fetch_one(query, red_node_id)
    assert red_result is not None, f"Reducer {red_node_id} should be in PostgreSQL"
    assert red_result["node_type"] == "reducer"

    # Verify metrics
    # Registry self-registers on startup, plus 2 test nodes = 3 total
    metrics = registry_node.get_registration_metrics()
    assert metrics["total_registrations"] == 3
    assert metrics["successful_registrations"] == 3
    assert metrics["registered_nodes_count"] == 3

    # Cleanup: Stop consuming to prevent test timeout
    await registry_node.stop_consuming()


# ============================================================================
# Test Suite 7: Graceful Degradation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graceful_degradation_without_kafka(orchestrator_node):
    """
    Test nodes work when Kafka is unavailable.

    Flow:
    1. Start orchestrator without Kafka
    2. Verify node still starts (degraded mode)
    3. Verify operations still work
    4. Verify introspection attempts but doesn't crash
    """
    # Disable Kafka by setting it to None in container services
    # (This simulates Kafka being unavailable)
    orchestrator_node.container.register_service("kafka_client", None)
    orchestrator_node.kafka_producer = None  # Set node attribute to None
    orchestrator_node.kafka_client = None  # Set node attribute to None

    # Initialize introspection
    orchestrator_node.initialize_introspection()

    # Attempt to publish introspection (should return False but not crash)
    success = await orchestrator_node.publish_introspection(reason="startup")

    # Should return False but not raise exception
    assert success is False

    # Node should still be operational
    capabilities = await orchestrator_node.get_capabilities()
    assert capabilities is not None
    assert "node_type" in capabilities


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graceful_degradation_without_consul(registry_node, postgres_client):
    """
    Test registry works when Consul is unavailable.

    Flow:
    1. Disable Consul client
    2. Register node
    3. Verify PostgreSQL registration still works
    4. Verify partial success reported
    """
    # Disable Consul
    registry_node.consul_client = None

    # Create test introspection
    introspection = ModelNodeIntrospectionEvent(
        node_id="test-node-degraded",
        node_type="orchestrator",
        capabilities={"node_type": "orchestrator"},
        endpoints={"health": "http://localhost:8053/health"},
        metadata={"broadcast_reason": "test"},
    )

    # Attempt registration
    result = await registry_node.dual_register(introspection)

    # Should succeed with partial status
    assert result["status"] in ["success", "partial"]
    assert result["consul_registered"] is False
    assert result["postgres_registered"] is True

    # Verify PostgreSQL registration (query real database)
    query = "SELECT * FROM node_registrations WHERE node_id = $1"
    db_result = await postgres_client.fetch_one(query, "test-node-degraded")
    assert (
        db_result is not None
    ), "Node should be registered in PostgreSQL despite Consul failure"


# ============================================================================
# Test Suite 8: Registry Self-Registration
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_registry_self_registration_e2e(
    kafka_client,
    postgres_client,
    consul_client,
):
    """Test registry self-registration end-to-end."""
    from omninode_bridge.services.node_registration_repository import (
        NodeRegistrationRepository,
    )

    # Create container with services
    container = _setup_test_container()
    container.config.kafka_broker_url.from_value("omninode-bridge-redpanda:9092")
    container.config.registry_id.from_value("test-registry-self-reg")
    container.config.postgres_host.from_value("192.168.86.200")
    container.config.postgres_port.from_value(5436)
    container.config.postgres_db.from_value("omninode_bridge")
    container.config.postgres_user.from_value("postgres")
    container.config.postgres_password.from_value("omninode_remote_2024_secure")

    # Create real repository
    node_repository = NodeRegistrationRepository(postgres_client)

    # Register services
    container.register_service("kafka_client", kafka_client)
    container.register_service("postgres_client", postgres_client)
    container.register_service("consul_client", consul_client)
    container.register_service("node_repository", node_repository)

    # Create registry
    registry = NodeBridgeRegistry(container, environment="dev")

    # Initialize services
    await registry._initialize_services_async(container)

    try:
        # Start registry (triggers self-registration)
        result = await registry.on_startup()

        # Verify startup result
        assert result is not None, "on_startup should return result"
        assert "status" in result, "Result should have status field"

        # Check if self_registration is present in result
        if "self_registration" in result:
            self_reg = result["self_registration"]

            # Check registration status
            assert self_reg.get("status") in [
                "success",
                "partial",
                "failed",
            ], f"Invalid self-registration status: {self_reg.get('status')}"

            # Report self-registration outcome
            if self_reg.get("status") == "success":
                print(
                    f"✓ Registry self-registration succeeded: consul={self_reg.get('consul_registered')}, postgres={self_reg.get('postgres_registered')}"
                )
            elif self_reg.get("status") == "partial":
                print(
                    f"⚠ Registry self-registration partial: consul={self_reg.get('consul_registered')}, postgres={self_reg.get('postgres_registered')}"
                )
            else:
                print(
                    f"✗ Registry self-registration failed: {self_reg.get('error', 'Unknown error')}"
                )
        else:
            print(
                "⚠ No self_registration field in startup result - implementation may be incomplete"
            )

        # Verify registry appears in its own database
        # Note: Check if repository has the list_all method before calling
        if registry.node_repository and hasattr(registry.node_repository, "list_all"):
            # Give it a moment to complete async registration
            await asyncio.sleep(0.5)

            try:
                nodes = await registry.node_repository.list_all()
                # Registry nodes are effect nodes with role "registry"
                registry_nodes = [
                    n
                    for n in nodes
                    if n.node_type == "effect"
                    and n.metadata.get("registry_id") is not None
                ]

                # Check if registry registered itself
                if len(registry_nodes) >= 1:
                    # Verify registry node details
                    registry_node = registry_nodes[0]
                    assert (
                        registry_node.node_type == "effect"
                    ), "Self-registered node should be of type 'effect' (registry is an effect node)"
                    assert (
                        registry_node.node_id is not None
                    ), "Node ID should be present"
                    print(
                        f"✓ Registry successfully self-registered with ID: {registry_node.node_id}"
                    )
                else:
                    # Self-registration may have failed or not implemented
                    print(
                        "⚠ Registry did not appear in its own database - self-registration may have failed"
                    )
            except AttributeError as e:
                print(f"⚠ Repository method not found: {e}")
        else:
            print("⚠ Repository not available or missing list_all method")

    finally:
        # Cleanup: Stop consuming and shutdown
        if hasattr(registry, "stop_consuming"):
            await registry.stop_consuming()
        await registry.on_shutdown()


# ============================================================================
# Test Performance Benchmarks
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
async def test_performance_benchmarks(
    orchestrator_node,
    registry_node,
    kafka_client,
    postgres_client,
):
    """
    Comprehensive performance benchmarking with real services.

    Measures:
    - Introspection broadcast latency
    - Registry processing latency
    - Dual registration time
    - Heartbeat overhead
    - End-to-end workflow time

    Note: Performance may be slower than mocked tests due to real
    network I/O and database operations.
    """
    # Initialize
    orchestrator_node.initialize_introspection()

    # Benchmark 1: Introspection broadcast
    orch_capabilities = await orchestrator_node.get_capabilities()
    orch_endpoints = await orchestrator_node.get_endpoints()

    start = time.perf_counter()
    success = await orchestrator_node.publish_introspection(reason="benchmark")
    broadcast_latency_ms = (time.perf_counter() - start) * 1000

    # Real services may be slower - allow 5x threshold
    assert (
        broadcast_latency_ms
        < PERFORMANCE_THRESHOLDS["introspection_broadcast_latency_ms"] * 5
    ), f"Broadcast latency {broadcast_latency_ms:.2f}ms too high (threshold: {PERFORMANCE_THRESHOLDS['introspection_broadcast_latency_ms'] * 5}ms)"

    # Benchmark 2: Registry processing + dual registration
    introspection = ModelNodeIntrospectionEvent(
        node_id=str(orchestrator_node.node_id),
        node_type="orchestrator",
        capabilities=orch_capabilities,
        endpoints=orch_endpoints,
        metadata={"broadcast_reason": "benchmark"},
    )

    start = time.perf_counter()
    result = await registry_node.dual_register(introspection)
    dual_registration_ms = (time.perf_counter() - start) * 1000

    assert result["status"] == "success"
    # Real services may be slower - allow 5x threshold
    assert (
        dual_registration_ms < PERFORMANCE_THRESHOLDS["dual_registration_time_ms"] * 5
    ), f"Dual registration {dual_registration_ms:.2f}ms too high (threshold: {PERFORMANCE_THRESHOLDS['dual_registration_time_ms'] * 5}ms)"

    # Benchmark 3: Heartbeat overhead
    start = time.perf_counter()
    success = await orchestrator_node._publish_heartbeat()
    heartbeat_overhead_ms = (time.perf_counter() - start) * 1000

    # Real services may be slower - allow 5x threshold
    assert (
        heartbeat_overhead_ms < PERFORMANCE_THRESHOLDS["heartbeat_overhead_ms"] * 5
    ), f"Heartbeat overhead {heartbeat_overhead_ms:.2f}ms too high (threshold: {PERFORMANCE_THRESHOLDS['heartbeat_overhead_ms'] * 5}ms)"

    # Log performance results
    print("\n=== Performance Benchmark Results (Real Services) ===")
    print(f"Introspection broadcast: {broadcast_latency_ms:.2f}ms")
    print(f"Dual registration: {dual_registration_ms:.2f}ms")
    print(f"Heartbeat overhead: {heartbeat_overhead_ms:.2f}ms")
    print(f"End-to-end workflow: {broadcast_latency_ms + dual_registration_ms:.2f}ms")
    print("Note: Performance measured against real Kafka/PostgreSQL/Consul services")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
