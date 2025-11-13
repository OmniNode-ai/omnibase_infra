#!/usr/bin/env python3
"""Fixtures for reducer node unit tests."""

from unittest.mock import patch
from uuid import uuid4

import pytest

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
except ImportError:
    # Fallback to stub ModelContainer from node's _stubs module
    from omninode_bridge.nodes.reducer.v1_0_0._stubs import ModelONEXContainer

# Import ConfigLoader settings for mocking
from omninode_bridge.config.settings import (
    AggregationConfig,
    CacheNodeConfig,
    CircuitBreakerNodeConfig,
    ConsulNodeConfig,
    DatabaseNodeConfig,
    EventsAggregationConfig,
    KafkaConsumerConfig,
    KafkaNodeConfig,
    KafkaProducerConfig,
    KafkaTopicsConfig,
    LoggingNodeConfig,
    MetricsAggregationConfig,
    MonitoringNodeConfig,
    NodeConfig,
    ReducerConfig,
    ReducerSettings,
    SessionWindowConfig,
    SlidingWindowConfig,
    TumblingWindowConfig,
    WindowingConfig,
    WorkflowStateAggregationConfig,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_input_state import (
    ModelReducerInputState,
)
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer


@pytest.fixture
def mock_reducer_config() -> ReducerSettings:
    """Create a mock ReducerSettings for testing.

    This fixture provides a complete, valid ReducerSettings object
    that can be used to mock get_reducer_config() in tests.
    """
    return ReducerSettings(
        environment="test",
        node=NodeConfig(
            type="reducer",
            name="test-reducer",
            version="1.0.0",
            namespace="omninode.test.reducer",
        ),
        reducer=ReducerConfig(
            aggregation_window_seconds=60,
            aggregation_batch_size=100,
            max_aggregation_buffer_size=10000,
            enable_incremental_aggregation=True,
            state_persistence_interval_seconds=30,
            state_snapshot_interval_seconds=300,
            enable_state_compression=True,
            max_state_size_mb=100,
            batch_size=50,
            batch_timeout_seconds=5,
            max_batch_retry_attempts=3,
            batch_retry_delay_seconds=2,
            retain_aggregated_data_hours=24,
            retain_raw_data_hours=6,
            enable_automatic_cleanup=True,
            worker_pool_size=8,
            aggregation_worker_threads=4,
            io_worker_threads=4,
            event_processing_buffer_size=500,
        ),
        consul=ConsulNodeConfig(
            host="localhost",
            port=8500,
            enable_registration=False,
            registration_timeout_seconds=10,
            health_check_interval_seconds=30,
            health_check_timeout_seconds=5,
        ),
        database=DatabaseNodeConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password=None,
            pool_min_size=3,
            pool_max_size=15,
            pool_timeout_seconds=10,
            query_timeout_seconds=30,
            command_timeout_seconds=60,
        ),
        kafka=KafkaNodeConfig(
            bootstrap_servers="localhost:9092",
            producer=KafkaProducerConfig(
                compression_type="snappy",
                batch_size=16384,
                linger_ms=5,
                acks="all",
                max_in_flight_requests=5,
            ),
            consumer=KafkaConsumerConfig(
                group_id="test-reducer-group",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                max_poll_records=500,
            ),
            topics=KafkaTopicsConfig(
                workflow_events="test.workflow.events",
                task_events="test.task.events",
                aggregated_metrics="test.aggregated.metrics",
                state_snapshots="test.state.snapshots",
            ),
        ),
        aggregation=AggregationConfig(
            metrics=MetricsAggregationConfig(
                enabled=True,
                window_seconds=60,
                functions=["count", "sum", "avg", "min", "max"],
                dimensions=["node_type", "operation", "status"],
            ),
            events=EventsAggregationConfig(
                enabled=True,
                window_seconds=30,
                group_by=["event_type", "source"],
                count_threshold=100,
            ),
            workflow_state=WorkflowStateAggregationConfig(
                enabled=True,
                window_seconds=120,
                track_transitions=True,
                track_duration=True,
            ),
        ),
        logging=LoggingNodeConfig(
            level="INFO",
            format="json",
            enable_structured_logging=True,
            log_requests=True,
            log_responses=False,
        ),
        monitoring=MonitoringNodeConfig(
            enable_prometheus=True,
            prometheus_port=9091,
            metrics_interval_seconds=15,
            health_check_interval_seconds=30,
        ),
        circuit_breaker=CircuitBreakerNodeConfig(
            enabled=True,
            failure_threshold=5,
            recovery_timeout_seconds=60,
            half_open_max_requests=3,
        ),
        cache=CacheNodeConfig(
            enabled=True,
            aggregation_state_ttl_seconds=600,
            dimension_cache_ttl_seconds=1800,
            max_cache_size_mb=128,
        ),
        windowing=WindowingConfig(
            tumbling=TumblingWindowConfig(enabled=True, sizes=[60, 300, 3600]),
            sliding=SlidingWindowConfig(
                enabled=False, size_seconds=300, slide_seconds=60
            ),
            session=SessionWindowConfig(enabled=False, gap_seconds=300),
        ),
    )


@pytest.fixture
def mock_container() -> ModelONEXContainer:
    """Create a mock ONEX container for testing with health check mode enabled."""
    container = ModelONEXContainer()

    # Enable health check mode to skip Kafka and DB adapter initialization
    # Use from_dict() method for dependency_injector Configuration objects
    container.config.from_dict(
        {
            "health_check_mode": True,
            "default_namespace": "omninode.test",
            "kafka_broker_url": "localhost:9092",
        }
    )

    # Mock the get_service method to return None for services not available in health check mode
    original_get_service = container.get_service

    def mock_get_service(service_name: str):
        # Return None for services that aren't available in health check mode
        return None

    container.get_service = mock_get_service

    return container


@pytest.fixture
def reducer_node(
    mock_container: ModelONEXContainer, mock_reducer_config: ReducerSettings
) -> NodeBridgeReducer:
    """Create a NodeBridgeReducer instance for testing with mocked ConfigLoader.

    This fixture patches get_reducer_config to return mock_reducer_config,
    allowing tests to run without needing actual YAML config files.
    """
    # Patch get_reducer_config at the import location (config_loader module)
    with patch(
        "omninode_bridge.config.config_loader.get_reducer_config",
        return_value=mock_reducer_config,
    ):
        # Create node - ConfigLoader will use mocked config
        node = NodeBridgeReducer(container=mock_container)
        return node


@pytest.fixture
def sample_metadata() -> ModelReducerInputState:
    """Create sample stamp metadata for testing."""
    return ModelReducerInputState(
        stamp_id=str(uuid4()),
        file_hash="abc123def456789",
        file_path="/data/test/document.pdf",
        file_size=1024000,
        namespace="omninode.services.metadata",
        content_type="application/pdf",
        workflow_id=uuid4(),
        workflow_state="COMPLETED",
        processing_time_ms=1.5,
    )


@pytest.fixture
def sample_metadata_batch() -> list[ModelReducerInputState]:
    """Create a batch of sample metadata for testing."""
    workflow_id = uuid4()
    return [
        ModelReducerInputState(
            stamp_id=str(uuid4()),
            file_hash=f"hash_{i}",
            file_path=f"/data/test/file_{i}.txt",
            file_size=1024 * (i + 1),
            namespace=(
                "omninode.services.metadata"
                if i % 2 == 0
                else "omninode.services.onextree"
            ),
            content_type="text/plain" if i % 3 == 0 else "application/json",
            workflow_id=workflow_id,
            workflow_state="COMPLETED" if i % 2 == 0 else "PROCESSING",
            processing_time_ms=float(i),
        )
        for i in range(10)
    ]
