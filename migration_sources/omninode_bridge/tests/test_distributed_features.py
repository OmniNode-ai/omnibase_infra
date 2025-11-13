"""
Comprehensive tests for distributed features of MetadataStampingService Phase 3.

Tests database sharding, circuit breakers, OpenTelemetry integration,
and distributed architecture components.
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from redis import asyncio as redis_async

from metadata_stamping.distributed.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerStateManager,
    CircuitState,
    DistributedCircuitBreaker,
)

# Import our distributed components
from metadata_stamping.distributed.sharding import (
    ConsistentHashRing,
    DatabaseShardManager,
    ShardConfig,
    ShardHealthMonitor,
    ShardRouter,
    ShardStatus,
    load_shard_configs_from_env,
)
from metadata_stamping.telemetry.opentelemetry import (
    MetricsCollector,
    TelemetryConfig,
    TelemetryManager,
    TraceEnricher,
)

# Test fixtures and utilities


@pytest.fixture
def shard_configs():
    """Create test shard configurations."""
    return [
        ShardConfig(
            shard_id="shard_0",
            database_url="postgresql://test:test@localhost:5432/test_shard_0",
            weight=1.0,
            max_connections=10,
            min_connections=2,
            region="us-west-2",
            availability_zone="us-west-2a",
        ),
        ShardConfig(
            shard_id="shard_1",
            database_url="postgresql://test:test@localhost:5432/test_shard_1",
            weight=1.5,
            max_connections=15,
            min_connections=3,
            region="us-west-2",
            availability_zone="us-west-2b",
        ),
        ShardConfig(
            shard_id="shard_2",
            database_url="postgresql://test:test@localhost:5432/test_shard_2",
            weight=0.5,
            max_connections=8,
            min_connections=2,
            region="us-east-1",
            availability_zone="us-east-1a",
        ),
    ]


@pytest.fixture
async def mock_redis():
    """Create mock Redis client for testing."""
    redis_mock = AsyncMock(spec=redis_async.Redis)
    # Ensure methods are AsyncMock instances so they can be awaited
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock(return_value=None)
    return redis_mock


@pytest.fixture
def circuit_breaker_config():
    """Create test circuit breaker configuration."""
    return CircuitBreakerConfig(
        name="test_circuit",
        failure_threshold=3,
        recovery_timeout=10,
        half_open_max_calls=2,
        timeout=5.0,
    )


@pytest.fixture
def telemetry_config():
    """Create test telemetry configuration."""
    return TelemetryConfig(
        service_name="test-metadata-stamping",
        service_version="3.0.0-test",
        environment="test",
        prometheus_port=9091,
        enable_custom_metrics=True,
        enable_runtime_metrics=False,
    )


@pytest.fixture
def in_memory_tracer():
    """Create in-memory tracer for testing."""
    tracer_provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    span_processor = SimpleSpanProcessor(memory_exporter)
    tracer_provider.add_span_processor(span_processor)

    tracer = tracer_provider.get_tracer("test")
    return tracer, memory_exporter


# Database Sharding Tests


class TestConsistentHashRing:
    """Test consistent hash ring implementation."""

    def test_hash_ring_initialization(self):
        """Test hash ring initialization and basic operations."""
        ring = ConsistentHashRing(virtual_nodes=100)

        assert len(ring.ring) == 0
        assert len(ring.nodes) == 0

    def test_add_remove_nodes(self):
        """Test adding and removing nodes from hash ring."""
        ring = ConsistentHashRing(virtual_nodes=100)

        # Add nodes
        ring.add_node("shard_0", weight=1.0)
        ring.add_node("shard_1", weight=2.0)

        assert "shard_0" in ring.nodes
        assert "shard_1" in ring.nodes
        assert len(ring.ring) > 0

        # shard_1 should have more virtual nodes due to higher weight
        shard_0_count = sum(1 for node in ring.ring.values() if node == "shard_0")
        shard_1_count = sum(1 for node in ring.ring.values() if node == "shard_1")
        assert shard_1_count > shard_0_count

        # Remove node
        ring.remove_node("shard_0")
        assert "shard_0" not in ring.nodes
        assert all(node != "shard_0" for node in ring.ring.values())

    def test_key_distribution(self):
        """Test key distribution across nodes."""
        ring = ConsistentHashRing(virtual_nodes=150)
        ring.add_node("shard_0", weight=1.0)
        ring.add_node("shard_1", weight=1.0)
        ring.add_node("shard_2", weight=1.0)

        # Test key distribution
        distribution = {}
        for i in range(1000):
            key = f"test_key_{i}"
            node = ring.get_node(key)
            distribution[node] = distribution.get(node, 0) + 1

        # Check that distribution is reasonably balanced
        assert len(distribution) == 3
        for count in distribution.values():
            assert 250 <= count <= 450  # Roughly balanced

    def test_consistent_hashing(self):
        """Test that same key always maps to same node."""
        ring = ConsistentHashRing()
        ring.add_node("shard_0")
        ring.add_node("shard_1")

        test_key = "consistent_test_key"
        node1 = ring.get_node(test_key)
        node2 = ring.get_node(test_key)
        node3 = ring.get_node(test_key)

        assert node1 == node2 == node3

    def test_get_multiple_nodes(self):
        """Test getting multiple nodes for replication."""
        ring = ConsistentHashRing()
        ring.add_node("shard_0")
        ring.add_node("shard_1")
        ring.add_node("shard_2")

        nodes = ring.get_nodes("test_key", count=2)
        assert len(nodes) == 2
        assert len(set(nodes)) == 2  # Should be unique nodes


class TestShardRouter:
    """Test shard routing functionality."""

    def test_router_initialization(self, shard_configs):
        """Test shard router initialization."""
        router = ShardRouter(shard_configs)

        assert len(router.shards) == 3
        assert "shard_0" in router.shards
        assert "shard_1" in router.shards
        assert "shard_2" in router.shards

    def test_write_routing(self, shard_configs):
        """Test write operation routing."""
        router = ShardRouter(shard_configs)

        # Test that same key always routes to same shard
        test_key = "test_write_key"
        shard1 = router.route_write(test_key)
        shard2 = router.route_write(test_key)
        shard3 = router.route_write(test_key)

        assert shard1 == shard2 == shard3
        assert shard1 in ["shard_0", "shard_1", "shard_2"]

    def test_read_routing(self, shard_configs):
        """Test read operation routing."""
        router = ShardRouter(shard_configs)

        test_key = "test_read_key"
        shard = router.route_read(test_key, prefer_replica=False)

        assert shard in ["shard_0", "shard_1", "shard_2"]

    def test_shard_management(self, shard_configs):
        """Test adding and removing shards."""
        router = ShardRouter(shard_configs)

        # Remove a shard
        original_shards = router.get_all_shards(include_replicas=False)
        router.remove_shard("shard_1")

        # Verify routing still works with remaining shards
        test_key = "test_after_removal"
        shard = router.route_write(test_key)
        assert shard in ["shard_0", "shard_2"]

        # Add new shard
        new_shard = ShardConfig(
            shard_id="shard_3",
            database_url="postgresql://test:test@localhost:5432/test_shard_3",
            weight=1.0,
        )
        router.add_shard(new_shard)

        updated_shards = router.get_all_shards(include_replicas=False)
        assert "shard_3" in updated_shards


class TestShardHealthMonitor:
    """Test shard health monitoring."""

    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = ShardHealthMonitor(check_interval=5)

        assert monitor.check_interval == 5
        assert len(monitor.shard_metrics) == 0
        assert not monitor.running

    @pytest.mark.asyncio
    async def test_health_check_recording(self):
        """Test recording health check results."""
        monitor = ShardHealthMonitor()

        # Simulate successful health check
        monitor.health_history["shard_0"].append((time.time(), True))
        monitor.health_history["shard_0"].append((time.time(), True))
        monitor.health_history["shard_0"].append((time.time(), False))

        error_rate = monitor._calculate_error_rate("shard_0")
        assert 0.3 <= error_rate <= 0.4  # 1 failure out of 3

    @pytest.mark.asyncio
    async def test_healthy_shard_identification(self):
        """Test identification of healthy shards."""
        monitor = ShardHealthMonitor()

        # Mock some metrics
        from metadata_stamping.distributed.sharding import ShardMetrics

        monitor.shard_metrics["shard_0"] = ShardMetrics(
            shard_id="shard_0",
            status=ShardStatus.HEALTHY,
            connection_count=5,
            active_queries=2,
            avg_response_time=50.0,
            error_rate=0.01,
            last_health_check=time.time(),
            cpu_usage=25.0,
            memory_usage=512.0,
            disk_usage=10.0,
        )

        monitor.shard_metrics["shard_1"] = ShardMetrics(
            shard_id="shard_1",
            status=ShardStatus.UNHEALTHY,
            connection_count=0,
            active_queries=0,
            avg_response_time=0.0,
            error_rate=1.0,
            last_health_check=time.time(),
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
        )

        healthy_shards = monitor.get_healthy_shards()
        assert "shard_0" in healthy_shards
        assert "shard_1" not in healthy_shards

        assert monitor.is_shard_healthy("shard_0")
        assert not monitor.is_shard_healthy("shard_1")


@patch.dict(
    os.environ,
    {
        "SHARD_DATABASE_URLS": "shard_0=postgresql://user:pass@host:5432/db0\nshard_1=postgresql://user:pass@host:5432/db1",
        "SHARD_SHARD_0_WEIGHT": "1.5",
        "SHARD_SHARD_1_READ_ONLY": "true",
    },
)
class TestShardConfigLoading:
    """Test loading shard configurations from environment."""

    def test_load_configs_from_env(self):
        """Test loading shard configurations from environment variables."""
        configs = load_shard_configs_from_env()

        assert len(configs) == 2

        shard_0 = next(c for c in configs if c.shard_id == "shard_0")
        shard_1 = next(c for c in configs if c.shard_id == "shard_1")

        assert shard_0.weight == 1.5
        assert not shard_0.read_only

        assert shard_1.weight == 1.0  # Default
        assert shard_1.read_only


# Circuit Breaker Tests


class TestCircuitBreakerStateManager:
    """Test circuit breaker state management."""

    @pytest.mark.asyncio
    async def test_state_manager_initialization(self, mock_redis):
        """Test state manager initialization."""
        manager = CircuitBreakerStateManager(mock_redis)

        assert manager.redis == mock_redis
        assert manager.key_prefix == "cb:"

    @pytest.mark.asyncio
    async def test_get_default_state(self, mock_redis):
        """Test getting default state for new circuit breaker."""
        mock_redis.get.return_value = None

        manager = CircuitBreakerStateManager(mock_redis)
        state = await manager.get_state("test_circuit")

        assert state["state"] == CircuitState.CLOSED.value
        assert state["failure_count"] == 0
        assert state["success_count"] == 0

    @pytest.mark.asyncio
    async def test_record_success_and_failure(self, mock_redis, circuit_breaker_config):
        """Test recording successes and failures."""
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.CLOSED.value,
                "failure_count": 0,
                "success_count": 0,
                "total_calls": 0,
                "response_times": [],
            }
        )

        manager = CircuitBreakerStateManager(mock_redis)

        # Record success (requires config parameter)
        await manager.record_success("test_circuit", 0.5, circuit_breaker_config)

        # Record failure (requires config parameter)
        await manager.record_failure(
            "test_circuit", Exception("Test error"), circuit_breaker_config
        )

        # Verify calls were made to Redis
        assert mock_redis.setex.call_count >= 2

    @pytest.mark.asyncio
    async def test_state_transitions(self, mock_redis):
        """Test circuit breaker state transitions."""
        # Configure mock to return proper state data
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.CLOSED.value,
                "failure_count": 0,
                "success_count": 0,
                "total_calls": 0,
                "last_failure_time": None,
                "state_change_time": time.time(),
                "slow_calls": 0,
                "response_times": [],
                "half_open_calls": 0,
            }
        )

        manager = CircuitBreakerStateManager(mock_redis)

        # Test transition to open
        await manager.transition_state("test_circuit", CircuitState.OPEN)

        # Test transition to half-open
        await manager.transition_state("test_circuit", CircuitState.HALF_OPEN)

        # Test transition to closed
        await manager.transition_state("test_circuit", CircuitState.CLOSED)

        # Each transition calls setex once (3 total)
        assert mock_redis.setex.call_count >= 3

    @pytest.mark.asyncio
    async def test_should_allow_request(self, mock_redis, circuit_breaker_config):
        """Test request allowance logic."""
        # Mock closed state
        mock_redis.get.return_value = json.dumps(
            {"state": CircuitState.CLOSED.value, "state_change_time": time.time()}
        )

        manager = CircuitBreakerStateManager(mock_redis)

        # Should allow in closed state
        allowed = await manager.should_allow_request(
            "test_circuit", circuit_breaker_config
        )
        assert allowed

        # Clear local cache to force fetching new state from Redis
        manager.local_cache.clear()

        # Mock open state (recent)
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.OPEN.value,
                "state_change_time": time.time(),  # Recent
            }
        )

        allowed = await manager.should_allow_request(
            "test_circuit", circuit_breaker_config
        )
        assert not allowed

        # Clear local cache to force fetching new state from Redis
        manager.local_cache.clear()

        # Mock open state (old, should transition to half-open)
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.OPEN.value,
                "state_change_time": time.time() - 100,  # Old
            }
        )

        allowed = await manager.should_allow_request(
            "test_circuit", circuit_breaker_config
        )
        assert allowed  # Should transition to half-open and allow


class TestDistributedCircuitBreaker:
    """Test distributed circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(
        self, mock_redis, circuit_breaker_config
    ):
        """Test successful operation flow."""
        state_manager = CircuitBreakerStateManager(mock_redis)
        circuit_breaker = DistributedCircuitBreaker(
            circuit_breaker_config, state_manager
        )

        # Mock closed state with all required fields
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.CLOSED.value,
                "state_change_time": time.time(),
                "failure_count": 0,
                "success_count": 0,
                "total_calls": 0,
                "slow_calls": 0,
                "response_times": [],
                "half_open_calls": 0,
                "last_failure_time": None,
            }
        )

        # Test successful call
        async with circuit_breaker.call():
            # Simulate successful operation
            await asyncio.sleep(0.01)

        # Should have recorded success
        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_flow(
        self, mock_redis, circuit_breaker_config
    ):
        """Test failure operation flow."""
        state_manager = CircuitBreakerStateManager(mock_redis)
        circuit_breaker = DistributedCircuitBreaker(
            circuit_breaker_config, state_manager
        )

        # Mock closed state with all required fields
        mock_redis.get.return_value = json.dumps(
            {
                "state": CircuitState.CLOSED.value,
                "state_change_time": time.time(),
                "failure_count": 0,
                "success_count": 0,
                "total_calls": 0,
                "slow_calls": 0,
                "response_times": [],
                "half_open_calls": 0,
                "last_failure_time": None,
            }
        )

        # Test failed call
        with pytest.raises(ValueError):
            async with circuit_breaker.call():
                raise ValueError("Test failure")

        # Should have recorded failure
        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejection(
        self, mock_redis, circuit_breaker_config
    ):
        """Test that open circuit breaker rejects calls."""
        state_manager = CircuitBreakerStateManager(mock_redis)
        circuit_breaker = DistributedCircuitBreaker(
            circuit_breaker_config, state_manager
        )

        # Mock open state
        mock_redis.get.return_value = json.dumps(
            {"state": CircuitState.OPEN.value, "state_change_time": time.time()}
        )

        # Should reject call
        with pytest.raises(CircuitBreakerOpenError):
            async with circuit_breaker.call():
                pass

    @pytest.mark.asyncio
    async def test_get_circuit_state(self, mock_redis, circuit_breaker_config):
        """Test getting current circuit state."""
        state_manager = CircuitBreakerStateManager(mock_redis)
        circuit_breaker = DistributedCircuitBreaker(
            circuit_breaker_config, state_manager
        )

        # Mock half-open state
        mock_redis.get.return_value = json.dumps(
            {"state": CircuitState.HALF_OPEN.value, "state_change_time": time.time()}
        )

        state = await circuit_breaker.get_state()
        assert state == CircuitState.HALF_OPEN


class TestCircuitBreakerManager:
    """Test circuit breaker manager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, mock_redis):
        """Test circuit breaker manager initialization."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            manager = CircuitBreakerManager("redis://localhost:6379")
            await manager.initialize()

            assert manager._initialized
            assert manager.redis_client == mock_redis
            assert len(manager.default_configs) > 0

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self, mock_redis):
        """Test getting circuit breaker instances."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            manager = CircuitBreakerManager("redis://localhost:6379")
            await manager.initialize()

            # Get existing circuit breaker
            cb1 = manager.get_circuit_breaker("database")
            assert cb1.name == "database"

            # Should return same instance
            cb2 = manager.get_circuit_breaker("database")
            assert cb1 is cb2

            # Get custom circuit breaker
            custom_config = CircuitBreakerConfig(name="custom", failure_threshold=10)
            cb3 = manager.get_circuit_breaker("custom", custom_config)
            assert cb3.config.failure_threshold == 10

    @pytest.mark.asyncio
    async def test_health_check(self, mock_redis):
        """Test circuit breaker system health check."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            manager = CircuitBreakerManager("redis://localhost:6379")
            await manager.initialize()

            health = await manager.health_check()

            assert health["status"] == "healthy"
            assert "total_circuits" in health
            assert "redis_connected" in health


# OpenTelemetry Integration Tests


class TestMetricsCollector:
    """Test custom metrics collection."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        # Mock meter provider
        mock_meter_provider = MagicMock()
        mock_meter = MagicMock()
        mock_meter_provider.get_meter.return_value = mock_meter

        collector = MetricsCollector(mock_meter_provider)

        assert collector.meter == mock_meter
        assert mock_meter.create_histogram.called
        assert mock_meter.create_counter.called

    def test_record_hash_operation(self):
        """Test recording hash operation metrics."""
        mock_meter_provider = MagicMock()
        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_gauge = MagicMock()

        mock_meter_provider.get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_up_down_counter.return_value = mock_gauge

        collector = MetricsCollector(mock_meter_provider)
        collector.record_hash_operation(0.001, 1024, "A")

        mock_histogram.record.assert_called()
        mock_counter.add.assert_called()

    def test_file_size_categorization(self):
        """Test file size categorization for metrics."""
        mock_meter_provider = MagicMock()
        mock_meter = MagicMock()
        mock_meter_provider.get_meter.return_value = mock_meter

        collector = MetricsCollector(mock_meter_provider)

        assert collector._get_file_size_category(500) == "small"
        assert collector._get_file_size_category(50000) == "medium"
        assert collector._get_file_size_category(5000000) == "large"
        assert collector._get_file_size_category(50000000) == "xlarge"


class TestTraceEnricher:
    """Test trace enrichment functionality."""

    def test_enrich_span_with_request_info(self, in_memory_tracer):
        """Test enriching spans with request information."""
        tracer, memory_exporter = in_memory_tracer

        with tracer.start_as_current_span("test_span") as span:
            request_data = {
                "method": "POST",
                "url": "/stamp",
                "user_agent": "test-agent",
                "content_length": 1024,
                "file_size": 2048,
                "file_type": "image/jpeg",
                "hash_algorithm": "blake3",
            }

            TraceEnricher.enrich_span_with_request_info(span, request_data)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.attributes["http.method"] == "POST"
        assert span.attributes["metadata_stamping.file_size"] == 2048
        assert span.attributes["metadata_stamping.hash_algorithm"] == "blake3"

    def test_enrich_span_with_database_info(self, in_memory_tracer):
        """Test enriching spans with database information."""
        tracer, memory_exporter = in_memory_tracer

        with tracer.start_as_current_span("db_operation") as span:
            TraceEnricher.enrich_span_with_database_info(span, "shard_1", "SELECT")

        spans = memory_exporter.get_finished_spans()
        span = spans[0]

        assert span.attributes["db.shard_id"] == "shard_1"
        assert span.attributes["db.operation"] == "SELECT"
        assert span.attributes["db.system"] == "postgresql"

    def test_enrich_span_with_performance_info(self, in_memory_tracer):
        """Test enriching spans with performance metrics."""
        tracer, memory_exporter = in_memory_tracer

        with tracer.start_as_current_span("performance_test") as span:
            performance_data = {
                "execution_time_ms": 1.5,
                "performance_grade": "A",
                "cpu_usage_percent": 25.0,
            }

            TraceEnricher.enrich_span_with_performance_info(span, performance_data)

        spans = memory_exporter.get_finished_spans()
        span = spans[0]

        assert span.attributes["metadata_stamping.execution_time_ms"] == 1.5
        assert span.attributes["metadata_stamping.performance_grade"] == "A"
        assert span.attributes["metadata_stamping.cpu_usage_percent"] == 25.0


class TestTelemetryManager:
    """Test telemetry manager."""

    def test_telemetry_manager_initialization(self, telemetry_config):
        """Test telemetry manager initialization."""
        with (
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.metrics.MeterProvider"),
            patch("prometheus_client.start_http_server"),
        ):

            manager = TelemetryManager(telemetry_config)
            manager.initialize()

            assert manager._initialized
            assert manager.config == telemetry_config

    def test_resource_creation(self, telemetry_config):
        """Test resource creation with metadata."""
        with patch.dict(
            os.environ,
            {
                "POD_NAME": "test-pod",
                "POD_NAMESPACE": "test-namespace",
                "NODE_NAME": "test-node",
                "POD_IP": "10.0.0.1",
            },
        ):
            manager = TelemetryManager(telemetry_config)
            resource = manager._create_resource()

            attributes = resource.attributes
            assert attributes["service.name"] == "test-metadata-stamping"
            assert attributes["service.version"] == "3.0.0-test"
            assert attributes["k8s.pod.name"] == "test-pod"
            assert attributes["k8s.namespace.name"] == "test-namespace"

    @patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "env-service",
            "OTEL_SERVICE_VERSION": "env-version",
            "PROMETHEUS_PORT": "9092",
        },
    )
    def test_config_loading_from_env(self):
        """Test loading configuration from environment variables."""
        from metadata_stamping.telemetry.opentelemetry import _load_config_from_env

        config = _load_config_from_env()

        assert config.service_name == "env-service"
        assert config.service_version == "env-version"
        assert config.prometheus_port == 9092


# Integration Tests


class TestDistributedIntegration:
    """Integration tests for distributed components."""

    @pytest.mark.asyncio
    async def test_sharding_with_circuit_breaker(self, shard_configs, mock_redis):
        """Test database sharding with circuit breaker protection."""
        # Mock database connections
        with patch("asyncpg.create_pool") as mock_create_pool:
            mock_pool = AsyncMock()
            mock_connection = AsyncMock()
            mock_connection.fetchval.return_value = 1
            mock_connection.fetch.return_value = [{"result": 1}]
            # Make pool.acquire() return an async context manager (not a coroutine!)
            # Use MagicMock for acquire() so it returns the context manager directly
            from unittest.mock import MagicMock

            mock_acquire_cm = AsyncMock()
            mock_acquire_cm.__aenter__.return_value = mock_connection
            mock_acquire_cm.__aexit__.return_value = None
            mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

            # asyncpg.create_pool is an async function, so we need to make it awaitable
            async def create_mock_pool(*args, **kwargs):
                return mock_pool

            mock_create_pool.side_effect = create_mock_pool

            # Initialize shard manager
            shard_manager = DatabaseShardManager(shard_configs)
            await shard_manager.initialize()

            # Initialize circuit breaker
            with patch("redis.asyncio.from_url", return_value=mock_redis):
                cb_manager = CircuitBreakerManager("redis://localhost:6379")
                await cb_manager.initialize()

                cb = cb_manager.get_circuit_breaker("database")

                # Test protected database operation
                test_key = "integration_test_key"
                shard_id = shard_manager.get_shard_for_key(test_key)

                async with cb.call():
                    result = await shard_manager.execute_on_shard(shard_id, "SELECT 1")

                assert mock_connection.fetch.called
                await shard_manager.close()

    @pytest.mark.asyncio
    async def test_telemetry_with_sharding(self, shard_configs, telemetry_config):
        """Test telemetry integration with sharding operations."""
        with (
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.metrics.MeterProvider"),
            patch("prometheus_client.start_http_server"),
            patch("asyncpg.create_pool") as mock_create_pool,
        ):

            mock_pool = AsyncMock()

            # asyncpg.create_pool is an async function, so make it awaitable
            async def create_mock_pool(*args, **kwargs):
                return mock_pool

            mock_create_pool.side_effect = create_mock_pool

            # Initialize telemetry
            telemetry_manager = TelemetryManager(telemetry_config)
            telemetry_manager.initialize()

            # Initialize shard manager
            shard_manager = DatabaseShardManager(shard_configs)
            await shard_manager.initialize()

            # Test traced database operation
            tracer = telemetry_manager.get_tracer("test")

            with tracer.start_as_current_span("database_operation") as span:
                test_key = "telemetry_test_key"
                shard_id = shard_manager.get_shard_for_key(test_key)

                TraceEnricher.enrich_span_with_database_info(span, shard_id, "INSERT")

            await shard_manager.close()

    @pytest.mark.asyncio
    async def test_end_to_end_distributed_flow(
        self, shard_configs, mock_redis, telemetry_config
    ):
        """Test complete end-to-end distributed flow."""
        with (
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.metrics.MeterProvider"),
            patch("prometheus_client.start_http_server"),
            patch("asyncpg.create_pool") as mock_create_pool,
            patch("redis.asyncio.from_url", return_value=mock_redis),
        ):

            mock_pool = AsyncMock()
            mock_connection = AsyncMock()
            mock_connection.fetchval.return_value = 1
            mock_connection.fetch.return_value = [{"id": 1, "hash": "test_hash"}]
            # Make pool.acquire() return an async context manager (not a coroutine!)
            # Use MagicMock for acquire() so it returns the context manager directly
            from unittest.mock import MagicMock

            mock_acquire_cm = AsyncMock()
            mock_acquire_cm.__aenter__.return_value = mock_connection
            mock_acquire_cm.__aexit__.return_value = None
            mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

            # asyncpg.create_pool is an async function, so make it awaitable
            async def create_mock_pool(*args, **kwargs):
                return mock_pool

            mock_create_pool.side_effect = create_mock_pool

            # Initialize all components
            telemetry_manager = TelemetryManager(telemetry_config)
            telemetry_manager.initialize()

            shard_manager = DatabaseShardManager(shard_configs)
            await shard_manager.initialize()

            cb_manager = CircuitBreakerManager("redis://localhost:6379")
            await cb_manager.initialize()

            # Simulate metadata stamping operation with all features
            tracer = telemetry_manager.get_tracer("integration_test")

            with tracer.start_as_current_span("metadata_stamp_operation") as span:
                # Enrich span with request info
                TraceEnricher.enrich_span_with_request_info(
                    span,
                    {
                        "method": "POST",
                        "url": "/stamp",
                        "file_size": 1024,
                        "file_type": "text/plain",
                    },
                )

                # Route to shard
                test_key = "end_to_end_test_key"
                shard_id = shard_manager.get_shard_for_key(test_key)

                # Execute with circuit breaker protection
                cb = cb_manager.get_circuit_breaker("database")

                async with cb.call():
                    result = await shard_manager.execute_write(
                        test_key,
                        "INSERT INTO metadata_stamps (file_hash, file_path) VALUES ($1, $2)",
                        "test_hash_123",
                        "/test/file.txt",
                    )

                # Record metrics
                if telemetry_manager.metrics_collector:
                    telemetry_manager.metrics_collector.record_stamp_request(
                        duration=0.05, status_code=200, operation="create"
                    )
                    telemetry_manager.metrics_collector.record_database_operation(
                        duration=0.01, shard_id=shard_id, operation="INSERT"
                    )

            # Verify operations completed
            assert mock_connection.fetch.called

            # Cleanup
            await shard_manager.close()
            await cb_manager.close()


# Performance and Load Tests


class TestDistributedPerformance:
    """Performance tests for distributed components."""

    @pytest.mark.asyncio
    async def test_shard_routing_performance(self, shard_configs):
        """Test performance of shard routing operations."""
        router = ShardRouter(shard_configs)

        # Warm up
        for i in range(100):
            router.route_write(f"warmup_key_{i}")

        # Performance test
        start_time = time.time()
        for i in range(10000):
            router.route_write(f"perf_key_{i}")
        end_time = time.time()

        total_time = end_time - start_time
        operations_per_second = 10000 / total_time

        # Should be able to route > 1k operations per second
        # Lower threshold acceptable in CI environments with limited resources
        assert operations_per_second > 1000
        print(f"Routing performance: {operations_per_second:.0f} ops/sec")

    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self, mock_redis, circuit_breaker_config):
        """Test circuit breaker performance overhead."""
        mock_redis.get.return_value = json.dumps(
            {"state": CircuitState.CLOSED.value, "state_change_time": time.time()}
        )

        state_manager = CircuitBreakerStateManager(mock_redis)
        circuit_breaker = DistributedCircuitBreaker(
            circuit_breaker_config, state_manager
        )

        # Test without circuit breaker
        start_time = time.time()
        for _ in range(1000):
            await asyncio.sleep(0.0001)  # Simulate 0.1ms operation
        baseline_time = time.time() - start_time

        # Test with circuit breaker
        start_time = time.time()
        for _ in range(1000):
            async with circuit_breaker.call():
                await asyncio.sleep(0.0001)
        protected_time = time.time() - start_time

        # Overhead should be minimal (< 100% increase)
        overhead = (protected_time - baseline_time) / baseline_time
        assert overhead < 1.0  # Less than 100% overhead
        print(f"Circuit breaker overhead: {overhead*100:.1f}%")


# Cleanup and utility functions


@pytest.fixture(autouse=True)
async def cleanup_global_state():
    """Clean up global state after each test."""
    yield

    # Reset global telemetry manager
    import metadata_stamping.telemetry.opentelemetry

    metadata_stamping.telemetry.opentelemetry._telemetry_manager = None

    # Reset global shard manager
    import metadata_stamping.distributed.sharding

    metadata_stamping.distributed.sharding._shard_manager = None

    # Reset global circuit breaker manager
    import metadata_stamping.distributed.circuit_breaker

    metadata_stamping.distributed.circuit_breaker._circuit_breaker_manager = None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
