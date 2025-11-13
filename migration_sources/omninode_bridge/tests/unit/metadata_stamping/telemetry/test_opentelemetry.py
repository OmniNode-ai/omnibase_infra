"""Unit tests for OpenTelemetry telemetry module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from metadata_stamping.telemetry.opentelemetry import (
    MetricsCollector,
    TelemetryConfig,
    TelemetryManager,
    _load_config_from_env,
)


class TestTelemetryConfig:
    """Test TelemetryConfig dataclass."""

    def test_config_defaults(self):
        """Test TelemetryConfig has correct default values."""
        config = TelemetryConfig()

        assert config.service_name == "metadata-stamping"
        assert config.service_version == "3.0.0"
        assert config.environment == "production"
        assert config.otlp_endpoint is None
        assert config.otlp_insecure is False
        assert config.prometheus_port == 9090
        assert config.prometheus_endpoint == "/metrics"
        assert config.sampling_ratio == 1.0
        assert config.cluster_name == "metadata-stamping-cluster"
        assert config.namespace == "metadata-stamping"
        assert config.region == "us-west-2"
        assert config.availability_zone == "us-west-2a"

    def test_config_custom_values(self):
        """Test TelemetryConfig accepts custom values."""
        config = TelemetryConfig(
            service_name="custom-service",
            service_version="2.0.0",
            environment="development",
            otlp_endpoint="http://localhost:4317",
            otlp_insecure=True,
            prometheus_port=9091,
            cluster_name="custom-cluster",
            namespace="custom-namespace",
            region="us-east-1",
            availability_zone="us-east-1a",
        )

        assert config.service_name == "custom-service"
        assert config.service_version == "2.0.0"
        assert config.environment == "development"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.otlp_insecure is True
        assert config.prometheus_port == 9091
        assert config.cluster_name == "custom-cluster"
        assert config.namespace == "custom-namespace"
        assert config.region == "us-east-1"
        assert config.availability_zone == "us-east-1a"

    def test_config_instrumentation_flags(self):
        """Test instrumentation configuration flags."""
        config = TelemetryConfig(
            instrument_asyncpg=False,
            instrument_redis=False,
            instrument_http_client=False,
            instrument_fastapi=False,
            instrument_logging=False,
        )

        assert config.instrument_asyncpg is False
        assert config.instrument_redis is False
        assert config.instrument_http_client is False
        assert config.instrument_fastapi is False
        assert config.instrument_logging is False

    def test_config_metrics_flags(self):
        """Test metrics configuration flags."""
        config = TelemetryConfig(
            enable_custom_metrics=False,
            enable_runtime_metrics=False,
            metrics_export_interval=60,
        )

        assert config.enable_custom_metrics is False
        assert config.enable_runtime_metrics is False
        assert config.metrics_export_interval == 60


class TestLoadConfigFromEnv:
    """Test _load_config_from_env function."""

    def test_load_config_with_defaults(self):
        """Test loading config with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_env()

            assert config.service_name == "metadata-stamping"
            assert config.service_version == "3.0.0"
            assert config.environment == "production"
            assert config.otlp_endpoint is None
            assert config.prometheus_port == 9090

    def test_load_config_with_env_vars(self):
        """Test loading config from environment variables."""
        env_vars = {
            "OTEL_SERVICE_NAME": "test-service",
            "OTEL_SERVICE_VERSION": "1.0.0",
            "DEPLOYMENT_ENVIRONMENT": "staging",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
            "PROMETHEUS_PORT": "9095",
            "CLUSTER_ID": "test-cluster",
            "POD_NAMESPACE": "test-namespace",
            "REGION": "eu-west-1",
            "AVAILABILITY_ZONE": "eu-west-1b",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = _load_config_from_env()

            assert config.service_name == "test-service"
            assert config.service_version == "1.0.0"
            assert config.environment == "staging"
            assert config.otlp_endpoint == "http://otel-collector:4317"
            assert config.prometheus_port == 9095
            assert config.cluster_name == "test-cluster"
            assert config.namespace == "test-namespace"
            assert config.region == "eu-west-1"
            assert config.availability_zone == "eu-west-1b"

    def test_load_config_prometheus_port_parsing(self):
        """Test Prometheus port is correctly parsed as integer."""
        with patch.dict(os.environ, {"PROMETHEUS_PORT": "8080"}, clear=True):
            config = _load_config_from_env()
            assert config.prometheus_port == 8080
            assert isinstance(config.prometheus_port, int)

    def test_load_config_invalid_prometheus_port(self):
        """Test invalid Prometheus port raises ValueError."""
        with patch.dict(os.environ, {"PROMETHEUS_PORT": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                _load_config_from_env()

    def test_load_config_all_default_values(self):
        """Test all config fields have correct default values when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_env()

            # Core service identification
            assert config.service_name == "metadata-stamping"
            assert config.service_version == "3.0.0"
            assert config.environment == "production"

            # OTLP configuration (should be None when not set)
            assert config.otlp_endpoint is None

            # Prometheus configuration (not loaded from env, uses TelemetryConfig defaults)
            assert config.prometheus_port == 9090
            assert config.prometheus_endpoint == "/metrics"

            # Kubernetes/cluster configuration
            assert config.cluster_name == "metadata-stamping-cluster"
            assert config.namespace == "metadata-stamping"
            assert config.region == "us-west-2"
            assert config.availability_zone == "us-west-2a"

            # Instrumentation flags (use TelemetryConfig defaults)
            assert config.instrument_asyncpg is True
            assert config.instrument_redis is True
            assert config.instrument_http_client is True
            assert config.instrument_fastapi is True
            assert config.instrument_logging is True

            # Metrics flags (use TelemetryConfig defaults)
            assert config.enable_custom_metrics is True
            assert config.enable_runtime_metrics is True
            assert config.metrics_export_interval == 30

            # OTLP security (use TelemetryConfig defaults)
            assert config.otlp_insecure is False
            assert config.otlp_headers == {}

            # Sampling configuration (use TelemetryConfig defaults)
            assert config.sampling_ratio == 1.0

    def test_load_config_with_valid_otlp_endpoint(self):
        """Test loading config with valid OTLP endpoint."""
        with patch.dict(
            os.environ,
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"},
            clear=True,
        ):
            config = _load_config_from_env()

            assert config.otlp_endpoint == "http://localhost:4317"
            # Verify other fields still have defaults
            assert config.service_name == "metadata-stamping"
            assert config.prometheus_port == 9090

    def test_load_config_without_otlp_endpoint(self):
        """Test loading config without OTLP endpoint (should be None)."""
        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_env()

            assert config.otlp_endpoint is None
            # Should still have all other defaults
            assert config.service_name == "metadata-stamping"
            assert config.service_version == "3.0.0"

    def test_load_config_no_jaeger_endpoint(self):
        """Test that jaeger_endpoint is NOT present in config (deprecated)."""
        with patch.dict(os.environ, {}, clear=True):
            config = _load_config_from_env()

            # Verify jaeger_endpoint attribute does not exist
            assert not hasattr(config, "jaeger_endpoint")
            assert not hasattr(config, "jaeger_agent_host")
            assert not hasattr(config, "jaeger_agent_port")

            # Only OTLP endpoint should be present
            assert hasattr(config, "otlp_endpoint")

    def test_load_config_partial_env_vars(self):
        """Test loading config with only some environment variables set."""
        with patch.dict(
            os.environ,
            {
                "OTEL_SERVICE_NAME": "custom-service",
                "REGION": "ap-south-1",
                # Other vars not set, should use defaults
            },
            clear=True,
        ):
            config = _load_config_from_env()

            # Verify env vars were loaded
            assert config.service_name == "custom-service"
            assert config.region == "ap-south-1"

            # Verify defaults for unset vars
            assert config.service_version == "3.0.0"
            assert config.environment == "production"
            assert config.otlp_endpoint is None
            assert config.prometheus_port == 9090
            assert config.cluster_name == "metadata-stamping-cluster"
            assert config.namespace == "metadata-stamping"
            assert config.availability_zone == "us-west-2a"

    def test_load_config_empty_string_values(self):
        """Test loading config with empty string environment variables."""
        with patch.dict(
            os.environ,
            {
                "OTEL_SERVICE_NAME": "",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "",
                "REGION": "",
            },
            clear=True,
        ):
            config = _load_config_from_env()

            # Empty strings should be used as-is (not converted to None)
            assert config.service_name == ""
            assert config.otlp_endpoint == ""
            assert config.region == ""

    def test_load_config_otlp_endpoint_variations(self):
        """Test various valid OTLP endpoint formats."""
        test_cases = [
            ("http://localhost:4317", "http://localhost:4317"),
            (
                "https://otel-collector.svc.cluster.local:4317",
                "https://otel-collector.svc.cluster.local:4317",
            ),
            ("grpc://collector:4317", "grpc://collector:4317"),
        ]

        for env_value, expected_value in test_cases:
            with patch.dict(
                os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": env_value}, clear=True
            ):
                config = _load_config_from_env()
                assert config.otlp_endpoint == expected_value


class TestTelemetryManager:
    """Test TelemetryManager initialization and setup."""

    def test_manager_initialization(self):
        """Test TelemetryManager initializes with config."""
        config = TelemetryConfig(service_name="test-service")
        manager = TelemetryManager(config)

        assert manager.config == config
        assert manager.tracer_provider is None
        assert manager.meter_provider is None
        assert manager.metrics_collector is None
        assert manager.runtime_collector is None
        assert manager._initialized is False

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_initialize_without_otlp_endpoint(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test telemetry initialization without OTLP endpoint (should not crash)."""
        config = TelemetryConfig(otlp_endpoint=None)
        manager = TelemetryManager(config)

        # Should initialize successfully even without OTLP endpoint
        manager.initialize()

        assert manager._initialized is True
        assert manager.tracer_provider is not None

        # Meter provider should be None when no OTLP endpoint is configured
        assert manager.meter_provider is None

        # Prometheus server should still start
        mock_prometheus_server.assert_called_once_with(9090)

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPSpanExporter")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_initialize_with_otlp_endpoint(
        self,
        mock_metric_exporter,
        mock_span_exporter,
        mock_fastapi_instrumentor,
        mock_prometheus_server,
    ):
        """Test telemetry initialization with OTLP endpoint."""
        config = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",
            otlp_insecure=True,
            otlp_headers={"api-key": "secret"},
        )
        manager = TelemetryManager(config)

        manager.initialize()

        assert manager._initialized is True
        assert manager.tracer_provider is not None
        assert manager.meter_provider is not None

        # Verify OTLP exporters were created with correct config
        mock_span_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=True,
            headers={"api-key": "secret"},
        )
        mock_metric_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=True,
            headers={"api-key": "secret"},
        )

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_initialize_idempotent(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test initialize is idempotent (calling twice is safe)."""
        config = TelemetryConfig()
        manager = TelemetryManager(config)

        manager.initialize()
        first_tracer = manager.tracer_provider

        # Second call should be no-op
        manager.initialize()
        assert manager.tracer_provider is first_tracer

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.AsyncPGInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.RedisInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.AioHttpClientInstrumentor")
    def test_instrumentation_setup(
        self,
        mock_aiohttp_instrumentor,
        mock_redis_instrumentor,
        mock_asyncpg_instrumentor,
        mock_fastapi_instrumentor,
        mock_prometheus_server,
    ):
        """Test automatic instrumentation setup."""
        config = TelemetryConfig(
            instrument_fastapi=True,
            instrument_asyncpg=True,
            instrument_redis=True,
            instrument_http_client=True,
        )
        manager = TelemetryManager(config)

        manager.initialize()

        # Verify all instrumentors were called
        mock_fastapi_instrumentor.instrument.assert_called_once()
        mock_asyncpg_instrumentor.instrument.assert_called_once()
        mock_redis_instrumentor.instrument.assert_called_once()
        mock_aiohttp_instrumentor.return_value.instrument.assert_called_once()

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_instrumentation_disabled(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test instrumentation can be disabled."""
        config = TelemetryConfig(
            instrument_fastapi=False,
            instrument_asyncpg=False,
            instrument_redis=False,
            instrument_http_client=False,
        )
        manager = TelemetryManager(config)

        manager.initialize()

        # FastAPI instrumentation should not be called
        mock_fastapi_instrumentor.instrument.assert_not_called()

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    def test_prometheus_server_failure(self, mock_prometheus_server):
        """Test graceful handling of Prometheus server start failure."""
        mock_prometheus_server.side_effect = Exception("Port already in use")

        config = TelemetryConfig()
        manager = TelemetryManager(config)

        # Should not raise exception, just log error
        manager.initialize()
        assert manager._initialized is True

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_get_tracer_before_init_raises_error(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test getting tracer before initialization raises error."""
        config = TelemetryConfig()
        manager = TelemetryManager(config)

        with pytest.raises(RuntimeError, match="Telemetry not initialized"):
            manager.get_tracer()

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_get_tracer_after_init(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test getting tracer after initialization succeeds."""
        config = TelemetryConfig(service_name="test-service")
        manager = TelemetryManager(config)
        manager.initialize()

        tracer = manager.get_tracer()
        assert tracer is not None

        # Test with custom name
        tracer2 = manager.get_tracer("custom-tracer")
        assert tracer2 is not None

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPSpanExporter")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_init_with_otlp_endpoint_creates_meter_provider(
        self,
        mock_metric_exporter,
        mock_span_exporter,
        mock_fastapi_instrumentor,
        mock_prometheus_server,
    ):
        """Test initialization WITH OTLP endpoint creates meter_provider."""
        # Arrange
        config = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",
            otlp_insecure=True,
            enable_custom_metrics=True,
            enable_runtime_metrics=True,
        )
        manager = TelemetryManager(config)

        # Assert initial state
        assert manager.meter_provider is None
        assert manager.metrics_collector is None
        assert manager.runtime_collector is None

        # Act
        manager.initialize()

        # Assert
        assert manager._initialized is True
        assert manager.tracer_provider is not None
        assert (
            manager.meter_provider is not None
        ), "meter_provider should be created with OTLP endpoint"
        assert (
            manager.metrics_collector is not None
        ), "metrics_collector should be created when meter_provider exists"
        assert (
            manager.runtime_collector is not None
        ), "runtime_collector should be created when metrics_collector exists"

        # Verify exporters were called with correct config
        mock_span_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=True,
            headers={},
        )
        mock_metric_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=True,
            headers={},
        )

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_init_without_otlp_endpoint_no_meter_provider(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test initialization WITHOUT OTLP endpoint does NOT create meter_provider."""
        # Arrange
        config = TelemetryConfig(
            otlp_endpoint=None,  # Explicitly no OTLP endpoint
            enable_custom_metrics=True,
            enable_runtime_metrics=True,
        )
        manager = TelemetryManager(config)

        # Assert initial state
        assert manager.meter_provider is None
        assert manager.metrics_collector is None

        # Act
        manager.initialize()

        # Assert
        assert manager._initialized is True
        assert (
            manager.tracer_provider is not None
        ), "tracer_provider should still be created"
        assert (
            manager.meter_provider is None
        ), "meter_provider should be None without OTLP endpoint"

        # Prometheus server should still start
        mock_prometheus_server.assert_called_once_with(9090)

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    def test_no_metrics_collector_when_meter_provider_is_none(
        self, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test MetricsCollector is NOT created when meter_provider is None."""
        # Arrange
        config = TelemetryConfig(
            otlp_endpoint=None,  # No OTLP = no meter_provider
            enable_custom_metrics=True,  # Flags are on, but no meter_provider
            enable_runtime_metrics=True,
        )
        manager = TelemetryManager(config)

        # Act
        manager.initialize()

        # Assert
        assert (
            manager.meter_provider is None
        ), "Precondition: meter_provider should be None"
        assert (
            manager.metrics_collector is None
        ), "MetricsCollector should NOT be created when meter_provider is None"
        assert (
            manager.runtime_collector is None
        ), "RuntimeMetricsCollector should NOT be created when metrics_collector is None"

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_metrics_collector_created_when_meter_provider_exists(
        self, mock_metric_exporter, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test MetricsCollector IS created when meter_provider exists."""
        # Arrange
        config = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",  # OTLP endpoint = meter_provider
            enable_custom_metrics=True,
            enable_runtime_metrics=True,
        )
        manager = TelemetryManager(config)

        # Act
        manager.initialize()

        # Assert
        assert (
            manager.meter_provider is not None
        ), "Precondition: meter_provider should exist"
        assert (
            manager.metrics_collector is not None
        ), "MetricsCollector should be created when meter_provider exists"
        assert isinstance(
            manager.metrics_collector, MetricsCollector
        ), "metrics_collector should be MetricsCollector instance"
        assert (
            manager.runtime_collector is not None
        ), "RuntimeMetricsCollector should be created when metrics_collector exists"

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_metrics_collector_not_created_when_disabled_via_config(
        self, mock_metric_exporter, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test MetricsCollector respects enable_custom_metrics=False even with meter_provider."""
        # Arrange
        config = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",  # meter_provider will exist
            enable_custom_metrics=False,  # But custom metrics disabled
            enable_runtime_metrics=False,  # Runtime metrics also disabled
        )
        manager = TelemetryManager(config)

        # Act
        manager.initialize()

        # Assert
        assert (
            manager.meter_provider is not None
        ), "meter_provider should exist with OTLP endpoint"
        assert (
            manager.metrics_collector is None
        ), "MetricsCollector should NOT be created when enable_custom_metrics=False"
        assert (
            manager.runtime_collector is None
        ), "RuntimeMetricsCollector should NOT be created when enable_runtime_metrics=False"


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_metrics_collector_only_created_with_meter_provider(
        self, mock_metric_exporter, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test MetricsCollector is only created when meter_provider exists."""
        # Case 1: No OTLP endpoint = no meter_provider = no metrics_collector
        config1 = TelemetryConfig(
            otlp_endpoint=None, enable_custom_metrics=True, enable_runtime_metrics=True
        )
        manager1 = TelemetryManager(config1)
        manager1.initialize()

        assert manager1.meter_provider is None
        assert manager1.metrics_collector is None
        assert manager1.runtime_collector is None

        # Case 2: With OTLP endpoint = meter_provider exists = metrics_collector created
        config2 = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",
            enable_custom_metrics=True,
            enable_runtime_metrics=True,
        )
        manager2 = TelemetryManager(config2)
        manager2.initialize()

        assert manager2.meter_provider is not None
        assert manager2.metrics_collector is not None
        assert manager2.runtime_collector is not None

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.FastAPIInstrumentor")
    @patch("metadata_stamping.telemetry.opentelemetry.OTLPMetricExporter")
    def test_metrics_collector_disabled_via_config(
        self, mock_metric_exporter, mock_fastapi_instrumentor, mock_prometheus_server
    ):
        """Test metrics collector can be disabled via config flags."""
        config = TelemetryConfig(
            otlp_endpoint="http://localhost:4317",
            enable_custom_metrics=False,
            enable_runtime_metrics=False,
        )
        manager = TelemetryManager(config)
        manager.initialize()

        assert manager.meter_provider is not None
        assert manager.metrics_collector is None
        assert manager.runtime_collector is None

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initializes with meter provider."""
        mock_meter_provider = MagicMock()
        mock_meter = MagicMock()
        mock_meter_provider.get_meter.return_value = mock_meter

        collector = MetricsCollector(mock_meter_provider)

        assert collector.meter == mock_meter
        mock_meter_provider.get_meter.assert_called_once_with(
            "metadata_stamping_metrics"
        )

    def test_file_size_categorization(self):
        """Test file size categorization logic."""
        mock_meter_provider = MagicMock()
        collector = MetricsCollector(mock_meter_provider)

        assert collector._get_file_size_category(500) == "small"  # 500 bytes
        assert collector._get_file_size_category(1024) == "small"  # 1KB
        assert collector._get_file_size_category(5000) == "medium"  # ~5KB
        assert collector._get_file_size_category(1024 * 1024) == "medium"  # 1MB
        assert collector._get_file_size_category(5 * 1024 * 1024) == "large"  # 5MB
        assert collector._get_file_size_category(10 * 1024 * 1024) == "large"  # 10MB
        assert collector._get_file_size_category(20 * 1024 * 1024) == "xlarge"  # 20MB

    def test_update_connection_pool_size(self):
        """Test updating connection pool size."""
        mock_meter_provider = MagicMock()
        collector = MetricsCollector(mock_meter_provider)

        collector.update_connection_pool_size("shard-1", 25)
        assert collector._current_pool_sizes["shard-1"] == 25

        collector.update_connection_pool_size("shard-2", 30)
        assert collector._current_pool_sizes["shard-2"] == 30

    def test_update_shard_health(self):
        """Test updating shard health status."""
        mock_meter_provider = MagicMock()
        collector = MetricsCollector(mock_meter_provider)

        collector.update_shard_health("shard-1", healthy=True)
        assert collector._current_shard_health["shard-1"] == 1

        collector.update_shard_health("shard-2", healthy=False)
        assert collector._current_shard_health["shard-2"] == 0

    def test_update_circuit_breaker_state(self):
        """Test updating circuit breaker state."""
        mock_meter_provider = MagicMock()
        collector = MetricsCollector(mock_meter_provider)

        collector.update_circuit_breaker_state("db-circuit", 0)  # CLOSED
        assert collector._current_circuit_breaker_states["db-circuit"] == 0

        collector.update_circuit_breaker_state("db-circuit", 1)  # OPEN
        assert collector._current_circuit_breaker_states["db-circuit"] == 1


class TestResourceCreation:
    """Test resource creation with service metadata."""

    @patch("metadata_stamping.telemetry.opentelemetry.start_http_server")
    @patch("metadata_stamping.telemetry.opentelemetry.socket.gethostname")
    @patch.dict(
        os.environ,
        {
            "POD_NAME": "test-pod-123",
            "POD_NAMESPACE": "test-namespace",
            "NODE_NAME": "test-node",
            "POD_IP": "10.0.0.1",
        },
    )
    def test_resource_creation_with_k8s_env(
        self, mock_gethostname, mock_prometheus_server
    ):
        """Test resource includes Kubernetes environment variables."""
        mock_gethostname.return_value = "test-host"

        config = TelemetryConfig(
            cluster_name="test-cluster",
            namespace="test-ns",
            region="us-west-2",
            availability_zone="us-west-2a",
        )
        manager = TelemetryManager(config)
        resource = manager._create_resource()

        # Verify resource attributes
        attrs = resource.attributes
        assert attrs["service.name"] == "metadata-stamping"
        assert attrs["k8s.cluster.name"] == "test-cluster"
        assert attrs["k8s.namespace.name"] == "test-namespace"  # From env var
        assert attrs["k8s.pod.name"] == "test-pod-123"
        assert attrs["k8s.node.name"] == "test-node"
        assert attrs["k8s.pod.ip"] == "10.0.0.1"
        assert attrs["cloud.region"] == "us-west-2"
        assert attrs["cloud.availability_zone"] == "us-west-2a"


class TestRuntimeMetricsCollector:
    """Test RuntimeMetricsCollector functionality."""

    @pytest.mark.asyncio
    async def test_runtime_collector_start_stop(self):
        """Test runtime collector can be started and stopped."""
        from metadata_stamping.telemetry.opentelemetry import RuntimeMetricsCollector

        mock_metrics_collector = MagicMock()
        runtime_collector = RuntimeMetricsCollector(mock_metrics_collector)

        assert runtime_collector.running is False
        assert runtime_collector.collection_task is None

        # Start collection
        await runtime_collector.start_collection(interval=1)
        assert runtime_collector.running is True
        assert runtime_collector.collection_task is not None

        # Stop collection
        await runtime_collector.stop_collection()
        assert runtime_collector.running is False

    @pytest.mark.asyncio
    async def test_runtime_collector_idempotent_start(self):
        """Test starting runtime collector multiple times is safe."""
        from metadata_stamping.telemetry.opentelemetry import RuntimeMetricsCollector

        mock_metrics_collector = MagicMock()
        runtime_collector = RuntimeMetricsCollector(mock_metrics_collector)

        await runtime_collector.start_collection()
        first_task = runtime_collector.collection_task

        # Second start should be no-op
        await runtime_collector.start_collection()
        assert runtime_collector.collection_task is first_task

        await runtime_collector.stop_collection()
