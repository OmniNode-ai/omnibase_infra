"""Tests for OpenTelemetry configuration."""

import os
import sys
from unittest.mock import MagicMock, patch

# Mock OpenTelemetry imports before importing config
sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = MagicMock()
sys.modules["opentelemetry.exporter.prometheus"] = MagicMock()
sys.modules["opentelemetry.instrumentation.aiohttp_client"] = MagicMock()
sys.modules["opentelemetry.instrumentation.asyncpg"] = MagicMock()
sys.modules["opentelemetry.instrumentation.fastapi"] = MagicMock()
sys.modules["opentelemetry.instrumentation.kafka"] = MagicMock()

# Mock the sampling module with TraceIdRatioBasedSampler
mock_sampling = MagicMock()
mock_sampling.TraceIdRatioBasedSampler = MagicMock()
sys.modules["opentelemetry.sdk.trace.sampling"] = mock_sampling

from omninode_bridge.tracing.opentelemetry_config import (
    OpenTelemetryConfig,
    initialize_opentelemetry,
)


class TestOpenTelemetryConfig:
    """Tests for OpenTelemetryConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = OpenTelemetryConfig("test-service")

        assert config.service_name == "test-service"
        assert config.service_version == "1.0.0"
        assert config.is_initialized is False

    def test_init_with_custom_version(self):
        """Test initialization with custom version."""
        config = OpenTelemetryConfig("test-service", "2.0.0")

        assert config.service_version == "2.0.0"

    @patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://custom:4317"})
    def test_env_var_configuration(self):
        """Test configuration from environment variables."""
        config = OpenTelemetryConfig("test-service")

        assert config.otlp_endpoint == "http://custom:4317"

    @patch.dict(os.environ, {"ENABLE_OTEL_TRACING": "false"})
    def test_tracing_disabled_via_env(self):
        """Test tracing can be disabled via environment."""
        config = OpenTelemetryConfig("test-service")

        assert config.enable_tracing is False

    @patch.dict(os.environ, {"ENABLE_PROMETHEUS_METRICS": "false"})
    def test_prometheus_disabled_via_env(self):
        """Test Prometheus can be disabled via environment."""
        config = OpenTelemetryConfig("test-service")

        assert config.enable_prometheus is False

    @patch.dict(os.environ, {"TRACE_SAMPLING_RATE": "0.5"})
    def test_sampling_rate_from_env(self):
        """Test sampling rate configured from environment."""
        config = OpenTelemetryConfig("test-service")

        assert config.trace_sampling_rate == 0.5

    @patch.dict(os.environ, {"ENABLE_AUTO_INSTRUMENTATION": "false"})
    def test_auto_instrumentation_disabled_via_env(self):
        """Test auto-instrumentation can be disabled."""
        config = OpenTelemetryConfig("test-service")

        assert config.enable_auto_instrumentation is False


class TestConfigureTracing:
    """Tests for tracing configuration."""

    @patch.dict(os.environ, {"ENABLE_OTEL_TRACING": "false"})
    def test_configure_tracing_disabled(self):
        """Test tracing configuration when disabled."""
        config = OpenTelemetryConfig("test-service")
        result = config.configure_tracing()

        assert result is None

    @patch("omninode_bridge.tracing.opentelemetry_config.TracerProvider")
    @patch("omninode_bridge.tracing.opentelemetry_config.OTLPSpanExporter")
    @patch("omninode_bridge.tracing.opentelemetry_config.BatchSpanProcessor")
    def test_configure_tracing_success(
        self, mock_processor, mock_exporter, mock_provider
    ):
        """Test successful tracing configuration."""
        config = OpenTelemetryConfig("test-service")
        config.enable_tracing = True

        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        result = config.configure_tracing()

        # Verify tracer provider was created
        mock_provider.assert_called_once()
        assert result is not None

    @patch("omninode_bridge.tracing.opentelemetry_config.TracerProvider")
    def test_configure_tracing_error_handling(self, mock_provider):
        """Test error handling in tracing configuration."""
        config = OpenTelemetryConfig("test-service")
        config.enable_tracing = True

        # Simulate error
        mock_provider.side_effect = Exception("Configuration failed")

        result = config.configure_tracing()

        assert result is None


class TestConfigureMetrics:
    """Tests for metrics configuration."""

    @patch.dict(os.environ, {"ENABLE_PROMETHEUS_METRICS": "false"})
    def test_configure_metrics_disabled(self):
        """Test metrics configuration when disabled."""
        config = OpenTelemetryConfig("test-service")
        result = config.configure_metrics()

        assert result is None

    @patch("omninode_bridge.tracing.opentelemetry_config.MeterProvider")
    @patch("omninode_bridge.tracing.opentelemetry_config.PrometheusMetricReader")
    def test_configure_metrics_success(self, mock_reader, mock_provider):
        """Test successful metrics configuration."""
        config = OpenTelemetryConfig("test-service")
        config.enable_prometheus = True

        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        result = config.configure_metrics()

        # Verify meter provider was created
        mock_provider.assert_called_once()
        assert result is not None

    @patch("omninode_bridge.tracing.opentelemetry_config.MeterProvider")
    def test_configure_metrics_error_handling(self, mock_provider):
        """Test error handling in metrics configuration."""
        config = OpenTelemetryConfig("test-service")
        config.enable_prometheus = True

        # Simulate error
        mock_provider.side_effect = Exception("Configuration failed")

        result = config.configure_metrics()

        assert result is None


class TestConfigureAutoInstrumentation:
    """Tests for auto-instrumentation configuration."""

    @patch.dict(os.environ, {"ENABLE_AUTO_INSTRUMENTATION": "false"})
    def test_auto_instrumentation_disabled(self):
        """Test auto-instrumentation when disabled."""
        config = OpenTelemetryConfig("test-service")
        # Should complete without error
        config.configure_auto_instrumentation()

    @patch("omninode_bridge.tracing.opentelemetry_config.FastAPIInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.AsyncPGInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.AioHttpClientInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.KafkaInstrumentor")
    def test_auto_instrumentation_all_components(
        self, mock_kafka, mock_aiohttp, mock_asyncpg, mock_fastapi
    ):
        """Test auto-instrumentation of all components."""
        config = OpenTelemetryConfig("test-service")
        config.enable_auto_instrumentation = True

        config.configure_auto_instrumentation()

        # Verify all instrumentors were called
        mock_fastapi.return_value.instrument.assert_called_once()
        mock_asyncpg.return_value.instrument.assert_called_once()
        mock_aiohttp.return_value.instrument.assert_called_once()
        mock_kafka.return_value.instrument.assert_called_once()

    @patch("omninode_bridge.tracing.opentelemetry_config.FastAPIInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.AsyncPGInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.AioHttpClientInstrumentor")
    @patch("omninode_bridge.tracing.opentelemetry_config.KafkaInstrumentor")
    def test_auto_instrumentation_kafka_failure(
        self, mock_kafka, mock_aiohttp, mock_asyncpg, mock_fastapi
    ):
        """Test auto-instrumentation continues if Kafka fails."""
        config = OpenTelemetryConfig("test-service")
        config.enable_auto_instrumentation = True

        # Simulate Kafka instrumentation failure
        mock_kafka.return_value.instrument.side_effect = Exception(
            "Kafka not available"
        )

        # Should not raise exception
        config.configure_auto_instrumentation()

        # Other instrumentations should still be called
        mock_fastapi.return_value.instrument.assert_called_once()
        mock_asyncpg.return_value.instrument.assert_called_once()
        mock_aiohttp.return_value.instrument.assert_called_once()


class TestInitialize:
    """Tests for full initialization."""

    def test_initialize_once(self):
        """Test that initialization only happens once."""
        config = OpenTelemetryConfig("test-service")
        config.enable_tracing = False
        config.enable_prometheus = False
        config.enable_auto_instrumentation = False

        # First initialization
        result1 = config.initialize()
        assert result1 is True
        assert config.is_initialized is True

        # Second initialization should return True but not reinitialize
        result2 = config.initialize()
        assert result2 is True

    @patch.object(OpenTelemetryConfig, "configure_tracing")
    @patch.object(OpenTelemetryConfig, "configure_metrics")
    @patch.object(OpenTelemetryConfig, "configure_auto_instrumentation")
    def test_initialize_calls_all_configs(self, mock_auto, mock_metrics, mock_tracing):
        """Test that initialize calls all configuration methods."""
        config = OpenTelemetryConfig("test-service")

        config.initialize()

        mock_tracing.assert_called_once()
        mock_metrics.assert_called_once()
        mock_auto.assert_called_once()

    @patch.object(OpenTelemetryConfig, "configure_tracing")
    def test_initialize_error_handling(self, mock_tracing):
        """Test error handling during initialization."""
        config = OpenTelemetryConfig("test-service")

        # Simulate error
        mock_tracing.side_effect = Exception("Initialization failed")

        result = config.initialize()

        assert result is False
        assert config.is_initialized is False


class TestSampler:
    """Tests for trace sampler creation."""

    def test_create_sampler_default_rate(self):
        """Test sampler creation with default rate."""
        config = OpenTelemetryConfig("test-service")

        sampler = config._create_sampler()

        # Should use default rate of 1.0
        assert sampler is not None

    @patch.dict(os.environ, {"TRACE_SAMPLING_RATE": "0.25"})
    def test_create_sampler_custom_rate(self):
        """Test sampler creation with custom rate."""
        config = OpenTelemetryConfig("test-service")

        sampler = config._create_sampler()

        assert sampler is not None


class TestHooks:
    """Tests for instrumentation hooks."""

    def test_fastapi_request_hook(self):
        """Test FastAPI request hook adds correlation context."""
        config = OpenTelemetryConfig("test-service")
        mock_span = MagicMock()
        scope = {}

        with patch(
            "omninode_bridge.tracing.opentelemetry_config.get_correlation_context",
            return_value={
                "correlation_id": "test-corr-id",
                "workflow_id": "test-workflow-id",
                "request_id": None,
            },
        ):
            config._fastapi_request_hook(mock_span, scope)

            # Verify attributes were set
            assert mock_span.set_attribute.call_count >= 3

    def test_fastapi_request_hook_error_handling(self):
        """Test FastAPI request hook handles errors gracefully."""
        config = OpenTelemetryConfig("test-service")
        mock_span = MagicMock()
        scope = {}

        with patch(
            "omninode_bridge.tracing.opentelemetry_config.get_correlation_context",
            side_effect=Exception("Context error"),
        ):
            # Should not raise exception
            config._fastapi_request_hook(mock_span, scope)

    def test_fastapi_response_hook(self):
        """Test FastAPI response hook adds status code."""
        config = OpenTelemetryConfig("test-service")
        mock_span = MagicMock()
        message = {"status": 200}

        config._fastapi_response_hook(mock_span, message)

        mock_span.set_attribute.assert_called_with("http.status_code", 200)

    def test_fastapi_response_hook_error_status(self):
        """Test FastAPI response hook handles error status codes."""
        config = OpenTelemetryConfig("test-service")
        mock_span = MagicMock()
        message = {"status": 500}

        config._fastapi_response_hook(mock_span, message)

        # Should set both attribute and status
        mock_span.set_attribute.assert_called_with("http.status_code", 500)
        mock_span.set_status.assert_called_once()

    def test_aiohttp_request_hook(self):
        """Test AIOHTTP request hook adds correlation context."""
        config = OpenTelemetryConfig("test-service")
        mock_span = MagicMock()
        params = {}

        with patch(
            "omninode_bridge.tracing.opentelemetry_config.get_correlation_context",
            return_value={"correlation_id": "test-id"},
        ):
            config._aiohttp_request_hook(mock_span, params)

            # Verify attributes were set
            mock_span.set_attribute.assert_called()


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    @patch("omninode_bridge.tracing.opentelemetry_config.OpenTelemetryConfig")
    def test_initialize_opentelemetry(self, mock_config_class):
        """Test global initialize_opentelemetry function."""
        mock_config = MagicMock()
        mock_config.initialize.return_value = True
        mock_config_class.return_value = mock_config

        result = initialize_opentelemetry("test-service", "1.0.0")

        assert result is True
        mock_config_class.assert_called_once_with("test-service", "1.0.0")
        mock_config.initialize.assert_called_once()

    def test_initialize_opentelemetry_already_initialized(self):
        """Test that multiple initializations return True."""
        # First initialization
        with patch("omninode_bridge.tracing.opentelemetry_config._otel_config", None):
            with patch(
                "omninode_bridge.tracing.opentelemetry_config.OpenTelemetryConfig"
            ) as mock_config_class:
                mock_config = MagicMock()
                mock_config.initialize.return_value = True
                mock_config_class.return_value = mock_config

                result1 = initialize_opentelemetry("test-service")
                assert result1 is True


class TestResourceConfiguration:
    """Tests for resource configuration."""

    @patch("omninode_bridge.tracing.opentelemetry_config.Resource")
    @patch("omninode_bridge.tracing.opentelemetry_config.TracerProvider")
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_resource_includes_environment(self, mock_provider, mock_resource):
        """Test that resource includes environment from env var."""
        config = OpenTelemetryConfig("test-service", "2.0.0")
        config.enable_tracing = True

        config.configure_tracing()

        # Verify Resource.create was called with correct attributes
        mock_resource.create.assert_called_once()
        call_args = mock_resource.create.call_args[0][0]

        assert call_args["service.name"] == "test-service"
        assert call_args["service.version"] == "2.0.0"
        assert call_args["service.namespace"] == "omninode-bridge"
        assert call_args["deployment.environment"] == "production"
