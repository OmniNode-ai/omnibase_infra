"""
Tests for Metrics Registry.

Tests Prometheus metrics collection and registry management.
Note: These are unit tests with mocked Prometheus components.
"""

from unittest.mock import Mock, patch

import pytest

from omnibase_infra.infrastructure.observability.metrics_registry import (
    MetricsRegistry,
    create_database_metrics,
    create_http_metrics,
    create_kafka_metrics,
)


class TestMetricsRegistryInit:
    """Test metrics registry initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        registry = MetricsRegistry(namespace="test")

        assert registry.namespace == "test"
        assert registry.metrics == {}

    def test_init_with_labels(self):
        """Test initialization with default labels."""
        registry = MetricsRegistry(
            namespace="test",
            default_labels={"service": "test_service"},
        )

        assert registry.default_labels["service"] == "test_service"


class TestMetricsRegistryCounter:
    """Test counter metric creation and operations."""

    @patch("prometheus_client.Counter")
    def test_create_counter(self, mock_counter_class):
        """Test creating a counter metric."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        registry = MetricsRegistry(namespace="test")

        counter = registry.create_counter(
            name="test_counter",
            description="Test counter",
        )

        assert counter == mock_counter
        mock_counter_class.assert_called_once()

    @patch("prometheus_client.Counter")
    def test_increment_counter(self, mock_counter_class):
        """Test incrementing a counter."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        registry = MetricsRegistry(namespace="test")
        counter = registry.create_counter("test_counter")

        counter.inc()

        mock_counter.inc.assert_called_once()

    @patch("prometheus_client.Counter")
    def test_counter_with_labels(self, mock_counter_class):
        """Test counter with labels."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        registry = MetricsRegistry(namespace="test")
        counter = registry.create_counter(
            "test_counter",
            labels=["method", "endpoint"],
        )

        # Use labels
        counter.labels(method="GET", endpoint="/api/users").inc()


class TestMetricsRegistryGauge:
    """Test gauge metric creation and operations."""

    @patch("prometheus_client.Gauge")
    def test_create_gauge(self, mock_gauge_class):
        """Test creating a gauge metric."""
        mock_gauge = Mock()
        mock_gauge_class.return_value = mock_gauge

        registry = MetricsRegistry(namespace="test")

        gauge = registry.create_gauge(
            name="test_gauge",
            description="Test gauge",
        )

        assert gauge == mock_gauge

    @patch("prometheus_client.Gauge")
    def test_set_gauge(self, mock_gauge_class):
        """Test setting gauge value."""
        mock_gauge = Mock()
        mock_gauge_class.return_value = mock_gauge

        registry = MetricsRegistry(namespace="test")
        gauge = registry.create_gauge("test_gauge")

        gauge.set(42)

        mock_gauge.set.assert_called_with(42)

    @patch("prometheus_client.Gauge")
    def test_gauge_inc_dec(self, mock_gauge_class):
        """Test incrementing and decrementing gauge."""
        mock_gauge = Mock()
        mock_gauge_class.return_value = mock_gauge

        registry = MetricsRegistry(namespace="test")
        gauge = registry.create_gauge("test_gauge")

        gauge.inc()
        gauge.dec()

        mock_gauge.inc.assert_called_once()
        mock_gauge.dec.assert_called_once()


class TestMetricsRegistryHistogram:
    """Test histogram metric creation and operations."""

    @patch("prometheus_client.Histogram")
    def test_create_histogram(self, mock_histogram_class):
        """Test creating a histogram metric."""
        mock_histogram = Mock()
        mock_histogram_class.return_value = mock_histogram

        registry = MetricsRegistry(namespace="test")

        histogram = registry.create_histogram(
            name="test_histogram",
            description="Test histogram",
        )

        assert histogram == mock_histogram

    @patch("prometheus_client.Histogram")
    def test_observe_histogram(self, mock_histogram_class):
        """Test observing histogram values."""
        mock_histogram = Mock()
        mock_histogram_class.return_value = mock_histogram

        registry = MetricsRegistry(namespace="test")
        histogram = registry.create_histogram("test_histogram")

        histogram.observe(0.5)

        mock_histogram.observe.assert_called_with(0.5)

    @patch("prometheus_client.Histogram")
    def test_histogram_custom_buckets(self, mock_histogram_class):
        """Test histogram with custom buckets."""
        mock_histogram = Mock()
        mock_histogram_class.return_value = mock_histogram

        registry = MetricsRegistry(namespace="test")

        histogram = registry.create_histogram(
            "test_histogram",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        # Verify buckets parameter was passed
        call_kwargs = mock_histogram_class.call_args.kwargs
        assert "buckets" in call_kwargs


class TestDatabaseMetrics:
    """Test database-specific metrics."""

    @patch("prometheus_client.Counter")
    @patch("prometheus_client.Histogram")
    def test_create_database_metrics(self, mock_histogram, mock_counter):
        """Test creating database metrics collection."""
        metrics = create_database_metrics(namespace="test")

        assert isinstance(metrics, dict)

    @patch("prometheus_client.Counter")
    def test_database_query_counter(self, mock_counter_class):
        """Test database query counter."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        metrics = create_database_metrics(namespace="test")

        # Simulate query counting
        if "queries_total" in metrics:
            metrics["queries_total"].inc()


class TestHttpMetrics:
    """Test HTTP-specific metrics."""

    @patch("prometheus_client.Counter")
    @patch("prometheus_client.Histogram")
    def test_create_http_metrics(self, mock_histogram, mock_counter):
        """Test creating HTTP metrics collection."""
        metrics = create_http_metrics(namespace="test")

        assert isinstance(metrics, dict)

    @patch("prometheus_client.Counter")
    def test_http_request_counter(self, mock_counter_class):
        """Test HTTP request counter."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        metrics = create_http_metrics(namespace="test")

        # Simulate request counting
        if "requests_total" in metrics:
            metrics["requests_total"].labels(
                method="GET",
                endpoint="/api/users",
                status="200",
            ).inc()


class TestKafkaMetrics:
    """Test Kafka-specific metrics."""

    @patch("prometheus_client.Counter")
    @patch("prometheus_client.Histogram")
    def test_create_kafka_metrics(self, mock_histogram, mock_counter):
        """Test creating Kafka metrics collection."""
        metrics = create_kafka_metrics(namespace="test")

        assert isinstance(metrics, dict)

    @patch("prometheus_client.Counter")
    def test_kafka_message_counter(self, mock_counter_class):
        """Test Kafka message counter."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        metrics = create_kafka_metrics(namespace="test")

        # Simulate message counting
        if "messages_sent_total" in metrics:
            metrics["messages_sent_total"].labels(topic="test_topic").inc()


class TestMetricsRegistryIntegration:
    """Integration tests for metrics registry."""

    @patch("prometheus_client.Counter")
    @patch("prometheus_client.Gauge")
    @patch("prometheus_client.Histogram")
    def test_full_metrics_workflow(
        self, mock_histogram, mock_gauge, mock_counter
    ):
        """Test complete metrics collection workflow."""
        registry = MetricsRegistry(namespace="test")

        # Create various metrics
        counter = registry.create_counter("requests")
        gauge = registry.create_gauge("connections")
        histogram = registry.create_histogram("latency")

        # Use metrics
        counter.inc()
        gauge.set(10)
        histogram.observe(0.5)

    @patch("prometheus_client.Counter")
    def test_metrics_with_default_labels(self, mock_counter_class):
        """Test metrics with default labels."""
        registry = MetricsRegistry(
            namespace="test",
            default_labels={"service": "test_service", "env": "test"},
        )

        counter = registry.create_counter("test_counter")

        # Default labels should be applied
        assert registry.default_labels["service"] == "test_service"

    @patch("prometheus_client.Counter")
    def test_duplicate_metric_registration(self, mock_counter_class):
        """Test handling of duplicate metric registration."""
        mock_counter = Mock()
        mock_counter_class.return_value = mock_counter

        registry = MetricsRegistry(namespace="test")

        # Register same metric twice
        counter1 = registry.create_counter("test_counter")
        counter2 = registry.create_counter("test_counter")

        # Should return same instance or handle appropriately
