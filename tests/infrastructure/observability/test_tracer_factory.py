"""
Tests for Tracer Factory.

Tests OpenTelemetry tracer creation and span operations.
Note: These are unit tests with mocked OpenTelemetry components.
"""

from unittest.mock import Mock, patch

import pytest

from omnibase_infra.infrastructure.observability.tracer_factory import (
    TracerFactory,
    create_database_tracer,
    create_http_tracer,
    create_kafka_tracer,
)


class TestTracerFactoryConfiguration:
    """Test tracer factory configuration."""

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    @patch("omnibase_infra.infrastructure.observability.tracer_factory.TracerProvider")
    def test_configure_tracer_provider(self, mock_provider, mock_trace):
        """Test configuring tracer provider."""
        factory = TracerFactory()
        factory.configure(
            service_name="test_service",
            environment="test",
        )

        # Should have configured provider
        assert factory.service_name == "test_service"

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_get_tracer(self, mock_trace):
        """Test getting tracer instance."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        factory = TracerFactory()
        factory.configure(service_name="test")

        tracer = factory.get_tracer("test_component")

        assert tracer == mock_tracer


class TestDatabaseTracer:
    """Test database-specific tracer."""

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_create_database_tracer(self, mock_trace):
        """Test creating database tracer."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_database_tracer(service_name="test_service")

        assert tracer == mock_tracer

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_database_tracer_span(self, mock_trace):
        """Test database tracer span creation."""
        mock_span = Mock()
        mock_tracer = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_database_tracer(service_name="test")

        with tracer.start_as_current_span("db_query"):
            pass


class TestHttpTracer:
    """Test HTTP-specific tracer."""

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_create_http_tracer(self, mock_trace):
        """Test creating HTTP tracer."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_http_tracer(service_name="test_service")

        assert tracer == mock_tracer


class TestKafkaTracer:
    """Test Kafka-specific tracer."""

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_create_kafka_tracer(self, mock_trace):
        """Test creating Kafka tracer."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_kafka_tracer(service_name="test_service")

        assert tracer == mock_tracer


class TestSpanOperations:
    """Test span operations."""

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_span_attributes(self, mock_trace):
        """Test setting span attributes."""
        mock_span = Mock()
        mock_tracer = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_database_tracer(service_name="test")

        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("key", "value")

        # Verify attribute was set
        mock_span.set_attribute.assert_called_with("key", "value")

    @patch("omnibase_infra.infrastructure.observability.tracer_factory.trace")
    def test_span_events(self, mock_trace):
        """Test adding span events."""
        mock_span = Mock()
        mock_tracer = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = create_database_tracer(service_name="test")

        with tracer.start_as_current_span("test_span") as span:
            span.add_event("test_event")

        # Verify event was added
        mock_span.add_event.assert_called_with("test_event")
