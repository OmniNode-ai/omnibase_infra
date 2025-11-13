"""
Unit tests for Prometheus metrics integration in bridge nodes.

Tests comprehensive metrics collection, feature flag support, and
industry-standard Prometheus instrumentation.
"""

from prometheus_client import CollectorRegistry

from omninode_bridge.nodes.metrics.prometheus_metrics import (
    BridgeMetricsCollector,
    NodeType,
    create_orchestrator_metrics,
    create_reducer_metrics,
    create_registry_metrics,
)


class TestBridgeMetricsCollector:
    """Test BridgeMetricsCollector core functionality."""

    def test_initialization_with_prometheus_enabled(self):
        """Test metrics collector initializes correctly when enabled."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR,
            registry=registry,
            enable_prometheus=True,
        )

        assert collector.enable_prometheus is True
        assert collector.node_type == NodeType.ORCHESTRATOR
        assert collector.registry is registry

        # Verify metrics are initialized
        assert hasattr(collector, "workflow_counter")
        assert hasattr(collector, "workflow_duration")
        assert hasattr(collector, "event_published_counter")

    def test_initialization_with_prometheus_disabled(self):
        """Test metrics collector gracefully handles disabled state."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER,
            registry=registry,
            enable_prometheus=False,
        )

        assert collector.enable_prometheus is False
        assert collector.node_type == NodeType.REDUCER

        # Metrics should not be initialized
        assert not hasattr(collector, "workflow_counter")

    def test_workflow_timing_context_manager(self):
        """Test workflow timing context manager."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Execute workflow with timing
        with collector.time_workflow(status="success"):
            # Simulate workflow execution
            pass

        # Verify metrics were recorded
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_workflow_total" in metrics
        assert "bridge_workflow_duration_seconds" in metrics
        assert 'status="success"' in metrics

    def test_aggregation_timing_context_manager(self):
        """Test aggregation timing context manager."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER, registry=registry, enable_prometheus=True
        )

        # Execute aggregation with timing
        with collector.time_aggregation(
            aggregation_type="namespace_grouping", status="success"
        ):
            # Simulate aggregation
            pass

        # Verify metrics were recorded
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_aggregation_total" in metrics
        assert "bridge_aggregation_duration_seconds" in metrics
        assert 'aggregation_type="namespace_grouping"' in metrics

    def test_event_publish_timing_context_manager(self):
        """Test event publishing timing context manager."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Execute event publishing with timing
        with collector.time_event_publish(
            event_type="workflow_started", status="success"
        ):
            # Simulate event publishing
            pass

        # Verify metrics were recorded
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_event_published_total" in metrics
        assert "bridge_event_publish_duration_seconds" in metrics
        assert 'event_type="workflow_started"' in metrics

    def test_record_aggregation_items(self):
        """Test recording aggregation items count."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER, registry=registry, enable_prometheus=True
        )

        # Record items
        collector.record_aggregation_items("namespace_grouping", 100)
        collector.record_aggregation_items("namespace_grouping", 50)

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_aggregation_items_processed_total" in metrics
        assert 'aggregation_type="namespace_grouping"' in metrics

    def test_set_aggregation_buffer_size(self):
        """Test setting aggregation buffer size gauge."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER, registry=registry, enable_prometheus=True
        )

        # Set buffer size
        collector.set_aggregation_buffer_size("test.namespace", 500)

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_aggregation_buffer_size" in metrics
        assert 'namespace="test.namespace"' in metrics

    def test_record_error(self):
        """Test error recording."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Record errors
        collector.record_error("ValueError", "execute_orchestration")
        collector.record_error("TimeoutError", "event_publish")

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_errors_total" in metrics
        assert 'error_type="ValueError"' in metrics
        assert 'operation="execute_orchestration"' in metrics

    def test_record_node_registration(self):
        """Test node registration recording."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REGISTRY, registry=registry, enable_prometheus=True
        )

        # Record registrations
        collector.record_node_registration(status="success")
        collector.record_node_registration(status="success")
        collector.record_node_registration(status="failed")

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_node_registration_total" in metrics
        assert 'status="success"' in metrics
        assert 'status="failed"' in metrics

    def test_set_registered_nodes_count(self):
        """Test setting registered nodes gauge."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REGISTRY, registry=registry, enable_prometheus=True
        )

        # Set node count
        collector.set_registered_nodes_count(25)

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_registered_nodes" in metrics

    def test_metrics_disabled_no_op(self):
        """Test that disabled metrics do not record anything."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR,
            registry=registry,
            enable_prometheus=False,
        )

        # Attempt to record metrics
        with collector.time_workflow(status="success"):
            pass

        collector.record_error("TestError", "test_operation")

        # Metrics should indicate disabled state
        metrics = collector.get_metrics().decode("utf-8")
        assert "Prometheus metrics disabled" in metrics

    def test_get_metrics_summary(self):
        """Test getting human-readable metrics summary."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Record some metrics
        with collector.time_workflow(status="success"):
            pass

        # Get summary
        summary = collector.get_metrics_summary()

        assert summary["enabled"] is True
        assert summary["node_type"] == "orchestrator"
        assert "metrics" in summary
        assert len(summary["metrics"]) > 0

    def test_get_metrics_summary_disabled(self):
        """Test metrics summary when disabled."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER, registry=registry, enable_prometheus=False
        )

        summary = collector.get_metrics_summary()

        assert summary["enabled"] is False
        assert "message" in summary
        assert "disabled" in summary["message"].lower()

    def test_db_query_timing(self):
        """Test database query timing context manager."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Time database query
        with collector.time_db_query(operation="insert", status="success"):
            # Simulate DB query
            pass

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_db_query_duration_seconds" in metrics
        assert "bridge_db_query_total" in metrics
        assert 'operation="insert"' in metrics

    def test_http_request_timing(self):
        """Test HTTP request timing context manager."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Time HTTP request
        with collector.time_http_request(
            method="POST", endpoint="/workflow/submit", status_code=201
        ):
            # Simulate HTTP request
            pass

        # Verify metrics
        metrics = collector.get_metrics().decode("utf-8")
        assert "bridge_http_requests_total" in metrics
        assert "bridge_http_request_duration_seconds" in metrics
        assert 'method="POST"' in metrics
        assert 'endpoint="/workflow/submit"' in metrics


class TestFactoryFunctions:
    """Test factory functions for creating metrics collectors."""

    def test_create_orchestrator_metrics(self):
        """Test creating orchestrator metrics collector."""
        collector = create_orchestrator_metrics(enable_prometheus=True)

        assert collector.node_type == NodeType.ORCHESTRATOR
        assert collector.enable_prometheus is True

    def test_create_reducer_metrics(self):
        """Test creating reducer metrics collector."""
        collector = create_reducer_metrics(enable_prometheus=True)

        assert collector.node_type == NodeType.REDUCER
        assert collector.enable_prometheus is True

    def test_create_registry_metrics(self):
        """Test creating registry metrics collector."""
        collector = create_registry_metrics(enable_prometheus=True)

        assert collector.node_type == NodeType.REGISTRY
        assert collector.enable_prometheus is True

    def test_factory_with_custom_registry(self):
        """Test factory functions with custom registry."""
        custom_registry = CollectorRegistry()
        collector = create_orchestrator_metrics(
            registry=custom_registry, enable_prometheus=True
        )

        assert collector.registry is custom_registry

    def test_factory_with_disabled_prometheus(self):
        """Test factory functions with disabled Prometheus."""
        collector = create_reducer_metrics(enable_prometheus=False)

        assert collector.enable_prometheus is False


class TestNodeTypeEnum:
    """Test NodeType enum."""

    def test_node_type_values(self):
        """Test NodeType enum values."""
        assert NodeType.ORCHESTRATOR.value == "orchestrator"
        assert NodeType.REDUCER.value == "reducer"
        assert NodeType.REGISTRY.value == "registry"

    def test_node_type_membership(self):
        """Test NodeType enum membership."""
        assert "orchestrator" in [t.value for t in NodeType]
        assert "reducer" in [t.value for t in NodeType]
        assert "registry" in [t.value for t in NodeType]


class TestPrometheusIntegration:
    """Test integration with Prometheus client library."""

    def test_metrics_format_is_valid_prometheus(self):
        """Test that metrics output is valid Prometheus format."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Record some metrics
        with collector.time_workflow(status="success"):
            pass

        # Get metrics
        metrics = collector.get_metrics().decode("utf-8")

        # Verify Prometheus format
        assert "# HELP" in metrics
        assert "# TYPE" in metrics
        assert "bridge_workflow_total" in metrics
        assert "bridge_workflow_duration_seconds" in metrics

    def test_metrics_include_labels(self):
        """Test that metrics include proper labels."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.REDUCER, registry=registry, enable_prometheus=True
        )

        # Record metrics with labels
        with collector.time_aggregation("namespace_grouping", "success"):
            pass

        collector.record_aggregation_items("namespace_grouping", 50)

        # Get metrics
        metrics = collector.get_metrics().decode("utf-8")

        # Verify labels are present
        assert 'node_type="reducer"' in metrics
        assert 'aggregation_type="namespace_grouping"' in metrics
        assert 'status="success"' in metrics

    def test_counter_increments(self):
        """Test that counters increment correctly."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Record multiple workflow executions
        with collector.time_workflow(status="success"):
            pass
        with collector.time_workflow(status="success"):
            pass
        with collector.time_workflow(status="failed"):
            pass

        # Get metrics
        metrics = collector.get_metrics().decode("utf-8")

        # Verify counters (should have incremented)
        assert "bridge_workflow_total" in metrics

    def test_histogram_buckets(self):
        """Test that histograms have proper buckets."""
        registry = CollectorRegistry()
        collector = BridgeMetricsCollector(
            node_type=NodeType.ORCHESTRATOR, registry=registry, enable_prometheus=True
        )

        # Record workflow duration
        with collector.time_workflow(status="success"):
            pass

        # Get metrics
        metrics = collector.get_metrics().decode("utf-8")

        # Verify histogram buckets exist
        assert "bridge_workflow_duration_seconds_bucket" in metrics
        assert 'le="0.01"' in metrics
        assert 'le="0.1"' in metrics
        assert 'le="1.0"' in metrics
        assert 'le="10.0"' in metrics
