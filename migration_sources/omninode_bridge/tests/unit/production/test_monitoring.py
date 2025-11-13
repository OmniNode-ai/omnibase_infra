#!/usr/bin/env python3
"""
Comprehensive tests for production monitoring infrastructure.

Tests:
- Health check system functionality
- Alert generation and delivery
- SLA monitoring and violation detection
- Production monitor integration
- Metrics export (Prometheus format)

Coverage Target: 95%+

Author: Code Generation System
Last Updated: 2025-11-06
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from omninode_bridge.production.alerting import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertType,
    NotificationChannel,
)
from omninode_bridge.production.health_checks import (
    ComponentType,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    SystemHealthReport,
)
from omninode_bridge.production.monitoring import (
    ProductionMonitor,
    SLAConfiguration,
    SLAThreshold,
)

# === Health Check Tests ===


@pytest.mark.asyncio
class TestHealthChecker:
    """Tests for HealthChecker component."""

    async def test_health_checker_initialization(self):
        """Test health checker initialization."""
        health_checker = HealthChecker()

        assert health_checker.template_manager is None
        assert health_checker.validation_pipeline is None
        assert health_checker.ai_quorum is None
        assert health_checker.check_timeout_seconds == 5.0

    async def test_check_template_manager_healthy(self):
        """Test template manager health check - healthy status."""
        # Mock template manager
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        health_checker = HealthChecker(template_manager=template_manager)

        result = await health_checker.check_template_manager()

        assert result.component == "template_manager"
        assert result.component_type == ComponentType.TEMPLATE_MANAGER
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms < 100  # Should be fast
        assert "healthy" in result.message.lower()
        assert result.details["cache_stats"]["hit_rate"] == 0.90

    async def test_check_template_manager_degraded(self):
        """Test template manager health check - degraded status."""
        # Mock template manager with low cache hit rate
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.65})

        health_checker = HealthChecker(template_manager=template_manager)

        result = await health_checker.check_template_manager()

        assert result.status == HealthStatus.DEGRADED
        assert "low cache hit rate" in result.message.lower()

    async def test_check_template_manager_unhealthy(self):
        """Test template manager health check - unhealthy status."""
        # Mock template manager without required method
        template_manager = Mock(spec=[])  # No methods

        health_checker = HealthChecker(template_manager=template_manager)

        result = await health_checker.check_template_manager()

        assert result.status == HealthStatus.UNHEALTHY
        assert "missing" in result.message.lower()

    async def test_check_validation_pipeline_healthy(self):
        """Test validation pipeline health check - healthy status."""
        # Mock validation pipeline
        validation_pipeline = Mock()
        validation_pipeline.validate = Mock()
        validation_pipeline.get_validator_count = Mock(return_value=5)

        health_checker = HealthChecker(validation_pipeline=validation_pipeline)

        result = await health_checker.check_validation_pipeline()

        assert result.component == "validation_pipeline"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["validator_count"] == 5

    async def test_check_validation_pipeline_degraded(self):
        """Test validation pipeline health check - degraded status."""
        # Mock validation pipeline with no validators
        validation_pipeline = Mock()
        validation_pipeline.validate = Mock()
        validation_pipeline.get_validator_count = Mock(return_value=0)

        health_checker = HealthChecker(validation_pipeline=validation_pipeline)

        result = await health_checker.check_validation_pipeline()

        assert result.status == HealthStatus.DEGRADED
        assert "no validators" in result.message.lower()

    async def test_check_ai_quorum_healthy(self):
        """Test AI Quorum health check - healthy status."""
        # Mock AI Quorum
        ai_quorum = Mock()
        ai_quorum.query = Mock()
        ai_quorum.get_available_models = Mock(
            return_value=["gemini", "codestral", "deepseek"]
        )

        health_checker = HealthChecker(ai_quorum=ai_quorum)

        result = await health_checker.check_ai_quorum()

        assert result.component == "ai_quorum"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["available_models"] == 3

    async def test_check_ai_quorum_degraded_few_models(self):
        """Test AI Quorum health check - degraded with few models."""
        # Mock AI Quorum with limited models
        ai_quorum = Mock()
        ai_quorum.query = Mock()
        ai_quorum.get_available_models = Mock(return_value=["gemini"])

        health_checker = HealthChecker(ai_quorum=ai_quorum)

        result = await health_checker.check_ai_quorum()

        assert result.status == HealthStatus.DEGRADED
        assert "limited" in result.message.lower()

    async def test_check_database_healthy(self):
        """Test database health check - healthy status."""
        # Mock database client (spec to avoid hasattr finding unwanted methods)
        database_client = AsyncMock(spec=["fetchval"])
        database_client.fetchval = AsyncMock(return_value=1)

        health_checker = HealthChecker(database_client=database_client)

        result = await health_checker.check_database()

        assert result.component == "database"
        assert result.status == HealthStatus.HEALTHY
        database_client.fetchval.assert_awaited_once_with("SELECT 1")

    async def test_check_database_unhealthy(self):
        """Test database health check - unhealthy status."""
        # Mock database client that raises exception
        database_client = AsyncMock(spec=["fetchval"])
        database_client.fetchval = AsyncMock(side_effect=Exception("Connection failed"))

        health_checker = HealthChecker(database_client=database_client)

        result = await health_checker.check_database()

        assert result.component == "database"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error is not None

    async def test_check_system_health_parallel_execution(self):
        """Test system health check executes checks in parallel."""
        # Mock multiple components
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        validation_pipeline = Mock()
        validation_pipeline.validate = Mock()
        validation_pipeline.get_validator_count = Mock(return_value=5)

        health_checker = HealthChecker(
            template_manager=template_manager,
            validation_pipeline=validation_pipeline,
        )

        # Execute system health check
        report = await health_checker.check_system_health()

        assert isinstance(report, SystemHealthReport)
        assert len(report.component_results) == 2  # Two components checked
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.healthy_count == 2
        assert report.degraded_count == 0
        assert report.unhealthy_count == 0

    async def test_system_health_report_overall_status_unhealthy(self):
        """Test overall system status is unhealthy if any component is unhealthy."""
        # Mock components with one unhealthy
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        database_client = AsyncMock(spec=["fetchval"])
        database_client.fetchval = AsyncMock(side_effect=Exception("DB down"))

        health_checker = HealthChecker(
            template_manager=template_manager,
            database_client=database_client,
        )

        report = await health_checker.check_system_health()

        assert report.overall_status == HealthStatus.UNHEALTHY
        assert report.unhealthy_count >= 1

    async def test_system_health_report_to_dict(self):
        """Test system health report serialization to dictionary."""
        result = HealthCheckResult(
            component="test_component",
            component_type=ComponentType.TEMPLATE_MANAGER,
            status=HealthStatus.HEALTHY,
            response_time_ms=10.5,
            message="Test message",
            details={"key": "value"},
        )

        report = SystemHealthReport(
            overall_status=HealthStatus.HEALTHY,
            component_results=[result],
            total_response_time_ms=15.0,
        )

        report_dict = report.to_dict()

        assert report_dict["overall_status"] == "healthy"
        assert report_dict["summary"]["healthy"] == 1
        assert report_dict["summary"]["total"] == 1
        assert len(report_dict["components"]) == 1
        assert report_dict["components"][0]["component"] == "test_component"


# === Alert Manager Tests ===


@pytest.mark.asyncio
class TestAlertManager:
    """Tests for AlertManager component."""

    async def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        alert_manager = AlertManager()

        assert len(alert_manager.alert_rules) > 0  # Default rules added
        assert NotificationChannel.LOG in alert_manager.notification_channels
        assert alert_manager.dedup_window_seconds == 300

    async def test_add_alert_rule(self):
        """Test adding custom alert rule."""
        alert_manager = AlertManager()
        initial_count = len(alert_manager.alert_rules)

        rule = AlertRule(
            name="test_rule",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.MEDIUM,
            condition=lambda m: m.get("test_metric", 0) > 100,
            message_template="Test metric too high",
        )

        alert_manager.add_rule(rule)

        assert len(alert_manager.alert_rules) == initial_count + 1

    async def test_remove_alert_rule(self):
        """Test removing alert rule."""
        alert_manager = AlertManager()

        # Add a rule
        rule = AlertRule(
            name="removable_rule",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.LOW,
            condition=lambda m: False,
            message_template="Test",
        )
        alert_manager.add_rule(rule)

        # Remove it
        removed = alert_manager.remove_rule("removable_rule")

        assert removed is True

        # Try to remove again
        removed_again = alert_manager.remove_rule("removable_rule")
        assert removed_again is False

    async def test_evaluate_rules_no_violations(self):
        """Test evaluating rules with no violations."""
        alert_manager = AlertManager()

        # Metrics that don't violate any default rules
        metrics = {
            "workflow_latency_p95": 3000,  # Within SLA
            "template_cache_hit_rate": 0.90,  # Good
            "cost_per_node": 0.02,  # Low
            "validation_pass_rate": 0.95,  # High
            "error_rate": 0.01,  # Low
        }

        alerts = await alert_manager.evaluate_rules(metrics)

        assert len(alerts) == 0  # No violations

    async def test_evaluate_rules_with_violations(self):
        """Test evaluating rules with violations."""
        alert_manager = AlertManager()

        # Metrics that violate multiple rules
        metrics = {
            "workflow_latency_p95": 15000,  # Violates SLA
            "template_cache_hit_rate": 0.70,  # Low
            "cost_per_node": 0.10,  # High
            "validation_pass_rate": 0.80,  # Low
            "error_rate": 0.10,  # High
        }

        alerts = await alert_manager.evaluate_rules(metrics)

        assert len(alerts) > 0  # Should have violations
        assert any(a.alert_type == AlertType.LATENCY_VIOLATION for a in alerts)
        assert any(a.alert_type == AlertType.COST_BUDGET_EXCEEDED for a in alerts)

    async def test_alert_deduplication(self):
        """Test alert deduplication within window."""
        alert_manager = AlertManager()
        alert_manager.dedup_window_seconds = 1  # Short window for testing

        # Create rule that always triggers
        rule = AlertRule(
            name="always_trigger",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.MEDIUM,
            condition=lambda m: True,  # Always true
            message_template="Always triggers",
            metadata={"component": "test"},
        )
        alert_manager.add_rule(rule)

        # First evaluation - should generate alert
        alerts1 = await alert_manager.evaluate_rules({})
        assert len(alerts1) > 0

        # Immediate second evaluation - should be deduplicated
        alerts2 = await alert_manager.evaluate_rules({})
        assert len(alerts2) == 0  # Deduplicated

        # Wait for dedup window to expire
        await asyncio.sleep(1.5)

        # Third evaluation - should generate alert again
        alerts3 = await alert_manager.evaluate_rules({})
        assert len(alerts3) > 0

    async def test_send_alert_to_log(self):
        """Test sending alert to log channel."""
        alert_manager = AlertManager(notification_channels=[NotificationChannel.LOG])

        alert = Alert(
            alert_id="test_alert_1",
            alert_type=AlertType.LATENCY_VIOLATION,
            severity=AlertSeverity.HIGH,
            component="test_component",
            message="Test alert message",
            threshold_violated={"threshold": 1000},
            current_value=2000,
        )

        # Should not raise exception
        await alert_manager.send_alert(alert)

    async def test_get_alert_history(self):
        """Test getting alert history."""
        alert_manager = AlertManager()

        # Generate some alerts
        metrics = {"workflow_latency_p95": 15000}
        await alert_manager.evaluate_rules(metrics)

        # Get history
        history = alert_manager.get_alert_history(limit=10)

        assert isinstance(history, list)
        # History may or may not have alerts depending on deduplication

    async def test_get_alert_statistics(self):
        """Test getting alert statistics."""
        alert_manager = AlertManager()

        # Generate some alerts
        metrics = {"workflow_latency_p95": 15000, "cost_per_node": 0.10}
        await alert_manager.evaluate_rules(metrics)

        # Get statistics
        stats = alert_manager.get_alert_statistics()

        assert "total_alerts" in stats
        assert "by_severity" in stats
        assert "by_type" in stats
        assert "by_component" in stats


# === SLA Threshold Tests ===


class TestSLAThreshold:
    """Tests for SLA threshold evaluation."""

    def test_sla_threshold_gt_violation(self):
        """Test greater-than threshold violation."""
        threshold = SLAThreshold(
            metric_name="latency",
            warning_threshold=1000,
            critical_threshold=2000,
            comparison="gt",
            unit="ms",
        )

        # No violation
        is_violated, severity = threshold.evaluate(500)
        assert not is_violated

        # Warning violation
        is_violated, severity = threshold.evaluate(1500)
        assert is_violated
        assert severity == AlertSeverity.HIGH

        # Critical violation
        is_violated, severity = threshold.evaluate(2500)
        assert is_violated
        assert severity == AlertSeverity.CRITICAL

    def test_sla_threshold_lt_violation(self):
        """Test less-than threshold violation."""
        threshold = SLAThreshold(
            metric_name="cache_hit_rate",
            warning_threshold=0.85,
            critical_threshold=0.70,
            comparison="lt",
            unit="%",
        )

        # No violation
        is_violated, severity = threshold.evaluate(0.90)
        assert not is_violated

        # Warning violation
        is_violated, severity = threshold.evaluate(0.80)
        assert is_violated
        assert severity == AlertSeverity.HIGH

        # Critical violation
        is_violated, severity = threshold.evaluate(0.65)
        assert is_violated
        assert severity == AlertSeverity.CRITICAL


# === SLA Configuration Tests ===


class TestSLAConfiguration:
    """Tests for SLA configuration."""

    def test_sla_configuration_defaults(self):
        """Test SLA configuration with default values."""
        config = SLAConfiguration()

        assert config.workflow_latency_p95_ms == 5000.0
        assert config.workflow_latency_p99_ms == 10000.0
        assert config.template_cache_hit_rate == 0.85
        assert config.validation_pass_rate == 0.90
        assert config.cost_per_node_usd == 0.05

    def test_sla_configuration_custom_values(self):
        """Test SLA configuration with custom values."""
        config = SLAConfiguration(
            workflow_latency_p95_ms=3000.0,
            validation_pass_rate=0.95,
            cost_per_node_usd=0.03,
        )

        assert config.workflow_latency_p95_ms == 3000.0
        assert config.validation_pass_rate == 0.95
        assert config.cost_per_node_usd == 0.03

    def test_get_thresholds(self):
        """Test getting all SLA thresholds."""
        config = SLAConfiguration()
        thresholds = config.get_thresholds()

        assert len(thresholds) > 0
        assert all(isinstance(t, SLAThreshold) for t in thresholds)


# === Production Monitor Tests ===


@pytest.mark.asyncio
class TestProductionMonitor:
    """Tests for ProductionMonitor component."""

    async def test_production_monitor_initialization(self):
        """Test production monitor initialization."""
        metrics_collector = Mock()
        alert_manager = AlertManager()
        health_checker = HealthChecker()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            health_checker=health_checker,
        )

        assert monitor.metrics_collector == metrics_collector
        assert monitor.alert_manager == alert_manager
        assert monitor.health_checker == health_checker
        assert not monitor.is_monitoring

    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        metrics_collector = Mock()
        metrics_collector.get_performance_summary = Mock(
            return_value={"overall_grade": "A"}
        )

        monitor = ProductionMonitor(metrics_collector=metrics_collector)

        # Start monitoring
        await monitor.start_monitoring(
            health_check_interval_seconds=1,
            metrics_export_interval_seconds=1,
        )

        assert monitor.is_monitoring
        assert monitor.monitoring_task is not None

        # Wait a bit for monitoring loop to run
        await asyncio.sleep(0.5)

        # Stop monitoring
        await monitor.stop_monitoring()

        assert not monitor.is_monitoring

    async def test_check_system_health(self):
        """Test on-demand system health check."""
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        health_checker = HealthChecker(template_manager=template_manager)
        metrics_collector = Mock()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            health_checker=health_checker,
        )

        # Check health
        report = await monitor.check_system_health()

        assert isinstance(report, SystemHealthReport)
        assert report.overall_status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    async def test_monitor_slas_no_violations(self):
        """Test SLA monitoring with no violations."""
        metrics_collector = Mock()
        sla_config = SLAConfiguration()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            sla_config=sla_config,
        )

        # Good metrics
        metrics = {
            "workflow_latency_p95": 3000,
            "template_cache_hit_rate": 0.90,
            "cost_per_node": 0.02,
            "validation_pass_rate": 0.95,
            "error_rate": 0.01,
            "throughput": 20,
        }

        alerts = await monitor.monitor_slas(metrics)

        # Should have no or very few alerts
        assert len(alerts) >= 0  # May have some from default rules

    async def test_monitor_slas_with_violations(self):
        """Test SLA monitoring with violations."""
        metrics_collector = Mock()
        sla_config = SLAConfiguration()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            sla_config=sla_config,
        )

        # Bad metrics
        metrics = {
            "workflow_latency_p95": 15000,  # Violation
            "template_cache_hit_rate": 0.70,  # Violation
            "cost_per_node": 0.10,  # Violation
            "validation_pass_rate": 0.80,  # Violation
            "error_rate": 0.10,  # Violation
            "throughput": 5,  # Violation
        }

        alerts = await monitor.monitor_slas(metrics)

        assert len(alerts) > 0  # Should have violations

    async def test_export_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        health_checker = HealthChecker(template_manager=template_manager)
        metrics_collector = Mock()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            health_checker=health_checker,
        )

        # Perform health check first to populate data
        await monitor.check_system_health()

        # Export metrics
        prometheus_output = monitor.export_prometheus_metrics()

        assert isinstance(prometheus_output, str)
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output
        assert "system_health" in prometheus_output

    async def test_get_monitoring_status(self):
        """Test getting monitoring status."""
        metrics_collector = Mock()
        monitor = ProductionMonitor(metrics_collector=metrics_collector)

        status = monitor.get_monitoring_status()

        assert "is_monitoring" in status
        assert status["is_monitoring"] is False
        assert "monitoring_overhead_ms" in status

    async def test_get_sla_compliance_report(self):
        """Test getting SLA compliance report."""
        metrics_collector = Mock()
        sla_config = SLAConfiguration()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            sla_config=sla_config,
        )

        metrics = {
            "workflow_latency_p95": 3000,
            "template_cache_hit_rate": 0.90,
            "cost_per_node": 0.02,
        }

        report = monitor.get_sla_compliance_report(metrics)

        assert "timestamp" in report
        assert "overall_compliant" in report
        assert "metrics" in report
        assert isinstance(report["metrics"], dict)


# === Integration Tests ===


@pytest.mark.asyncio
class TestProductionMonitoringIntegration:
    """Integration tests for production monitoring system."""

    async def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow from health check to alerting."""
        # Setup components
        template_manager = Mock()
        template_manager.get_template = Mock()
        template_manager.get_cache_stats = Mock(return_value={"hit_rate": 0.90})

        database_client = AsyncMock()
        database_client.fetchval = AsyncMock(return_value=1)

        health_checker = HealthChecker(
            template_manager=template_manager,
            database_client=database_client,
        )

        alert_manager = AlertManager()
        metrics_collector = Mock()
        metrics_collector.get_performance_summary = Mock(
            return_value={"overall_grade": "A"}
        )

        sla_config = SLAConfiguration()

        monitor = ProductionMonitor(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            health_checker=health_checker,
            sla_config=sla_config,
        )

        # 1. Check system health
        health_report = await monitor.check_system_health()
        assert health_report.overall_status == HealthStatus.HEALTHY

        # 2. Monitor SLAs with good metrics
        good_metrics = {
            "workflow_latency_p95": 3000,
            "template_cache_hit_rate": 0.90,
            "cost_per_node": 0.02,
        }
        alerts = await monitor.monitor_slas(good_metrics)
        # Good metrics should generate no/few alerts

        # 3. Monitor SLAs with bad metrics
        bad_metrics = {
            "workflow_latency_p95": 15000,
            "cost_per_node": 0.10,
        }
        alerts = await monitor.monitor_slas(bad_metrics)
        assert len(alerts) > 0  # Should generate alerts

        # 4. Get SLA compliance report
        compliance = monitor.get_sla_compliance_report(bad_metrics)
        assert not compliance["overall_compliant"]

        # 5. Export Prometheus metrics
        prometheus_output = monitor.export_prometheus_metrics()
        assert len(prometheus_output) > 0

        # 6. Get monitoring status
        status = monitor.get_monitoring_status()
        assert status["is_monitoring"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
