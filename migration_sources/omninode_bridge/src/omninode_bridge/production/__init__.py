"""
Production Hardening Infrastructure for ONEX v2.0 Code Generation.

This module provides comprehensive production monitoring, alerting, and health
checking capabilities for the code generation system.

Components:
- health_checks: System and component health monitoring
- alerting: Alert generation and notification delivery
- monitoring: Comprehensive production monitoring with SLA tracking

ONEX v2.0 Compliance:
- Type-safe monitoring and alerting
- Async/non-blocking execution
- Structured monitoring data
- Prometheus integration

Example Usage:
    >>> from omninode_bridge.production import (
    ...     ProductionMonitor,
    ...     AlertManager,
    ...     HealthChecker,
    ...     SLAConfiguration,
    ... )
    >>>
    >>> # Initialize monitoring
    >>> alert_manager = AlertManager()
    >>> health_checker = HealthChecker()
    >>> sla_config = SLAConfiguration(
    ...     workflow_latency_p95_ms=5000,
    ...     validation_pass_rate=0.90,
    ... )
    >>>
    >>> # Create production monitor
    >>> monitor = ProductionMonitor(
    ...     metrics_collector=metrics_collector,
    ...     alert_manager=alert_manager,
    ...     health_checker=health_checker,
    ...     sla_config=sla_config,
    ... )
    >>>
    >>> # Start monitoring
    >>> await monitor.start_monitoring()
    >>>
    >>> # Check health
    >>> health_report = await monitor.check_system_health()
    >>> print(health_report.overall_status)

Author: Code Generation System
Last Updated: 2025-11-06
"""

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

__all__ = [
    # Alerting
    "Alert",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertType",
    "NotificationChannel",
    # Health Checks
    "ComponentType",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
    "SystemHealthReport",
    # Monitoring
    "ProductionMonitor",
    "SLAConfiguration",
    "SLAThreshold",
]

__version__ = "1.0.0"
