#!/usr/bin/env python3
"""
Production Monitoring System for ONEX v2.0 Code Generation.

Comprehensive production monitoring with:
- System health monitoring
- SLA tracking and enforcement
- Alert generation and delivery
- Metrics aggregation
- Prometheus integration
- Dashboard-ready exports

ONEX v2.0 Compliance:
- Type-safe monitoring configuration
- Async/non-blocking execution
- Structured monitoring data
- Comprehensive observability

Performance Requirements:
- Monitoring overhead: <5ms per operation
- Health check interval: 30 seconds
- Metrics export interval: 10 seconds
- Alert evaluation: <10ms

Author: Code Generation System
Last Updated: 2025-11-06
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable

from omninode_bridge.production.alerting import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
)
from omninode_bridge.production.health_checks import (
    HealthChecker,
    HealthStatus,
    SystemHealthReport,
)

logger = logging.getLogger(__name__)


# === Circuit Breaker ===


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure threshold exceeded, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker pattern for monitoring operations.

    Prevents cascading failures by:
    - Tracking consecutive failures
    - Opening circuit after threshold exceeded
    - Half-open state for recovery attempts
    - Automatic reset after success

    Performance:
    - Minimal overhead (<1ms)
    - Automatic recovery
    - Graceful degradation
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            timeout_seconds: Seconds before attempting recovery (open → half-open)
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"timeout={timeout_seconds}s"
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout expired for recovery attempt
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.timeout_seconds
            ):
                self._transition_to_half_open()
            else:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN (failures: {self.failure_count})"
                )

        # Half-open state: limit calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(
                    "Circuit breaker is HALF_OPEN (max calls reached)"
                )
            self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # Transition to closed after successful recovery
            self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Stay closed
            pass

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Recovery failed, reopen circuit
            self._transition_to_open()

        elif self.state == CircuitState.CLOSED:
            # Check if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker opening: {self.failure_count} consecutive failures"
            )
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            self.success_count = 0

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(
            f"Circuit breaker half-open: attempting recovery after "
            f"{time.time() - self.last_failure_time:.0f}s"
        )
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info("Circuit breaker closed: service recovered")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.success_count = 0

    def get_state(self) -> dict[str, Any]:
        """
        Get current circuit breaker state.

        Returns:
            State dictionary
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "is_available": self.state != CircuitState.OPEN,
        }


# === SLA Models ===


@dataclass
class SLAThreshold:
    """
    SLA threshold definition.

    Attributes:
        metric_name: Name of metric to monitor
        warning_threshold: Warning threshold value
        critical_threshold: Critical threshold value
        comparison: Comparison operator (gt, lt, gte, lte, eq)
        unit: Unit of measurement for display
    """

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "lt"  # gt, lt, gte, lte, eq
    unit: str = ""

    def evaluate(self, value: float) -> tuple[bool, Optional[AlertSeverity]]:
        """
        Evaluate if value violates SLA threshold.

        Args:
            value: Current metric value

        Returns:
            Tuple of (is_violated, severity)
        """
        if self.comparison == "gt":
            if value > self.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif value > self.warning_threshold:
                return True, AlertSeverity.HIGH
        elif self.comparison == "lt":
            if value < self.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif value < self.warning_threshold:
                return True, AlertSeverity.HIGH
        elif self.comparison == "gte":
            if value >= self.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return True, AlertSeverity.HIGH
        elif self.comparison == "lte":
            if value <= self.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return True, AlertSeverity.HIGH
        elif self.comparison == "eq":
            if value == self.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif value == self.warning_threshold:
                return True, AlertSeverity.HIGH

        return False, None


@dataclass
class SLAConfiguration:
    """
    SLA configuration for production monitoring.

    Attributes:
        workflow_latency_p95_ms: P95 workflow latency SLA (ms)
        workflow_latency_p99_ms: P99 workflow latency SLA (ms)
        template_cache_hit_rate: Minimum cache hit rate (0-1)
        validation_pass_rate: Minimum validation pass rate (0-1)
        cost_per_node_usd: Maximum cost per node (USD)
        error_rate_max: Maximum error rate (0-1)
        throughput_min_ops_sec: Minimum throughput (ops/sec)
    """

    workflow_latency_p95_ms: float = 5000.0  # <5s
    workflow_latency_p99_ms: float = 10000.0  # <10s
    template_cache_hit_rate: float = 0.85  # >85%
    validation_pass_rate: float = 0.90  # >90%
    cost_per_node_usd: float = 0.05  # <$0.05
    error_rate_max: float = 0.05  # <5%
    throughput_min_ops_sec: float = 10.0  # >10 ops/sec

    def get_thresholds(self) -> list[SLAThreshold]:
        """Get all SLA thresholds."""
        return [
            SLAThreshold(
                metric_name="workflow_latency_p95",
                warning_threshold=self.workflow_latency_p95_ms * 0.8,
                critical_threshold=self.workflow_latency_p95_ms,
                comparison="gt",
                unit="ms",
            ),
            SLAThreshold(
                metric_name="workflow_latency_p99",
                warning_threshold=self.workflow_latency_p99_ms * 0.8,
                critical_threshold=self.workflow_latency_p99_ms,
                comparison="gt",
                unit="ms",
            ),
            SLAThreshold(
                metric_name="template_cache_hit_rate",
                warning_threshold=self.template_cache_hit_rate,
                critical_threshold=self.template_cache_hit_rate * 0.8,
                comparison="lt",
                unit="%",
            ),
            SLAThreshold(
                metric_name="validation_pass_rate",
                warning_threshold=self.validation_pass_rate,
                critical_threshold=self.validation_pass_rate * 0.8,
                comparison="lt",
                unit="%",
            ),
            SLAThreshold(
                metric_name="cost_per_node",
                warning_threshold=self.cost_per_node_usd * 0.8,
                critical_threshold=self.cost_per_node_usd,
                comparison="gt",
                unit="USD",
            ),
            SLAThreshold(
                metric_name="error_rate",
                warning_threshold=self.error_rate_max * 0.5,
                critical_threshold=self.error_rate_max,
                comparison="gt",
                unit="%",
            ),
            SLAThreshold(
                metric_name="throughput",
                warning_threshold=self.throughput_min_ops_sec,
                critical_threshold=self.throughput_min_ops_sec * 0.5,
                comparison="lt",
                unit="ops/sec",
            ),
        ]


# === Production Monitor ===


class ProductionMonitor:
    """
    Comprehensive production monitoring system.

    Monitors:
    - System health (all components)
    - SLA compliance
    - Performance metrics
    - Cost metrics
    - Quality metrics

    Provides:
    - Real-time monitoring
    - Alert generation
    - Metrics aggregation
    - Prometheus export
    - Dashboard data

    Performance:
    - Monitoring overhead: <5ms
    - Health check interval: 30s
    - Metrics export interval: 10s
    """

    def __init__(
        self,
        metrics_collector: Any,
        alert_manager: Optional[AlertManager] = None,
        health_checker: Optional[HealthChecker] = None,
        sla_config: Optional[SLAConfiguration] = None,
    ):
        """
        Initialize production monitor.

        Args:
            metrics_collector: Metrics collector instance
            alert_manager: Alert manager instance
            health_checker: Health checker instance
            sla_config: SLA configuration
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager or AlertManager()
        self.health_checker = health_checker or HealthChecker()
        self.sla_config = sla_config or SLAConfiguration()

        # Monitoring state
        self.is_monitoring = False
        self.last_health_check: Optional[SystemHealthReport] = None
        self.last_health_check_time: Optional[datetime] = None
        self.last_metrics_export_time: Optional[datetime] = None

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.monitoring_overhead_ms: float = 0.0

        # Circuit breakers for resilience
        self.health_check_circuit = CircuitBreaker(
            failure_threshold=5, timeout_seconds=60, half_open_max_calls=1
        )
        self.metrics_export_circuit = CircuitBreaker(
            failure_threshold=5, timeout_seconds=60, half_open_max_calls=1
        )

    async def start_monitoring(
        self,
        health_check_interval_seconds: int = 30,
        metrics_export_interval_seconds: int = 10,
    ):
        """
        Start continuous monitoring.

        Args:
            health_check_interval_seconds: Health check interval (default: 30s)
            metrics_export_interval_seconds: Metrics export interval (default: 10s)

        Performance:
            - Non-blocking background execution
            - Configurable intervals
            - Automatic error recovery
        """
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        logger.info("Starting production monitoring")
        self.is_monitoring = True

        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(
                health_check_interval_seconds,
                metrics_export_interval_seconds,
            )
        )

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.is_monitoring:
            return

        logger.info("Stopping production monitoring")
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(
        self,
        health_check_interval: int,
        metrics_export_interval: int,
    ):
        """
        Main monitoring loop with circuit breaker protection.

        Args:
            health_check_interval: Seconds between health checks
            metrics_export_interval: Seconds between metrics exports

        Performance:
            - Circuit breaker prevents cascading failures
            - Graceful degradation when services unavailable
            - Automatic recovery attempts
        """
        last_health_check = time.time()
        last_metrics_export = time.time()

        while self.is_monitoring:
            try:
                current_time = time.time()

                # Health check interval (with circuit breaker)
                if current_time - last_health_check >= health_check_interval:
                    try:
                        await self.health_check_circuit.call(self._perform_health_check)
                        last_health_check = current_time
                    except CircuitOpenError as e:
                        logger.warning(
                            f"Health check skipped: {e} (will retry after timeout)"
                        )
                        # Continue monitoring even if health check circuit is open
                        last_health_check = current_time
                    except Exception as e:
                        logger.error(f"Health check failed: {e}", exc_info=True)
                        # Circuit breaker will track failure

                # Metrics export interval (with circuit breaker)
                if current_time - last_metrics_export >= metrics_export_interval:
                    try:
                        await self.metrics_export_circuit.call(self._export_metrics)
                        last_metrics_export = current_time
                    except CircuitOpenError as e:
                        logger.warning(
                            f"Metrics export skipped: {e} (will retry after timeout)"
                        )
                        # Continue monitoring even if metrics export circuit is open
                        last_metrics_export = current_time
                    except Exception as e:
                        logger.error(f"Metrics export failed: {e}", exc_info=True)
                        # Circuit breaker will track failure

                # Small sleep to prevent busy loop
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def _perform_health_check(self):
        """Perform system health check."""
        start_time = time.perf_counter()

        try:
            # Execute health check
            health_report = await self.health_checker.check_system_health()

            self.last_health_check = health_report
            self.last_health_check_time = datetime.now()

            # Generate alerts for unhealthy components
            for result in health_report.component_results:
                if result.status == HealthStatus.UNHEALTHY:
                    alert = Alert(
                        alert_id=f"health_{result.component}_{datetime.now().timestamp()}",
                        alert_type=AlertType.HEALTH_CHECK_FAILED,
                        severity=AlertSeverity.CRITICAL,
                        component=result.component,
                        message=f"Health check failed: {result.message}",
                        threshold_violated={"status": HealthStatus.UNHEALTHY.value},
                        current_value=result.status.value,
                        metadata=result.details,
                    )
                    await self.alert_manager.send_alert(alert)

            logger.debug(
                f"Health check completed: {health_report.overall_status.value} "
                f"({health_report.healthy_count}/{len(health_report.component_results)} healthy)"
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)

        finally:
            self.monitoring_overhead_ms = (time.perf_counter() - start_time) * 1000

    async def _export_metrics(self):
        """Export metrics for Prometheus/dashboards."""
        try:
            # Get current metrics summary
            summary = self.metrics_collector.get_performance_summary()

            # Log metrics summary
            logger.debug(f"Metrics export: {summary.get('overall_grade', 'UNKNOWN')}")

            self.last_metrics_export_time = datetime.now()

        except Exception as e:
            logger.error(f"Metrics export failed: {e}", exc_info=True)

    async def check_system_health(self) -> SystemHealthReport:
        """
        Check system health on-demand.

        Returns:
            System health report
        """
        health_report = await self.health_checker.check_system_health()
        self.last_health_check = health_report
        self.last_health_check_time = datetime.now()
        return health_report

    async def monitor_slas(self, metrics: dict[str, Any]) -> list[Alert]:
        """
        Monitor SLAs and generate alerts for violations.

        Args:
            metrics: Current system metrics

        Returns:
            List of SLA violation alerts

        Performance:
            - <10ms SLA evaluation
            - Parallel threshold evaluation
        """
        alerts: list[Alert] = []

        # Evaluate all SLA thresholds
        for threshold in self.sla_config.get_thresholds():
            metric_value = metrics.get(threshold.metric_name)

            if metric_value is None:
                continue

            is_violated, severity = threshold.evaluate(metric_value)

            if is_violated:
                # Create SLA violation alert
                alert = Alert(
                    alert_id=f"sla_{threshold.metric_name}_{datetime.now().timestamp()}",
                    alert_type=(
                        AlertType.LATENCY_VIOLATION
                        if "latency" in threshold.metric_name
                        else AlertType.CUSTOM
                    ),
                    severity=severity,
                    component="sla_monitor",
                    message=(
                        f"SLA violation: {threshold.metric_name} = "
                        f"{metric_value}{threshold.unit} "
                        f"(threshold: {threshold.warning_threshold if severity == AlertSeverity.HIGH else threshold.critical_threshold}{threshold.unit})"
                    ),
                    threshold_violated={
                        "metric": threshold.metric_name,
                        "threshold": (
                            threshold.warning_threshold
                            if severity == AlertSeverity.HIGH
                            else threshold.critical_threshold
                        ),
                        "comparison": threshold.comparison,
                    },
                    current_value=metric_value,
                )

                alerts.append(alert)
                await self.alert_manager.send_alert(alert)

        # Also evaluate alert manager rules
        rule_alerts = await self.alert_manager.evaluate_rules(metrics)
        alerts.extend(rule_alerts)

        # Send alerts
        for alert in rule_alerts:
            await self.alert_manager.send_alert(alert)

        return alerts

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        metrics_lines = []

        # Health metrics
        if self.last_health_check:
            metrics_lines.append("# HELP system_health System health status")
            metrics_lines.append("# TYPE system_health gauge")

            health_values = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0.0,
                HealthStatus.UNKNOWN: -1.0,
            }

            metrics_lines.append(
                f"system_health {health_values[self.last_health_check.overall_status]}"
            )

            # Component health
            metrics_lines.append("# HELP component_health Component health status")
            metrics_lines.append("# TYPE component_health gauge")

            for result in self.last_health_check.component_results:
                health_value = health_values[result.status]
                metrics_lines.append(
                    f'component_health{{component="{result.component}"}} {health_value}'
                )

        # Monitoring overhead
        metrics_lines.append(
            "# HELP monitoring_overhead_ms Monitoring overhead in milliseconds"
        )
        metrics_lines.append("# TYPE monitoring_overhead_ms gauge")
        metrics_lines.append(f"monitoring_overhead_ms {self.monitoring_overhead_ms}")

        # Circuit breaker metrics
        metrics_lines.append(
            "# HELP circuit_breaker_state Circuit breaker state (0=open, 0.5=half_open, 1=closed)"
        )
        metrics_lines.append("# TYPE circuit_breaker_state gauge")

        circuit_state_values = {
            CircuitState.CLOSED: 1.0,
            CircuitState.HALF_OPEN: 0.5,
            CircuitState.OPEN: 0.0,
        }

        health_check_state = self.health_check_circuit.get_state()
        metrics_export_state = self.metrics_export_circuit.get_state()

        metrics_lines.append(
            f'circuit_breaker_state{{component="health_check"}} '
            f'{circuit_state_values[CircuitState(health_check_state["state"])]}'
        )
        metrics_lines.append(
            f'circuit_breaker_state{{component="metrics_export"}} '
            f'{circuit_state_values[CircuitState(metrics_export_state["state"])]}'
        )

        metrics_lines.append(
            "# HELP circuit_breaker_failures Circuit breaker failure count"
        )
        metrics_lines.append("# TYPE circuit_breaker_failures counter")
        metrics_lines.append(
            f'circuit_breaker_failures{{component="health_check"}} '
            f'{health_check_state["failure_count"]}'
        )
        metrics_lines.append(
            f'circuit_breaker_failures{{component="metrics_export"}} '
            f'{metrics_export_state["failure_count"]}'
        )

        # Alert statistics
        alert_stats = self.alert_manager.get_alert_statistics()

        metrics_lines.append("# HELP alerts_total Total number of alerts")
        metrics_lines.append("# TYPE alerts_total counter")
        metrics_lines.append(f"alerts_total {alert_stats['total_alerts']}")

        # Alerts by severity
        for severity, count in alert_stats.get("by_severity", {}).items():
            metrics_lines.append(f'alerts_total{{severity="{severity}"}} {count}')

        return "\n".join(metrics_lines)

    def get_monitoring_status(self) -> dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Monitoring status dictionary with circuit breaker states
        """
        status = {
            "is_monitoring": self.is_monitoring,
            "monitoring_overhead_ms": round(self.monitoring_overhead_ms, 2),
            "last_health_check_time": (
                self.last_health_check_time.isoformat()
                if self.last_health_check_time
                else None
            ),
            "last_metrics_export_time": (
                self.last_metrics_export_time.isoformat()
                if self.last_metrics_export_time
                else None
            ),
            "circuit_breakers": {
                "health_check": self.health_check_circuit.get_state(),
                "metrics_export": self.metrics_export_circuit.get_state(),
            },
        }

        if self.last_health_check:
            status["health_summary"] = {
                "overall_status": self.last_health_check.overall_status.value,
                "healthy_count": self.last_health_check.healthy_count,
                "degraded_count": self.last_health_check.degraded_count,
                "unhealthy_count": self.last_health_check.unhealthy_count,
                "total_components": len(self.last_health_check.component_results),
            }

        return status

    def get_sla_compliance_report(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Get SLA compliance report.

        Args:
            metrics: Current system metrics

        Returns:
            SLA compliance report
        """
        compliance = {}

        for threshold in self.sla_config.get_thresholds():
            metric_value = metrics.get(threshold.metric_name)

            if metric_value is None:
                compliance[threshold.metric_name] = {
                    "status": "unknown",
                    "value": None,
                }
                continue

            is_violated, severity = threshold.evaluate(metric_value)

            compliance[threshold.metric_name] = {
                "status": "violated" if is_violated else "compliant",
                "value": metric_value,
                "unit": threshold.unit,
                "threshold": {
                    "warning": threshold.warning_threshold,
                    "critical": threshold.critical_threshold,
                },
                "severity": severity.value if severity else None,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_compliant": all(
                c["status"] == "compliant" for c in compliance.values()
            ),
            "metrics": compliance,
        }
