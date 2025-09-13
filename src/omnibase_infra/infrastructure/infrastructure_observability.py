"""Infrastructure Observability and Metrics Collection.

Comprehensive observability framework for ONEX infrastructure including:
- Prometheus-style metrics for circuit breakers and event publishing
- Performance tracking and trend analysis  
- Health check aggregation and monitoring
- Alert generation for critical infrastructure issues
- Dashboard-ready metrics export
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
from uuid import UUID

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_infra.infrastructure.event_bus_circuit_breaker import (
    EventBusCircuitBreaker,
    CircuitBreakerState
)


class MetricType(Enum):
    """Types of metrics collected by observability system."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Summary statistics


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"        # Service-affecting issues
    HIGH = "high"               # Performance degradation
    MEDIUM = "medium"           # Potential issues
    LOW = "low"                 # Informational


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Infrastructure alert."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """Performance snapshot for trend analysis."""
    timestamp: datetime
    circuit_breaker_health: Dict[str, Any]
    event_publishing_metrics: Dict[str, float]
    infrastructure_health: Dict[str, Any]
    performance_indicators: Dict[str, float]


class InfrastructureObservability:
    """
    Comprehensive observability system for ONEX infrastructure.
    
    Provides:
    - Real-time metrics collection and aggregation
    - Performance trend analysis and predictions
    - Health monitoring and alerting
    - Dashboard-ready metrics export
    - Integration with external monitoring systems
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize observability system.
        
        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self.metrics: Deque[MetricPoint] = deque()
        self.alerts: List[Alert] = []
        self.performance_snapshots: Deque[PerformanceSnapshot] = deque()
        
        # Circuit breaker references for monitoring
        self.circuit_breakers: Dict[str, EventBusCircuitBreaker] = {}
        
        # Performance tracking
        self.event_latencies: Deque[float] = deque(maxlen=1000)
        self.error_rates: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert thresholds (configurable)
        self.alert_thresholds = {
            'circuit_breaker_open': {'severity': AlertSeverity.CRITICAL, 'threshold': 1},
            'high_error_rate': {'severity': AlertSeverity.HIGH, 'threshold': 0.1},  # 10%
            'high_latency': {'severity': AlertSeverity.HIGH, 'threshold': 5.0},     # 5 seconds
            'queue_capacity': {'severity': AlertSeverity.MEDIUM, 'threshold': 0.8}, # 80% full
            'dead_letter_growth': {'severity': AlertSeverity.MEDIUM, 'threshold': 10}, # 10 events
        }
        
        self.logger = logging.getLogger(f"{__name__}.InfrastructureObservability")
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_background_monitoring()
    
    def register_circuit_breaker(self, name: str, circuit_breaker: EventBusCircuitBreaker):
        """Register circuit breaker for monitoring."""
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker for monitoring: {name}")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric data point."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.metrics.append(metric)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def record_event_latency(self, latency_seconds: float):
        """Record event publishing latency."""
        self.event_latencies.append(latency_seconds)
        
        # Record as metric
        self.record_metric(
            "event_publishing_latency_seconds",
            latency_seconds,
            metric_type=MetricType.HISTOGRAM
        )
        
        # Check for high latency alert
        if latency_seconds > self.alert_thresholds['high_latency']['threshold']:
            self._create_alert(
                "high_latency",
                f"High event publishing latency: {latency_seconds:.2f}s",
                AlertSeverity.HIGH,
                details={"latency_seconds": latency_seconds}
            )
    
    def record_error_rate(self, service: str, error_rate: float):
        """Record error rate for a service."""
        self.error_rates[service].append(error_rate)
        
        # Record as metric
        self.record_metric(
            "service_error_rate",
            error_rate,
            labels={"service": service},
            metric_type=MetricType.GAUGE
        )
        
        # Check for high error rate alert
        if error_rate > self.alert_thresholds['high_error_rate']['threshold']:
            self._create_alert(
                f"high_error_rate_{service}",
                f"High error rate for {service}: {error_rate:.1%}",
                AlertSeverity.HIGH,
                details={"service": service, "error_rate": error_rate}
            )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current infrastructure metrics snapshot."""
        current_time = datetime.now()
        
        # Circuit breaker metrics
        circuit_breaker_metrics = {}
        for name, cb in self.circuit_breakers.items():
            status = cb.get_health_status()
            circuit_breaker_metrics[name] = {
                "state": status["circuit_state"],
                "is_healthy": status["is_healthy"],
                "failure_count": status["failure_count"],
                "total_events": status["metrics"]["total_events"],
                "success_rate": status["metrics"]["success_rate"],
                "queued_events": status["queued_events"],
                "dead_letter_events": status["dead_letter_events"]
            }
        
        # Performance metrics
        performance_metrics = {
            "avg_latency": sum(self.event_latencies) / len(self.event_latencies) if self.event_latencies else 0,
            "p95_latency": self._calculate_percentile(list(self.event_latencies), 95) if self.event_latencies else 0,
            "p99_latency": self._calculate_percentile(list(self.event_latencies), 99) if self.event_latencies else 0,
        }
        
        # Error rate metrics
        error_rate_metrics = {}
        for service, rates in self.error_rates.items():
            if rates:
                error_rate_metrics[service] = {
                    "current": rates[-1],
                    "average": sum(rates) / len(rates),
                    "max": max(rates)
                }
        
        return {
            "timestamp": current_time.isoformat(),
            "circuit_breakers": circuit_breaker_metrics,
            "performance": performance_metrics,
            "error_rates": error_rate_metrics,
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "total_metrics": len(self.metrics)
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall infrastructure health summary."""
        healthy_circuits = 0
        total_circuits = len(self.circuit_breakers)
        
        for cb in self.circuit_breakers.values():
            if cb.is_healthy():
                healthy_circuits += 1
        
        # Calculate overall health score
        circuit_health_score = (healthy_circuits / total_circuits * 100) if total_circuits > 0 else 100
        
        # Performance health score based on latency
        avg_latency = sum(self.event_latencies) / len(self.event_latencies) if self.event_latencies else 0
        performance_health_score = max(0, 100 - (avg_latency / 0.1 * 10))  # Penalize high latency
        
        # Error rate health score
        recent_error_rates = []
        for rates in self.error_rates.values():
            if rates:
                recent_error_rates.append(rates[-1])
        
        avg_error_rate = sum(recent_error_rates) / len(recent_error_rates) if recent_error_rates else 0
        error_health_score = max(0, 100 - (avg_error_rate * 1000))  # Penalize error rates
        
        # Overall health score (weighted average)
        overall_health_score = (
            circuit_health_score * 0.4 +
            performance_health_score * 0.3 +
            error_health_score * 0.3
        )
        
        # Determine health status
        if overall_health_score >= 90:
            health_status = "healthy"
        elif overall_health_score >= 70:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return {
            "overall_health_score": round(overall_health_score, 1),
            "health_status": health_status,
            "circuit_breaker_health": {
                "healthy_circuits": healthy_circuits,
                "total_circuits": total_circuits,
                "health_score": round(circuit_health_score, 1)
            },
            "performance_health": {
                "avg_latency_seconds": round(avg_latency, 3),
                "health_score": round(performance_health_score, 1)
            },
            "error_rate_health": {
                "avg_error_rate": round(avg_error_rate, 4),
                "health_score": round(error_health_score, 1)
            },
            "active_alerts": len([a for a in self.alerts if not a.resolved])
        }
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent snapshots
        recent_snapshots = [
            snapshot for snapshot in self.performance_snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {"message": "No performance data available for specified time period"}
        
        # Calculate trends
        latency_trend = []
        error_rate_trend = []
        health_score_trend = []
        
        for snapshot in recent_snapshots:
            latency_trend.append(snapshot.performance_indicators.get("avg_latency", 0))
            error_rate_trend.append(snapshot.performance_indicators.get("error_rate", 0))
            health_score_trend.append(snapshot.performance_indicators.get("health_score", 100))
        
        # Trend analysis
        latency_direction = self._calculate_trend_direction(latency_trend)
        error_rate_direction = self._calculate_trend_direction(error_rate_trend)
        health_direction = self._calculate_trend_direction(health_score_trend)
        
        return {
            "time_period_hours": hours,
            "data_points": len(recent_snapshots),
            "latency_trend": {
                "direction": latency_direction,
                "current": latency_trend[-1] if latency_trend else 0,
                "average": sum(latency_trend) / len(latency_trend) if latency_trend else 0,
                "min": min(latency_trend) if latency_trend else 0,
                "max": max(latency_trend) if latency_trend else 0
            },
            "error_rate_trend": {
                "direction": error_rate_direction,
                "current": error_rate_trend[-1] if error_rate_trend else 0,
                "average": sum(error_rate_trend) / len(error_rate_trend) if error_rate_trend else 0
            },
            "health_trend": {
                "direction": health_direction,
                "current": health_score_trend[-1] if health_score_trend else 100,
                "average": sum(health_score_trend) / len(health_score_trend) if health_score_trend else 100
            }
        }
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None, 
                  active_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        filtered_alerts = self.alerts
        
        if active_only:
            filtered_alerts = [alert for alert in filtered_alerts if not alert.resolved]
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]
        
        return [
            {
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "details": alert.details,
                "resolved": alert.resolved,
                "resolution_timestamp": alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
            }
            for alert in filtered_alerts
        ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                self.logger.info(f"Resolved alert: {alert.name} (ID: {alert_id})")
                return True
        
        return False
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in self.metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Generate Prometheus format
        for metric_name, metric_points in metrics_by_name.items():
            # Add help and type comments
            lines.append(f"# HELP {metric_name} Infrastructure metric")
            lines.append(f"# TYPE {metric_name} {metric_points[0].metric_type.value}")
            
            # Add metric points
            for point in metric_points:
                labels_str = ""
                if point.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                timestamp_ms = int(point.timestamp.timestamp() * 1000)
                lines.append(f"{metric_name}{labels_str} {point.value} {timestamp_ms}")
        
        return "\n".join(lines)
    
    async def _start_background_monitoring(self):
        """Start background monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while True:
                await self._collect_periodic_metrics()
                await self._check_alert_conditions()
                await self._take_performance_snapshot()
                
                # Monitor every 30 seconds
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    async def _collect_periodic_metrics(self):
        """Collect periodic metrics from all monitored components."""
        # Collect circuit breaker metrics
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            
            self.record_metric(f"circuit_breaker_total_events", metrics.total_events, {"circuit": name})
            self.record_metric(f"circuit_breaker_successful_events", metrics.successful_events, {"circuit": name})
            self.record_metric(f"circuit_breaker_failed_events", metrics.failed_events, {"circuit": name})
            self.record_metric(f"circuit_breaker_queued_events", metrics.queued_events, {"circuit": name})
            self.record_metric(f"circuit_breaker_dead_letter_events", metrics.dead_letter_events, {"circuit": name})
            
            # Circuit breaker state as numeric
            state_value = {"closed": 0, "half_open": 1, "open": 2}.get(cb.get_state().value, -1)
            self.record_metric(f"circuit_breaker_state", state_value, {"circuit": name})
    
    async def _check_alert_conditions(self):
        """Check for alert conditions across all monitored components."""
        for name, cb in self.circuit_breakers.items():
            state = cb.get_state()
            
            # Circuit breaker open alert
            if state == CircuitBreakerState.OPEN:
                self._create_alert(
                    f"circuit_breaker_open_{name}",
                    f"Circuit breaker {name} is open",
                    AlertSeverity.CRITICAL,
                    details={"circuit_breaker": name, "state": state.value}
                )
            
            # Queue capacity alert
            metrics = cb.get_metrics()
            if metrics.queued_events > 0:
                queue_utilization = metrics.queued_events / cb.config.max_queue_size
                if queue_utilization > self.alert_thresholds['queue_capacity']['threshold']:
                    self._create_alert(
                        f"queue_capacity_{name}",
                        f"Queue capacity high for {name}: {queue_utilization:.1%}",
                        AlertSeverity.MEDIUM,
                        details={"circuit_breaker": name, "utilization": queue_utilization}
                    )
            
            # Dead letter growth alert
            if metrics.dead_letter_events > self.alert_thresholds['dead_letter_growth']['threshold']:
                self._create_alert(
                    f"dead_letter_growth_{name}",
                    f"Dead letter queue growth for {name}: {metrics.dead_letter_events} events",
                    AlertSeverity.MEDIUM,
                    details={"circuit_breaker": name, "dead_letter_events": metrics.dead_letter_events}
                )
    
    async def _take_performance_snapshot(self):
        """Take performance snapshot for trend analysis."""
        current_metrics = self.get_current_metrics()
        health_summary = self.get_health_summary()
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            circuit_breaker_health=current_metrics["circuit_breakers"],
            event_publishing_metrics=current_metrics["performance"],
            infrastructure_health=health_summary,
            performance_indicators={
                "avg_latency": current_metrics["performance"]["avg_latency"],
                "error_rate": sum(rates[-1] for rates in self.error_rates.values() if rates) / len(self.error_rates) if self.error_rates else 0,
                "health_score": health_summary["overall_health_score"]
            }
        )
        
        self.performance_snapshots.append(snapshot)
        
        # Clean old snapshots
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        while self.performance_snapshots and self.performance_snapshots[0].timestamp < cutoff_time:
            self.performance_snapshots.popleft()
    
    def _create_alert(self, name: str, description: str, severity: AlertSeverity,
                     details: Optional[Dict[str, Any]] = None):
        """Create new alert if not already exists."""
        # Check for existing unresolved alert with same name
        for alert in self.alerts:
            if alert.name == name and not alert.resolved:
                return  # Alert already exists
        
        alert = Alert(
            id=f"{name}_{int(time.time())}",
            name=name,
            description=description,
            severity=severity,
            timestamp=datetime.now(),
            source="infrastructure_observability",
            details=details or {}
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Created {severity.value} alert: {description}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile / 100.0 * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation using first and last values
        start_value = values[0]
        end_value = values[-1]
        
        if end_value > start_value * 1.1:  # 10% increase threshold
            return "increasing"
        elif end_value < start_value * 0.9:  # 10% decrease threshold
            return "decreasing"
        else:
            return "stable"
    
    async def close(self):
        """Clean up observability system."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Infrastructure observability system closed")