"""Monitoring dashboard endpoints.

Provides REST API endpoints for accessing performance metrics,
resource utilization, and health information.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .alerts import AlertManager
from .metrics_collector import MetricsCollector, OperationType, PerformanceGrade
from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics."""

    operation_type: str
    count: int
    avg_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    success_rate: float
    throughput_per_second: float
    grade_distribution: dict[str, int]
    last_updated: datetime


class PerformanceSummaryResponse(BaseModel):
    """Response model for performance summary."""

    overall_grade: str
    operations: dict[str, dict[str, Any]]
    alerts: list[dict[str, Any]]
    last_updated: str


class ResourceSummaryResponse(BaseModel):
    """Response model for resource summary."""

    status: str
    timestamp: str
    current: dict[str, Any]
    statistics_15min: dict[str, Any]
    alerts: list[dict[str, Any]]
    monitoring_info: dict[str, Any]


class AlertResponse(BaseModel):
    """Response model for alerts."""

    active_alerts: list[dict[str, Any]]
    alert_summary: dict[str, int]
    last_updated: str


class HealthStatusResponse(BaseModel):
    """Response model for overall health status."""

    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Status timestamp")
    components: dict[str, str] = Field(description="Component health status")
    performance_grade: str = Field(description="Overall performance grade")
    active_alerts: int = Field(description="Number of active alerts")
    uptime_seconds: float = Field(description="Service uptime in seconds")


class MonitoringDashboard:
    """REST API dashboard for monitoring metrics."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        resource_monitor: ResourceMonitor,
        alert_manager: AlertManager,
    ):
        """Initialize monitoring dashboard.

        Args:
            metrics_collector: Metrics collector instance
            resource_monitor: Resource monitor instance
            alert_manager: Alert manager instance
        """
        self.metrics_collector = metrics_collector
        self.resource_monitor = resource_monitor
        self.alert_manager = alert_manager

        # Create FastAPI router
        self.router = APIRouter(prefix="/monitoring", tags=["monitoring"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup monitoring API routes."""

        @self.router.get("/health", response_model=HealthStatusResponse)
        async def get_health_status():
            """Get overall health status with component breakdown."""
            try:
                # Get performance summary
                perf_summary = self.metrics_collector.get_performance_summary()

                # Get resource summary
                resource_summary = await self.resource_monitor.get_resource_summary()

                # Get active alerts
                active_alerts = await self.alert_manager.get_active_alerts()

                # Determine overall status
                overall_status = "healthy"
                if resource_summary.get("status") == "critical" or any(
                    alert.get("severity") == "critical" for alert in active_alerts
                ):
                    overall_status = "critical"
                elif resource_summary.get("status") == "warning" or any(
                    alert.get("severity") == "warning" for alert in active_alerts
                ):
                    overall_status = "warning"

                # Component health breakdown
                components = {
                    "performance": perf_summary.get("overall_grade", "B"),
                    "resources": resource_summary.get("status", "unknown"),
                    "database": "healthy",  # Will be updated by service health check
                    "monitoring": "healthy",
                }

                return HealthStatusResponse(
                    status=overall_status,
                    timestamp=datetime.now().isoformat(),
                    components=components,
                    performance_grade=perf_summary.get("overall_grade", "B"),
                    active_alerts=len(active_alerts),
                    uptime_seconds=resource_summary.get("monitoring_info", {}).get(
                        "uptime_seconds", 0
                    ),
                )

            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving health status"
                )

        @self.router.get(
            "/performance/summary", response_model=PerformanceSummaryResponse
        )
        async def get_performance_summary():
            """Get comprehensive performance summary."""
            try:
                summary = self.metrics_collector.get_performance_summary()
                return PerformanceSummaryResponse(**summary)
            except Exception as e:
                logger.error(f"Error getting performance summary: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving performance summary"
                )

        @self.router.get(
            "/performance/stats/{operation_type}",
            response_model=PerformanceStatsResponse,
        )
        async def get_performance_stats(
            operation_type: str,
            force_refresh: bool = Query(False, description="Force cache refresh"),
        ):
            """Get detailed performance statistics for specific operation type."""
            try:
                # Validate operation type
                try:
                    op_type = OperationType(operation_type)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid operation type: {operation_type}",
                    )

                stats = self.metrics_collector.calculate_performance_stats(
                    op_type, force_refresh
                )
                if not stats:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No performance data available for {operation_type}",
                    )

                # Convert grade distribution to strings
                grade_dist = {
                    grade.value: count
                    for grade, count in stats.grade_distribution.items()
                }

                return PerformanceStatsResponse(
                    operation_type=stats.operation_type.value,
                    count=stats.count,
                    avg_time_ms=stats.avg_time_ms,
                    p50_time_ms=stats.p50_time_ms,
                    p95_time_ms=stats.p95_time_ms,
                    p99_time_ms=stats.p99_time_ms,
                    success_rate=stats.success_rate,
                    throughput_per_second=stats.throughput_per_second,
                    grade_distribution=grade_dist,
                    last_updated=stats.last_updated,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"Error getting performance stats for {operation_type}: {e}"
                )
                raise HTTPException(
                    status_code=500, detail="Error retrieving performance statistics"
                )

        @self.router.get("/performance/operations")
        async def list_available_operations():
            """List all available operation types with current statistics."""
            try:
                all_stats = self.metrics_collector.get_all_performance_stats(
                    force_refresh=True
                )

                operations = []
                for op_type, stats in all_stats.items():
                    operations.append(
                        {
                            "operation_type": op_type.value,
                            "description": self._get_operation_description(op_type),
                            "count": stats.count,
                            "avg_time_ms": round(stats.avg_time_ms, 2),
                            "success_rate": round(stats.success_rate, 3),
                            "throughput_per_second": round(
                                stats.throughput_per_second, 2
                            ),
                            "last_updated": stats.last_updated.isoformat(),
                        }
                    )

                return {
                    "operations": operations,
                    "total_operations": len(operations),
                    "last_updated": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Error listing operations: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving operation list"
                )

        @self.router.get("/resources", response_model=ResourceSummaryResponse)
        async def get_resource_summary():
            """Get comprehensive resource utilization summary."""
            try:
                summary = await self.resource_monitor.get_resource_summary()
                return ResourceSummaryResponse(**summary)
            except Exception as e:
                logger.error(f"Error getting resource summary: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving resource summary"
                )

        @self.router.get("/resources/history")
        async def get_resource_history(
            minutes: int = Query(
                60, description="Time window in minutes", ge=1, le=1440
            )
        ):
            """Get resource utilization history."""
            try:
                time_window = timedelta(minutes=minutes)
                history = self.resource_monitor.get_resource_history(time_window)

                # Convert to serializable format
                history_data = []
                for snapshot in history:
                    history_data.append(
                        {
                            "timestamp": snapshot.timestamp.isoformat(),
                            "cpu_percent": snapshot.cpu_percent,
                            "memory_mb": snapshot.memory_mb,
                            "memory_percent": snapshot.memory_percent,
                            "disk_usage_percent": snapshot.disk_usage_percent,
                            "database_connections": snapshot.database_connections,
                            "active_threads": snapshot.active_threads,
                            "file_descriptors": snapshot.file_descriptors,
                        }
                    )

                return {
                    "time_window_minutes": minutes,
                    "sample_count": len(history_data),
                    "history": history_data,
                    "statistics": self.resource_monitor.get_resource_statistics(
                        time_window
                    ),
                }

            except Exception as e:
                logger.error(f"Error getting resource history: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving resource history"
                )

        @self.router.get("/alerts", response_model=AlertResponse)
        async def get_alerts(
            severity: Optional[str] = Query(None, description="Filter by severity"),
            limit: int = Query(
                100, description="Maximum number of alerts", ge=1, le=1000
            ),
        ):
            """Get current alerts with optional filtering."""
            try:
                active_alerts = await self.alert_manager.get_active_alerts()

                # Filter by severity if requested
                if severity:
                    active_alerts = [
                        alert
                        for alert in active_alerts
                        if alert.get("severity", "").lower() == severity.lower()
                    ]

                # Limit results
                active_alerts = active_alerts[:limit]

                # Create alert summary
                alert_summary = {}
                for alert in active_alerts:
                    severity_level = alert.get("severity", "unknown")
                    alert_summary[severity_level] = (
                        alert_summary.get(severity_level, 0) + 1
                    )

                return AlertResponse(
                    active_alerts=active_alerts,
                    alert_summary=alert_summary,
                    last_updated=datetime.now().isoformat(),
                )

            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail="Error retrieving alerts")

        @self.router.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Get metrics in Prometheus format."""
            try:
                from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

                # Update resource metrics in Prometheus
                current_snapshot = self.resource_monitor.get_current_snapshot()
                if current_snapshot:
                    self.metrics_collector.update_resource_metrics(
                        cpu_percent=current_snapshot.cpu_percent,
                        memory_mb=current_snapshot.memory_mb,
                        db_connections=current_snapshot.database_connections,
                    )

                # Generate Prometheus metrics
                metrics_output = generate_latest()

                return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)

            except Exception as e:
                logger.error(f"Error generating Prometheus metrics: {e}")
                raise HTTPException(status_code=500, detail="Error generating metrics")

        @self.router.post("/alerts/acknowledge/{alert_id}")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge a specific alert."""
            try:
                success = await self.alert_manager.acknowledge_alert(alert_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Alert not found")

                return {"status": "acknowledged", "alert_id": alert_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error acknowledging alert {alert_id}: {e}")
                raise HTTPException(status_code=500, detail="Error acknowledging alert")

        @self.router.get("/dashboard")
        async def get_dashboard_data():
            """Get comprehensive dashboard data in single request."""
            try:
                # Gather all dashboard data
                perf_summary = self.metrics_collector.get_performance_summary()
                resource_summary = await self.resource_monitor.get_resource_summary()
                active_alerts = await self.alert_manager.get_active_alerts()

                # Get top performing and underperforming operations
                all_stats = self.metrics_collector.get_all_performance_stats()
                operations_summary = []

                for op_type, stats in all_stats.items():
                    operations_summary.append(
                        {
                            "operation_type": op_type.value,
                            "avg_time_ms": round(stats.avg_time_ms, 2),
                            "p95_time_ms": round(stats.p95_time_ms, 2),
                            "success_rate": round(stats.success_rate, 3),
                            "throughput_per_second": round(
                                stats.throughput_per_second, 2
                            ),
                            "grade": self._determine_operation_grade_display(
                                op_type, stats
                            ),
                        }
                    )

                return {
                    "overview": {
                        "performance_grade": perf_summary.get("overall_grade", "B"),
                        "resource_status": resource_summary.get("status", "unknown"),
                        "active_alerts": len(active_alerts),
                        "critical_alerts": len(
                            [
                                a
                                for a in active_alerts
                                if a.get("severity") == "critical"
                            ]
                        ),
                        "timestamp": datetime.now().isoformat(),
                    },
                    "performance": {
                        "summary": perf_summary,
                        "operations": operations_summary,
                    },
                    "resources": resource_summary,
                    "alerts": {
                        "active": active_alerts[:10],  # Top 10 alerts
                        "summary": self._summarize_alerts(active_alerts),
                    },
                }

            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(
                    status_code=500, detail="Error retrieving dashboard data"
                )

    def _get_operation_description(self, operation_type: OperationType) -> str:
        """Get human-readable description for operation type.

        Args:
            operation_type: Operation type

        Returns:
            Human-readable description
        """
        descriptions = {
            OperationType.DATABASE_QUERY: "Database query operations",
            OperationType.DATABASE_INSERT: "Database insert operations",
            OperationType.DATABASE_UPDATE: "Database update operations",
            OperationType.DATABASE_BATCH: "Database batch operations",
            OperationType.HASH_GENERATION: "BLAKE3 hash generation",
            OperationType.BATCH_PROCESSING: "Batch processing operations",
            OperationType.EVENT_PUBLISHING: "Event publishing operations",
            OperationType.API_REQUEST: "API request handling",
            OperationType.HEALTH_CHECK: "Health check operations",
        }
        return descriptions.get(operation_type, "Unknown operation")

    def _determine_operation_grade_display(self, operation_type, stats):
        """Determine display grade for operation.

        Args:
            operation_type: Operation type
            stats: Performance statistics

        Returns:
            Grade string for display
        """
        total_operations = sum(stats.grade_distribution.values())
        if total_operations == 0:
            return "C"

        a_grade_ratio = (
            stats.grade_distribution.get(PerformanceGrade.A, 0) / total_operations
        )
        if a_grade_ratio >= 0.8:
            return "A"
        elif a_grade_ratio >= 0.6:
            return "B"
        else:
            return "C"

    def _summarize_alerts(self, alerts: list[dict[str, Any]]) -> dict[str, int]:
        """Summarize alerts by type and severity.

        Args:
            alerts: List of alerts

        Returns:
            Alert summary dictionary
        """
        summary = {
            "total": len(alerts),
            "critical": 0,
            "warning": 0,
            "info": 0,
            "by_type": {},
        }

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            alert_type = alert.get("type", "unknown")

            if severity == "critical":
                summary["critical"] += 1
            elif severity == "warning":
                summary["warning"] += 1
            else:
                summary["info"] += 1

            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1

        return summary


# Add missing import for Response
from fastapi import Response
