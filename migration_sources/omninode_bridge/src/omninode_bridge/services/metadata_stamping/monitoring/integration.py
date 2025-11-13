"""Monitoring integration for metadata stamping service.

Provides seamless integration of performance monitoring, resource tracking,
and alerting with the existing service components.
"""

import asyncio
import logging
from typing import Optional

from .alerts import AlertManager, AlertSeverity
from .dashboard import MonitoringDashboard
from .metrics_collector import MetricsCollector, OperationType
from .performance_tracker import PerformanceTracker, set_global_tracker
from .resource_monitor import ResourceMonitor, ResourceThresholds

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """Complete monitoring integration for metadata stamping service."""

    def __init__(
        self,
        enable_resource_monitoring: bool = True,
        enable_alerts: bool = True,
        enable_dashboard: bool = True,
        resource_sample_interval: float = 5.0,
        max_samples: int = 10000,
    ):
        """Initialize monitoring integration.

        Args:
            enable_resource_monitoring: Enable resource monitoring
            enable_alerts: Enable alert management
            enable_dashboard: Enable monitoring dashboard
            resource_sample_interval: Resource sampling interval in seconds
            max_samples: Maximum samples to keep in memory
        """
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_alerts = enable_alerts
        self.enable_dashboard = enable_dashboard

        # Core monitoring components
        self.metrics_collector = MetricsCollector(max_samples=max_samples)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)

        # Resource monitoring
        self.resource_monitor: Optional[ResourceMonitor] = None
        if enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(
                sample_interval=resource_sample_interval,
                thresholds=ResourceThresholds(),
            )

        # Alert management
        self.alert_manager: Optional[AlertManager] = None
        if enable_alerts:
            self.alert_manager = AlertManager()

        # Monitoring dashboard
        self.dashboard: Optional[MonitoringDashboard] = None
        if enable_dashboard and self.resource_monitor and self.alert_manager:
            self.dashboard = MonitoringDashboard(
                self.metrics_collector, self.resource_monitor, self.alert_manager
            )

        # Set global tracker for decorators
        set_global_tracker(self.performance_tracker)

        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    async def initialize(self, db_client=None) -> bool:
        """Initialize monitoring components.

        Args:
            db_client: Database client for connection monitoring

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing monitoring integration...")

            # Set database client for resource monitoring
            if self.resource_monitor and db_client:
                self.resource_monitor.set_database_client(db_client)

            # Start resource monitoring
            if self.resource_monitor:
                await self.resource_monitor.start_monitoring()

            # Start alert manager
            if self.alert_manager:
                await self.alert_manager.start()

            # Start background monitoring
            await self.start_monitoring()

            logger.info("Monitoring integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize monitoring integration: {e}")
            return False

    async def cleanup(self):
        """Cleanup monitoring resources."""
        try:
            logger.info("Cleaning up monitoring integration...")

            # Stop background monitoring
            await self.stop_monitoring()

            # Stop alert manager
            if self.alert_manager:
                await self.alert_manager.stop()

            # Stop resource monitoring
            if self.resource_monitor:
                await self.resource_monitor.stop_monitoring()

            logger.info("Monitoring integration cleanup completed")

        except Exception as e:
            logger.error(f"Error during monitoring cleanup: {e}")

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started background monitoring")

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop for performance alerts."""
        try:
            while self.is_monitoring:
                await self._check_performance_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    async def _check_performance_alerts(self):
        """Check for performance threshold violations."""
        if not self.alert_manager:
            return

        try:
            # Get current performance summary
            perf_summary = self.metrics_collector.get_performance_summary()

            # Check for performance alerts
            for alert in perf_summary.get("alerts", []):
                await self._create_performance_alert(alert)

            # Check resource alerts
            if self.resource_monitor:
                resource_alerts = self.resource_monitor.check_resource_alerts()
                for alert in resource_alerts:
                    await self._create_resource_alert(alert)

        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")

    async def _create_performance_alert(self, alert_data: dict):
        """Create performance alert.

        Args:
            alert_data: Alert data from performance summary
        """
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "warning": AlertSeverity.WARNING,
            "info": AlertSeverity.INFO,
        }

        severity = severity_map.get(
            alert_data.get("severity", "warning"), AlertSeverity.WARNING
        )

        await self.alert_manager.create_alert(
            alert_type="performance_violation",
            severity=severity,
            title=f"Performance Alert: {alert_data.get('operation', 'Unknown')}",
            message=alert_data.get("message", "Performance threshold exceeded"),
            source="performance_monitor",
            metadata=alert_data,
        )

    async def _create_resource_alert(self, alert_data: dict):
        """Create resource alert.

        Args:
            alert_data: Alert data from resource monitor
        """
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "warning": AlertSeverity.WARNING,
            "info": AlertSeverity.INFO,
        }

        severity = severity_map.get(
            alert_data.get("severity", "warning"), AlertSeverity.WARNING
        )

        await self.alert_manager.create_alert(
            alert_type=f"resource_{alert_data.get('resource', 'unknown')}",
            severity=severity,
            title=f"Resource Alert: {alert_data.get('resource', 'Unknown').upper()}",
            message=alert_data.get("message", "Resource threshold exceeded"),
            source="resource_monitor",
            metadata=alert_data,
        )

    def get_router(self):
        """Get monitoring dashboard router.

        Returns:
            FastAPI router for monitoring endpoints
        """
        if self.dashboard:
            return self.dashboard.router
        return None

    async def get_monitoring_status(self) -> dict:
        """Get comprehensive monitoring status.

        Returns:
            Monitoring status and metrics
        """
        status = {
            "monitoring_enabled": True,
            "components": {
                "metrics_collector": "enabled",
                "performance_tracker": "enabled",
                "resource_monitor": "enabled" if self.resource_monitor else "disabled",
                "alert_manager": "enabled" if self.alert_manager else "disabled",
                "dashboard": "enabled" if self.dashboard else "disabled",
            },
            "background_monitoring": self.is_monitoring,
        }

        # Get performance summary
        perf_summary = self.metrics_collector.get_performance_summary()
        status["performance"] = perf_summary

        # Get resource summary if available
        if self.resource_monitor:
            resource_summary = await self.resource_monitor.get_resource_summary()
            status["resources"] = resource_summary

        # Get alert statistics if available
        if self.alert_manager:
            alert_stats = await self.alert_manager.get_alert_statistics()
            status["alerts"] = alert_stats

        return status

    # Convenience methods for common operations
    async def track_database_operation(
        self, operation_func, operation_type: str = "query", *args, **kwargs
    ):
        """Track database operation with automatic performance monitoring.

        Args:
            operation_func: Database function to execute
            operation_type: Type of database operation
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        operation_mapping = {
            "query": OperationType.DATABASE_QUERY,
            "select": OperationType.DATABASE_QUERY,
            "insert": OperationType.DATABASE_INSERT,
            "update": OperationType.DATABASE_UPDATE,
            "delete": OperationType.DATABASE_UPDATE,
            "batch": OperationType.DATABASE_BATCH,
        }

        op_type = operation_mapping.get(
            operation_type.lower(), OperationType.DATABASE_QUERY
        )

        return await self.metrics_collector.record_operation(
            *args,
            operation_type=op_type,
            operation_func=operation_func,
            metadata={"operation_subtype": operation_type},
            **kwargs,
        )

    async def track_hash_generation(
        self, hash_func, file_size: Optional[int] = None, *args, **kwargs
    ):
        """Track hash generation with automatic performance monitoring.

        Args:
            hash_func: Hash function to execute
            file_size: Size of file being hashed
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        metadata = {}
        if file_size is not None:
            metadata["file_size_bytes"] = file_size

        return await self.metrics_collector.record_operation(
            *args,
            operation_type=OperationType.HASH_GENERATION,
            operation_func=hash_func,
            metadata=metadata,
            **kwargs,
        )

    async def track_batch_processing(
        self, batch_func, batch_size: Optional[int] = None, *args, **kwargs
    ):
        """Track batch processing with automatic performance monitoring.

        Args:
            batch_func: Batch function to execute
            batch_size: Number of items in batch
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        metadata = {}
        if batch_size is not None:
            metadata["batch_size"] = batch_size

        return await self.metrics_collector.record_operation(
            *args,
            operation_type=OperationType.BATCH_PROCESSING,
            operation_func=batch_func,
            metadata=metadata,
            **kwargs,
        )

    async def track_api_request(
        self, api_func, endpoint: str, method: str = "GET", *args, **kwargs
    ):
        """Track API request with automatic performance monitoring.

        Args:
            api_func: API function to execute
            endpoint: API endpoint
            method: HTTP method
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        return await self.metrics_collector.record_operation(
            *args,
            operation_type=OperationType.API_REQUEST,
            operation_func=api_func,
            metadata={"endpoint": endpoint, "method": method},
            **kwargs,
        )

    def create_batch_tracker(self):
        """Create a batch operation tracker.

        Returns:
            Batch tracker instance
        """
        return self.performance_tracker.create_batch_tracker()


# Global monitoring integration instance
_global_monitoring: Optional[MonitoringIntegration] = None


def set_global_monitoring(monitoring: MonitoringIntegration):
    """Set global monitoring integration.

    Args:
        monitoring: Monitoring integration instance
    """
    global _global_monitoring
    _global_monitoring = monitoring


def get_global_monitoring() -> Optional[MonitoringIntegration]:
    """Get global monitoring integration.

    Returns:
        Monitoring integration instance or None
    """
    return _global_monitoring


# Convenience functions for easy integration
async def track_database_operation(
    operation_func, operation_type: str = "query", *args, **kwargs
):
    """Global function to track database operations.

    Args:
        operation_func: Database function to execute
        operation_type: Type of database operation
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    if _global_monitoring:
        return await _global_monitoring.track_database_operation(
            operation_func, operation_type, *args, **kwargs
        )
    else:
        # Fallback to direct execution if monitoring not available
        if asyncio.iscoroutinefunction(operation_func):
            return await operation_func(*args, **kwargs)
        else:
            return operation_func(*args, **kwargs)


async def track_hash_generation(
    hash_func, file_size: Optional[int] = None, *args, **kwargs
):
    """Global function to track hash generation.

    Args:
        hash_func: Hash function to execute
        file_size: Size of file being hashed
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    if _global_monitoring:
        return await _global_monitoring.track_hash_generation(
            hash_func, file_size, *args, **kwargs
        )
    else:
        # Fallback to direct execution if monitoring not available
        if asyncio.iscoroutinefunction(hash_func):
            return await hash_func(*args, **kwargs)
        else:
            return hash_func(*args, **kwargs)
