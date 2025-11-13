"""Resource utilization monitoring.

Monitors CPU, memory, and database connection usage to detect resource
constraints and performance bottlenecks.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource utilization at a point in time."""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_usage_percent: float
    database_connections: dict[str, int]
    active_threads: int
    file_descriptors: int


@dataclass
class ResourceThresholds:
    """Resource utilization thresholds for alerting."""

    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 70.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    db_connections_warning: int = 80  # Percentage of max connections
    db_connections_critical: int = 95


class ResourceMonitor:
    """Monitor system and application resource utilization."""

    def __init__(
        self,
        sample_interval: float = 5.0,
        history_retention_minutes: int = 60,
        thresholds: Optional[ResourceThresholds] = None,
    ):
        """Initialize resource monitor.

        Args:
            sample_interval: Seconds between resource samples
            history_retention_minutes: Minutes to retain resource history
            thresholds: Resource thresholds for alerting
        """
        self.sample_interval = sample_interval
        self.history_retention = timedelta(minutes=history_retention_minutes)
        self.thresholds = thresholds or ResourceThresholds()

        # Resource history
        self.resource_history: list[ResourceSnapshot] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Database client reference (set externally)
        self.db_client = None

        # Process information
        self.process = psutil.Process()

    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.is_monitoring:
            logger.warning("Resource monitoring already started")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Started resource monitoring with {self.sample_interval}s interval"
        )

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped resource monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                snapshot = await self._take_snapshot()
                self._add_snapshot(snapshot)
                self._cleanup_old_snapshots()

                await asyncio.sleep(self.sample_interval)
        except asyncio.CancelledError:
            logger.info("Resource monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring loop: {e}")

    async def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource utilization.

        Returns:
            Resource snapshot
        """
        try:
            # CPU and memory (system-wide)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Process-specific metrics
            process_memory = self.process.memory_info()
            memory_mb = process_memory.rss / (1024 * 1024)  # Convert to MB

            # Thread and file descriptor counts
            try:
                active_threads = self.process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                active_threads = 0

            try:
                file_descriptors = (
                    self.process.num_fds() if hasattr(self.process, "num_fds") else 0
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                file_descriptors = 0

            # Database connection information
            db_connections = await self._get_database_connections()

            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                database_connections=db_connections,
                active_threads=active_threads,
                file_descriptors=file_descriptors,
            )

        except Exception as e:
            logger.error(f"Error taking resource snapshot: {e}")
            # Return minimal snapshot on error
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                database_connections={},
                active_threads=0,
                file_descriptors=0,
            )

    async def _get_database_connections(self) -> dict[str, int]:
        """Get database connection statistics.

        Returns:
            Dictionary with connection counts by state
        """
        if not self.db_client:
            return {"active": 0, "idle": 0, "total": 0}

        try:
            # Get connection pool statistics
            if hasattr(self.db_client, "get_pool_stats"):
                pool_stats = await self.db_client.get_pool_stats()
                return {
                    "active": pool_stats.get("open_connections", 0),
                    "idle": pool_stats.get("free_connections", 0),
                    "total": pool_stats.get("open_connections", 0)
                    + pool_stats.get("free_connections", 0),
                }
            else:
                # Fallback if pool stats not available
                return {"active": 0, "idle": 0, "total": 0}

        except Exception as e:
            logger.error(f"Error getting database connection stats: {e}")
            return {"active": 0, "idle": 0, "total": 0}

    def _add_snapshot(self, snapshot: ResourceSnapshot):
        """Add snapshot to history.

        Args:
            snapshot: Resource snapshot to add
        """
        self.resource_history.append(snapshot)

    def _cleanup_old_snapshots(self):
        """Remove old snapshots beyond retention period."""
        cutoff_time = datetime.now() - self.history_retention
        self.resource_history = [
            snapshot
            for snapshot in self.resource_history
            if snapshot.timestamp >= cutoff_time
        ]

    def get_current_snapshot(self) -> Optional[ResourceSnapshot]:
        """Get the most recent resource snapshot.

        Returns:
            Most recent snapshot or None if no data
        """
        return self.resource_history[-1] if self.resource_history else None

    def get_resource_history(
        self, time_window: Optional[timedelta] = None
    ) -> list[ResourceSnapshot]:
        """Get resource history within time window.

        Args:
            time_window: Time window (default: all retained history)

        Returns:
            List of resource snapshots
        """
        if not time_window:
            return self.resource_history.copy()

        cutoff_time = datetime.now() - time_window
        return [
            snapshot
            for snapshot in self.resource_history
            if snapshot.timestamp >= cutoff_time
        ]

    def get_resource_statistics(
        self, time_window: Optional[timedelta] = None
    ) -> dict[str, Any]:
        """Get aggregated resource statistics.

        Args:
            time_window: Time window for statistics

        Returns:
            Dictionary with resource statistics
        """
        history = self.get_resource_history(time_window)

        if not history:
            return {}

        # Calculate statistics
        cpu_values = [s.cpu_percent for s in history]
        memory_values = [s.memory_mb for s in history]
        memory_percent_values = [s.memory_percent for s in history]
        disk_values = [s.disk_usage_percent for s in history]

        # Database connection statistics
        total_connections = [s.database_connections.get("total", 0) for s in history]
        active_connections = [s.database_connections.get("active", 0) for s in history]

        return {
            "time_window_minutes": (
                (time_window.total_seconds() / 60) if time_window else None
            ),
            "sample_count": len(history),
            "cpu": {
                "avg_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
                "min_percent": min(cpu_values),
                "current_percent": cpu_values[-1] if cpu_values else 0,
            },
            "memory": {
                "avg_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "min_mb": min(memory_values),
                "current_mb": memory_values[-1] if memory_values else 0,
                "avg_percent": sum(memory_percent_values) / len(memory_percent_values),
                "max_percent": max(memory_percent_values),
                "current_percent": (
                    memory_percent_values[-1] if memory_percent_values else 0
                ),
            },
            "disk": {
                "avg_percent": sum(disk_values) / len(disk_values),
                "max_percent": max(disk_values),
                "current_percent": disk_values[-1] if disk_values else 0,
            },
            "database_connections": {
                "avg_total": (
                    sum(total_connections) / len(total_connections)
                    if total_connections
                    else 0
                ),
                "max_total": max(total_connections) if total_connections else 0,
                "avg_active": (
                    sum(active_connections) / len(active_connections)
                    if active_connections
                    else 0
                ),
                "max_active": max(active_connections) if active_connections else 0,
                "current_total": total_connections[-1] if total_connections else 0,
                "current_active": active_connections[-1] if active_connections else 0,
            },
            "threads": {
                "current": history[-1].active_threads if history else 0,
                "avg": sum(s.active_threads for s in history) / len(history),
                "max": max(s.active_threads for s in history),
            },
            "file_descriptors": {
                "current": history[-1].file_descriptors if history else 0,
                "avg": sum(s.file_descriptors for s in history) / len(history),
                "max": max(s.file_descriptors for s in history),
            },
        }

    def check_resource_alerts(
        self, snapshot: Optional[ResourceSnapshot] = None
    ) -> list[dict[str, Any]]:
        """Check for resource utilization alerts.

        Args:
            snapshot: Snapshot to check (default: most recent)

        Returns:
            List of resource alerts
        """
        if snapshot is None:
            snapshot = self.get_current_snapshot()

        if not snapshot:
            return []

        alerts = []

        # CPU alerts
        if snapshot.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(
                {
                    "type": "resource_critical",
                    "resource": "cpu",
                    "severity": "critical",
                    "message": f"CPU usage ({snapshot.cpu_percent:.1f}%) exceeds critical threshold ({self.thresholds.cpu_critical}%)",
                    "current_value": snapshot.cpu_percent,
                    "threshold": self.thresholds.cpu_critical,
                }
            )
        elif snapshot.cpu_percent >= self.thresholds.cpu_warning:
            alerts.append(
                {
                    "type": "resource_warning",
                    "resource": "cpu",
                    "severity": "warning",
                    "message": f"CPU usage ({snapshot.cpu_percent:.1f}%) exceeds warning threshold ({self.thresholds.cpu_warning}%)",
                    "current_value": snapshot.cpu_percent,
                    "threshold": self.thresholds.cpu_warning,
                }
            )

        # Memory alerts
        if snapshot.memory_percent >= self.thresholds.memory_critical:
            alerts.append(
                {
                    "type": "resource_critical",
                    "resource": "memory",
                    "severity": "critical",
                    "message": f"Memory usage ({snapshot.memory_percent:.1f}%) exceeds critical threshold ({self.thresholds.memory_critical}%)",
                    "current_value": snapshot.memory_percent,
                    "threshold": self.thresholds.memory_critical,
                }
            )
        elif snapshot.memory_percent >= self.thresholds.memory_warning:
            alerts.append(
                {
                    "type": "resource_warning",
                    "resource": "memory",
                    "severity": "warning",
                    "message": f"Memory usage ({snapshot.memory_percent:.1f}%) exceeds warning threshold ({self.thresholds.memory_warning}%)",
                    "current_value": snapshot.memory_percent,
                    "threshold": self.thresholds.memory_warning,
                }
            )

        # Disk alerts
        if snapshot.disk_usage_percent >= self.thresholds.disk_critical:
            alerts.append(
                {
                    "type": "resource_critical",
                    "resource": "disk",
                    "severity": "critical",
                    "message": f"Disk usage ({snapshot.disk_usage_percent:.1f}%) exceeds critical threshold ({self.thresholds.disk_critical}%)",
                    "current_value": snapshot.disk_usage_percent,
                    "threshold": self.thresholds.disk_critical,
                }
            )
        elif snapshot.disk_usage_percent >= self.thresholds.disk_warning:
            alerts.append(
                {
                    "type": "resource_warning",
                    "resource": "disk",
                    "severity": "warning",
                    "message": f"Disk usage ({snapshot.disk_usage_percent:.1f}%) exceeds warning threshold ({self.thresholds.disk_warning}%)",
                    "current_value": snapshot.disk_usage_percent,
                    "threshold": self.thresholds.disk_warning,
                }
            )

        return alerts

    def set_database_client(self, db_client):
        """Set database client for connection monitoring.

        Args:
            db_client: Database client instance
        """
        self.db_client = db_client

    async def get_resource_summary(self) -> dict[str, Any]:
        """Get comprehensive resource summary.

        Returns:
            Resource summary with current state and alerts
        """
        current_snapshot = self.get_current_snapshot()
        if not current_snapshot:
            return {"status": "no_data", "message": "No resource data available"}

        # Get statistics for last 15 minutes
        stats_15min = self.get_resource_statistics(timedelta(minutes=15))

        # Check for alerts
        alerts = self.check_resource_alerts(current_snapshot)

        # Determine overall resource health
        health_status = "healthy"
        if any(alert["severity"] == "critical" for alert in alerts):
            health_status = "critical"
        elif any(alert["severity"] == "warning" for alert in alerts):
            health_status = "warning"

        return {
            "status": health_status,
            "timestamp": current_snapshot.timestamp.isoformat(),
            "current": {
                "cpu_percent": current_snapshot.cpu_percent,
                "memory_mb": current_snapshot.memory_mb,
                "memory_percent": current_snapshot.memory_percent,
                "disk_percent": current_snapshot.disk_usage_percent,
                "database_connections": current_snapshot.database_connections,
                "active_threads": current_snapshot.active_threads,
                "file_descriptors": current_snapshot.file_descriptors,
            },
            "statistics_15min": stats_15min,
            "alerts": alerts,
            "monitoring_info": {
                "sample_interval_seconds": self.sample_interval,
                "history_retention_minutes": self.history_retention.total_seconds()
                / 60,
                "samples_in_history": len(self.resource_history),
            },
        }
