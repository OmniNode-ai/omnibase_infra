"""Comprehensive health check system for OmniNode Bridge services."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram

from ..middleware.request_correlation import get_correlation_context

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""

    LIVENESS = "liveness"  # Service is running
    READINESS = "readiness"  # Service is ready to accept traffic
    STARTUP = "startup"  # Service has started successfully


class ComponentHealthCheckType(Enum):
    """Types of components to health check."""

    SELF = "self"  # Node runtime status
    DATABASE = "database"  # PostgreSQL connection
    KAFKA = "kafka"  # Kafka producer/consumer
    CONSUL = "consul"  # Service discovery
    HTTP_SERVICE = "http_service"  # External HTTP services


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any]
    check_duration_ms: float
    timestamp: datetime
    check_type: HealthCheckType

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["status"] = self.status.value
        result["check_type"] = self.check_type.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ServiceHealthStatus:
    """Overall service health status."""

    service_name: str
    status: HealthStatus
    uptime_seconds: float
    timestamp: datetime
    version: str
    environment: str
    checks: list[HealthCheckResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "environment": self.environment,
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "total_checks": len(self.checks),
                "healthy_checks": len(
                    [c for c in self.checks if c.status == HealthStatus.HEALTHY]
                ),
                "degraded_checks": len(
                    [c for c in self.checks if c.status == HealthStatus.DEGRADED]
                ),
                "unhealthy_checks": len(
                    [c for c in self.checks if c.status == HealthStatus.UNHEALTHY]
                ),
            },
        }


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        environment: str = "development",
        timeout_seconds: float = 30.0,
        enable_metrics: bool = True,
    ):
        """Initialize health checker.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Environment (development, staging, production)
            timeout_seconds: Default timeout for health checks
            enable_metrics: Enable Prometheus metrics collection
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.enable_metrics = enable_metrics
        self.startup_time = datetime.now(UTC)

        # Health check registrations
        self._liveness_checks: dict[str, Callable] = {}
        self._readiness_checks: dict[str, Callable] = {}
        self._startup_checks: dict[str, Callable] = {}

        # Metrics
        if enable_metrics:
            self.setup_metrics()

    def setup_metrics(self) -> None:
        """Setup Prometheus metrics for health checks."""
        self.health_check_total = Counter(
            "omninode_health_checks_total",
            "Total number of health checks performed",
            ["service", "check_name", "check_type", "status"],
        )

        self.health_check_duration = Histogram(
            "omninode_health_check_duration_seconds",
            "Duration of health checks in seconds",
            ["service", "check_name", "check_type"],
        )

        self.health_status_gauge = Gauge(
            "omninode_health_status",
            "Current health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            ["service", "check_name", "check_type"],
        )

    def register_liveness_check(
        self, name: str, check_func: Callable, timeout: Optional[float] = None
    ) -> None:
        """Register a liveness check.

        Liveness checks determine if the service is running and should be restarted if failing.

        Args:
            name: Name of the check
            check_func: Async function that returns (HealthStatus, message, details)
            timeout: Timeout for this specific check
        """
        self._liveness_checks[name] = {
            "func": check_func,
            "timeout": timeout or self.timeout_seconds,
        }
        logger.debug(
            "Registered liveness check", check_name=name, service=self.service_name
        )

    def register_readiness_check(
        self, name: str, check_func: Callable, timeout: Optional[float] = None
    ) -> None:
        """Register a readiness check.

        Readiness checks determine if the service is ready to accept traffic.

        Args:
            name: Name of the check
            check_func: Async function that returns (HealthStatus, message, details)
            timeout: Timeout for this specific check
        """
        self._readiness_checks[name] = {
            "func": check_func,
            "timeout": timeout or self.timeout_seconds,
        }
        logger.debug(
            "Registered readiness check", check_name=name, service=self.service_name
        )

    def register_startup_check(
        self, name: str, check_func: Callable, timeout: Optional[float] = None
    ) -> None:
        """Register a startup check.

        Startup checks determine if the service has started successfully.

        Args:
            name: Name of the check
            check_func: Async function that returns (HealthStatus, message, details)
            timeout: Timeout for this specific check
        """
        self._startup_checks[name] = {
            "func": check_func,
            "timeout": timeout or self.timeout_seconds,
        }
        logger.debug(
            "Registered startup check", check_name=name, service=self.service_name
        )

    async def run_check(
        self, name: str, check_config: dict[str, Any], check_type: HealthCheckType
    ) -> HealthCheckResult:
        """Run a single health check.

        Args:
            name: Name of the check
            check_config: Check configuration
            check_type: Type of health check

        Returns:
            Health check result
        """
        start_time = time.time()
        correlation_context = get_correlation_context()

        try:
            check_func = check_config["func"]
            timeout = check_config["timeout"]

            # Run the health check with timeout
            status, message, details = await asyncio.wait_for(
                check_func(), timeout=timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            # Update metrics
            if self.enable_metrics:
                self.health_check_total.labels(
                    service=self.service_name,
                    check_name=name,
                    check_type=check_type.value,
                    status=status.value,
                ).inc()

                self.health_check_duration.labels(
                    service=self.service_name,
                    check_name=name,
                    check_type=check_type.value,
                ).observe(duration_ms / 1000)

                status_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0.0,
                    HealthStatus.UNKNOWN: 0.0,
                }[status]

                self.health_status_gauge.labels(
                    service=self.service_name,
                    check_name=name,
                    check_type=check_type.value,
                ).set(status_value)

            # Log check result
            logger.info(
                "Health check completed",
                check_name=name,
                check_type=check_type.value,
                status=status.value,
                duration_ms=duration_ms,
                service=self.service_name,
                **correlation_context,
            )

            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                details=details or {},
                check_duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                check_type=check_type,
            )

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000

            error_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout} seconds",
                details={"timeout_seconds": timeout},
                check_duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                check_type=check_type,
            )

            # Update metrics for timeout
            if self.enable_metrics:
                self.health_check_total.labels(
                    service=self.service_name,
                    check_name=name,
                    check_type=check_type.value,
                    status="timeout",
                ).inc()

            logger.warning(
                "Health check timed out",
                check_name=name,
                check_type=check_type.value,
                timeout_seconds=timeout,
                service=self.service_name,
                **correlation_context,
            )

            return error_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            error_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                details={"error": str(e), "error_type": type(e).__name__},
                check_duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                check_type=check_type,
            )

            # Update metrics for error
            if self.enable_metrics:
                self.health_check_total.labels(
                    service=self.service_name,
                    check_name=name,
                    check_type=check_type.value,
                    status="error",
                ).inc()

            logger.error(
                "Health check failed with exception",
                check_name=name,
                check_type=check_type.value,
                error=str(e),
                service=self.service_name,
                **correlation_context,
            )

            return error_result

    async def run_liveness_checks(self) -> list[HealthCheckResult]:
        """Run all liveness checks."""
        tasks = []
        for name, config in self._liveness_checks.items():
            task = self.run_check(name, config, HealthCheckType.LIVENESS)
            tasks.append(task)

        if not tasks:
            return []

        return await asyncio.gather(*tasks)

    async def run_readiness_checks(self) -> list[HealthCheckResult]:
        """Run all readiness checks."""
        tasks = []
        for name, config in self._readiness_checks.items():
            task = self.run_check(name, config, HealthCheckType.READINESS)
            tasks.append(task)

        if not tasks:
            return []

        return await asyncio.gather(*tasks)

    async def run_startup_checks(self) -> list[HealthCheckResult]:
        """Run all startup checks."""
        tasks = []
        for name, config in self._startup_checks.items():
            task = self.run_check(name, config, HealthCheckType.STARTUP)
            tasks.append(task)

        if not tasks:
            return []

        return await asyncio.gather(*tasks)

    async def get_service_health(
        self, check_types: Optional[list[HealthCheckType]] = None
    ) -> ServiceHealthStatus:
        """Get overall service health status.

        Args:
            check_types: Types of checks to run (default: all)

        Returns:
            Service health status
        """
        if check_types is None:
            check_types = [HealthCheckType.LIVENESS, HealthCheckType.READINESS]

        all_results = []

        # Run requested check types
        if HealthCheckType.LIVENESS in check_types:
            liveness_results = await self.run_liveness_checks()
            all_results.extend(liveness_results)

        if HealthCheckType.READINESS in check_types:
            readiness_results = await self.run_readiness_checks()
            all_results.extend(readiness_results)

        if HealthCheckType.STARTUP in check_types:
            startup_results = await self.run_startup_checks()
            all_results.extend(startup_results)

        # Determine overall status
        overall_status = self._determine_overall_status(all_results)

        # Calculate uptime
        uptime = (datetime.now(UTC) - self.startup_time).total_seconds()

        return ServiceHealthStatus(
            service_name=self.service_name,
            status=overall_status,
            uptime_seconds=uptime,
            timestamp=datetime.now(UTC),
            version=self.service_version,
            environment=self.environment,
            checks=all_results,
        )

    def _determine_overall_status(
        self, results: list[HealthCheckResult]
    ) -> HealthStatus:
        """Determine overall health status from individual check results.

        Args:
            results: List of health check results

        Returns:
            Overall health status
        """
        if not results:
            return HealthStatus.UNKNOWN

        # Count statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0,
        }

        for result in results:
            status_counts[result.status] += 1

        # Determine overall status based on worst-case principle
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


# Common health check functions
async def check_memory_usage(
    threshold_percent: float = 90.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check system memory usage.

    Args:
        threshold_percent: Memory usage threshold for unhealthy status

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        if usage_percent > threshold_percent:
            return (
                HealthStatus.UNHEALTHY,
                f"High memory usage: {usage_percent:.1f}%",
                {"memory_usage_percent": usage_percent, "threshold": threshold_percent},
            )
        elif usage_percent > threshold_percent * 0.8:
            return (
                HealthStatus.DEGRADED,
                f"Elevated memory usage: {usage_percent:.1f}%",
                {"memory_usage_percent": usage_percent, "threshold": threshold_percent},
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Memory usage normal: {usage_percent:.1f}%",
                {"memory_usage_percent": usage_percent, "threshold": threshold_percent},
            )

    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not available for memory checking",
            {"psutil_available": False},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Memory check failed: {e!s}",
            {"error": str(e)},
        )


async def check_disk_space(
    path: str = "/", threshold_percent: float = 90.0
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check disk space usage.

    Args:
        path: Path to check disk space for
        threshold_percent: Disk usage threshold for unhealthy status

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import shutil

        total, used, free = shutil.disk_usage(path)
        usage_percent = (used / total) * 100

        if usage_percent > threshold_percent:
            return (
                HealthStatus.UNHEALTHY,
                f"High disk usage on {path}: {usage_percent:.1f}%",
                {
                    "disk_usage_percent": usage_percent,
                    "threshold": threshold_percent,
                    "path": path,
                    "free_gb": free / (1024**3),
                    "total_gb": total / (1024**3),
                },
            )
        elif usage_percent > threshold_percent * 0.8:
            return (
                HealthStatus.DEGRADED,
                f"Elevated disk usage on {path}: {usage_percent:.1f}%",
                {
                    "disk_usage_percent": usage_percent,
                    "threshold": threshold_percent,
                    "path": path,
                    "free_gb": free / (1024**3),
                    "total_gb": total / (1024**3),
                },
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Disk usage normal on {path}: {usage_percent:.1f}%",
                {
                    "disk_usage_percent": usage_percent,
                    "threshold": threshold_percent,
                    "path": path,
                    "free_gb": free / (1024**3),
                    "total_gb": total / (1024**3),
                },
            )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Disk check failed for {path}: {e!s}",
            {"error": str(e), "path": path},
        )
