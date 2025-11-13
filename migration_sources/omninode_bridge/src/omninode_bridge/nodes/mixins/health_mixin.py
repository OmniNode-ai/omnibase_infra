"""
ONEX-compliant Health Check Mixin for Bridge Nodes.

Provides comprehensive health checking capabilities for ONEX v2.0 nodes:
- Component-level health checks (database, kafka, external services)
- Status reporting (HEALTHY, DEGRADED, UNHEALTHY)
- Docker HEALTHCHECK compatibility
- Prometheus metrics integration
- Async and sync health check methods

Usage:
    class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin):
        def __init__(self, container: ModelContainer):
            super().__init__(container)
            self.initialize_health_checks()

        def _register_component_checks(self):
            self.register_component_check(
                "metadata_stamping",
                self._check_metadata_stamping_health
            )
"""

import asyncio
import time
from abc import ABCMeta
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """
    ONEX-compliant health status enumeration.

    Status Meanings:
    - HEALTHY: All components operational, accepting requests
    - DEGRADED: Some non-critical components failing, limited functionality
    - UNHEALTHY: Critical components failing, not accepting requests
    - UNKNOWN: Health status cannot be determined
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    def to_http_status(self) -> int:
        """
        Convert health status to HTTP status code.

        Returns:
            HTTP status code (200, 503, or 500)
        """
        return {
            HealthStatus.HEALTHY: 200,
            HealthStatus.DEGRADED: 200,  # Still accepting requests
            HealthStatus.UNHEALTHY: 503,  # Service unavailable
            HealthStatus.UNKNOWN: 500,  # Internal server error
        }[self]

    def to_docker_exit_code(self) -> int:
        """
        Convert health status to Docker HEALTHCHECK exit code.

        Returns:
            Exit code (0 for healthy, 1 for unhealthy)
        """
        return 0 if self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED) else 1


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    check_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "check_duration_ms": self.check_duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "is_critical": self.is_critical,
        }


@dataclass
class NodeHealthCheckResult:
    """Overall health check result for a node."""

    node_id: str
    node_type: str
    overall_status: HealthStatus
    components: list[ComponentHealth] = field(default_factory=list)
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "overall_status": self.overall_status.value,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "components": [comp.to_dict() for comp in self.components],
            "summary": {
                "total_components": len(self.components),
                "healthy_components": len(
                    [c for c in self.components if c.status == HealthStatus.HEALTHY]
                ),
                "degraded_components": len(
                    [c for c in self.components if c.status == HealthStatus.DEGRADED]
                ),
                "unhealthy_components": len(
                    [c for c in self.components if c.status == HealthStatus.UNHEALTHY]
                ),
                "critical_unhealthy": len(
                    [
                        c
                        for c in self.components
                        if c.status == HealthStatus.UNHEALTHY and c.is_critical
                    ]
                ),
            },
        }


class HealthCheckMixin(metaclass=ABCMeta):
    """
    ONEX-compliant health check mixin for bridge nodes.

    Provides comprehensive health checking capabilities:
    - Component-level health checks (database, kafka, services)
    - Overall status calculation
    - Docker HEALTHCHECK support
    - Prometheus metrics (optional)
    - Graceful degradation support

    Components:
    - self._health_checks: Registry of component health check functions
    - self._critical_components: Set of components required for basic operation
    - self._startup_time: Node startup timestamp for uptime calculation

    Methods:
    - initialize_health_checks(): Setup health check system
    - register_component_check(): Register a component health check
    - check_health(): Async comprehensive health check
    - get_health_status(): Sync health status retrieval
    - is_healthy(): Simple boolean health check
    """

    _startup_time: datetime | float

    def initialize_health_checks(self) -> None:
        """
        Initialize health check system.

        Call this in your node's __init__() method after super().__init__().
        Sets up internal state for health checking.
        """
        # Health check registry
        self._health_checks: dict[str, dict[str, Any]] = {}
        self._critical_components: set[str] = set()
        self._startup_time = datetime.now(UTC)
        self._last_health_check: Optional[NodeHealthCheckResult] = None
        self._health_check_cache_ttl_seconds = 5.0

        # Register node-specific checks
        self._register_component_checks()

        logger.info(
            "Health check system initialized",
            node_id=getattr(self, "node_id", "unknown"),
            node_type=type(self).__name__,
        )

    def _register_component_checks(self) -> None:
        """
        Register node-specific component health checks.

        Override this method in your node to register component checks:

        Example:
            def _register_component_checks(self):
                self.register_component_check(
                    "database",
                    self._check_database_health,
                    critical=True
                )
                self.register_component_check(
                    "kafka",
                    self._check_kafka_health,
                    critical=False
                )
        """
        # Default: register basic node health check
        self.register_component_check(
            "node_runtime",
            self._check_node_runtime,
            critical=True,
            timeout_seconds=1.0,
        )

    def register_component_check(
        self,
        component_name: str,
        check_func: Callable[[], Awaitable[tuple[HealthStatus, str, dict[str, Any]]]],
        critical: bool = False,
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Register a component health check function.

        Args:
            component_name: Unique name for the component
            check_func: Async function returning (status, message, details)
            critical: Whether component is critical for node operation
            timeout_seconds: Timeout for this check
        """
        self._health_checks[component_name] = {
            "func": check_func,
            "critical": critical,
            "timeout": timeout_seconds,
        }

        if critical:
            self._critical_components.add(component_name)

        logger.debug(
            "Registered health check",
            component=component_name,
            critical=critical,
            timeout_seconds=timeout_seconds,
        )

    async def check_health(self, force: bool = False) -> NodeHealthCheckResult:
        """
        Perform comprehensive health check of all components.

        Args:
            force: Force fresh check, bypass cache

        Returns:
            NodeHealthCheckResult with overall status and component details
        """
        # Return cached result if available and fresh
        if (
            not force
            and self._last_health_check
            and (datetime.now(UTC) - self._last_health_check.timestamp).total_seconds()
            < self._health_check_cache_ttl_seconds
        ):
            return self._last_health_check

        # Execute all component checks
        component_results: list[ComponentHealth] = []

        for component_name, check_config in self._health_checks.items():
            component_health = await self._execute_component_check(
                component_name, check_config
            )
            component_results.append(component_health)

        # Determine overall status
        overall_status = self._determine_overall_status(component_results)

        # Calculate uptime - handle both datetime and float types for _startup_time
        uptime_seconds = self._calculate_uptime()

        # Create result
        result = NodeHealthCheckResult(
            node_id=getattr(self, "node_id", "unknown"),
            node_type=type(self).__name__,
            overall_status=overall_status,
            components=component_results,
            uptime_seconds=uptime_seconds,
            version=getattr(self, "version", "1.0.0"),
        )

        # Cache result
        self._last_health_check = result

        logger.debug(
            "Health check completed",
            overall_status=overall_status.value,
            components_checked=len(component_results),
            critical_unhealthy=result.to_dict()["summary"]["critical_unhealthy"],
        )

        return result

    async def _execute_component_check(
        self,
        component_name: str,
        check_config: dict[str, Any],
    ) -> ComponentHealth:
        """
        Execute a single component health check.

        Args:
            component_name: Name of the component
            check_config: Check configuration with func, timeout, critical

        Returns:
            ComponentHealth result
        """
        start_time = time.time()
        check_func = check_config["func"]
        timeout = check_config["timeout"]
        is_critical = check_config["critical"]

        try:
            # Execute check with timeout
            status, message, details = await asyncio.wait_for(
                check_func(), timeout=timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=component_name,
                status=status,
                message=message,
                details=details,
                check_duration_ms=duration_ms,
                is_critical=is_critical,
            )

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
                details={"timeout_seconds": timeout},
                check_duration_ms=duration_ms,
                is_critical=is_critical,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            logger.warning(
                "Component health check failed",
                component=component_name,
                error=str(e),
                error_type=type(e).__name__,
            )

            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                details={"error": str(e), "error_type": type(e).__name__},
                check_duration_ms=duration_ms,
                is_critical=is_critical,
            )

    def _determine_overall_status(
        self, components: list[ComponentHealth]
    ) -> HealthStatus:
        """
        Determine overall health status from component checks.

        Logic:
        - UNHEALTHY: Any critical component is unhealthy
        - DEGRADED: Some non-critical components unhealthy or any component degraded
        - HEALTHY: All components healthy
        - UNKNOWN: No components or all unknown

        Args:
            components: List of component health results

        Returns:
            Overall health status
        """
        if not components:
            return HealthStatus.UNKNOWN

        # Check for critical component failures
        critical_unhealthy = any(
            c.is_critical and c.status == HealthStatus.UNHEALTHY for c in components
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # Check for any degraded components
        any_degraded = any(c.status == HealthStatus.DEGRADED for c in components)

        # Check for non-critical unhealthy components
        any_unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in components)

        if any_degraded or any_unhealthy:
            return HealthStatus.DEGRADED

        # Check if all are healthy
        all_healthy = all(c.status == HealthStatus.HEALTHY for c in components)

        if all_healthy:
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_status(self) -> dict[str, Any]:
        """
        Synchronous health status retrieval.

        Returns cached health status or 'unknown' if no check has been performed.
        Use this for non-async contexts like Docker HEALTHCHECK scripts.

        Returns:
            Dictionary with health status information
        """
        if self._last_health_check:
            return self._last_health_check.to_dict()

        # Calculate uptime - handle both datetime and float types for _startup_time
        uptime_seconds = self._calculate_uptime()

        # Return minimal status if no check has been performed
        return {
            "node_id": getattr(self, "node_id", "unknown"),
            "node_type": type(self).__name__,
            "overall_status": HealthStatus.UNKNOWN.value,
            "message": "No health check performed yet",
            "uptime_seconds": uptime_seconds,
        }

    def is_healthy(self) -> bool:
        """
        Simple boolean health check.

        Returns True if overall status is HEALTHY or DEGRADED.
        Use this for quick health checks in routing decisions.

        Returns:
            True if node is operational (healthy or degraded)
        """
        if self._last_health_check:
            return self._last_health_check.overall_status in (
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED,
            )

        # Conservative: assume healthy if no check yet
        return True

    def _calculate_uptime(self) -> float:
        """
        Calculate uptime in seconds, handling both datetime and float types.

        The _startup_time attribute can be either:
        - datetime object (from HealthCheckMixin)
        - float (from NodeIntrospectionMixin)

        Returns:
            Uptime in seconds
        """
        if isinstance(self._startup_time, datetime):
            # _startup_time is a datetime object
            return (datetime.now(UTC) - self._startup_time).total_seconds()
        elif isinstance(self._startup_time, int | float):
            # _startup_time is a unix timestamp
            return time.time() - self._startup_time
        else:
            # Fallback: assume node just started
            return 0.0

    async def _check_node_runtime(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Basic node runtime health check.

        Checks that the node is running and responsive.
        This is the default critical check for all nodes.

        Returns:
            Tuple of (status, message, details)
        """
        try:
            # Check node has required attributes
            node_id = getattr(self, "node_id", None)
            container = getattr(self, "container", None)

            if not node_id:
                return (
                    HealthStatus.UNHEALTHY,
                    "Node ID not set",
                    {"missing": "node_id"},
                )

            if not container:
                return (
                    HealthStatus.DEGRADED,
                    "Container not available",
                    {"missing": "container"},
                )

            # Node is operational - calculate uptime safely
            uptime = self._calculate_uptime()

            return (
                HealthStatus.HEALTHY,
                "Node runtime operational",
                {
                    "node_id": str(node_id),
                    "uptime_seconds": uptime,
                    "has_container": container is not None,
                },
            )

        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Runtime check failed: {e!s}",
                {"error": str(e)},
            )

    # Docker HEALTHCHECK compatibility methods

    def docker_healthcheck_exit_code(self) -> int:
        """
        Get Docker HEALTHCHECK compatible exit code.

        Returns 0 for healthy/degraded, 1 for unhealthy/unknown.
        Use this in Docker HEALTHCHECK CMD scripts.

        Returns:
            Exit code (0 = healthy, 1 = unhealthy)
        """
        if self._last_health_check:
            return self._last_health_check.overall_status.to_docker_exit_code()

        # Conservative: return unhealthy if no check yet
        return 1

    def http_status_code(self) -> int:
        """
        Get HTTP status code for health endpoint.

        Returns:
            HTTP status code (200, 503, or 500)
        """
        if self._last_health_check:
            return self._last_health_check.overall_status.to_http_status()

        return 500  # Internal server error if no check yet


# Common component health check functions for reuse


async def check_database_connection(
    db_client: Any,
    timeout_seconds: float = 3.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """
    Check database connection health.

    Args:
        db_client: Database client instance
        timeout_seconds: Check timeout

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Attempt simple query with timeout
        start_time = time.time()

        # Check if client has async query method
        if hasattr(db_client, "execute"):
            await asyncio.wait_for(
                db_client.execute("SELECT 1"),
                timeout=timeout_seconds,
            )
        else:
            # Assume client is healthy if it exists
            pass

        query_time_ms = (time.time() - start_time) * 1000

        return (
            HealthStatus.HEALTHY,
            "Database connection healthy",
            {"query_time_ms": query_time_ms},
        )

    except TimeoutError:
        return (
            HealthStatus.UNHEALTHY,
            "Database query timed out",
            {"timeout_seconds": timeout_seconds},
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database connection failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )


async def check_kafka_connection(
    kafka_client: Any,
    timeout_seconds: float = 3.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """
    Check Kafka connection health.

    Args:
        kafka_client: Kafka producer/consumer instance
        timeout_seconds: Check timeout

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Check if client is connected
        if hasattr(kafka_client, "bootstrap_connected"):
            if not await asyncio.wait_for(
                kafka_client.bootstrap_connected(),
                timeout=timeout_seconds,
            ):
                return (
                    HealthStatus.UNHEALTHY,
                    "Kafka not connected",
                    {},
                )

        return (
            HealthStatus.HEALTHY,
            "Kafka connection healthy",
            {},
        )

    except TimeoutError:
        return (
            HealthStatus.DEGRADED,
            "Kafka connection check timed out",
            {"timeout_seconds": timeout_seconds},
        )

    except Exception as e:
        return (
            HealthStatus.DEGRADED,
            f"Kafka connection check failed: {e!s}",
            {"error": str(e)},
        )


async def check_http_service(
    service_url: str,
    timeout_seconds: float = 3.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """
    Check HTTP service health via health endpoint.

    Args:
        service_url: Service base URL or health endpoint URL
        timeout_seconds: Request timeout

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import aiohttp

        # Append /health if not already present
        health_url = (
            service_url if service_url.endswith("/health") else f"{service_url}/health"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                if response.status == 200:
                    return (
                        HealthStatus.HEALTHY,
                        f"Service {service_url} healthy",
                        {"status_code": response.status, "url": health_url},
                    )
                else:
                    return (
                        HealthStatus.DEGRADED,
                        f"Service {service_url} returned {response.status}",
                        {"status_code": response.status, "url": health_url},
                    )

    except TimeoutError:
        return (
            HealthStatus.DEGRADED,
            f"Service {service_url} timed out",
            {"timeout_seconds": timeout_seconds, "url": service_url},
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Service {service_url} check failed: {e!s}",
            {"error": str(e), "url": service_url},
        )
