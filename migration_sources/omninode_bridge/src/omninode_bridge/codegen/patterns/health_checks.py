#!/usr/bin/env python3
"""
Health Check Pattern Generator for ONEX Code Generation.

Generates comprehensive health check implementations for ONEX v2.0 nodes with:
- Self health checks (node operational status)
- Database health checks (PostgreSQL connection pooling)
- Kafka health checks (producer/consumer status)
- Consul health checks (service registration)
- HTTP service health checks (external dependencies)

Generated code uses omnibase_core types and follows ONEX v2.0 patterns.

Phase 2 Goals:
- Reduce manual completion from 50% → 10%
- Generate production-ready health check implementations
- Include proper error handling and logging
- Support all major infrastructure dependencies

Example:
    >>> from omninode_bridge.codegen.patterns.health_checks import HealthCheckGenerator
    >>> generator = HealthCheckGenerator()
    >>> code = generator.generate_health_check_method(
    ...     node_type="NodePostgresCRUDEffect",
    ...     dependencies=["postgres", "kafka"],
    ...     operations=["read", "write"]
    ... )
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from omninode_bridge.health.health_checker import ComponentHealthCheckType

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthCheckConfig:
    """
    Configuration for health check generation.

    Attributes:
        check_type: Type of health check (self/database/kafka/consul/http_service)
        component_name: Human-readable component name
        is_critical: Whether this component is critical for node operation
        timeout_seconds: Timeout for health check execution
        custom_checks: Additional custom check logic
        dependencies: List of dependencies this check requires
    """

    check_type: ComponentHealthCheckType
    component_name: str
    is_critical: bool = False
    timeout_seconds: float = 5.0
    custom_checks: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


class HealthCheckGenerator:
    """
    Generator for production-ready health check implementations.

    Creates complete health check methods that:
    - Follow ONEX v2.0 patterns
    - Use omnibase_core types (HealthStatus, ComponentHealth)
    - Include proper error handling
    - Use structured logging (emit_log_event)
    - Support async operations
    - Handle timeouts gracefully
    """

    def __init__(self):
        """Initialize health check generator."""
        self.generated_checks: list[str] = []

    def generate_health_check_method(
        self,
        node_type: str,
        dependencies: list[str],
        operations: list[str] | None = None,
    ) -> str:
        """
        Generate complete health check implementation for a node.

        Creates a comprehensive health check method that:
        1. Registers all component checks (_register_component_checks)
        2. Generates individual check methods for each dependency
        3. Includes proper error handling and logging

        Args:
            node_type: Node class name (e.g., "NodePostgresCRUDEffect")
            dependencies: List of dependencies (e.g., ["postgres", "kafka", "consul"])
            operations: Optional list of operations to validate (e.g., ["read", "write"])

        Returns:
            Complete Python code for health check implementation

        Example:
            >>> generator = HealthCheckGenerator()
            >>> code = generator.generate_health_check_method(
            ...     node_type="NodePostgresCRUDEffect",
            ...     dependencies=["postgres", "kafka"],
            ...     operations=["read", "write"]
            ... )
        """
        # Input validation
        if not node_type or not isinstance(node_type, str):
            raise ValueError(
                f"node_type must be a non-empty string, got: {node_type!r}. "
                f"Valid examples: 'NodePostgresCRUDEffect', 'NodeKafkaConsumerEffect', 'effect', 'compute'"
            )

        if not isinstance(dependencies, list):
            raise TypeError(
                f"dependencies must be a list, got: {type(dependencies).__name__}. "
                f"Example: ['postgres', 'kafka', 'consul']"
            )

        VALID_DEPENDENCIES = {
            "postgres",
            "database",
            "kafka",
            "redpanda",
            "consul",
            "redis",
            "http_service",
            "vault",
        }
        invalid_deps = {
            dep for dep in dependencies if not dep.startswith(("http://", "https://"))
        } - VALID_DEPENDENCIES
        if invalid_deps:
            raise ValueError(
                f"Invalid dependencies: {invalid_deps}. "
                f"Valid options: {', '.join(sorted(VALID_DEPENDENCIES))} or URLs starting with http:// or https://"
            )

        if operations is not None and not isinstance(operations, list):
            raise TypeError(
                f"operations must be a list or None, got: {type(operations).__name__}. "
                f"Example: ['read', 'write', 'query']"
            )

        operations = operations or []

        # Generate _register_component_checks method
        register_method = self._generate_register_method(dependencies)

        # Generate individual check methods
        check_methods = []
        for dep in dependencies:
            if dep == "postgres" or dep == "database":
                check_methods.append(self.generate_database_health_check())
            elif dep == "kafka":
                check_methods.append(self.generate_kafka_health_check())
            elif dep == "consul":
                check_methods.append(self.generate_consul_health_check())
            elif dep.startswith("http://") or dep.startswith("https://"):
                # Extract service name from URL to avoid collisions with multiple HTTP dependencies
                hostname_with_port = dep.split("://", 1)[1].split("/", 1)[0]
                service_name = hostname_with_port.split(":")[0].replace("-", "_")
                check_methods.append(
                    self.generate_http_service_health_check(
                        service_name=service_name, service_url=dep
                    )
                )

        # Always include self check
        check_methods.insert(0, self.generate_self_health_check())

        # Combine all methods
        all_methods = [register_method] + check_methods

        return "\n\n".join(all_methods)

    def _generate_register_method(self, dependencies: list[str]) -> str:
        """
        Generate _register_component_checks method.

        This method is called during node initialization to register
        all health check functions with the HealthCheckMixin.

        Args:
            dependencies: List of dependencies to register checks for

        Returns:
            Python code for _register_component_checks method
        """
        checks = []

        # Self check (always critical)
        checks.append(
            "        self.register_component_check(\n"
            '            "node_runtime",\n'
            "            self._check_node_runtime,\n"
            "            critical=True,\n"
            "            timeout_seconds=1.0,\n"
            "        )"
        )

        # Dependency checks
        for dep in dependencies:
            if dep == "postgres" or dep == "database":
                checks.append(
                    "        self.register_component_check(\n"
                    '            "postgres",\n'
                    "            self._check_postgres_health,\n"
                    "            critical=True,\n"
                    "            timeout_seconds=5.0,\n"
                    "        )"
                )
            elif dep == "kafka":
                checks.append(
                    "        self.register_component_check(\n"
                    '            "kafka",\n'
                    "            self._check_kafka_health,\n"
                    "            critical=False,\n"
                    "            timeout_seconds=5.0,\n"
                    "        )"
                )
            elif dep == "consul":
                checks.append(
                    "        self.register_component_check(\n"
                    '            "consul",\n'
                    "            self._check_consul_health,\n"
                    "            critical=False,\n"
                    "            timeout_seconds=3.0,\n"
                    "        )"
                )
            elif dep.startswith("http://") or dep.startswith("https://"):
                # Extract service key from URL (e.g., "http://service-name:8080" -> "service_name")
                hostname_with_port = dep.split("://", 1)[1].split("/", 1)[0]
                service_key = hostname_with_port.split(":")[0].replace("-", "_")
                checks.append(
                    f"        self.register_component_check(\n"
                    f'            "{service_key}",\n'
                    f"            self._check_{service_key}_health,\n"
                    "            critical=False,\n"
                    "            timeout_seconds=3.0,\n"
                    "        )"
                )

        checks_code = "\n\n".join(checks)

        return f'''    def _register_component_checks(self) -> None:
        """
        Register node-specific component health checks.

        Called during initialization to set up health monitoring for:
        - Node runtime (critical)
{self._format_dependency_list(dependencies)}

        Critical components must be healthy for node operation.
        Non-critical components can degrade without failing the node.
        """
{checks_code}'''

    def _format_dependency_list(self, dependencies: list[str]) -> str:
        """Format dependency list for docstring."""
        dep_lines = []
        for dep in dependencies:
            criticality = (
                "critical" if dep in ["postgres", "database"] else "non-critical"
            )
            dep_lines.append(f"        - {dep.capitalize()} ({criticality})")
        return "\n".join(dep_lines)

    def generate_self_health_check(self) -> str:
        """
        Generate self health check pattern.

        Checks node operational status:
        - Node ID is set
        - Container is available
        - Uptime is being tracked
        - Memory usage is acceptable

        Returns:
            Python code for _check_node_runtime method
        """
        return '''    async def _check_node_runtime(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check node runtime health.

        Validates that the node is operational:
        - Node ID is configured
        - Container is available
        - Uptime tracking is active
        - Basic runtime metrics are healthy

        Returns:
            Tuple of (status, message, details)

        Example:
            >>> status, message, details = await node._check_node_runtime()
            >>> assert status == HealthStatus.HEALTHY
            >>> assert "operational" in message.lower()
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

            # Calculate uptime
            uptime = self._calculate_uptime()

            # Check memory usage (if available)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_details = {"memory_mb": round(memory_mb, 2)}

                # Warn if memory usage is high (>512MB for typical nodes)
                if memory_mb > 512:
                    return (
                        HealthStatus.DEGRADED,
                        f"High memory usage: {memory_mb:.1f}MB",
                        {**memory_details, "threshold_mb": 512},
                    )
            except ImportError:
                memory_details = {"memory_mb": "unavailable"}

            return (
                HealthStatus.HEALTHY,
                "Node runtime operational",
                {
                    "node_id": str(node_id),
                    "uptime_seconds": uptime,
                    "has_container": container is not None,
                    **memory_details,
                },
            )

        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Runtime check failed: {e!s}",
                {"error": str(e), "error_type": type(e).__name__},
            )'''

    def generate_database_health_check(self) -> str:
        """
        Generate PostgreSQL database health check pattern.

        Checks:
        - Database client is configured
        - Connection pool is available
        - Can execute simple query (SELECT 1)
        - Connection pool has capacity
        - Query performance is acceptable

        Returns:
            Python code for _check_postgres_health method
        """
        return '''    async def _check_postgres_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check PostgreSQL database health.

        Validates database connectivity and performance:
        - Client is configured
        - Connection pool is available
        - Can execute queries (SELECT 1)
        - Pool has available connections
        - Query latency is acceptable (<100ms)

        Returns:
            Tuple of (status, message, details)

        Example:
            >>> status, message, details = await node._check_postgres_health()
            >>> assert status == HealthStatus.HEALTHY
            >>> assert details["query_time_ms"] < 100
        """
        try:
            # Check if database client exists
            if not hasattr(self.container, "db_client") or not self.container.db_client:
                return (
                    HealthStatus.UNHEALTHY,
                    "Database client not configured",
                    {"missing": "db_client"},
                )

            db_client = self.container.db_client

            # Test connection with simple query
            import time
            start_time = time.time()

            try:
                # Try multiple query methods (different clients have different APIs)
                if hasattr(db_client, "fetch_one"):
                    result = await db_client.fetch_one("SELECT 1")
                elif hasattr(db_client, "fetchrow"):
                    result = await db_client.fetchrow("SELECT 1")
                elif hasattr(db_client, "execute"):
                    result = await db_client.execute("SELECT 1")
                else:
                    return (
                        HealthStatus.DEGRADED,
                        "Database client missing query methods",
                        {"available_methods": dir(db_client)},
                    )
            except Exception as query_error:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Database query failed: {query_error!s}",
                    {"error": str(query_error), "error_type": type(query_error).__name__},
                )

            query_time_ms = (time.time() - start_time) * 1000

            # Check connection pool status (if available)
            pool_details = {}
            if hasattr(db_client, "pool"):
                pool = db_client.pool
                if hasattr(pool, "get_size"):
                    pool_details["pool_size"] = pool.get_size()
                if hasattr(pool, "get_idle_size"):
                    pool_details["idle_connections"] = pool.get_idle_size()

            # Determine status based on query time
            if query_time_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"Slow database query: {query_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"

            return (
                status,
                message,
                {
                    "query_time_ms": round(query_time_ms, 2),
                    "has_result": result is not None,
                    **pool_details,
                },
            )

        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Database health check failed: {e!s}",
                {"error": str(e), "error_type": type(e).__name__},
            )'''

    def generate_kafka_health_check(self) -> str:
        """
        Generate Kafka health check pattern.

        Checks:
        - Kafka client is configured
        - Producer/consumer is connected
        - Bootstrap servers are reachable
        - No connection errors

        Returns:
            Python code for _check_kafka_health method
        """
        return '''    async def _check_kafka_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check Kafka connection health.

        Validates Kafka connectivity and status:
        - Client is configured
        - Bootstrap servers are reachable
        - Producer/consumer is connected
        - No connection errors

        Returns:
            Tuple of (status, message, details)

        Example:
            >>> status, message, details = await node._check_kafka_health()
            >>> assert status == HealthStatus.HEALTHY
            >>> assert details["connected"] is True
        """
        try:
            # Check if Kafka client exists
            if not hasattr(self.container, "kafka_producer") or not self.container.kafka_producer:
                # Kafka is often optional, so degraded not unhealthy
                return (
                    HealthStatus.DEGRADED,
                    "Kafka producer not configured",
                    {"missing": "kafka_producer"},
                )

            kafka_client = self.container.kafka_producer

            # Check if connected (aiokafka)
            if hasattr(kafka_client, "bootstrap_connected"):
                try:
                    connected = await kafka_client.bootstrap_connected()
                    if not connected:
                        return (
                            HealthStatus.DEGRADED,
                            "Kafka not connected to bootstrap servers",
                            {"connected": False},
                        )
                except Exception as connect_error:
                    return (
                        HealthStatus.DEGRADED,
                        f"Kafka connection check failed: {connect_error!s}",
                        {"error": str(connect_error)},
                    )

            # Check if started (aiokafka requires start() to be called)
            if hasattr(kafka_client, "_closed"):
                is_closed = kafka_client._closed
                if is_closed:
                    return (
                        HealthStatus.DEGRADED,
                        "Kafka producer is closed",
                        {"closed": True},
                    )

            # Get cluster metadata (if available)
            cluster_details = {}
            if hasattr(kafka_client, "client") and hasattr(kafka_client.client, "cluster"):
                cluster = kafka_client.client.cluster
                if hasattr(cluster, "brokers"):
                    cluster_details["broker_count"] = len(cluster.brokers())

            return (
                HealthStatus.HEALTHY,
                "Kafka connection healthy",
                {
                    "connected": True,
                    "client_type": type(kafka_client).__name__,
                    **cluster_details,
                },
            )

        except Exception as e:
            # Kafka failures are non-critical - degrade not fail
            return (
                HealthStatus.DEGRADED,
                f"Kafka health check failed: {e!s}",
                {"error": str(e), "error_type": type(e).__name__},
            )'''

    def generate_consul_health_check(self) -> str:
        """
        Generate Consul health check pattern.

        Checks:
        - Consul client is configured
        - Can connect to Consul API
        - Service registration is active
        - Health check is registered

        Returns:
            Python code for _check_consul_health method
        """
        return '''    async def _check_consul_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check Consul service discovery health.

        Validates Consul connectivity and registration:
        - Client is configured
        - Can connect to Consul API
        - Service registration is active
        - Health check is registered

        Returns:
            Tuple of (status, message, details)

        Example:
            >>> status, message, details = await node._check_consul_health()
            >>> assert status == HealthStatus.HEALTHY
            >>> assert details["registered"] is True
        """
        try:
            # Check if Consul client exists
            if not hasattr(self.container, "consul_client") or not self.container.consul_client:
                # Consul is often optional, so degraded not unhealthy
                return (
                    HealthStatus.DEGRADED,
                    "Consul client not configured",
                    {"missing": "consul_client"},
                )

            consul_client = self.container.consul_client

            # Check if service is registered
            if hasattr(consul_client, "agent") and hasattr(consul_client.agent, "services"):
                try:
                    services = await consul_client.agent.services()
                    node_id = getattr(self, "node_id", None)

                    if node_id and str(node_id) in services:
                        service_info = services[str(node_id)]
                        return (
                            HealthStatus.HEALTHY,
                            "Consul service registered",
                            {
                                "registered": True,
                                "service_id": str(node_id),
                                "service_name": service_info.get("Service", "unknown"),
                            },
                        )
                    else:
                        return (
                            HealthStatus.DEGRADED,
                            "Service not registered in Consul",
                            {"registered": False, "node_id": str(node_id)},
                        )
                except Exception as consul_error:
                    return (
                        HealthStatus.DEGRADED,
                        f"Consul service check failed: {consul_error!s}",
                        {"error": str(consul_error)},
                    )

            # Basic connectivity check if no agent API
            if hasattr(consul_client, "health"):
                try:
                    # Simple health check to validate connectivity
                    await consul_client.health.node("consul")
                    return (
                        HealthStatus.HEALTHY,
                        "Consul connectivity healthy",
                        {"connected": True},
                    )
                except Exception as health_error:
                    return (
                        HealthStatus.DEGRADED,
                        f"Consul health API failed: {health_error!s}",
                        {"error": str(health_error)},
                    )

            return (
                HealthStatus.HEALTHY,
                "Consul client available",
                {"connected": True},
            )

        except Exception as e:
            # Consul failures are non-critical - degrade not fail
            return (
                HealthStatus.DEGRADED,
                f"Consul health check failed: {e!s}",
                {"error": str(e), "error_type": type(e).__name__},
            )'''

    def generate_http_service_health_check(
        self,
        service_name: str = "external_service",
        service_url: str = "http://service:8080",
    ) -> str:
        """
        Generate HTTP service health check pattern.

        Checks:
        - Service URL is configured
        - Can connect to service
        - Service /health endpoint returns 200
        - Response time is acceptable

        Args:
            service_name: Name of the service (for logging)
            service_url: Base URL of the service

        Returns:
            Python code for _check_{service_name}_health method
        """
        # Input validation
        if not service_name or not isinstance(service_name, str):
            raise ValueError(
                f"service_name must be a non-empty string, got: {service_name!r}. "
                f"Valid examples: 'external_service', 'api_gateway', 'metadata_service'"
            )

        if not service_url or not isinstance(service_url, str):
            raise ValueError(
                f"service_url must be a non-empty string, got: {service_url!r}. "
                f"Valid examples: 'http://service:8080', 'https://api.example.com'"
            )

        if not service_url.startswith(("http://", "https://")):
            raise ValueError(
                f"service_url must start with 'http://' or 'https://', got: {service_url!r}. "
                f"Valid examples: 'http://service:8080', 'https://api.example.com'"
            )

        method_name = f"_check_{service_name.lower().replace('-', '_')}_health"

        return f'''    async def {method_name}(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check {service_name} HTTP service health.

        Validates HTTP service connectivity:
        - Service URL is configured
        - Can connect to service
        - /health endpoint returns 200
        - Response time is acceptable (<3s)

        Returns:
            Tuple of (status, message, details)

        Example:
            >>> status, message, details = await node.{method_name}()
            >>> assert status == HealthStatus.HEALTHY
            >>> assert details["status_code"] == 200
        """
        try:
            import asyncio
            import aiohttp

            # Get service URL from container config or use default
            service_url = getattr(
                self.container.config,
                "{service_name.lower().replace('-', '_')}_url",
                "{service_url}",
            )

            # Append /health if not already present
            health_url = (
                service_url
                if service_url.endswith("/health")
                else f"{{service_url}}/health"
            )

            # Make health check request
            import time
            start_time = time.time()

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        health_url,
                        timeout=aiohttp.ClientTimeout(total=3.0),
                    ) as response:
                        request_time_ms = (time.time() - start_time) * 1000

                        if response.status == 200:
                            return (
                                HealthStatus.HEALTHY,
                                f"{service_name} service healthy",
                                {{
                                    "status_code": response.status,
                                    "url": health_url,
                                    "response_time_ms": round(request_time_ms, 2),
                                }},
                            )
                        else:
                            return (
                                HealthStatus.DEGRADED,
                                f"{service_name} returned {{response.status}}",
                                {{
                                    "status_code": response.status,
                                    "url": health_url,
                                    "response_time_ms": round(request_time_ms, 2),
                                }},
                            )
            except asyncio.TimeoutError:
                return (
                    HealthStatus.DEGRADED,
                    f"{service_name} health check timed out",
                    {{"url": health_url, "timeout_seconds": 3.0}},
                )
            except aiohttp.ClientError as client_error:
                return (
                    HealthStatus.DEGRADED,
                    f"{service_name} connection failed: {{client_error!s}}",
                    {{"url": health_url, "error": str(client_error)}},
                )

        except ImportError:
            return (
                HealthStatus.DEGRADED,
                "aiohttp not available for HTTP health checks",
                {{"missing_dependency": "aiohttp"}},
            )
        except Exception as e:
            return (
                HealthStatus.DEGRADED,
                f"{service_name} health check failed: {{e!s}}",
                {{"error": str(e), "error_type": type(e).__name__}},
            )'''


# Convenience functions for direct use


def generate_health_check_method(
    node_type: str,
    dependencies: list[str],
    operations: list[str] | None = None,
) -> str:
    """
    Generate complete health check implementation.

    Convenience function that creates a HealthCheckGenerator and generates
    health check methods for the specified node type and dependencies.

    Args:
        node_type: Node class name (e.g., "NodePostgresCRUDEffect")
        dependencies: List of dependencies (e.g., ["postgres", "kafka", "consul"])
        operations: Optional list of operations to validate

    Returns:
        Complete Python code for health check implementation

    Example:
        >>> code = generate_health_check_method(
        ...     node_type="NodePostgresCRUDEffect",
        ...     dependencies=["postgres", "kafka"],
        ...     operations=["read", "write"]
        ... )
        >>> print(code)
    """
    generator = HealthCheckGenerator()
    return generator.generate_health_check_method(node_type, dependencies, operations)


def generate_self_health_check() -> str:
    """
    Generate self health check pattern.

    Returns:
        Python code for _check_node_runtime method
    """
    generator = HealthCheckGenerator()
    return generator.generate_self_health_check()


def generate_database_health_check() -> str:
    """
    Generate PostgreSQL health check pattern.

    Returns:
        Python code for _check_postgres_health method
    """
    generator = HealthCheckGenerator()
    return generator.generate_database_health_check()


def generate_kafka_health_check() -> str:
    """
    Generate Kafka health check pattern.

    Returns:
        Python code for _check_kafka_health method
    """
    generator = HealthCheckGenerator()
    return generator.generate_kafka_health_check()


def generate_consul_health_check() -> str:
    """
    Generate Consul health check pattern.

    Returns:
        Python code for _check_consul_health method
    """
    generator = HealthCheckGenerator()
    return generator.generate_consul_health_check()


def generate_http_service_health_check(
    service_name: str = "external_service",
    service_url: str = "http://service:8080",
) -> str:
    """
    Generate HTTP service health check pattern.

    Args:
        service_name: Name of the service
        service_url: Base URL of the service

    Returns:
        Python code for health check method
    """
    generator = HealthCheckGenerator()
    return generator.generate_http_service_health_check(service_name, service_url)


__all__ = [
    "HealthCheckGenerator",
    "ModelHealthCheckConfig",
    "generate_health_check_method",
    "generate_self_health_check",
    "generate_database_health_check",
    "generate_kafka_health_check",
    "generate_consul_health_check",
    "generate_http_service_health_check",
]
