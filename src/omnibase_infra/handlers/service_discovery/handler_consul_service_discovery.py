# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Service Discovery Handler.

This module provides a Consul-backed implementation of the service discovery
handler protocol, wrapping existing Consul functionality with circuit breaker
resilience.

Thread Pool Management:
    - All synchronous consul operations run in a dedicated thread pool
    - Configurable max workers (default: 10)
    - Thread pool is lazily initialized on first use
    - Thread pool gracefully shutdown on handler shutdown

Circuit Breaker:
    - Uses MixinAsyncCircuitBreaker for consistent resilience
    - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
    - Configurable via ModelCircuitBreakerConfig model
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import consul

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
)
from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelHandlerRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
from omnibase_infra.nodes.node_service_discovery_effect.models import (
    ModelDiscoveryQuery,
    ModelServiceDiscoveryHealthCheckDetails,
    ModelServiceDiscoveryHealthCheckResult,
)
from omnibase_infra.nodes.node_service_discovery_effect.models.enum_health_status import (
    EnumHealthStatus,
)
from omnibase_infra.nodes.node_service_discovery_effect.models.enum_service_discovery_operation import (
    EnumServiceDiscoveryOperation,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_WORKERS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0

# Custom namespace for ONEX service IDs (deterministic but distinct from DNS namespace)
# This ensures UUID5 generation for non-UUID service IDs is specific to ONEX
# and won't collide with DNS-based UUID5 values from other systems.
NAMESPACE_ONEX_SERVICE = uuid5(NAMESPACE_DNS, "omninode.service.discovery")


class ConsulServiceDiscoveryHandler(MixinAsyncCircuitBreaker):
    """Consul implementation of ProtocolServiceDiscoveryHandler.

    Wraps existing Consul client functionality with circuit breaker resilience
    and proper error handling.

    Thread Safety:
        This handler is coroutine-safe. All Consul operations are executed
        in a dedicated thread pool, and circuit breaker state is protected
        by asyncio.Lock.

    Thread Pool Initialization:
        The thread pool executor is lazily initialized on first use via
        ``_ensure_executor()``. This avoids resource allocation until the
        handler is actually used.

    Attributes:
        handler_type: Returns "consul" identifier.

    Example:
        >>> handler = ConsulServiceDiscoveryHandler(
        ...     consul_client=consul_client,
        ...     circuit_breaker_config=ModelCircuitBreakerConfig(threshold=5),
        ... )
        >>> result = await handler.register_service(service_info)
    """

    def __init__(
        self,
        consul_client: ProtocolConsulClient | None = None,
        consul_host: str = "localhost",
        consul_port: int = 8500,
        consul_scheme: str = "http",
        consul_token: str | None = None,
        circuit_breaker_config: ModelCircuitBreakerConfig
        | dict[str, object]
        | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize ConsulServiceDiscoveryHandler.

        Args:
            consul_client: Optional existing Consul client (ProtocolConsulClient).
                If not provided, a new python-consul client will be created.
            consul_host: Consul server hostname (default: "localhost").
            consul_port: Consul server port (default: 8500).
            consul_scheme: HTTP scheme "http" or "https" (default: "http").
            consul_token: Optional Consul ACL token.
            circuit_breaker_config: Optional circuit breaker configuration.
                Can be a ModelCircuitBreakerConfig instance or a dict with keys:
                - threshold: Max failures before opening (default: 5)
                - reset_timeout_seconds: Seconds before reset (default: 60.0)
                - service_name: Service identifier (default: "consul.discovery")
                If not provided, uses ModelCircuitBreakerConfig defaults.
            max_workers: Thread pool max workers (default: 10).
            timeout_seconds: Operation timeout in seconds (default: 30.0).
        """
        # Parse circuit breaker configuration using ModelCircuitBreakerConfig
        if isinstance(circuit_breaker_config, ModelCircuitBreakerConfig):
            cb_config = circuit_breaker_config
        elif circuit_breaker_config is not None:
            # Handle legacy dict format with key mapping
            config_dict = dict(circuit_breaker_config)
            # Map legacy 'reset_timeout' key to 'reset_timeout_seconds'
            if (
                "reset_timeout" in config_dict
                and "reset_timeout_seconds" not in config_dict
            ):
                config_dict["reset_timeout_seconds"] = config_dict.pop("reset_timeout")
            # Set defaults for service_name and transport_type if not provided
            config_dict.setdefault("service_name", "consul.discovery")
            config_dict.setdefault("transport_type", EnumInfraTransportType.CONSUL)
            cb_config = ModelCircuitBreakerConfig(**config_dict)
        else:
            cb_config = ModelCircuitBreakerConfig(
                service_name="consul.discovery",
                transport_type=EnumInfraTransportType.CONSUL,
            )

        self._init_circuit_breaker(
            threshold=cb_config.threshold,
            reset_timeout=cb_config.reset_timeout_seconds,
            service_name=cb_config.service_name,
            transport_type=cb_config.transport_type,
        )

        # Store configuration
        self._consul_host = consul_host
        self._consul_port = consul_port
        self._consul_scheme = consul_scheme
        self._consul_token = consul_token
        self._timeout_seconds = timeout_seconds

        # Initialize Consul client
        # Note: We use consul.Consul type since that's what we create internally.
        # External clients are expected to duck-type as consul.Consul.
        self._consul_client: consul.Consul | None
        if consul_client is not None:
            # Use provided client (duck-typed ProtocolConsulClient)
            self._consul_client = consul_client
            self._owns_client = False
        else:
            # Create python-consul client
            self._consul_client = consul.Consul(
                host=consul_host,
                port=consul_port,
                scheme=consul_scheme,
                token=consul_token,
            )
            self._owns_client = True

        # Lazy thread pool initialization
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = asyncio.Lock()
        self._max_workers = max_workers

        logger.info(
            "ConsulServiceDiscoveryHandler initialized",
            extra={
                "consul_host": consul_host,
                "consul_port": consul_port,
                "max_workers": max_workers,
            },
        )

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            "consul" identifier string.
        """
        return "consul"

    async def _ensure_executor(self) -> ThreadPoolExecutor:
        """Ensure thread pool executor is initialized.

        Uses double-checked locking for thread-safe lazy initialization.
        The executor is created on first use rather than at handler
        construction time to avoid allocating resources for handlers
        that may never be used.

        Returns:
            The ThreadPoolExecutor instance.
        """
        if self._executor is not None:
            return self._executor

        async with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
                logger.debug(
                    "Thread pool executor initialized",
                    extra={"max_workers": self._max_workers},
                )
            return self._executor

    def _check_not_shutdown(
        self,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Check that handler has not been shut down.

        Args:
            operation: Name of the operation being attempted.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConnectionError: If handler has been shut down.
        """
        if self._consul_client is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Handler has been shut down, cannot perform operation",
                context=context,
            )

    async def register_service(
        self,
        service_info: ModelServiceInfo,
        correlation_id: UUID | None = None,
    ) -> ModelHandlerRegistrationResult:
        """Register a service with Consul.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelHandlerRegistrationResult with registration outcome.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Guard against use-after-shutdown
        self._check_not_shutdown("register_service", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="register_service",
                correlation_id=correlation_id,
            )

        try:
            # Build health check config if URL provided
            check_config: dict[str, str] | None = None
            if service_info.health_check_url:
                check_config = {
                    "http": service_info.health_check_url,
                    "interval": "10s",
                    "timeout": "5s",
                }

            # Execute registration in thread pool
            # Client is typed as consul.Consul (duck-typed for injected clients)
            client = self._consul_client
            executor = await self._ensure_executor()
            loop = asyncio.get_running_loop()
            # Convert UUID to string for Consul API compatibility
            service_id_str = str(service_info.service_id)
            await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: client.agent.service.register(  # type: ignore[union-attr]
                        name=service_info.service_name,
                        service_id=service_id_str,
                        address=service_info.address,
                        port=service_info.port,
                        tags=list(service_info.tags),
                        meta=service_info.metadata,
                        check=check_config,
                    ),
                ),
                timeout=self._timeout_seconds,
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Service registered with Consul",
                extra={
                    "service_id": service_id_str,
                    "service_name": service_info.service_name,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerRegistrationResult(
                success=True,
                service_id=service_info.service_id,
                operation=EnumServiceDiscoveryOperation.REGISTER,
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="register_service",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register_service",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Consul registration timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except consul.ConsulException as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="register_service",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register_service",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Consul registration failed",
                context=context,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="register_service",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register_service",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Consul registration failed: {type(e).__name__}",
                context=context,
            ) from e

    async def deregister_service(
        self,
        service_id: UUID,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from Consul.

        Args:
            service_id: UUID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        # Convert UUID to string for Consul API
        service_id_str = str(service_id)

        # Guard against use-after-shutdown
        self._check_not_shutdown("deregister_service", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="deregister_service",
                correlation_id=correlation_id,
            )

        try:
            # Client is typed as consul.Consul (duck-typed for injected clients)
            client = self._consul_client
            executor = await self._ensure_executor()
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: client.agent.service.deregister(service_id_str),  # type: ignore[union-attr]
                ),
                timeout=self._timeout_seconds,
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "Service deregistered from Consul",
                extra={
                    "service_id": service_id_str,
                    "correlation_id": str(correlation_id),
                },
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="deregister_service",
                    correlation_id=correlation_id,
                )
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="deregister_service",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Consul deregistration timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except consul.ConsulException as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="deregister_service",
                    correlation_id=correlation_id,
                )
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="deregister_service",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Consul deregistration failed",
                context=context,
            ) from e

    async def discover_services(
        self,
        query: ModelDiscoveryQuery,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the query criteria.

        Args:
            query: Query parameters including service_name, tags,
                and health_filter for filtering services.
            correlation_id: Optional correlation ID for tracing.
                If not provided, uses query.correlation_id, then generates new.

        Returns:
            ModelDiscoveryResult with list of matching services.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Note:
            Service IDs from Consul are converted to UUIDs. If the Consul service
            ID is not a valid UUID string, a deterministic UUID5 is generated
            using the ONEX service namespace. This ensures consistent IDs across
            discovery calls while maintaining UUID format compatibility.
        """
        # Standardized correlation ID handling: explicit > query > generate
        correlation_id = correlation_id or query.correlation_id or uuid4()
        service_name = query.service_name or ""
        tags = query.tags
        start_time = time.monotonic()

        # Guard against use-after-shutdown
        self._check_not_shutdown("discover_services", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="discover_services",
                correlation_id=correlation_id,
            )

        try:
            # Query Consul catalog
            # Client is typed as consul.Consul (duck-typed for injected clients)
            client = self._consul_client
            executor = await self._ensure_executor()
            loop = asyncio.get_running_loop()

            def _query_services() -> tuple[int, list[dict[str, object]]]:
                # Use health endpoint for service discovery (includes health status)
                tag = tags[0] if tags else None
                result: tuple[int, list[dict[str, object]]] = client.health.service(  # type: ignore[union-attr]
                    service_name,
                    tag=tag,
                    passing=True,  # Only healthy services
                )
                return result

            _, services = await asyncio.wait_for(
                loop.run_in_executor(executor, _query_services),
                timeout=self._timeout_seconds,
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Convert to ModelServiceInfo list
            service_infos: list[ModelServiceInfo] = []
            for svc in services:
                # Cast nested dicts for type safety
                svc_data = cast(dict[str, object], svc.get("Service", {}))
                node_data = cast(dict[str, object], svc.get("Node", {}))
                svc_id_str = str(svc_data.get("ID", ""))
                svc_name = svc_data.get("Service", "")
                address = svc_data.get("Address", "") or node_data.get("Address", "")
                port_raw = svc_data.get("Port", 0)
                port = int(port_raw) if isinstance(port_raw, (int, float, str)) else 0
                svc_tags = cast(list[str], svc_data.get("Tags", []))
                svc_meta = cast(dict[str, str], svc_data.get("Meta", {}))

                if svc_id_str and svc_name and address and port:
                    # Convert Consul service ID string to UUID
                    # Try to parse as UUID, otherwise generate deterministic UUID from string
                    try:
                        svc_id = UUID(svc_id_str)
                    except ValueError:
                        # Non-UUID service ID from Consul - generate deterministic UUID5
                        # using ONEX-specific namespace to avoid collisions
                        logger.warning(
                            "Non-UUID service ID from Consul, using deterministic UUID5 conversion",
                            extra={
                                "original_id": svc_id_str,
                                "correlation_id": str(correlation_id),
                            },
                        )
                        svc_id = uuid5(NAMESPACE_ONEX_SERVICE, svc_id_str)

                    service_infos.append(
                        ModelServiceInfo(
                            service_id=svc_id,
                            service_name=str(svc_name),
                            address=str(address),
                            port=port,
                            tags=tuple(svc_tags or []),
                            health_status=EnumHealthStatus.HEALTHY,
                            metadata=svc_meta or {},
                            registered_at=datetime.now(UTC),
                            correlation_id=correlation_id,
                        )
                    )

            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Service discovery completed",
                extra={
                    "service_name": service_name,
                    "found_count": len(service_infos),
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelDiscoveryResult(
                success=True,
                services=tuple(service_infos),
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="discover_services",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="discover_services",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Consul discovery timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except consul.ConsulException as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="discover_services",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="discover_services",
                target_name="consul.discovery",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Consul discovery failed",
                context=context,
            ) from e

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelServiceDiscoveryHealthCheckResult:
        """Perform a health check on the Consul connection.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelServiceDiscoveryHealthCheckResult with health status information.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Guard against use-after-shutdown
        self._check_not_shutdown("health_check", correlation_id)

        try:
            # Client is typed as consul.Consul (duck-typed for injected clients)
            client = self._consul_client
            executor = await self._ensure_executor()
            loop = asyncio.get_running_loop()
            leader = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: client.status.leader(),  # type: ignore[union-attr]
                ),
                timeout=5.0,  # Short timeout for health check
            )

            duration_ms = (time.monotonic() - start_time) * 1000

            return ModelServiceDiscoveryHealthCheckResult(
                healthy=True,
                backend_type=self.handler_type,
                latency_ms=duration_ms,
                reason="ok",
                details=ModelServiceDiscoveryHealthCheckDetails(
                    agent_address=f"{self._consul_host}:{self._consul_port}",
                    leader=str(leader) if leader else None,
                ),
                correlation_id=correlation_id,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ModelServiceDiscoveryHealthCheckResult(
                healthy=False,
                backend_type=self.handler_type,
                latency_ms=duration_ms,
                reason=f"Health check failed: {type(e).__name__}",
                error_type=type(e).__name__,
                details=ModelServiceDiscoveryHealthCheckDetails(
                    agent_address=f"{self._consul_host}:{self._consul_port}",
                ),
                correlation_id=correlation_id,
            )

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources.

        Cleans up:
        - Thread pool executor (always)
        - Consul client (only if handler owns it)
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Clean up owned Consul client
        if self._owns_client:
            self._consul_client = None
            self._owns_client = False

        logger.info("ConsulServiceDiscoveryHandler shutdown complete")


__all__ = ["ConsulServiceDiscoveryHandler"]
