# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Service Discovery Handler.

This module provides a Consul-backed implementation of the service discovery
handler protocol, wrapping existing Consul functionality with circuit breaker
resilience.

Thread Pool Management:
    - All synchronous consul operations run in a dedicated thread pool
    - Configurable max workers (default: 10)
    - Thread pool gracefully shutdown on handler shutdown

Circuit Breaker:
    - Uses MixinAsyncCircuitBreaker for consistent resilience
    - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
    - Configurable failure threshold and reset timeout
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import consul

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT = 30.0
DEFAULT_MAX_WORKERS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0


class ConsulServiceDiscoveryHandler(MixinAsyncCircuitBreaker):
    """Consul implementation of ProtocolServiceDiscoveryHandler.

    Wraps existing Consul client functionality with circuit breaker resilience
    and proper error handling.

    Thread Safety:
        This handler is coroutine-safe. All Consul operations are executed
        in a dedicated thread pool, and circuit breaker state is protected
        by asyncio.Lock.

    Attributes:
        handler_type: Returns "consul" identifier.

    Example:
        >>> handler = ConsulServiceDiscoveryHandler(
        ...     consul_client=consul_client,
        ...     circuit_breaker_config={"threshold": 5, "reset_timeout": 30.0},
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
        circuit_breaker_config: dict[str, object] | None = None,
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
            circuit_breaker_config: Optional circuit breaker configuration with:
                - threshold: Max failures before opening (default: 5)
                - reset_timeout: Seconds before reset (default: 30.0)
                - service_name: Service identifier (default: "consul.discovery")
            max_workers: Thread pool max workers (default: 10).
            timeout_seconds: Operation timeout in seconds (default: 30.0).
        """
        config = circuit_breaker_config or {}
        _threshold_raw = config.get("threshold", DEFAULT_CIRCUIT_BREAKER_THRESHOLD)
        threshold = (
            int(_threshold_raw)
            if isinstance(_threshold_raw, (int, float, str))
            else DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )
        _reset_timeout_raw = config.get(
            "reset_timeout", DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT
        )
        reset_timeout = (
            float(_reset_timeout_raw)
            if isinstance(_reset_timeout_raw, (int, float, str))
            else DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT
        )
        _service_name_raw = config.get("service_name", "consul.discovery")
        service_name = (
            str(_service_name_raw)
            if _service_name_raw is not None
            else "consul.discovery"
        )

        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.CONSUL,
        )

        # Store configuration
        self._consul_host = consul_host
        self._consul_port = consul_port
        self._consul_scheme = consul_scheme
        self._consul_token = consul_token
        self._timeout_seconds = timeout_seconds

        # Initialize Consul client
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

        # Initialize thread pool
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max_workers
        )
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

    async def register_service(
        self,
        service_info: ModelServiceInfo,
        correlation_id: UUID | None = None,
    ) -> ModelRegistrationResult:
        """Register a service with Consul.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelRegistrationResult with registration outcome.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

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
            # Cast to Any for duck-typed consul client access
            client: Any = self._consul_client
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: client.agent.service.register(
                        name=service_info.service_name,
                        service_id=service_info.service_id,
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
                    "service_id": service_info.service_id,
                    "service_name": service_info.service_name,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelRegistrationResult(
                success=True,
                service_id=service_info.service_id,
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
        service_id: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from Consul.

        Args:
            service_id: ID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="deregister_service",
                correlation_id=correlation_id,
            )

        try:
            # Cast to Any for duck-typed consul client access
            client: Any = self._consul_client
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: client.agent.service.deregister(service_id),
                ),
                timeout=self._timeout_seconds,
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "Service deregistered from Consul",
                extra={
                    "service_id": service_id,
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
        service_name: str,
        tags: tuple[str, ...] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the given criteria.

        Args:
            service_name: Name of the service to discover.
            tags: Optional tags to filter services.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelDiscoveryResult with list of matching services.

        Raises:
            InfraConnectionError: If connection to Consul fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="discover_services",
                correlation_id=correlation_id,
            )

        try:
            # Query Consul catalog
            # Cast to Any for duck-typed consul client access
            client: Any = self._consul_client
            loop = asyncio.get_running_loop()

            def _query_services() -> tuple[int, list[dict[str, object]]]:
                # Use health endpoint for service discovery (includes health status)
                tag = tags[0] if tags else None
                result: tuple[int, list[dict[str, object]]] = client.health.service(
                    service_name,
                    tag=tag,
                    passing=True,  # Only healthy services
                )
                return result

            _, services = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _query_services),
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
                        # Generate deterministic UUID from service ID string using uuid5
                        svc_id = uuid5(NAMESPACE_DNS, svc_id_str)

                    service_infos.append(
                        ModelServiceInfo(
                            service_id=svc_id,
                            service_name=str(svc_name),
                            address=str(address),
                            port=port,
                            tags=tuple(svc_tags or []),
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
    ) -> dict[str, object]:
        """Perform a health check on the Consul connection.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Dict with health status information.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        try:
            # Cast to Any for duck-typed consul client access
            client: Any = self._consul_client
            loop = asyncio.get_running_loop()
            status = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: client.status.leader(),
                ),
                timeout=5.0,  # Short timeout for health check
            )

            duration_ms = (time.monotonic() - start_time) * 1000

            return {
                "healthy": True,
                "backend_type": self.handler_type,
                "consul_host": self._consul_host,
                "consul_port": self._consul_port,
                "leader": status,
                "circuit_breaker_open": self._circuit_breaker_open,
                "duration_ms": duration_ms,
                "correlation_id": str(correlation_id),
            }

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return {
                "healthy": False,
                "backend_type": self.handler_type,
                "consul_host": self._consul_host,
                "consul_port": self._consul_port,
                "error": f"Health check failed: {type(e).__name__}",
                "circuit_breaker_open": self._circuit_breaker_open,
                "duration_ms": duration_ms,
                "correlation_id": str(correlation_id),
            }

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("ConsulServiceDiscoveryHandler shutdown complete")


__all__ = ["ConsulServiceDiscoveryHandler"]
