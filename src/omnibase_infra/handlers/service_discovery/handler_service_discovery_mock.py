# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock Service Discovery Handler.

This module provides an in-memory mock implementation of the service discovery
handler protocol for testing purposes.

Thread Safety:
    This handler uses asyncio.Lock for coroutine-safe access to the
    in-memory service store.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelHandlerRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.nodes.node_service_discovery_effect.models import (
    ModelDiscoveryQuery,
    ModelServiceDiscoveryHealthCheckDetails,
    ModelServiceDiscoveryHealthCheckResult,
)
from omnibase_infra.nodes.node_service_discovery_effect.models.enum_service_discovery_operation import (
    EnumServiceDiscoveryOperation,
)

logger = logging.getLogger(__name__)


class HandlerServiceDiscoveryMock:
    """In-memory mock for testing service discovery.

    Provides a simple in-memory implementation of the service discovery
    protocol for unit and integration testing.

    Thread Safety:
        This handler is coroutine-safe. All operations on the internal
        service store are protected by asyncio.Lock.

    Attributes:
        handler_type: Returns "mock" identifier.

    Example:
        >>> handler = HandlerServiceDiscoveryMock()
        >>> result = await handler.register_service(service_info)
        >>> assert result.success
    """

    def __init__(self) -> None:
        """Initialize HandlerServiceDiscoveryMock with empty service store."""
        self._services: dict[UUID, ModelServiceInfo] = {}
        self._lock = asyncio.Lock()

        logger.debug("HandlerServiceDiscoveryMock initialized")

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            "mock" identifier string.
        """
        return "mock"

    async def register_service(
        self,
        service_info: ModelServiceInfo,
        correlation_id: UUID | None = None,
    ) -> ModelHandlerRegistrationResult:
        """Register a service in the mock store.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelHandlerRegistrationResult with registration outcome.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            # Add timestamp if not present
            if service_info.registered_at is None:
                service_info = ModelServiceInfo(
                    service_id=service_info.service_id,
                    service_name=service_info.service_name,
                    address=service_info.address,
                    port=service_info.port,
                    tags=service_info.tags,
                    health_status=service_info.health_status,
                    metadata=service_info.metadata,
                    health_check_url=service_info.health_check_url,
                    registered_at=datetime.now(UTC),
                    correlation_id=correlation_id,
                )

            self._services[service_info.service_id] = service_info

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock service registered",
            extra={
                "service_id": service_info.service_id,
                "service_name": service_info.service_name,
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

    async def deregister_service(
        self,
        service_id: UUID,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from the mock store.

        Args:
            service_id: UUID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.
        """
        correlation_id = correlation_id or uuid4()

        async with self._lock:
            self._services.pop(service_id, None)

        logger.debug(
            "Mock service deregistered",
            extra={
                "service_id": str(service_id),
                "correlation_id": str(correlation_id),
            },
        )

    async def discover_services(
        self,
        query: ModelDiscoveryQuery,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the query criteria in the mock store.

        Args:
            query: Query parameters including service_name, tags,
                and health_filter for filtering services.
            correlation_id: Optional correlation ID for tracing.
                If not provided, uses query.correlation_id.

        Returns:
            ModelDiscoveryResult with list of matching services.
        """
        correlation_id = correlation_id or query.correlation_id
        start_time = time.monotonic()

        async with self._lock:
            matching_services: list[ModelServiceInfo] = []

            for service in self._services.values():
                # Match by service name if provided
                if query.service_name and service.service_name != query.service_name:
                    continue

                # Match by tags if provided
                if query.tags:
                    service_tags = set(service.tags)
                    if not all(tag in service_tags for tag in query.tags):
                        continue

                matching_services.append(service)

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock service discovery completed",
            extra={
                "service_name": query.service_name,
                "found_count": len(matching_services),
                "correlation_id": str(correlation_id),
            },
        )

        return ModelDiscoveryResult(
            success=True,
            services=tuple(matching_services),
            duration_ms=duration_ms,
            backend_type=self.handler_type,
            correlation_id=correlation_id,
        )

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelServiceDiscoveryHealthCheckResult:
        """Perform a health check on the mock handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelServiceDiscoveryHealthCheckResult with health status information.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            service_count = len(self._services)

        latency_ms = (time.monotonic() - start_time) * 1000

        return ModelServiceDiscoveryHealthCheckResult(
            healthy=True,
            backend_type=self.handler_type,
            latency_ms=latency_ms,
            reason="ok",
            details=ModelServiceDiscoveryHealthCheckDetails(
                server_version="mock-1.0.0",
                service_count=service_count,
            ),
            correlation_id=correlation_id,
        )

    async def clear(self) -> None:
        """Clear all services from the mock store.

        Utility method for test cleanup.
        """
        async with self._lock:
            self._services.clear()

        logger.debug("Mock service store cleared")

    async def get_service_count(self) -> int:
        """Get the number of registered services.

        Returns:
            Number of services in the store.
        """
        async with self._lock:
            return len(self._services)


__all__ = ["HandlerServiceDiscoveryMock"]
