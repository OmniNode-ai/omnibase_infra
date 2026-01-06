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
from datetime import UTC, datetime, timezone
from uuid import UUID, uuid4

from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelRegistrationResult,
    ModelServiceInfo,
)

logger = logging.getLogger(__name__)


class MockServiceDiscoveryHandler:
    """In-memory mock for testing service discovery.

    Provides a simple in-memory implementation of the service discovery
    protocol for unit and integration testing.

    Thread Safety:
        This handler is coroutine-safe. All operations on the internal
        service store are protected by asyncio.Lock.

    Attributes:
        handler_type: Returns "mock" identifier.

    Example:
        >>> handler = MockServiceDiscoveryHandler()
        >>> result = await handler.register_service(service_info)
        >>> assert result.success
    """

    def __init__(self) -> None:
        """Initialize MockServiceDiscoveryHandler with empty service store."""
        self._services: dict[UUID, ModelServiceInfo] = {}
        self._lock = asyncio.Lock()

        logger.debug("MockServiceDiscoveryHandler initialized")

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
    ) -> ModelRegistrationResult:
        """Register a service in the mock store.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelRegistrationResult with registration outcome.
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

        return ModelRegistrationResult(
            success=True,
            service_id=service_info.service_id,
            duration_ms=duration_ms,
            backend_type=self.handler_type,
            correlation_id=correlation_id,
        )

    async def deregister_service(
        self,
        service_id: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from the mock store.

        Args:
            service_id: ID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.
        """
        correlation_id = correlation_id or uuid4()

        # Convert string service_id to UUID for dict lookup
        try:
            uuid_service_id = UUID(service_id)
        except ValueError:
            # Invalid UUID format, service won't be found
            logger.debug(
                "Invalid service_id format for deregistration",
                extra={
                    "service_id": service_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        async with self._lock:
            if uuid_service_id in self._services:
                del self._services[uuid_service_id]

        logger.debug(
            "Mock service deregistered",
            extra={
                "service_id": service_id,
                "correlation_id": str(correlation_id),
            },
        )

    async def discover_services(
        self,
        service_name: str,
        tags: tuple[str, ...] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the given criteria in the mock store.

        Args:
            service_name: Name of the service to discover.
            tags: Optional tags to filter services.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelDiscoveryResult with list of matching services.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            matching_services: list[ModelServiceInfo] = []

            for service in self._services.values():
                # Match by service name
                if service.service_name != service_name:
                    continue

                # Match by tags if provided
                if tags:
                    service_tags = set(service.tags)
                    if not all(tag in service_tags for tag in tags):
                        continue

                matching_services.append(service)

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock service discovery completed",
            extra={
                "service_name": service_name,
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
    ) -> dict[str, object]:
        """Perform a health check on the mock handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Dict with health status information.
        """
        correlation_id = correlation_id or uuid4()

        async with self._lock:
            service_count = len(self._services)

        return {
            "healthy": True,
            "backend_type": self.handler_type,
            "service_count": service_count,
            "correlation_id": str(correlation_id),
        }

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


__all__ = ["MockServiceDiscoveryHandler"]
