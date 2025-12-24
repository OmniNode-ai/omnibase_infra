# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test doubles for NodeRegistryEffect integration tests.

This module provides controllable test doubles that implement the protocol
interfaces required by NodeRegistryEffect. Unlike mocks, these test doubles:

1. Implement the actual protocol interface (type-safe)
2. Maintain internal state for verification
3. Can be configured to succeed or fail
4. Support async operation patterns
5. Track call history for assertions

Test Doubles:
    - StubConsulClient: Implements ProtocolConsulClient
    - StubPostgresAdapter: Implements ProtocolPostgresAdapter

Design Principles:
    - No mocking: Use real implementations with controllable behavior
    - State tracking: Track registrations for verification
    - Async-native: Full async support for realistic testing
    - Configurable failures: Set up failure scenarios programmatically
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from omnibase_infra.nodes.effects.models import ModelBackendResult


@dataclass
class ConsulRegistration:
    """Record of a Consul service registration.

    Attributes:
        service_id: Unique identifier for the service instance.
        service_name: Name of the service for discovery.
        tags: List of tags for filtering.
        health_check: Optional health check configuration.
    """

    service_id: str
    service_name: str
    tags: list[str]
    health_check: dict[str, str] | None = None


@dataclass
class PostgresRegistration:
    """Record of a PostgreSQL registration upsert.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: Type of ONEX node.
        node_version: Semantic version of the node.
        endpoints: Dict of endpoint type to URL.
        metadata: Additional metadata.
    """

    node_id: UUID
    node_type: str
    node_version: str
    endpoints: dict[str, str]
    metadata: dict[str, str]


class StubConsulClient:
    """Stub implementing ProtocolConsulClient for integration testing.

    A controllable Consul client that tracks registrations and can be
    configured to succeed or fail. Implements the full ProtocolConsulClient
    protocol for type-safe testing.

    Configuration:
        - Set should_fail=True to simulate Consul failures
        - Set failure_error to customize the error message
        - Set delay_seconds to simulate network latency

    State Tracking:
        - registrations: List of all successful registrations
        - call_count: Number of times register_service was called

    Example:
        >>> client = StubConsulClient()
        >>> result = await client.register_service("id-1", "svc", ["tag"])
        >>> assert result.success is True
        >>> assert len(client.registrations) == 1

        >>> # Simulate failure
        >>> client.should_fail = True
        >>> result = await client.register_service("id-2", "svc", ["tag"])
        >>> assert result.success is False
    """

    def __init__(
        self,
        *,
        should_fail: bool = False,
        failure_error: str = "Consul registration failed",
        delay_seconds: float = 0.0,
    ) -> None:
        """Initialize the test double.

        Args:
            should_fail: If True, register_service will return failure.
            failure_error: Error message to return on failure.
            delay_seconds: Simulated network delay in seconds.
        """
        self.should_fail = should_fail
        self.failure_error = failure_error
        self.delay_seconds = delay_seconds
        self.registrations: list[ConsulRegistration] = []
        self.call_count = 0
        self._raise_exception: Exception | None = None

    def set_exception(self, exception: Exception) -> None:
        """Configure the client to raise an exception.

        Args:
            exception: Exception to raise on next call.
        """
        self._raise_exception = exception

    def clear_exception(self) -> None:
        """Clear any configured exception."""
        self._raise_exception = None

    def reset(self) -> None:
        """Reset all state and configuration."""
        self.should_fail = False
        self.failure_error = "Consul registration failed"
        self.delay_seconds = 0.0
        self.registrations.clear()
        self.call_count = 0
        self._raise_exception = None

    async def register_service(
        self,
        service_id: str,
        service_name: str,
        tags: list[str],
        health_check: dict[str, str] | None = None,
    ) -> ModelBackendResult:
        """Register a service in Consul.

        Implements ProtocolConsulClient.register_service with controllable
        behavior for testing.

        Args:
            service_id: Unique identifier for the service instance.
            service_name: Name of the service for discovery.
            tags: List of tags for filtering.
            health_check: Optional health check configuration.

        Returns:
            ModelBackendResult with success status and optional error.

        Raises:
            Exception: If set_exception was called with an exception.
        """
        self.call_count += 1

        # Simulate network delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Raise configured exception
        if self._raise_exception is not None:
            raise self._raise_exception

        # Simulate failure
        if self.should_fail:
            return ModelBackendResult(success=False, error=self.failure_error)

        # Record successful registration
        registration = ConsulRegistration(
            service_id=service_id,
            service_name=service_name,
            tags=tags,
            health_check=health_check,
        )
        self.registrations.append(registration)

        return ModelBackendResult(success=True)


class StubPostgresAdapter:
    """Stub implementing ProtocolPostgresAdapter for integration testing.

    A controllable PostgreSQL adapter that tracks upserts and can be
    configured to succeed or fail. Implements the full ProtocolPostgresAdapter
    protocol for type-safe testing.

    Configuration:
        - Set should_fail=True to simulate PostgreSQL failures
        - Set failure_error to customize the error message
        - Set delay_seconds to simulate network latency

    State Tracking:
        - registrations: List of all successful upserts
        - call_count: Number of times upsert was called

    Example:
        >>> adapter = StubPostgresAdapter()
        >>> result = await adapter.upsert(uuid4(), "effect", "1.0.0", {}, {})
        >>> assert result.success is True
        >>> assert len(adapter.registrations) == 1

        >>> # Simulate failure
        >>> adapter.should_fail = True
        >>> result = await adapter.upsert(uuid4(), "effect", "1.0.0", {}, {})
        >>> assert result.success is False
    """

    def __init__(
        self,
        *,
        should_fail: bool = False,
        failure_error: str = "PostgreSQL upsert failed",
        delay_seconds: float = 0.0,
    ) -> None:
        """Initialize the test double.

        Args:
            should_fail: If True, upsert will return failure.
            failure_error: Error message to return on failure.
            delay_seconds: Simulated network delay in seconds.
        """
        self.should_fail = should_fail
        self.failure_error = failure_error
        self.delay_seconds = delay_seconds
        self.registrations: list[PostgresRegistration] = []
        self.call_count = 0
        self._raise_exception: Exception | None = None

    def set_exception(self, exception: Exception) -> None:
        """Configure the adapter to raise an exception.

        Args:
            exception: Exception to raise on next call.
        """
        self._raise_exception = exception

    def clear_exception(self) -> None:
        """Clear any configured exception."""
        self._raise_exception = None

    def reset(self) -> None:
        """Reset all state and configuration."""
        self.should_fail = False
        self.failure_error = "PostgreSQL upsert failed"
        self.delay_seconds = 0.0
        self.registrations.clear()
        self.call_count = 0
        self._raise_exception = None

    async def upsert(
        self,
        node_id: UUID,
        node_type: str,
        node_version: str,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> ModelBackendResult:
        """Upsert a node registration record.

        Implements ProtocolPostgresAdapter.upsert with controllable
        behavior for testing.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of ONEX node.
            node_version: Semantic version of the node.
            endpoints: Dict of endpoint type to URL.
            metadata: Additional metadata.

        Returns:
            ModelBackendResult with success status and optional error.

        Raises:
            Exception: If set_exception was called with an exception.
        """
        self.call_count += 1

        # Simulate network delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Raise configured exception
        if self._raise_exception is not None:
            raise self._raise_exception

        # Simulate failure
        if self.should_fail:
            return ModelBackendResult(success=False, error=self.failure_error)

        # Record successful upsert
        registration = PostgresRegistration(
            node_id=node_id,
            node_type=node_type,
            node_version=node_version,
            endpoints=endpoints,
            metadata=metadata,
        )
        self.registrations.append(registration)

        return ModelBackendResult(success=True)


__all__ = [
    "ConsulRegistration",
    "PostgresRegistration",
    "StubConsulClient",
    "StubPostgresAdapter",
]
