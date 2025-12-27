# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for circuit breaker awareness.

This module defines the protocol interface for components that support
circuit breaker functionality via MixinAsyncCircuitBreaker.

The protocol enables structural typing (duck typing) for circuit breaker
methods, allowing mixins like MixinRetryExecution to interact with circuit
breaker functionality without requiring inheritance relationships.

Usage:
    ```python
    from omnibase_infra.mixins import (
        MixinAsyncCircuitBreaker,
        MixinRetryExecution,
        ProtocolCircuitBreakerAware,
    )

    class MyHandler(MixinAsyncCircuitBreaker, MixinRetryExecution):
        # Automatically satisfies ProtocolCircuitBreakerAware via MixinAsyncCircuitBreaker
        ...
    ```

Design Rationale:
    This protocol exists to eliminate type: ignore comments in MixinRetryExecution
    when accessing circuit breaker methods. By defining the protocol, type checkers
    can verify that the required methods exist without runtime inheritance checks.

See Also:
    - MixinAsyncCircuitBreaker: Implementation of circuit breaker functionality
    - MixinRetryExecution: Consumer of this protocol for retry logic
    - docs/patterns/circuit_breaker_implementation.md: Full implementation details

.. versionadded:: 0.4.1
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolCircuitBreakerAware(Protocol):
    """Protocol for components with circuit breaker capability.

    This protocol defines the interface for circuit breaker functionality
    that MixinAsyncCircuitBreaker provides. Components that want to interact
    with circuit breakers can use this protocol for type hints.

    Attributes:
        _circuit_breaker_lock: Async lock for coroutine-safe circuit breaker access.
            All circuit breaker operations MUST be performed while holding this lock.

    Methods:
        _check_circuit_breaker: Verify circuit allows operation (raises if open).
        _record_circuit_failure: Record a failure and potentially open circuit.
        _reset_circuit_breaker: Reset circuit to closed state on success.

    Concurrency Safety:
        All circuit breaker methods MUST be called while holding ``_circuit_breaker_lock``.
        This is documented in each method's docstring using:
        "REQUIRES: self._circuit_breaker_lock must be held by caller."

        Example:
            ```python
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker("operation", correlation_id)
            ```

    Note:
        Method bodies use ``...`` (Ellipsis) per PEP 544 Protocol conventions.
    """

    _circuit_breaker_lock: asyncio.Lock
    """Async lock for coroutine-safe circuit breaker access."""

    async def _check_circuit_breaker(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Check if circuit breaker allows operation.

        Verifies circuit breaker state and raises InfraUnavailableError if
        circuit is open. Automatically transitions from OPEN to HALF_OPEN
        if reset timeout has elapsed.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.

        Args:
            operation: Operation name for error context (e.g., "publish", "register").
            correlation_id: Optional correlation ID for distributed tracing.

        Raises:
            InfraUnavailableError: If circuit breaker is open and reset timeout
                has not elapsed.
        """
        ...

    async def _record_circuit_failure(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Record a circuit breaker failure and potentially open the circuit.

        Increments the failure counter and opens the circuit if the threshold
        is reached. When the circuit opens, it sets the reset timestamp for
        automatic recovery.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.

        Args:
            operation: Operation name for logging context.
            correlation_id: Optional correlation ID for distributed tracing.
        """
        ...

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state.

        Resets the failure counter and closes the circuit, allowing all
        requests to proceed normally. Typically called after a successful
        operation.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.
        """
        ...


__all__: list[str] = ["ProtocolCircuitBreakerAware"]
