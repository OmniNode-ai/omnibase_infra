# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Coroutine-safe async circuit breaker mixin for infrastructure components.

This module provides a reusable circuit breaker implementation for infrastructure
components such as event buses, service adapters, HTTP clients, and database
connections. It implements the standard 3-state circuit breaker pattern with
coroutine-safe async operations.

Circuit Breaker States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Circuit tripped, requests blocked
    - HALF_OPEN: Testing recovery, limited requests allowed

Features:
    - Coroutine-safe state management using asyncio.Lock
    - Automatic state transitions based on failure thresholds
    - Time-based auto-reset with configurable timeout
    - Infrastructure error integration (InfraUnavailableError)
    - Correlation ID propagation for distributed tracing
    - Configurable failure thresholds and reset timeouts

Usage:
    ```python
    from omnibase_infra.mixins import MixinAsyncCircuitBreaker
    from omnibase_infra.enums import EnumInfraTransportType

    class EventBusKafka(MixinAsyncCircuitBreaker):
        def __init__(self, config):
            # Initialize circuit breaker with configuration
            self._init_circuit_breaker(
                threshold=config.circuit_breaker_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout,
                service_name=f"kafka.{config.environment}",
                transport_type=EnumInfraTransportType.KAFKA,
            )

        async def publish(
            self, topic: str, key: str, value: bytes, correlation_id: UUID | None = None
        ) -> None:
            # Check circuit before operation
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="publish",
                    correlation_id=correlation_id,
                )

            try:
                # Perform operation
                await self._kafka_producer.send(topic, key, value)

                # Record success
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

            except Exception:
                # Record failure
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="publish",
                        correlation_id=correlation_id,
                    )
                raise
    ```

Concurrency Safety:
    All circuit breaker methods require the caller to hold `_circuit_breaker_lock`
    before invocation. This is documented in each method's docstring using:
    "REQUIRES: self._circuit_breaker_lock must be held by caller."

    Note: This mixin uses asyncio.Lock which provides coroutine-safe access,
    not thread-safe access. For true multi-threaded usage, additional
    synchronization (e.g., threading.Lock) would be required.

    Example:
        ```python
        # Correct - lock held by caller
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("operation")

        # Incorrect - race condition between coroutines!
        await self._check_circuit_breaker("operation")
        ```

Integration Requirements:
    Classes using this mixin must:
    1. Call `_init_circuit_breaker()` during initialization
    2. Use `async with self._circuit_breaker_lock` before calling circuit methods
    3. Pass appropriate operation name and correlation_id to circuit methods
    4. Handle InfraUnavailableError when circuit is open

See Also:
    - docs/analysis/CIRCUIT_BREAKER_COMPARISON.md for design rationale
    - src/omnibase_infra/event_bus/event_bus_kafka.py for reference implementation
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID, uuid4

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumCircuitState, EnumInfraTransportType
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

logger = logging.getLogger(__name__)


class MixinAsyncCircuitBreaker:
    """Coroutine-safe async circuit breaker mixin for infrastructure components.

    Provides circuit breaker pattern implementation with:
    - Coroutine-safe state management using asyncio.Lock
    - Configurable failure thresholds and reset timeouts
    - Automatic state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
    - Infrastructure error integration (InfraUnavailableError)
    - Correlation ID propagation for distributed tracing

    State Variables:
        _circuit_breaker_failures: Failure counter (incremented on each failure)
        _circuit_breaker_open: Circuit open/closed state (True = open)
        _circuit_breaker_open_until: Timestamp for automatic reset
        _circuit_breaker_lock: asyncio.Lock for coroutine-safe access

    Configuration Variables (public attributes set by _init_circuit_breaker):
        circuit_breaker_threshold: Max failures before opening (default: 5)
        circuit_breaker_reset_timeout: Auto-reset timeout in seconds (default: 60.0)
        service_name: Service identifier for error context

    Concurrency Safety:
        All circuit breaker methods MUST be called while holding
        `_circuit_breaker_lock`. Callers are responsible for lock acquisition.
        Note: asyncio.Lock protects against concurrent coroutine access,
        not OS thread access. For multi-threaded scenarios, use threading.Lock.

        ```python
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("operation")
        ```

    Example:
        ```python
        class ConsulAdapter(MixinAsyncCircuitBreaker):
            def __init__(self, config):
                self._init_circuit_breaker(
                    threshold=5,
                    reset_timeout=30.0,
                    service_name="consul.dev",
                    transport_type=EnumInfraTransportType.CONSUL,
                )

            async def register_service(
                self, service: str, correlation_id: UUID | None = None
            ) -> None:
                # Check circuit (coroutine-safe)
                async with self._circuit_breaker_lock:
                    await self._check_circuit_breaker(
                        operation="register_service",
                        correlation_id=correlation_id,
                    )

                try:
                    # Perform operation
                    await self._consul_client.register(service)

                    # Record success (coroutine-safe)
                    async with self._circuit_breaker_lock:
                        await self._reset_circuit_breaker()

                except Exception:
                    # Record failure (coroutine-safe)
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure(
                            operation="register_service",
                            correlation_id=correlation_id,
                        )
                    raise
        ```
    """

    def _init_circuit_breaker(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "unknown",
        transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
    ) -> None:
        """Initialize circuit breaker state and configuration.

        Must be called during class initialization before any circuit breaker
        operations are performed.

        Args:
            threshold: Maximum failures before opening circuit (default: 5)
            reset_timeout: Seconds before automatic reset (default: 60.0)
            service_name: Service identifier for error context (e.g., "kafka.dev")
            transport_type: Transport type for error context (default: HTTP)

        Raises:
            ValueError: If threshold < 1 or reset_timeout < 0

        Example:
            ```python
            class MyService(MixinAsyncCircuitBreaker):
                def __init__(self, config):
                    self._init_circuit_breaker(
                        threshold=config.circuit_breaker_threshold,
                        reset_timeout=config.circuit_breaker_reset_timeout,
                        service_name=f"my-service.{config.environment}",
                        transport_type=EnumInfraTransportType.HTTP,
                    )
            ```
        """
        # Validate parameters
        if threshold < 1:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=transport_type,
                operation="init_circuit_breaker",
                target_name=service_name,
            )
            raise ProtocolConfigurationError(
                f"Circuit breaker threshold must be >= 1, got {threshold}",
                context=context,
                parameter="threshold",
                value=threshold,
            )
        if reset_timeout < 0:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=transport_type,
                operation="init_circuit_breaker",
                target_name=service_name,
            )
            raise ProtocolConfigurationError(
                f"Circuit breaker reset_timeout must be >= 0, got {reset_timeout}",
                context=context,
                parameter="reset_timeout",
                value=reset_timeout,
            )

        # State variables
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_open_until: float = 0.0

        # Configuration
        self.circuit_breaker_threshold = threshold
        self.circuit_breaker_reset_timeout = reset_timeout
        self.service_name = service_name
        self._cb_transport_type = (
            transport_type  # Use private name to avoid property conflicts
        )

        # Coroutine-safety lock (asyncio.Lock for concurrent async access, not thread-safe)
        self._circuit_breaker_lock = asyncio.Lock()

        logger.debug(
            f"Circuit breaker initialized for {service_name}",
            extra={
                "threshold": threshold,
                "reset_timeout": reset_timeout,
                "transport_type": transport_type.value,
            },
        )

    def _init_circuit_breaker_from_config(
        self,
        config: ModelCircuitBreakerConfig,
    ) -> None:
        """Initialize circuit breaker from a configuration model.

        This method provides an alternative initialization path using a
        configuration model instead of individual parameters. This reduces
        union types in calling code and follows ONEX patterns.

        Args:
            config: Configuration model containing all circuit breaker settings.
                See ModelCircuitBreakerConfig for available options.

        Raises:
            ValueError: If config contains invalid values (validated by Pydantic).

        Example:
            ```python
            from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
            from omnibase_infra.enums import EnumInfraTransportType

            class EventBusKafka(MixinAsyncCircuitBreaker):
                def __init__(self, environment: str):
                    config = ModelCircuitBreakerConfig(
                        threshold=5,
                        reset_timeout_seconds=60.0,
                        service_name=f"kafka.{environment}",
                        transport_type=EnumInfraTransportType.KAFKA,
                    )
                    self._init_circuit_breaker_from_config(config)
            ```

        See Also:
            _init_circuit_breaker: Original initialization method with parameters.
            ModelCircuitBreakerConfig: Configuration model with all options.
        """
        self._init_circuit_breaker(
            threshold=config.threshold,
            reset_timeout=config.reset_timeout_seconds,
            service_name=config.service_name,
            transport_type=config.transport_type,
        )

    async def _check_circuit_breaker(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Check if circuit breaker allows operation.

        Verifies circuit breaker state and raises InfraUnavailableError if
        circuit is open. Automatically transitions from OPEN to HALF_OPEN
        if reset timeout has elapsed.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.

            This method accesses shared state variables and MUST be called
            while holding the lock to prevent race conditions between coroutines:

            ```python
            # Correct
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker("operation")

            # Incorrect - race condition between coroutines!
            await self._check_circuit_breaker("operation")
            ```

        Args:
            operation: Operation name for error context (e.g., "publish", "register")
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, a new UUID will be generated.

        Raises:
            InfraUnavailableError: If circuit breaker is open and reset timeout
                has not elapsed. Error includes:
                - context.transport_type: Transport type from configuration
                - context.operation: Operation name from parameter
                - context.target_name: Service name from configuration
                - context.correlation_id: Correlation ID (provided or generated)
                - circuit_state: Current circuit state ("open")
                - retry_after_seconds: Seconds remaining until auto-reset

        Example:
            ```python
            async def perform_operation(
                self, correlation_id: UUID | None = None
            ) -> Result:
                # Check circuit before operation (coroutine-safe)
                async with self._circuit_breaker_lock:
                    await self._check_circuit_breaker(
                        operation="perform_operation",
                        correlation_id=correlation_id,
                    )

                # Proceed with operation if circuit allows
                result = await self._do_work()
                return result
            ```
        """
        # Verify lock is held (debug assertion)
        if not self._circuit_breaker_lock.locked():
            logger.error(
                "Circuit breaker lock not held during state check",
                extra={
                    "service": self.service_name,
                    "operation": operation,
                },
            )
            # Still proceed but log the violation for debugging

        current_time = time.time()

        # Check if circuit is open (atomic read protected by caller's lock)
        if self._circuit_breaker_open:
            # Check if reset timeout has passed
            if current_time >= self._circuit_breaker_open_until:
                # Transition to HALF_OPEN (atomic write protected by caller's lock)
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
                logger.info(
                    f"Circuit breaker transitioning to half-open for {self.service_name}",
                    extra={
                        "service": self.service_name,
                        "operation": operation,
                    },
                )
            else:
                # Circuit still open - block request
                retry_after = int(self._circuit_breaker_open_until - current_time)
                context = ModelInfraErrorContext(
                    transport_type=self._cb_transport_type,
                    operation=operation,
                    target_name=self.service_name,
                    correlation_id=correlation_id if correlation_id else uuid4(),
                )
                raise InfraUnavailableError(
                    f"Circuit breaker is open - {self.service_name} temporarily unavailable",
                    context=context,
                    circuit_state="open",
                    retry_after_seconds=retry_after,
                )

    async def _record_circuit_failure(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Record a circuit breaker failure and potentially open the circuit.

        Increments the failure counter and opens the circuit if the threshold
        is reached. When the circuit opens, it sets the reset timestamp for
        automatic recovery.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.

            This method mutates shared state variables and MUST be called
            while holding the lock to prevent race conditions between coroutines:

            ```python
            # Correct
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("operation")

            # Incorrect - race condition between coroutines!
            await self._record_circuit_failure("operation")
            ```

        Args:
            operation: Operation name for logging context
            correlation_id: Optional correlation ID for distributed tracing

        State Transitions:
            If failure count >= threshold:
                CLOSED → OPEN
                HALF_OPEN → OPEN

        Side Effects:
            - Increments _circuit_breaker_failures
            - If threshold reached:
                - Sets _circuit_breaker_open = True
                - Sets _circuit_breaker_open_until = current_time + reset_timeout
                - Logs warning message

        Example:
            ```python
            async def perform_operation(
                self, correlation_id: UUID | None = None
            ) -> Result:
                try:
                    result = await self._do_work()
                    return result
                except Exception:
                    # Record failure on exception (coroutine-safe)
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure(
                            operation="perform_operation",
                            correlation_id=correlation_id,
                        )
                    raise
            ```
        """
        # Verify lock is held (debug assertion)
        if not self._circuit_breaker_lock.locked():
            logger.error(
                "Circuit breaker lock not held during failure recording",
                extra={
                    "service": self.service_name,
                    "operation": operation,
                },
            )
            # Still proceed but log the violation for debugging

        # Increment failure counter (atomic write protected by caller's lock)
        self._circuit_breaker_failures += 1

        # Check if threshold reached
        if self._circuit_breaker_failures >= self.circuit_breaker_threshold:
            # Transition to OPEN state (atomic write protected by caller's lock)
            self._circuit_breaker_open = True
            self._circuit_breaker_open_until = (
                time.time() + self.circuit_breaker_reset_timeout
            )

            logger.warning(
                f"Circuit breaker opened for {self.service_name} after {self._circuit_breaker_failures} failures",
                extra={
                    "service": self.service_name,
                    "operation": operation,
                    "failure_count": self._circuit_breaker_failures,
                    "threshold": self.circuit_breaker_threshold,
                    "reset_timeout": self.circuit_breaker_reset_timeout,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state.

        Resets the failure counter and closes the circuit, allowing all
        requests to proceed normally. Typically called after a successful
        operation.

        Concurrency Safety:
            REQUIRES: self._circuit_breaker_lock must be held by caller.

            This method mutates shared state variables and MUST be called
            while holding the lock to prevent race conditions between coroutines:

            ```python
            # Correct
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Incorrect - race condition between coroutines!
            await self._reset_circuit_breaker()
            ```

        State Transitions:
            OPEN → CLOSED
            HALF_OPEN → CLOSED
            CLOSED → CLOSED (idempotent)

        Side Effects:
            - Sets _circuit_breaker_open = False
            - Sets _circuit_breaker_failures = 0
            - Sets _circuit_breaker_open_until = 0.0
            - Logs info message if circuit was not already closed

        Example:
            ```python
            async def perform_operation(self):
                try:
                    result = await self._do_work()

                    # Reset circuit on success (coroutine-safe)
                    async with self._circuit_breaker_lock:
                        await self._reset_circuit_breaker()

                    return result
                except Exception:
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure("perform_operation")
                    raise
            ```
        """
        # Verify lock is held (debug assertion)
        if not self._circuit_breaker_lock.locked():
            logger.error(
                "Circuit breaker lock not held during reset",
                extra={
                    "service": self.service_name,
                },
            )
            # Still proceed but log the violation for debugging

        # Log state transition if circuit was open or had failures
        if self._circuit_breaker_open or self._circuit_breaker_failures > 0:
            previous_state = "open" if self._circuit_breaker_open else "closed"
            logger.info(
                f"Circuit breaker reset from {previous_state} to closed for {self.service_name}",
                extra={
                    "service": self.service_name,
                    "previous_state": previous_state,
                    "previous_failures": self._circuit_breaker_failures,
                },
            )

        # Reset state (atomic write protected by caller's lock)
        self._circuit_breaker_open = False
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open_until = 0.0

    def _get_circuit_breaker_state(self) -> dict[str, JsonType]:
        """Return current circuit breaker state for introspection.

        This method encapsulates circuit breaker internals for safe access
        by subclasses implementing describe() or other introspection methods.
        It provides a stable interface for reading circuit breaker state without
        exposing internal attribute names.

        Note:
            This method does NOT require holding _circuit_breaker_lock because
            it only performs reads for observability purposes. The state may be
            slightly stale in concurrent scenarios, which is acceptable for
            introspection use cases.

        Returns:
            dict containing:
                - initialized: Whether circuit breaker has been initialized
                - state: Current state ("closed", "open", or "half_open")
                - failures: Current failure count
                - threshold: Configured failure threshold
                - reset_timeout_seconds: Configured reset timeout
                - seconds_until_half_open: Seconds until half_open (only when open)

        Example:
            ```python
            def describe(self) -> dict[str, object]:
                circuit_breaker_info = self._get_circuit_breaker_state()
                return {
                    "handler_type": self.handler_type.value,
                    "circuit_breaker": circuit_breaker_info,
                }
            ```
        """
        # Check if circuit breaker has been initialized by looking for key attributes
        cb_initialized = hasattr(self, "_circuit_breaker_lock") and hasattr(
            self, "circuit_breaker_threshold"
        )

        # Read state variables with safe defaults for uninitialized state
        cb_open = getattr(self, "_circuit_breaker_open", False)
        cb_open_until = getattr(self, "_circuit_breaker_open_until", 0.0)
        cb_failures = getattr(self, "_circuit_breaker_failures", 0)
        cb_threshold = getattr(self, "circuit_breaker_threshold", 5)
        cb_reset_timeout = getattr(self, "circuit_breaker_reset_timeout", 60.0)

        # Calculate state: closed, open, or half_open
        current_time = time.time()
        if cb_open:
            if current_time >= cb_open_until:
                cb_state = "half_open"
                seconds_until_half_open: float | None = None
            else:
                cb_state = "open"
                seconds_until_half_open = round(cb_open_until - current_time, 2)
        else:
            cb_state = "closed"
            seconds_until_half_open = None

        result: dict[str, JsonType] = {
            "initialized": cb_initialized,
            "state": cb_state,
            "failures": cb_failures,
            "threshold": cb_threshold,
            "reset_timeout_seconds": cb_reset_timeout,
        }

        if seconds_until_half_open is not None:
            result["seconds_until_half_open"] = seconds_until_half_open

        return result


__all__ = ["EnumCircuitState", "MixinAsyncCircuitBreaker", "ModelCircuitBreakerConfig"]
