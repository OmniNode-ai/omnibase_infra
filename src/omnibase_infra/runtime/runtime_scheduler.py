# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RuntimeScheduler - Core service for emitting RuntimeTick events.

This module implements the RuntimeScheduler service that emits RuntimeTick events
at configured intervals. The scheduler is the single source of truth for "now"
across all orchestrators in the ONEX infrastructure.

Key Features:
    - Configurable tick interval (10ms - 60,000ms)
    - Circuit breaker pattern for publish resilience
    - Restart-safe sequence number tracking
    - Jitter to prevent thundering herd
    - Graceful shutdown with proper lifecycle management
    - Metrics collection for observability

Architecture:
    The RuntimeScheduler is an INFRASTRUCTURE concern. It emits RuntimeTick events
    that orchestrators subscribe to for timeout decisions (DOMAIN concern). This
    separation ensures clear ownership and testability.

Concurrency Safety:
    This scheduler is coroutine-safe, not thread-safe. All locking uses
    asyncio primitives which protect against concurrent coroutine access:
    - Circuit breaker operations protected by `_circuit_breaker_lock` (asyncio.Lock)
    - State variables protected by `_state_lock` (asyncio.Lock)
    - Tick loop runs as background task with shutdown signaling via `asyncio.Event`
    For multi-threaded access, additional synchronization would be required.

Usage:
    ```python
    from omnibase_infra.runtime.runtime_scheduler import RuntimeScheduler
    from omnibase_infra.runtime.models import ModelRuntimeSchedulerConfig
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    # Create scheduler with configuration
    config = ModelRuntimeSchedulerConfig.default()
    event_bus = EventBusKafka.default()
    await event_bus.start()

    scheduler = RuntimeScheduler(config=config, event_bus=event_bus)
    await scheduler.start()

    try:
        # Scheduler runs, emitting ticks at configured interval
        while scheduler.is_running:
            await asyncio.sleep(1.0)
    finally:
        await scheduler.stop()
        metrics = scheduler.get_metrics()
        print(f"Emitted {metrics.ticks_emitted} ticks")
    ```

Related:
    - OMN-953: RuntimeTick scheduler implementation
    - ModelRuntimeTick: The event model emitted by the scheduler
    - ModelRuntimeSchedulerConfig: Configuration model
    - ModelRuntimeSchedulerMetrics: Metrics model for observability
    - ProtocolRuntimeScheduler: Protocol interface

.. versionadded:: 0.4.0
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.runtime.enums import EnumSchedulerStatus
from omnibase_infra.runtime.models import (
    ModelRuntimeSchedulerConfig,
    ModelRuntimeSchedulerMetrics,
    ModelRuntimeTick,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)


class RuntimeScheduler(MixinAsyncCircuitBreaker):
    """Runtime scheduler that emits RuntimeTick events at configured intervals.

    The scheduler is the single source of truth for "now" across all orchestrators.
    It emits RuntimeTick events that orchestrators subscribe to for timeout decisions.

    This is an INFRASTRUCTURE concern - orchestrators derive timeout decisions
    (DOMAIN concern) from the ticks.

    Attributes:
        scheduler_id: Unique identifier for this scheduler instance.
        is_running: Whether the scheduler is currently running.
        current_sequence_number: Current sequence number for restart-safety.

    Concurrency Safety:
        This scheduler is coroutine-safe using asyncio primitives:
        - Circuit breaker operations protected by `_circuit_breaker_lock` (asyncio.Lock)
        - State variables protected by `_state_lock` (asyncio.Lock)
        - Shutdown signaling via `asyncio.Event`
        Note: This is coroutine-safe, not thread-safe.

    Restart Safety:
        The `current_sequence_number` property returns a monotonically increasing
        value that helps orchestrators detect scheduler restarts. If the sequence
        number decreases or resets, orchestrators know a restart occurred.

    Example:
        ```python
        config = ModelRuntimeSchedulerConfig.default()
        event_bus = EventBusKafka.default()
        await event_bus.start()

        scheduler = RuntimeScheduler(config=config, event_bus=event_bus)
        await scheduler.start()

        # Scheduler runs in background
        await asyncio.sleep(10.0)

        await scheduler.stop()
        print(f"Emitted {scheduler.get_metrics().ticks_emitted} ticks")
        ```
    """

    def __init__(
        self,
        config: ModelRuntimeSchedulerConfig,
        event_bus: EventBusKafka,
    ) -> None:
        """Initialize the RuntimeScheduler.

        Args:
            config: Configuration model containing all scheduler settings.
            event_bus: EventBusKafka instance for publishing tick events.

        Raises:
            ProtocolConfigurationError: If config or event_bus is None.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="scheduler_init",
        )
        if config is None:
            raise ProtocolConfigurationError("config cannot be None", context=context)
        if event_bus is None:
            raise ProtocolConfigurationError(
                "event_bus cannot be None", context=context
            )

        # Store configuration
        self._config = config
        self._event_bus = event_bus

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name=f"runtime-scheduler.{config.scheduler_id}",
            transport_type=EnumInfraTransportType.KAFKA,
        )

        # State variables (protected by _state_lock)
        self._status = EnumSchedulerStatus.STOPPED
        self._sequence_number: int = 0
        self._started_at: datetime | None = None
        self._last_tick_at: datetime | None = None
        self._last_tick_duration_ms: float = 0.0
        self._total_tick_duration_ms: float = 0.0
        self._max_tick_duration_ms: float = 0.0
        self._ticks_emitted: int = 0
        self._ticks_failed: int = 0
        self._consecutive_failures: int = 0
        self._last_persisted_sequence: int = 0

        # Synchronization primitives
        self._state_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._tick_task: asyncio.Task[None] | None = None

        # Valkey client for sequence number persistence
        # Created lazily on first use to avoid blocking __init__
        self._valkey_client: Redis | None = None
        self._valkey_available: bool = True  # Assume available until proven otherwise

    # =========================================================================
    # Properties (ProtocolRuntimeScheduler interface)
    # =========================================================================

    @property
    def scheduler_id(self) -> str:
        """Return the unique identifier for this scheduler instance.

        Returns:
            Unique scheduler identifier from configuration.
        """
        return self._config.scheduler_id

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler is currently running.

        Returns:
            True if running and emitting ticks, False otherwise.
        """
        return self._status == EnumSchedulerStatus.RUNNING

    @property
    def current_sequence_number(self) -> int:
        """Return the current sequence number for restart-safety tracking.

        Returns:
            Current sequence number (non-negative).
        """
        return self._sequence_number

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the scheduler and begin emitting ticks.

        Initializes the scheduler and starts the tick emission loop.
        After calling start(), the scheduler will emit RuntimeTick events
        at its configured interval.

        Idempotency:
            Calling start() on an already-running scheduler is a no-op
            with a warning log.

        Raises:
            InfraUnavailableError: If the circuit breaker is open.
        """
        async with self._state_lock:
            if self._status.is_active():
                logger.warning(
                    "Scheduler already active, ignoring start()",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "status": str(self._status),
                    },
                )
                return

            # Transition to STARTING
            self._status = EnumSchedulerStatus.STARTING

        # Check circuit breaker before starting (outside state lock)
        async with self._circuit_breaker_lock:
            try:
                await self._check_circuit_breaker(
                    operation="start",
                    correlation_id=uuid4(),
                )
            except InfraUnavailableError:
                async with self._state_lock:
                    self._status = EnumSchedulerStatus.ERROR
                raise

        # Load persisted sequence number (if enabled)
        await self._load_sequence_number()

        # Start tick loop
        async with self._state_lock:
            self._shutdown_event.clear()
            self._started_at = datetime.now(UTC)
            self._status = EnumSchedulerStatus.RUNNING
            self._tick_task = asyncio.create_task(self._tick_loop())

        logger.info(
            "Scheduler started",
            extra={
                "scheduler_id": self.scheduler_id,
                "tick_interval_ms": self._config.tick_interval_ms,
                "max_jitter_ms": self._config.max_tick_jitter_ms,
            },
        )

    async def stop(self) -> None:
        """Stop the scheduler gracefully.

        Performs a graceful shutdown of the scheduler:
        - Signals the tick loop to stop
        - Waits for any in-flight tick emission to complete
        - Sets status to STOPPED
        - Persists sequence number if enabled

        Idempotency:
            Calling stop() on an already-stopped scheduler is a no-op.
        """
        async with self._state_lock:
            if self._status.is_terminal():
                logger.debug(
                    "Scheduler already stopped, ignoring stop()",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "status": str(self._status),
                    },
                )
                return

            # Transition to STOPPING
            self._status = EnumSchedulerStatus.STOPPING

        # Signal shutdown to tick loop
        self._shutdown_event.set()

        # Wait for tick task to complete
        if self._tick_task is not None:
            try:
                await asyncio.wait_for(self._tick_task, timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "Tick task did not complete within timeout, cancelling",
                    extra={"scheduler_id": self.scheduler_id},
                )
                self._tick_task.cancel()
                try:
                    await self._tick_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._tick_task = None

        # Persist sequence number (if enabled)
        await self._persist_sequence_number()

        # Finalize state
        async with self._state_lock:
            self._status = EnumSchedulerStatus.STOPPED

        logger.info(
            "Scheduler stopped",
            extra={
                "scheduler_id": self.scheduler_id,
                "ticks_emitted": self._ticks_emitted,
                "ticks_failed": self._ticks_failed,
                "final_sequence": self._sequence_number,
            },
        )

        # Always close Valkey client if it exists (idempotent)
        await self._close_valkey_client()

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def emit_tick(self, now: datetime | None = None) -> None:
        """Emit a single tick immediately.

        This method emits a RuntimeTick event outside the normal interval loop.
        It is primarily used for testing and manual intervention.

        Args:
            now: Optional override for current time. If None, uses actual
                current time (datetime.now(timezone.utc)).

        Raises:
            InfraUnavailableError: When the circuit breaker is open.
            InfraTimeoutError: When tick emission to Kafka times out.
            InfraConnectionError: When tick emission fails due to connection issues.

        Concurrency Safety:
            This method is safe for concurrent coroutine calls. State modifications
            are protected by `_state_lock` (asyncio.Lock).
        """
        tick_time = now or datetime.now(UTC)
        correlation_id = uuid4()
        tick_id = uuid4()
        start_time = time.monotonic()

        # Increment sequence number (protected)
        async with self._state_lock:
            self._sequence_number += 1
            current_sequence = self._sequence_number

        # Create tick event
        tick = ModelRuntimeTick(
            now=tick_time,
            tick_id=tick_id,
            sequence_number=current_sequence,
            scheduled_at=tick_time,
            correlation_id=correlation_id,
            scheduler_id=self.scheduler_id,
            tick_interval_ms=self._config.tick_interval_ms,
        )

        # Check circuit breaker before publishing
        async with self._circuit_breaker_lock:
            try:
                await self._check_circuit_breaker(
                    operation="emit_tick",
                    correlation_id=correlation_id,
                )
            except InfraUnavailableError:
                await self._record_tick_failure(correlation_id)
                raise

        # Create headers for the event
        headers = ModelEventHeaders(
            correlation_id=correlation_id,
            message_id=tick_id,
            timestamp=tick_time,
            source=f"runtime-scheduler.{self.scheduler_id}",
            event_type="runtime.tick.v1",
        )

        # Serialize tick to JSON bytes
        tick_bytes = tick.model_dump_json().encode("utf-8")

        # Prepare error context for ONEX error types
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="emit_tick",
            target_name=self._config.tick_topic,
            correlation_id=correlation_id,
        )

        try:
            # Publish tick event
            await self._event_bus.publish(
                topic=self._config.tick_topic,
                key=self.scheduler_id.encode("utf-8"),
                value=tick_bytes,
                headers=headers,
            )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            await self._record_tick_success(start_time, tick_time)

            logger.debug(
                "Tick emitted",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_number": current_sequence,
                    "tick_id": str(tick_id),
                    "correlation_id": str(correlation_id),
                },
            )

        except TimeoutError as e:
            # Record failure for timeout
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="emit_tick",
                    correlation_id=correlation_id,
                )

            await self._record_tick_failure(correlation_id)

            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="emit_tick",
                target_name=self._config.tick_topic,
                correlation_id=correlation_id,
                # timeout_seconds omitted - Kafka timeout is event bus level, not available here
            )
            raise InfraTimeoutError(
                f"Timeout emitting tick to topic {self._config.tick_topic}",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            # Record failure for other errors
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="emit_tick",
                    correlation_id=correlation_id,
                )

            await self._record_tick_failure(correlation_id)

            raise InfraConnectionError(
                f"Failed to emit tick to topic {self._config.tick_topic}",
                context=ctx,
            ) from e

    async def get_metrics(self) -> ModelRuntimeSchedulerMetrics:
        """Get current scheduler metrics.

        Returns a snapshot of the scheduler's operational metrics for
        observability and monitoring purposes.

        Returns:
            ModelRuntimeSchedulerMetrics: Current metrics snapshot.

        Concurrency Safety:
            This method acquires both ``_circuit_breaker_lock`` and ``_state_lock``
            (asyncio.Lock instances) to ensure a consistent snapshot of all metrics.
            Circuit breaker state is read under its own lock first (consistent with
            modification patterns), then scheduler state is read under the state lock.
            The returned Pydantic model is immutable and safe to use after locks are
            released. Note: This is coroutine-safe, not thread-safe.

        Example:
            >>> scheduler = RuntimeScheduler(config=config, event_bus=event_bus)
            >>> await scheduler.start()
            >>> # After some ticks have been emitted...
            >>> metrics = await scheduler.get_metrics()
            >>> print(f"Scheduler: {metrics.scheduler_id}")
            >>> print(f"Status: {metrics.status}")
            >>> print(f"Ticks emitted: {metrics.ticks_emitted}")
            >>> print(f"Ticks failed: {metrics.ticks_failed}")
            >>> print(f"Success rate: {metrics.tick_success_rate()}")
            >>> print(f"Average tick duration: {metrics.average_tick_duration_ms}ms")
            >>> print(f"Circuit breaker open: {metrics.circuit_breaker_open}")
            >>> print(f"Consecutive failures: {metrics.consecutive_failures}")
            >>> print(f"Uptime: {metrics.total_uptime_seconds}s")
            >>> if metrics.is_healthy():
            ...     print("Scheduler is healthy")
        """
        # First, capture circuit breaker state under its own lock
        # This ensures consistency with how _circuit_breaker_open is modified
        async with self._circuit_breaker_lock:
            circuit_breaker_open = self._circuit_breaker_open

        # Then capture scheduler state under state lock
        async with self._state_lock:
            # Calculate uptime
            uptime_seconds = 0.0
            if self._started_at is not None:
                uptime_seconds = (datetime.now(UTC) - self._started_at).total_seconds()

            # Calculate average tick duration
            average_tick_duration_ms = 0.0
            if self._ticks_emitted > 0:
                average_tick_duration_ms = (
                    self._total_tick_duration_ms / self._ticks_emitted
                )

            return ModelRuntimeSchedulerMetrics(
                scheduler_id=self.scheduler_id,
                status=self._status,
                ticks_emitted=self._ticks_emitted,
                ticks_failed=self._ticks_failed,
                last_tick_at=self._last_tick_at,
                last_tick_duration_ms=self._last_tick_duration_ms,
                average_tick_duration_ms=average_tick_duration_ms,
                max_tick_duration_ms=self._max_tick_duration_ms,
                current_sequence_number=self._sequence_number,
                last_persisted_sequence=self._last_persisted_sequence,
                circuit_breaker_open=circuit_breaker_open,
                consecutive_failures=self._consecutive_failures,
                started_at=self._started_at,
                total_uptime_seconds=uptime_seconds,
            )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _tick_loop(self) -> None:
        """Main tick loop that emits ticks at configured intervals.

        This method runs as a background task and continuously emits ticks
        until the shutdown event is set. It handles jitter and graceful
        shutdown.
        """
        logger.debug(
            "Tick loop started",
            extra={
                "scheduler_id": self.scheduler_id,
                "tick_interval_ms": self._config.tick_interval_ms,
            },
        )

        try:
            while not self._shutdown_event.is_set():
                # Calculate interval with jitter
                interval_seconds = self._config.tick_interval_ms / 1000.0
                if self._config.max_tick_jitter_ms > 0:
                    jitter_ms = random.randint(0, self._config.max_tick_jitter_ms)
                    interval_seconds += jitter_ms / 1000.0

                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval_seconds,
                    )
                    # Shutdown event was set - exit loop
                    break
                except TimeoutError:
                    # Timeout expired - time to emit tick
                    pass

                # Check if we should still be running
                if self._shutdown_event.is_set():
                    break

                # Emit tick (errors are logged but don't crash the loop)
                try:
                    await self.emit_tick()
                except Exception as e:
                    # Log but don't crash the loop
                    logger.exception(
                        "Error in tick loop, continuing",
                        extra={
                            "scheduler_id": self.scheduler_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    # Increment consecutive failures for monitoring
                    async with self._state_lock:
                        self._consecutive_failures += 1

        except asyncio.CancelledError:
            logger.info(
                "Tick loop cancelled",
                extra={"scheduler_id": self.scheduler_id},
            )
            raise

        except Exception as e:
            logger.exception(
                "Unexpected error in tick loop",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "error": str(e),
                },
            )
            async with self._state_lock:
                self._status = EnumSchedulerStatus.ERROR

        finally:
            logger.debug(
                "Tick loop exiting",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "ticks_emitted": self._ticks_emitted,
                },
            )

    async def _record_tick_success(
        self,
        start_time: float,
        tick_time: datetime,
    ) -> None:
        """Record a successful tick emission.

        Args:
            start_time: Monotonic time when tick started.
            tick_time: The timestamp used in the tick.
        """
        duration_ms = (time.monotonic() - start_time) * 1000.0

        async with self._state_lock:
            self._ticks_emitted += 1
            self._last_tick_at = tick_time
            self._last_tick_duration_ms = duration_ms
            self._total_tick_duration_ms += duration_ms
            self._max_tick_duration_ms = max(duration_ms, self._max_tick_duration_ms)
            self._consecutive_failures = 0

    async def _record_tick_failure(self, correlation_id: UUID) -> None:
        """Record a failed tick emission.

        Args:
            correlation_id: Correlation ID for tracing.
        """
        async with self._state_lock:
            self._ticks_failed += 1
            self._consecutive_failures += 1

    # =========================================================================
    # Valkey Persistence (for restart-safety)
    # =========================================================================

    async def _get_valkey_client(self) -> Redis | None:
        """Get or create the Valkey client for sequence number persistence.

        This method lazily creates a Valkey client on first use. If the client
        has been marked as unavailable (due to connection failures), it returns
        None without attempting to reconnect.

        Returns:
            Redis client instance if available, None if unavailable or disabled.

        Note:
            The client is created with the configured host, port, password, and
            timeout settings. Connection failures are handled gracefully with
            retry logic.
        """
        # Skip if persistence is disabled or Valkey was marked unavailable
        if not self._config.persist_sequence_number:
            return None

        if not self._valkey_available:
            return None

        # Return existing client if already created
        if self._valkey_client is not None:
            return self._valkey_client

        # Create new client with retry logic
        correlation_id = uuid4()
        retries = self._config.valkey_connection_retries

        for attempt in range(retries + 1):
            try:
                self._valkey_client = redis.Redis(
                    host=self._config.valkey_host,
                    port=self._config.valkey_port,
                    password=self._config.valkey_password,
                    socket_timeout=self._config.valkey_timeout_seconds,
                    socket_connect_timeout=self._config.valkey_timeout_seconds,
                    decode_responses=True,
                )

                # Test connection with a ping
                await asyncio.wait_for(
                    self._valkey_client.ping(),
                    timeout=self._config.valkey_timeout_seconds,
                )

                logger.info(
                    "Valkey client connected for sequence persistence",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "valkey_host": self._config.valkey_host,
                        "valkey_port": self._config.valkey_port,
                        "correlation_id": str(correlation_id),
                    },
                )
                return self._valkey_client

            except (RedisConnectionError, RedisTimeoutError, TimeoutError) as e:
                if attempt < retries:
                    # Calculate exponential backoff delay: 1s, 2s, 4s, 8s... max 60s
                    backoff_delay = min(1.0 * (2**attempt), 60.0)

                    logger.warning(
                        "Valkey connection attempt %d/%d failed, retrying in %.1fs",
                        attempt + 1,
                        retries + 1,
                        backoff_delay,
                        extra={
                            "scheduler_id": self.scheduler_id,
                            "valkey_host": self._config.valkey_host,
                            "valkey_port": self._config.valkey_port,
                            # SECURITY: Sanitize error to prevent credential exposure
                            "error": sanitize_error_string(str(e)),
                            "error_type": type(e).__name__,
                            "correlation_id": str(correlation_id),
                            "backoff_delay_seconds": backoff_delay,
                        },
                    )
                    await asyncio.sleep(backoff_delay)
                else:
                    # All retries exhausted - mark as unavailable
                    self._valkey_available = False
                    self._valkey_client = None

                    logger.warning(
                        "Valkey unavailable after %d attempts, using in-memory",
                        retries + 1,
                        extra={
                            "scheduler_id": self.scheduler_id,
                            "valkey_host": self._config.valkey_host,
                            "valkey_port": self._config.valkey_port,
                            # SECURITY: Sanitize error to prevent credential exposure
                            "error": sanitize_error_string(str(e)),
                            "error_type": type(e).__name__,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    return None

            except RedisError as e:
                # Unexpected Redis error - mark as unavailable
                self._valkey_available = False
                self._valkey_client = None

                logger.warning(
                    "Valkey error during connection, using in-memory fallback",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "valkey_host": self._config.valkey_host,
                        "valkey_port": self._config.valkey_port,
                        # SECURITY: Sanitize error to prevent credential exposure
                        "error": sanitize_error_string(str(e)),
                        "error_type": type(e).__name__,
                        "correlation_id": str(correlation_id),
                    },
                )
                return None

        return None

    async def _close_valkey_client(self) -> None:
        """Close the Valkey client connection.

        This method gracefully closes the Valkey client connection if one exists.
        It is called during scheduler shutdown to ensure proper resource cleanup.

        Idempotency:
            This method is safe to call multiple times. It atomically swaps the
            client reference to None before attempting close, preventing double-close
            scenarios even with concurrent coroutine access.
        """
        # Atomically swap client reference to None to prevent double-close
        client = self._valkey_client
        self._valkey_client = None

        if client is not None:
            try:
                await client.aclose()
                logger.debug(
                    "Valkey client closed",
                    extra={"scheduler_id": self.scheduler_id},
                )
            except Exception as e:
                logger.warning(
                    "Error closing Valkey client",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        # SECURITY: Sanitize error to prevent credential exposure
                        "error": sanitize_error_string(str(e)),
                        "error_type": type(e).__name__,
                    },
                )

    async def _load_sequence_number(self) -> None:
        """Load persisted sequence number from Valkey for restart-safety.

        Attempts to read the sequence number from Valkey using the configured
        `sequence_number_key`. If Valkey is unavailable or the key doesn't exist,
        gracefully falls back to starting from 0.

        Graceful Fallback:
            - If Valkey is unavailable: Logs warning and starts from 0
            - If key doesn't exist: Logs debug and starts from 0
            - If value is not a valid integer: Logs warning and starts from 0

        Note:
            This method is called during start() if `persist_sequence_number`
            is enabled in configuration.
        """
        if not self._config.persist_sequence_number:
            logger.debug(
                "Sequence number persistence disabled, starting from 0",
                extra={"scheduler_id": self.scheduler_id},
            )
            return

        correlation_id = uuid4()
        client = await self._get_valkey_client()

        if client is None:
            logger.warning(
                "Valkey unavailable for sequence load, starting from 0",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_key": self._config.sequence_number_key,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        try:
            # Read sequence number from Valkey
            value = await asyncio.wait_for(
                client.get(self._config.sequence_number_key),
                timeout=self._config.valkey_timeout_seconds,
            )

            if value is None:
                # Key doesn't exist - this is a fresh start
                logger.debug(
                    "No persisted sequence number found, starting from 0",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "sequence_key": self._config.sequence_number_key,
                        "correlation_id": str(correlation_id),
                    },
                )
                return

            # Parse the sequence number
            try:
                loaded_sequence = int(value)
                if loaded_sequence < 0:
                    logger.warning(
                        "Persisted sequence number is negative, starting from 0",
                        extra={
                            "scheduler_id": self.scheduler_id,
                            "sequence_key": self._config.sequence_number_key,
                            "persisted_value": value,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    return

                # Successfully loaded - update state
                async with self._state_lock:
                    self._sequence_number = loaded_sequence
                    self._last_persisted_sequence = loaded_sequence

                logger.info(
                    "Loaded persisted sequence number",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "sequence_number": loaded_sequence,
                        "sequence_key": self._config.sequence_number_key,
                        "correlation_id": str(correlation_id),
                    },
                )

            except ValueError:
                logger.warning(
                    "Persisted sequence number is not a valid integer, starting from 0",
                    extra={
                        "scheduler_id": self.scheduler_id,
                        "sequence_key": self._config.sequence_number_key,
                        "persisted_value": value,
                        "correlation_id": str(correlation_id),
                    },
                )

        except (RedisConnectionError, RedisTimeoutError, TimeoutError) as e:
            # Connection failed during operation - mark unavailable
            self._valkey_available = False

            logger.warning(
                "Valkey connection failed during sequence load, starting from 0",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_key": self._config.sequence_number_key,
                    # SECURITY: Sanitize error to prevent credential exposure
                    "error": sanitize_error_string(str(e)),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

        except RedisError as e:
            logger.warning(
                "Valkey error during sequence load, starting from 0",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_key": self._config.sequence_number_key,
                    # SECURITY: Sanitize error to prevent credential exposure
                    "error": sanitize_error_string(str(e)),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

    async def _persist_sequence_number(self) -> None:
        """Persist current sequence number to Valkey for restart-safety.

        Writes the current sequence number to Valkey using the configured
        `sequence_number_key`. If Valkey is unavailable, gracefully logs a
        warning and continues without persistence.

        Graceful Fallback:
            - If Valkey is unavailable: Logs warning and skips persistence
            - If write fails: Logs warning with error details

        Note:
            This method is called during stop() if `persist_sequence_number`
            is enabled in configuration.
        """
        if not self._config.persist_sequence_number:
            return

        correlation_id = uuid4()
        client = await self._get_valkey_client()

        if client is None:
            # Valkey unavailable - log but don't fail shutdown
            # Note: _last_persisted_sequence is NOT updated because persistence failed
            logger.warning(
                "Valkey unavailable for sequence persistence",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_number": self._sequence_number,
                    "sequence_key": self._config.sequence_number_key,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        try:
            # Write sequence number to Valkey
            await asyncio.wait_for(
                client.set(
                    self._config.sequence_number_key,
                    str(self._sequence_number),
                ),
                timeout=self._config.valkey_timeout_seconds,
            )

            self._last_persisted_sequence = self._sequence_number

            logger.info(
                "Persisted sequence number to Valkey",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_number": self._sequence_number,
                    "sequence_key": self._config.sequence_number_key,
                    "correlation_id": str(correlation_id),
                },
            )

        except (RedisConnectionError, RedisTimeoutError, TimeoutError) as e:
            # Connection failed during operation - log but don't fail shutdown
            # Note: _last_persisted_sequence is NOT updated because persistence failed
            self._valkey_available = False

            logger.warning(
                "Valkey connection failed during sequence persistence",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_number": self._sequence_number,
                    "sequence_key": self._config.sequence_number_key,
                    # SECURITY: Sanitize error to prevent credential exposure
                    "error": sanitize_error_string(str(e)),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

        except RedisError as e:
            # Note: _last_persisted_sequence is NOT updated because persistence failed
            logger.warning(
                "Valkey error during sequence persistence",
                extra={
                    "scheduler_id": self.scheduler_id,
                    "sequence_number": self._sequence_number,
                    "sequence_key": self._config.sequence_number_key,
                    # SECURITY: Sanitize error to prevent credential exposure
                    "error": sanitize_error_string(str(e)),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

        finally:
            # Close the Valkey client during shutdown
            await self._close_valkey_client()


__all__: list[str] = ["RuntimeScheduler"]
