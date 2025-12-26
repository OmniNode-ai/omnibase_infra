# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol interface for the RuntimeTick scheduler.

This module defines the ProtocolRuntimeScheduler interface for implementing
runtime tick schedulers. The scheduler is the single source of truth for 'now'
across all orchestrators in the ONEX infrastructure.

Architecture Note (Infrastructure vs Domain):
    The scheduler is an **infrastructure concern** - it emits RuntimeTick events
    at configured intervals, providing a consistent time reference.

    Orchestrators derive **timeout decisions** (a domain concern) from these ticks.
    This separation ensures:
    - Clear ownership: Infrastructure manages timing, domain manages decisions
    - Testability: Schedulers can be mocked or time-controlled for testing
    - Decoupling: Domain logic doesn't depend on system clock directly

Design Pattern:
    The scheduler follows an event-driven pattern where:
    1. Scheduler emits RuntimeTick events at regular intervals
    2. Orchestrators subscribe to tick events
    3. On each tick, orchestrators evaluate timeout conditions
    4. Timeout decisions are made based on elapsed time since last action

Restart Safety:
    The scheduler maintains a monotonically increasing sequence_number that
    survives restarts. This enables orchestrators to detect scheduler restarts
    and handle any missed ticks appropriately.

Concurrency Safety:
    Implementations MUST be safe for concurrent coroutine access. The scheduler
    may be accessed from multiple coroutines for status checks while the tick
    loop runs. Use asyncio.Lock for shared mutable state (coroutine-safe, not
    thread-safe).

Example:
    .. code-block:: python

        from omnibase_infra.runtime.protocols import ProtocolRuntimeScheduler

        class InMemoryScheduler:
            '''Simple in-memory scheduler for testing.'''

            def __init__(self, interval_seconds: float = 1.0) -> None:
                self._scheduler_id = "test-scheduler-001"
                self._running = False
                self._sequence = 0
                self._total_ticks_emitted = 0
                self._interval = interval_seconds
                self._state_lock = asyncio.Lock()

            @property
            def scheduler_id(self) -> str:
                return self._scheduler_id

            @property
            def is_running(self) -> bool:
                return self._running

            @property
            def current_sequence_number(self) -> int:
                return self._sequence

            async def start(self) -> None:
                self._running = True
                # Start tick loop...

            async def stop(self) -> None:
                self._running = False

            async def emit_tick(self, now: datetime | None = None) -> None:
                self._sequence += 1
                self._total_ticks_emitted += 1
                tick_time = now or datetime.now(timezone.utc)
                # Emit event to Kafka...

            async def get_metrics(self) -> ModelRuntimeSchedulerMetrics:
                # Lock ensures consistent snapshot of all metrics
                from omnibase_infra.runtime.enums import EnumSchedulerStatus
                async with self._state_lock:
                    return ModelRuntimeSchedulerMetrics(
                        scheduler_id=self._scheduler_id,
                        status=EnumSchedulerStatus.RUNNING if self._running else EnumSchedulerStatus.STOPPED,
                        ticks_emitted=self._total_ticks_emitted,
                    )

        # Protocol conformance check via duck typing (per ONEX conventions)
        scheduler = InMemoryScheduler()

        # Verify required methods/properties exist and are callable
        assert hasattr(scheduler, 'scheduler_id')
        assert hasattr(scheduler, 'is_running')
        assert hasattr(scheduler, 'start') and callable(scheduler.start)
        assert hasattr(scheduler, 'stop') and callable(scheduler.stop)
        assert hasattr(scheduler, 'emit_tick') and callable(scheduler.emit_tick)

Related:
    - OMN-953: RuntimeTick scheduler implementation
    - ModelRuntimeTick: The event model emitted by schedulers
    - ModelRuntimeSchedulerMetrics: Metrics model for observability

.. versionadded:: 0.4.0
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.runtime.models import ModelRuntimeSchedulerMetrics

__all__: list[str] = [
    "ProtocolRuntimeScheduler",
]


@runtime_checkable
class ProtocolRuntimeScheduler(Protocol):
    """Protocol for runtime tick scheduler.

    The scheduler is the single source of truth for 'now' across orchestrators.
    It emits RuntimeTick events at configured intervals, enabling time-based
    coordination without direct clock dependencies.

    This is an **infrastructure concern** - orchestrators derive timeout
    decisions (domain concern) from the ticks.

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing rather
        than isinstance checks. Verify required methods and properties exist:

        .. code-block:: python

            # Duck typing verification (preferred)
            assert hasattr(scheduler, 'scheduler_id')
            assert hasattr(scheduler, 'start') and callable(scheduler.start)
            assert hasattr(scheduler, 'stop') and callable(scheduler.stop)
            assert hasattr(scheduler, 'emit_tick') and callable(scheduler.emit_tick)

        **Note**: For complete type safety, use static type checking (mypy)
        in addition to duck typing verification.

    Concurrency Safety:
        Implementations MUST be safe for concurrent coroutine access. The scheduler
        may be:
        - Started/stopped from the main coroutine
        - Queried for status from multiple coroutines
        - Emitting ticks in a background task

        Use asyncio.Lock for state access (coroutine-safe, not OS thread-safe).

    Restart Safety:
        The ``current_sequence_number`` property returns a monotonically increasing
        value that helps orchestrators detect scheduler restarts. If the sequence
        number decreases or resets, orchestrators know a restart occurred and can
        handle any missed ticks appropriately.

    Lifecycle:
        1. Create scheduler instance with configuration
        2. Call ``start()`` to begin tick emission
        3. Scheduler emits ticks at configured intervals
        4. Call ``stop()`` for graceful shutdown
        5. Query ``get_metrics()`` for observability

    Example:
        .. code-block:: python

            from omnibase_infra.runtime.protocols import ProtocolRuntimeScheduler

            async def run_scheduler(scheduler: ProtocolRuntimeScheduler) -> None:
                '''Run scheduler with graceful shutdown.'''
                await scheduler.start()

                try:
                    # Scheduler runs, emitting ticks...
                    while scheduler.is_running:
                        await asyncio.sleep(1.0)
                finally:
                    await scheduler.stop()
                    metrics = await scheduler.get_metrics()
                    print(f"Scheduler stopped after {metrics.ticks_emitted} ticks")

    Attributes:
        scheduler_id: Unique identifier for this scheduler instance.
        is_running: Whether the scheduler is currently running.
        current_sequence_number: Current sequence number for restart-safety tracking.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544. Protocol classes define
        structural subtyping (duck typing) interfaces, not inheritance-based
        abstract base classes.

    .. versionadded:: 0.4.0
    """

    @property
    def scheduler_id(self) -> str:
        """Return the unique identifier for this scheduler instance.

        The scheduler_id is used for:
        - Registration and lookup in scheduler registries
        - Correlation in RuntimeTick events
        - Tracing and observability
        - Error reporting and debugging

        Format Recommendations:
            - Include environment/instance info for uniqueness
            - Use format: "{service}-{instance}-{uuid}" or similar
            - Keep reasonably short for log readability

        Returns:
            str: Unique scheduler identifier (e.g., "runtime-scheduler-prod-001")

        Example:
            .. code-block:: python

                @property
                def scheduler_id(self) -> str:
                    return f"runtime-scheduler-{self._instance_id}"
        """
        ...

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler is currently running.

        This property indicates the scheduler's operational state:
        - True: Scheduler is actively emitting ticks
        - False: Scheduler is stopped or not yet started

        Concurrency Safety:
            This property MUST be safe for concurrent coroutine access. It may
            be called from different coroutines while the tick loop runs.

        Returns:
            bool: True if running and emitting ticks, False otherwise.

        Example:
            .. code-block:: python

                @property
                def is_running(self) -> bool:
                    return self._running

                # Usage
                if scheduler.is_running:
                    print("Scheduler is active")
                else:
                    print("Scheduler is stopped")
        """
        ...

    @property
    def current_sequence_number(self) -> int:
        """Return the current sequence number for restart-safety tracking.

        The sequence number is a monotonically increasing value that:
        - Increments with each tick emitted
        - Helps detect scheduler restarts
        - Enables ordering of ticks
        - Aids in missed-tick detection

        Restart Safety:
            If orchestrators observe a sequence number that is lower than
            previously seen, they know a scheduler restart occurred. This
            allows them to:
            - Re-evaluate pending timeouts
            - Handle any ticks that may have been missed
            - Reset internal tick counters

        Persistence:
            Implementations MAY persist the sequence number to survive
            process restarts. This is recommended for production use but
            optional for testing scenarios.

        Returns:
            int: Current sequence number (non-negative).

        Example:
            .. code-block:: python

                @property
                def current_sequence_number(self) -> int:
                    return self._sequence_number

                # Restart detection in orchestrator
                if new_sequence < last_seen_sequence:
                    logger.warning("Scheduler restart detected, re-evaluating timeouts")
                    self._handle_scheduler_restart()
        """
        ...

    async def start(self) -> None:
        """Start the scheduler, begin emitting ticks.

        This method initializes the scheduler and starts the tick emission loop.
        After calling start(), the scheduler will emit RuntimeTick events at
        its configured interval.

        Idempotency:
            Calling start() on an already-running scheduler SHOULD be idempotent
            (no-op or warning log, not an error).

        Behavior:
            - Sets is_running to True
            - Starts the internal tick loop (background task)
            - Emits an initial tick immediately (optional, implementation-defined)

        Raises:
            InfraConnectionError: If unable to connect to event bus for tick emission.
            InfraUnavailableError: If required dependencies are unavailable.

        Example:
            .. code-block:: python

                async def start(self) -> None:
                    if self._running:
                        logger.warning("Scheduler already running, ignoring start()")
                        return

                    self._running = True
                    self._tick_task = asyncio.create_task(self._tick_loop())
                    logger.info("Scheduler started with interval=%s", self._interval)
        """
        ...

    async def stop(self) -> None:
        """Stop the scheduler gracefully.

        This method performs a graceful shutdown of the scheduler:
        - Signals the tick loop to stop
        - Waits for any in-flight tick emission to complete
        - Sets is_running to False
        - Releases any resources

        Idempotency:
            Calling stop() on an already-stopped scheduler SHOULD be idempotent
            (no-op or debug log, not an error).

        Graceful Shutdown:
            The stop() method SHOULD wait for the current tick to complete
            before returning. It SHOULD NOT emit partial ticks or leave
            the event bus in an inconsistent state.

        Example:
            .. code-block:: python

                async def stop(self) -> None:
                    if not self._running:
                        logger.debug("Scheduler already stopped, ignoring stop()")
                        return

                    self._running = False

                    if self._tick_task:
                        self._tick_task.cancel()
                        try:
                            await self._tick_task
                        except asyncio.CancelledError:
                            pass

                    logger.info("Scheduler stopped after %d ticks", self._sequence_number)
        """
        ...

    async def emit_tick(self, now: datetime | None = None) -> None:
        """Emit a single tick immediately.

        This method emits a RuntimeTick event outside the normal interval loop.
        It is primarily used for:
        - Testing: Manually trigger ticks with controlled timestamps
        - Manual intervention: Force a tick when needed
        - Initial tick: Emit a tick immediately on start

        Args:
            now: Optional override for current time. If None, uses actual
                current time (datetime.now(timezone.utc)). Pass a specific
                datetime for testing scenarios where time control is needed.

        Behavior:
            - Increments the sequence number
            - Creates a RuntimeTick event with the given (or current) time
            - Publishes the tick to the configured event bus/topic
            - Updates internal metrics

        Concurrency Safety:
            This method MUST be safe for concurrent coroutine calls. Use
            asyncio.Lock if internal state is modified.

        Example:
            .. code-block:: python

                from datetime import datetime, timezone

                async def emit_tick(self, now: datetime | None = None) -> None:
                    tick_time = now or datetime.now(timezone.utc)
                    self._sequence_number += 1

                    tick_event = ModelRuntimeTick(
                        scheduler_id=self._scheduler_id,
                        sequence_number=self._sequence_number,
                        timestamp=tick_time,
                    )

                    await self._event_bus.publish(
                        topic="runtime.tick.v1",
                        event=tick_event,
                    )

                    self._total_ticks_emitted += 1

                # Testing with controlled time
                test_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                await scheduler.emit_tick(now=test_time)
        """
        ...

    async def get_metrics(self) -> ModelRuntimeSchedulerMetrics:
        """Get current scheduler metrics.

        Returns a snapshot of the scheduler's operational metrics for
        observability and monitoring purposes.

        The metrics model typically includes:
            - scheduler_id: Scheduler identifier
            - status: Current scheduler status (EnumSchedulerStatus)
            - ticks_emitted: Total ticks emitted since start
            - ticks_failed: Number of failed tick emissions
            - current_sequence_number: Current sequence number
            - last_tick_at: Timestamp of last tick (if any)
            - consecutive_failures: Number of consecutive tick failures

        Returns:
            ModelRuntimeSchedulerMetrics: Current metrics snapshot.

        Concurrency Safety:
            This method acquires the internal state lock (asyncio.Lock) to ensure
            a consistent snapshot of all metrics. The returned metrics object is
            immutable and safe to use after the call returns. All state variables
            are read atomically within a single lock acquisition.

        Example:
            .. code-block:: python

                from omnibase_infra.runtime.enums import EnumSchedulerStatus

                async def get_metrics(self) -> ModelRuntimeSchedulerMetrics:
                    async with self._state_lock:
                        return ModelRuntimeSchedulerMetrics(
                            scheduler_id=self._scheduler_id,
                            status=EnumSchedulerStatus.RUNNING if self._running else EnumSchedulerStatus.STOPPED,
                            ticks_emitted=self._total_ticks_emitted,
                            ticks_failed=self._ticks_failed,
                            current_sequence_number=self._sequence_number,
                            last_tick_at=self._last_tick_time,
                            consecutive_failures=self._consecutive_failures,
                        )

                # Usage in monitoring
                metrics = await scheduler.get_metrics()
                if metrics.status != EnumSchedulerStatus.RUNNING:
                    alert("Scheduler is not running!")
        """
        ...
