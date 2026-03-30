# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler pool for per-handler-type instance pooling (OMN-477).

Provides ``HandlerPool`` which manages multiple handler instances of a single
handler type, enabling true parallel processing without contention.  When
``max_concurrent_handlers > 1`` (OMN-476) and a single handler instance is
shared across all in-flight envelopes, handler-level state (connections,
buffers, cursors) can become a contention point.  HandlerPool eliminates this
by maintaining a pool of independently-instantiated handler instances with
checkout/checkin semantics.

Key Features:
    - Configurable pool size per handler type
    - asyncio.Queue-based checkout/checkin (no busy-waiting)
    - Instance health checking and automatic recycling
    - Pool-level metrics exposed for health_check()
    - Dynamic pool growth up to max_size when load demands it

Usage::

    pool = HandlerPool(
        handler_type="db",
        factory=lambda: handler_cls(container=container),
        pool_size=4,
    )
    await pool.initialize()

    async with pool.checkout() as handler:
        response = await handler.execute(envelope)

    await pool.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolContainerAware

logger = logging.getLogger(__name__)

# Pool size bounds
MIN_POOL_SIZE = 1
MAX_POOL_SIZE = 64
DEFAULT_POOL_SIZE = 1


class HandlerPool:
    """Pool of handler instances for a single handler type (OMN-477).

    Maintains a fixed-size pool of handler instances.  Callers use
    ``checkout()`` to acquire an idle instance and it is automatically
    returned when the async context manager exits.

    The pool is initialized with ``pool_size`` instances created via
    the supplied factory callable.  Instances that fail health checks
    are recycled (shut down and replaced with a fresh instance).

    Attributes:
        handler_type: The protocol type identifier (e.g., "db", "http").
        pool_size: Configured number of instances in the pool.
    """

    def __init__(
        self,
        handler_type: str,
        factory: Callable[[], ProtocolContainerAware],
        pool_size: int = DEFAULT_POOL_SIZE,
        handler_config: dict[str, object] | None = None,
    ) -> None:
        """Initialize the handler pool.

        Args:
            handler_type: Protocol type identifier for this pool.
            factory: Callable that creates a new handler instance.
                Must return a ProtocolContainerAware-compatible handler.
            pool_size: Number of handler instances to maintain.
                Clamped to [MIN_POOL_SIZE, MAX_POOL_SIZE].
            handler_config: Optional configuration dict to pass to each
                handler's ``initialize(config)`` method.  When provided,
                ``instance.initialize(handler_config)`` is called; when
                ``None``, ``instance.initialize()`` is called with no
                arguments (backwards-compatible for handlers that accept
                no config).
        """
        self.handler_type: str = handler_type
        self.pool_size: int = max(MIN_POOL_SIZE, min(pool_size, MAX_POOL_SIZE))
        self._factory: Callable[[], ProtocolContainerAware] = factory
        self._handler_config: dict[str, object] | None = handler_config
        self._pool: asyncio.Queue[ProtocolContainerAware] = asyncio.Queue(
            maxsize=self.pool_size
        )
        self._all_instances: list[ProtocolContainerAware] = []
        self._initialized: bool = False
        self._shutting_down: bool = False

        # Metrics
        self._checkout_count: int = 0
        self._checkin_count: int = 0
        self._recycle_count: int = 0
        self._checkout_wait_total_ms: float = 0.0

    async def _initialize_instance(self, instance: ProtocolContainerAware) -> None:
        """Call ``initialize()`` on a handler instance, passing config if available.

        When ``_handler_config`` is set, calls ``instance.initialize(config)``;
        otherwise calls ``instance.initialize()`` with no arguments for
        backwards compatibility with handlers that accept no config.
        """
        if not (hasattr(instance, "initialize") and callable(instance.initialize)):
            return
        if self._handler_config is not None:
            await instance.initialize(self._handler_config)  # type: ignore[call-arg]
        else:
            await instance.initialize()  # type: ignore[call-arg]

    async def initialize(self) -> None:
        """Create and initialize all handler instances in the pool.

        Each instance is created via the factory, then ``initialize()``
        is called on it (if the method exists).  When ``handler_config``
        was provided at construction time, it is passed to each instance's
        ``initialize(config)`` call.

        Raises:
            RuntimeError: If pool is already initialized.
        """
        if self._initialized:
            msg = f"HandlerPool for {self.handler_type!r} is already initialized"
            raise RuntimeError(msg)

        logger.info(
            "Initializing handler pool",
            extra={
                "handler_type": self.handler_type,
                "pool_size": self.pool_size,
                "has_handler_config": self._handler_config is not None,
            },
        )

        for i in range(self.pool_size):
            instance = self._factory()
            await self._initialize_instance(instance)
            self._all_instances.append(instance)
            self._pool.put_nowait(instance)
            logger.debug(
                "Handler instance created",
                extra={
                    "handler_type": self.handler_type,
                    "instance_index": i,
                    "instance_class": type(instance).__name__,
                },
            )

        self._initialized = True

    @asynccontextmanager
    async def checkout(self) -> AsyncIterator[ProtocolContainerAware]:
        """Checkout a handler instance from the pool.

        Blocks until an instance is available.  The instance is
        automatically returned to the pool when the context manager exits.

        Yields:
            A handler instance ready for use.

        Raises:
            RuntimeError: If pool is not initialized or is shutting down.
        """
        if not self._initialized:
            msg = f"HandlerPool for {self.handler_type!r} is not initialized"
            raise RuntimeError(msg)
        if self._shutting_down:
            msg = f"HandlerPool for {self.handler_type!r} is shutting down"
            raise RuntimeError(msg)

        start = time.monotonic()
        instance = await self._pool.get()
        wait_ms = (time.monotonic() - start) * 1000
        self._checkout_count += 1
        self._checkout_wait_total_ms += wait_ms

        try:
            yield instance
        finally:
            # Check health before returning to pool; recycle if unhealthy
            if await self._is_healthy(instance):
                self._pool.put_nowait(instance)
            else:
                await self._recycle_instance(instance)
            self._checkin_count += 1

    async def shutdown(self) -> None:
        """Shut down all handler instances in the pool.

        Drains the pool and calls ``shutdown()`` on each instance.
        After shutdown, the pool cannot be used.
        """
        self._shutting_down = True
        logger.info(
            "Shutting down handler pool",
            extra={
                "handler_type": self.handler_type,
                "total_instances": len(self._all_instances),
            },
        )

        for instance in self._all_instances:
            try:
                if hasattr(instance, "shutdown") and callable(instance.shutdown):
                    await instance.shutdown()
            except Exception:
                logger.exception(
                    "Error shutting down pooled handler instance",
                    extra={"handler_type": self.handler_type},
                )

        self._all_instances.clear()
        # Drain the queue
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._initialized = False
        logger.info(
            "Handler pool shutdown complete",
            extra={"handler_type": self.handler_type},
        )

    async def health_check(self) -> dict[str, object]:
        """Return pool health metrics.

        Returns:
            Dict with pool health information including:
            - healthy: Whether the pool is operational
            - handler_type: The handler type this pool manages
            - pool_size: Configured pool size
            - available: Number of idle instances in the pool
            - total_instances: Number of live instances
            - checkout_count: Total checkouts performed
            - checkin_count: Total checkins performed
            - recycle_count: Number of instances recycled
            - avg_checkout_wait_ms: Average wait time for checkout
        """
        avg_wait = (
            self._checkout_wait_total_ms / self._checkout_count
            if self._checkout_count > 0
            else 0.0
        )

        return {
            "healthy": self._initialized and not self._shutting_down,
            "handler_type": self.handler_type,
            "pool_size": self.pool_size,
            "available": self._pool.qsize(),
            "total_instances": len(self._all_instances),
            "checkout_count": self._checkout_count,
            "checkin_count": self._checkin_count,
            "recycle_count": self._recycle_count,
            "avg_checkout_wait_ms": round(avg_wait, 2),
        }

    @property
    def available_count(self) -> int:
        """Number of handler instances currently available in the pool."""
        return self._pool.qsize()

    @property
    def total_instance_count(self) -> int:
        """Total number of handler instances managed by this pool."""
        return len(self._all_instances)

    async def _is_healthy(self, instance: ProtocolContainerAware) -> bool:
        """Check if a handler instance is healthy.

        Args:
            instance: The handler instance to check.

        Returns:
            True if the instance reports healthy, False otherwise.
        """
        try:
            if hasattr(instance, "health_check") and callable(instance.health_check):
                result = await instance.health_check()
                if isinstance(result, dict):
                    return bool(result.get("healthy", True))
            # If no health_check, assume healthy
            return True
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Handler instance health check failed",
                extra={"handler_type": self.handler_type},
            )
            return False

    async def _recycle_instance(self, instance: ProtocolContainerAware) -> None:
        """Replace an unhealthy instance with a fresh one.

        Shuts down the old instance, creates and initializes a new one,
        and adds it to the pool.

        Args:
            instance: The unhealthy instance to replace.
        """
        self._recycle_count += 1
        logger.info(
            "Recycling unhealthy handler instance",
            extra={
                "handler_type": self.handler_type,
                "recycle_count": self._recycle_count,
            },
        )

        # Remove old instance from tracking
        try:
            self._all_instances.remove(instance)
        except ValueError:
            pass  # Already removed

        # Shut down old instance
        try:
            if hasattr(instance, "shutdown") and callable(instance.shutdown):
                await instance.shutdown()
        except Exception:
            logger.exception(
                "Error shutting down recycled handler instance",
                extra={"handler_type": self.handler_type},
            )

        # Create and initialize replacement
        try:
            new_instance = self._factory()
            await self._initialize_instance(new_instance)
            self._all_instances.append(new_instance)
            self._pool.put_nowait(new_instance)
        except Exception:
            logger.exception(
                "Failed to create replacement handler instance",
                extra={
                    "handler_type": self.handler_type,
                    "pool_size": self.pool_size,
                    "remaining_instances": len(self._all_instances),
                },
            )
            # Pool is now degraded (fewer instances than pool_size)
