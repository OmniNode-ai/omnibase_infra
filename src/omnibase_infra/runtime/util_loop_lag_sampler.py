# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event-loop lag sampler and per-window reading publisher (OMN-26249).

A runtime member that is overloaded looks alive to every probe that runs on its
own event loop: the probe is late, never wrong. The sampler measures the loop
from inside the only way that works, by scheduling a short sleep and recording
how late it returns. A handler that blocks the loop for 0.6 s makes the next
sample late by about 0.6 s; an idle loop returns within a few milliseconds.

Each window the publisher turns what the sampler saw into one
``ModelRuntimeLoopReading`` and publishes it under the member's identity:

- the largest gap of the window, in milliseconds;
- the tick lag (consume time minus the newest tick's ``scheduled_at``), only
  for a member that subscribes to the runtime tick. A member that does not
  reports ``not_a_tick_subscriber``; a subscriber that has not yet consumed a
  tick reports ``no_tick_consumed_yet``, which a verdict must not read as
  not-applicable;
- the count the bus already keeps of dispatches abandoned at their deadline
  (OMN-19355).

Windows are contiguous: each starts where the previous one ended, so a gap in
the published windows means a reading was lost, never that time went unmeasured.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable
from datetime import UTC, datetime

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.operational_timeouts import TIMEOUTS
from omnibase_infra.protocols.protocol_dispatch_deadline_source import (
    ProtocolDispatchDeadlineSource,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.runtime.models.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
    TickLagAbsentReason,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry
from omnibase_infra.utils.correlation import generate_correlation_id

logger = logging.getLogger(__name__)

_MS_PER_SECOND: float = 1000.0


class LoopLagSampler:
    """Records how late a short scheduled sleep returns, and the newest tick lag."""

    def __init__(
        self,
        interval_seconds: float = TIMEOUTS.loop_lag_sample_interval_seconds,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        self._interval = interval_seconds
        self._max_gap_seconds = 0.0
        self._task: asyncio.Task[None] | None = None
        self._tick_subscriber = False
        self._newest_tick_lag_ms: float | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start sampling on the running loop. Idempotent."""
        if self.is_running:
            return
        self._task = asyncio.create_task(self._sample(), name="loop-lag-sampler")

    async def stop(self) -> None:
        """Stop sampling. Idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _sample(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(self._interval)
            gap = loop.time() - started - self._interval
            if gap > self._max_gap_seconds:
                self._max_gap_seconds = gap

    def drain_max_gap_ms(self) -> float:
        """Return the largest gap since the last drain, in ms, and reset it."""
        gap_ms = max(0.0, self._max_gap_seconds) * _MS_PER_SECOND
        self._max_gap_seconds = 0.0
        return gap_ms

    def mark_tick_subscriber(self) -> None:
        """Record that this member subscribes to the runtime tick."""
        self._tick_subscriber = True

    def record_tick(self, scheduled_at: datetime, consumed_at: datetime) -> None:
        """Record the lag of a consumed tick; the newest tick wins.

        A negative lag (the scheduling host's clock ahead of this one) reads as
        zero rather than as a negative number.
        """
        self._tick_subscriber = True
        lag_seconds = (consumed_at - scheduled_at).total_seconds()
        self._newest_tick_lag_ms = max(0.0, lag_seconds) * _MS_PER_SECOND

    def tick_lag(self) -> tuple[float | None, TickLagAbsentReason | None]:
        """Return ``(tick_lag_ms, None)`` or ``(None, reason)``, never both set."""
        if not self._tick_subscriber:
            return None, "not_a_tick_subscriber"
        if self._newest_tick_lag_ms is None:
            return None, "no_tick_consumed_yet"
        return self._newest_tick_lag_ms, None


class RuntimeLoopReadingPublisher:
    """Publishes one ``ModelRuntimeLoopReading`` per window under a member id."""

    def __init__(
        self,
        *,
        sampler: LoopLagSampler,
        event_bus: ProtocolEventBusLike,
        member_id: str,
        host: str,
        window_seconds: float = TIMEOUTS.loop_reading_window_seconds,
        topic_registry: ServiceTopicRegistry | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._sampler = sampler
        self._event_bus = event_bus
        self._member_id = member_id
        self._host = host
        self._window_seconds = window_seconds
        self._now = now
        registry = topic_registry or ServiceTopicRegistry.from_defaults()
        self._topic = registry.resolve(topic_keys.RUNTIME_LOOP_READING_RECORDED)
        self._task: asyncio.Task[None] | None = None
        self._window_start: datetime | None = None
        self._last_reading: ModelRuntimeLoopReading | None = None

    @property
    def last_reading(self) -> ModelRuntimeLoopReading | None:
        """The newest reading built, published or not (health payload source)."""
        return self._last_reading

    async def start(self) -> None:
        """Start the sampler and the window loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        await self._sampler.start()
        self._window_start = self._now()
        self._sampler.drain_max_gap_ms()
        self._task = asyncio.create_task(self._loop(), name="runtime-loop-reading")

    async def stop(self) -> None:
        """Stop the window loop and the sampler. Idempotent."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._sampler.stop()

    def build_reading(self) -> ModelRuntimeLoopReading:
        """Close the current window into a reading and open the next one."""
        window_end = self._now()
        window_start = self._window_start or window_end
        tick_lag_ms, absent_reason = self._sampler.tick_lag()
        reading = ModelRuntimeLoopReading(
            member_id=self._member_id,
            host=self._host,
            window_start=window_start,
            window_end=max(window_end, window_start),
            max_gap_ms=self._sampler.drain_max_gap_ms(),
            tick_lag_ms=tick_lag_ms,
            tick_lag_absent_reason=absent_reason,
            abandoned_count=self._abandoned_count(),
        )
        self._window_start = window_end
        self._last_reading = reading
        return reading

    def _abandoned_count(self) -> int:
        # A transport with no consume loop (the in-memory bus) has nothing to
        # abandon, so zero is the truthful reading there.
        if isinstance(self._event_bus, ProtocolDispatchDeadlineSource):
            return self._event_bus.dispatch_deadline_status().orphaned_dispatches
        return 0

    async def publish_window(self) -> ModelRuntimeLoopReading:
        """Build this window's reading and publish it; a failed publish is logged."""
        reading = self.build_reading()
        envelope: ModelEventEnvelope[ModelRuntimeLoopReading] = ModelEventEnvelope(
            payload=reading,
            correlation_id=generate_correlation_id(),
            event_type="runtime-loop-reading-recorded",
            source_tool="RuntimeLoopReadingPublisher",
            tenant_id=None,
        )
        try:
            await self._event_bus.publish_envelope(
                envelope=envelope,
                topic=self._topic,
                key=self._member_id.encode(),
            )
        except Exception:  # noqa: BLE001 -- a lost reading must not stop the loop
            logger.warning(
                "runtime loop reading for member %s window ending %s was not published",
                self._member_id,
                reading.window_end.isoformat(),
                exc_info=True,
            )
        return reading

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._window_seconds)
            try:
                await self.publish_window()
            except Exception:  # noqa: BLE001 -- never crash the window loop
                logger.warning("runtime loop reading window failed", exc_info=True)


def default_member_id(env: str, service: str, node_name: str, version: str, pid: int) -> str:
    """The member id a runtime host publishes under: identity plus process id."""
    return f"{env}.{service}.{node_name}.{version}:{pid}"


def default_host() -> str:
    """The host name this process runs on."""
    return socket.gethostname() or "unknown-host"


__all__: list[str] = [
    "LoopLagSampler",
    "RuntimeLoopReadingPublisher",
    "default_host",
    "default_member_id",
]
