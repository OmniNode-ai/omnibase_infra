# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event-loop lag sampler and window reading handler (OMN-26249).

The sampler schedules a short sleep and records how late the event loop woke
it. A handler that blocks the loop for 0.6 s makes the wake-up late by close to
0.6 s, so a stalled loop reads as a number. ``handle`` closes the window that
the sampler and the tick observations filled and returns the reading the member
publishes; the handler never touches the bus, the member publishes.

Tick lag is consume time minus the tick's ``scheduled_at`` for the newest tick
consumed in the window. Only a member that subscribes to the tick has one: the
event bus wiring marks the subscription when it creates the callback for the
tick topic and records every tick that callback receives.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.health.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)
from omnibase_infra.models.health.model_runtime_loop_window_request import (
    ModelRuntimeLoopWindowRequest,
)
from omnibase_infra.protocols.protocol_dispatch_deadline_source import (
    ProtocolDispatchDeadlineSource,
)
from omnibase_infra.runtime.runtime_profile import resolve_runtime_profile_name
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_RUNTIME_TICK

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.models import ModelNodeIdentity
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0
"""Length of one reading window: one reading is published per window."""

_READING_TOPIC = EnumOmnibaseInfraTopic.EVT_RUNTIME_LOOP_READING_RECORDED_V1.value

DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.05
"""Length of the sampler's scheduled sleep; the loop gap is how late it wakes."""

NOT_A_TICK_SUBSCRIBER_REASON = "member does not subscribe to the runtime tick topic"
NO_TICK_IN_WINDOW_REASON = (
    "member subscribes to the runtime tick but consumed none in the window"
)


class RuntimeLoopLagSampler:
    """Samples event-loop lag and tick lag, and closes reading windows."""

    def __init__(
        self, sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self._interval = sample_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._expected_wake: float | None = None
        self._max_gap_seconds = 0.0
        self._window_start = datetime.now(UTC)
        self._tick_subscriber = False
        self._newest_tick_lag_ms: float | None = None

    @property
    def is_running(self) -> bool:
        """Whether the sampling task is alive."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start sampling and open the first window."""
        if self.is_running:
            return
        self._window_start = datetime.now(UTC)
        self._max_gap_seconds = 0.0
        self._task = asyncio.get_running_loop().create_task(self._sample_loop())

    async def stop(self) -> None:
        """Stop sampling."""
        task, self._task = self._task, None
        self._expected_wake = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def mark_tick_subscriber(self) -> None:
        """Record that this member subscribes to the runtime tick topic."""
        self._tick_subscriber = True

    def record_tick_consumed(
        self, scheduled_at: datetime, consumed_at: datetime
    ) -> None:
        """Record a consumed tick; the newest one in a window is the one reported."""
        self._tick_subscriber = True
        self._newest_tick_lag_ms = (consumed_at - scheduled_at).total_seconds() * 1000.0

    def observe_subscription(self, topic: str) -> None:
        """Mark the member a tick subscriber when it wires a tick-topic callback."""
        if topic.endswith(SUFFIX_RUNTIME_TICK):
            self._tick_subscriber = True

    def observe_message(self, topic: str, payload: object) -> None:
        """Record a message the member consumed; only a runtime tick counts.

        A tick payload without a readable timezone-aware ``scheduled_at`` is
        skipped, so the reading says no tick was consumed rather than carrying
        a lag that was guessed.
        """
        if not topic.endswith(SUFFIX_RUNTIME_TICK):
            return
        raw = (
            payload.get("scheduled_at")
            if isinstance(payload, Mapping)
            else getattr(payload, "scheduled_at", None)
        )
        try:
            scheduled_at = datetime.fromisoformat(raw) if isinstance(raw, str) else raw
        except ValueError:
            return
        if isinstance(scheduled_at, datetime) and scheduled_at.tzinfo is not None:
            self.record_tick_consumed(scheduled_at, datetime.now(UTC))

    def close_window(
        self, request: ModelRuntimeLoopWindowRequest
    ) -> ModelRuntimeLoopReading:
        """Close the open window and return its reading."""
        # A block that began before the window closed has not woken the sampler
        # yet; its lateness so far still belongs to this window.
        gap = self._max_gap_seconds
        if self._expected_wake is not None:
            gap = max(gap, asyncio.get_running_loop().time() - self._expected_wake)
        if not self._tick_subscriber:
            tick_lag_ms, reason = None, NOT_A_TICK_SUBSCRIBER_REASON
        elif self._newest_tick_lag_ms is None:
            tick_lag_ms, reason = None, NO_TICK_IN_WINDOW_REASON
        else:
            tick_lag_ms, reason = self._newest_tick_lag_ms, None
        reading = ModelRuntimeLoopReading(
            correlation_id=request.correlation_id,
            runtime_member=request.runtime_member,
            host=request.host,
            window_start=self._window_start,
            window_end=request.window_end,
            max_loop_gap_ms=max(gap, 0.0) * 1000.0,
            tick_subscriber=self._tick_subscriber,
            tick_lag_ms=tick_lag_ms,
            tick_lag_not_applicable_reason=reason,
            abandoned_dispatches=request.abandoned_dispatches,
        )
        self._window_start = request.window_end
        self._max_gap_seconds = 0.0
        self._newest_tick_lag_ms = None
        return reading

    async def _sample_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._expected_wake = loop.time() + self._interval
            await asyncio.sleep(self._interval)
            late_by = loop.time() - self._expected_wake
            self._expected_wake = None
            self._max_gap_seconds = max(self._max_gap_seconds, late_by)


_PROCESS_SAMPLER = RuntimeLoopLagSampler()


def get_process_loop_lag_sampler() -> RuntimeLoopLagSampler:
    """Return the one sampler of this process, which the tick wiring reports to."""
    return _PROCESS_SAMPLER


class HandlerRuntimeLoopLagSampler:
    """Closes the process sampler's open window into a reading.

    The handler holds no bus access; the runtime host process publishes the
    reading this returns.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: compute handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral category: reads the clock and in-process state, no I/O."""
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    def handle(self, request: ModelRuntimeLoopWindowRequest) -> ModelRuntimeLoopReading:
        """Close the open window of this process's sampler and return its reading."""
        return get_process_loop_lag_sampler().close_window(request)


class RuntimeLoopReadingPublisher:
    """Publishes the process's loop reading once per window.

    The runtime host process starts it once the runtime is up. The publishing
    loop runs on the event loop it measures, so a stalled loop delays the
    reading it would have published, and the late reading carries the gap that
    stalled it: the stall is read as a number after the fact.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._handler: HandlerRuntimeLoopLagSampler | None = None

    async def start(
        self,
        event_bus: ProtocolEventBusLike,
        node_identity: ModelNodeIdentity,
        container: ModelONEXContainer,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        """Start the sampler and the per-window publishing task."""
        if self._task is not None and not self._task.done():
            return
        await get_process_loop_lag_sampler().start()
        self._handler = HandlerRuntimeLoopLagSampler(container)
        self._task = asyncio.create_task(
            self._publish_every_window(event_bus, node_identity, window_seconds),
            name="runtime-loop-reading",
        )

    async def stop(self) -> None:
        """Stop publishing and sampling."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await get_process_loop_lag_sampler().stop()

    async def publish_reading(
        self, event_bus: ProtocolEventBusLike, node_identity: ModelNodeIdentity
    ) -> None:
        """Close the open window and publish its reading."""
        if self._handler is None:
            raise RuntimeError("publish_reading called before start")
        abandoned = (
            event_bus.dispatch_deadline_status().orphaned_dispatches
            if isinstance(event_bus, ProtocolDispatchDeadlineSource)
            else 0
        )
        reading = self._handler.handle(
            ModelRuntimeLoopWindowRequest(
                correlation_id=uuid4(),
                runtime_member=(
                    f"{node_identity.env}.{node_identity.service}"
                    f".{node_identity.node_name}.{resolve_runtime_profile_name()}"
                ),
                host=socket.gethostname(),
                window_end=datetime.now(UTC),
                abandoned_dispatches=abandoned,
            )
        )
        try:
            await event_bus.publish_envelope(
                reading.model_dump(mode="json"), _READING_TOPIC
            )
        except (RuntimeHostError, OSError, TimeoutError):
            logger.warning(
                "Failed to publish runtime loop reading (correlation_id=%s)",
                reading.correlation_id,
                exc_info=True,
            )

    async def _publish_every_window(
        self,
        event_bus: ProtocolEventBusLike,
        node_identity: ModelNodeIdentity,
        window_seconds: float,
    ) -> None:
        """Publish on a fixed schedule so windows do not drift apart."""
        loop = asyncio.get_running_loop()
        due = loop.time() + window_seconds
        while True:
            await asyncio.sleep(max(due - loop.time(), 0.0))
            due += window_seconds
            await self.publish_reading(event_bus, node_identity)


__all__: list[str] = [
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "WINDOW_SECONDS",
    "NOT_A_TICK_SUBSCRIBER_REASON",
    "NO_TICK_IN_WINDOW_REASON",
    "HandlerRuntimeLoopLagSampler",
    "RuntimeLoopLagSampler",
    "RuntimeLoopReadingPublisher",
    "get_process_loop_lag_sampler",
]
