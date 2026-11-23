# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Longest event-loop gap per boot phase, sampled from boot step 0 on (OMN-26561).

:class:`ModelBootClock` is the process-wide clock the boot summary line reads. The
first phase entered starts a sampler that schedules a short sleep and records how late
the event loop wakes it; each gap is attributed to the phase running at the time. A
blocking call that holds the loop for 2 s inside boot step 0 therefore reads as a gap
of about 2 s in ``boot_step0``. The clock stops when the summary is written.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Awaitable, Iterator
from typing import Final

from pydantic import BaseModel, ConfigDict

OTHER_PHASE: Final = "other"
"""Where a gap lands while no named phase is running."""

SAMPLE_INTERVAL_SECONDS: Final = 0.05
"""Length of the scheduled sleep; the gap is the time between two wake-ups."""


class BootClockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: list[str] = []
        self.gaps: dict[str, float] = {}
        self.last_wake: float | None = None
        self.task: asyncio.Task[None] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.done = False
        self.kernel_started = time.monotonic()

    def record(self, now: float) -> None:
        if self.last_wake is None:
            return
        name = self.active[-1] if self.active else OTHER_PHASE
        self.gaps[name] = max(self.gaps.get(name, 0.0), now - self.last_wake)
        self.last_wake = now

    async def sample(self) -> None:
        while True:
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
            with self.lock:
                self.record(time.monotonic())

    def ensure_sampler(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self.loop is loop and self.task is not None and not self.task.done():
            return
        self.loop = loop
        self.last_wake = time.monotonic()
        self.task = loop.create_task(self.sample(), name="boot-summary-gap-sampler")

    def stop_sampler(self) -> None:
        task, self.task = self.task, None
        if task is not None:
            with contextlib.suppress(RuntimeError):  # its loop already closed
                task.cancel()


_STATE = BootClockState()


class ModelBootClock(BaseModel):
    """Process-wide boot phase clock; every method is a classmethod on one shared state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def begin(cls, name: str) -> float:
        """Start phase ``name`` and return the wall time, for a caller that times it too."""
        with _STATE.lock:
            if not _STATE.done:
                _STATE.ensure_sampler()
                _STATE.record(time.monotonic())
                _STATE.active.append(name)
        return time.time()

    @classmethod
    def end(cls, name: str, started: float) -> float:
        """End phase ``name``; return the seconds since ``started`` (a ``begin`` result)."""
        with _STATE.lock:
            _STATE.record(time.monotonic())
            for index in range(len(_STATE.active) - 1, -1, -1):
                if _STATE.active[index] == name:
                    del _STATE.active[index]
                    break
        return time.time() - started

    @classmethod
    @contextlib.contextmanager
    def phase(cls, name: str) -> Iterator[None]:
        """Attribute event-loop gaps to ``name`` while the block runs."""
        started = cls.begin(name)
        try:
            yield
        finally:
            cls.end(name, started)

    @classmethod
    async def run[T](cls, name: str, awaitable: Awaitable[T]) -> T:
        """Await ``awaitable`` as phase ``name``."""
        with cls.phase(name):
            return await awaitable

    @classmethod
    def note_kernel_start(cls) -> float:
        """Mark the kernel's start for the summary's seconds count; return the wall time."""
        _STATE.kernel_started = time.monotonic()
        return time.time()

    @classmethod
    def seconds_since_kernel_start(cls) -> float:
        return time.monotonic() - _STATE.kernel_started

    @classmethod
    def finish(cls) -> dict[str, float] | None:
        """Stop sampling and return the longest gap per phase; ``None`` when already finished."""
        with _STATE.lock:
            if _STATE.done:
                return None
            _STATE.done = True
            _STATE.record(time.monotonic())
            _STATE.stop_sampler()
            return dict(_STATE.gaps)

    @classmethod
    def reset(cls) -> None:
        """Forget everything and stop the sampler (tests)."""
        with _STATE.lock:
            _STATE.stop_sampler()
            _STATE.active.clear()
            _STATE.gaps.clear()
            _STATE.last_wake = None
            _STATE.loop = None
            _STATE.done = False


__all__ = ["OTHER_PHASE", "SAMPLE_INTERVAL_SECONDS", "ModelBootClock"]
