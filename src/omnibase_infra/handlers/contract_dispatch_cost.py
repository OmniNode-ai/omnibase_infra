# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-contract dispatch cost counter (OMN-26250).

Every dispatcher the runtime attaches for a contract is wrapped by
``measure_contract_dispatch``. The wrapper times each dispatch with the
monotonic clock, and the entry-method offload (OMN-24491) reports how much of
that time a plain-function handler spent on a worker thread. The remainder is
the time on the event loop. The bytes are the wire size of the bus message the
dispatch consumed, which the event bus wiring records for the dispatch.

``close_window`` returns one reading per attached contract and zeroes the
counters, so the member publishes a number for every attached contract every
window, including a contract that dispatched nothing. The counter holds no bus
access; the loop-reading publisher publishes the readings with the loop
reading.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime, timedelta

from omnibase_infra.models.health.model_contract_cost_reading import (
    ModelContractCostReading,
)

_MILLISECOND = timedelta(milliseconds=1)


def _to_ms(seconds: float) -> float:
    return timedelta(seconds=seconds) / _MILLISECOND


class ContractTotals:
    """Mutable per-window accumulator of one contract."""

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.on_loop_seconds = 0.0
        self.off_loop_seconds = 0.0
        self.payload_bytes = 0


class DispatchTimer:
    """Mutable per-dispatch record of the time spent on a worker thread."""

    def __init__(self) -> None:
        self.off_loop_seconds = 0.0


_current_timer: ContextVar[DispatchTimer | None] = ContextVar(
    "contract_dispatch_timer", default=None
)
_ingress_bytes: ContextVar[int] = ContextVar("contract_ingress_bytes", default=0)


def record_ingress_bytes(size: int) -> None:
    """Record the wire size of the bus message about to be dispatched."""
    _ingress_bytes.set(size)


async def run_in_worker_timed[T](fn: Callable[[object], T], arg: object) -> T:
    """Run ``fn(arg)`` on a worker thread and charge its run time off the loop.

    The clock runs inside the worker, so the time spent waiting for a free
    worker is not charged to the handler.
    """
    timer = _current_timer.get()

    def _run() -> T:
        started = time.perf_counter()
        try:
            return fn(arg)
        finally:
            if timer is not None:
                timer.off_loop_seconds += time.perf_counter() - started

    return await asyncio.to_thread(_run)


class ContractDispatchCostCounter:
    """Counts dispatch time, count and bytes per attached contract."""

    def __init__(self) -> None:
        self._totals: dict[str, ContractTotals] = {}

    def register(self, contract_label: str) -> None:
        """Make the contract report from the next window on, even with no dispatch."""
        self._totals.setdefault(contract_label, ContractTotals())

    def record(
        self,
        contract_label: str,
        *,
        wall_seconds: float,
        off_loop_seconds: float,
        payload_bytes: int,
    ) -> None:
        """Add one finished dispatch to its contract's window totals."""
        totals = self._totals.setdefault(contract_label, ContractTotals())
        totals.dispatch_count += 1
        totals.off_loop_seconds += off_loop_seconds
        totals.on_loop_seconds += max(wall_seconds - off_loop_seconds, 0.0)
        totals.payload_bytes += payload_bytes

    def close_window(
        self, window_start: datetime, window_end: datetime
    ) -> tuple[ModelContractCostReading, ...]:
        """Return one reading per attached contract, sorted by name, and zero them."""
        readings = tuple(
            ModelContractCostReading(
                contract_label=name,
                window_start=window_start,
                window_end=window_end,
                dispatch_count=totals.dispatch_count,
                on_loop_wall_ms=_to_ms(totals.on_loop_seconds),
                off_loop_wall_ms=_to_ms(totals.off_loop_seconds),
                payload_bytes=totals.payload_bytes,
            )
            for name, totals in sorted(self._totals.items())
        )
        self._totals = {name: ContractTotals() for name in self._totals}
        return readings


_PROCESS_COUNTER = ContractDispatchCostCounter()


def get_process_contract_cost_counter() -> ContractDispatchCostCounter:
    """Return the one counter of this process."""
    return _PROCESS_COUNTER


def attach_contract_dispatcher[**P, T](
    owner_contract_label: str | None,
    dispatcher: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T]]:
    """Return the dispatcher the engine registers for an attached contract.

    A dispatcher registered with an owner contract is registered with the
    counter, so it reads every window, and is wrapped to be measured. A manual
    registration with no owner is not an attached contract and is returned
    unchanged.
    """
    if not owner_contract_label:
        return dispatcher
    _PROCESS_COUNTER.register(owner_contract_label)
    return measure_contract_dispatch(owner_contract_label, dispatcher)


def measure_contract_dispatch[**P, T](
    contract_label: str,
    dispatcher: Callable[P, Awaitable[T]],
    counter: ContractDispatchCostCounter | None = None,
) -> Callable[P, Awaitable[T]]:
    """Wrap a dispatcher so every dispatch is charged to ``contract_label``.

    A dispatch that raises is charged too: the time it held the member is real.

    The wrapper forwards every argument and carries the dispatcher's own
    signature (``functools.wraps`` sets ``__wrapped__``, which
    ``inspect.signature`` follows). The dispatch engine probes that signature
    once at registration to decide whether to pass a keyword-only ``delivery``
    and a ``context``; a wrapper that took only the envelope hid both.
    """
    target = counter if counter is not None else _PROCESS_COUNTER

    @functools.wraps(dispatcher)
    async def _measured(*args: P.args, **kwargs: P.kwargs) -> T:
        timer = DispatchTimer()
        token = _current_timer.set(timer)
        started = time.perf_counter()
        try:
            return await dispatcher(*args, **kwargs)
        finally:
            wall = time.perf_counter() - started
            _current_timer.reset(token)
            target.record(
                contract_label,
                wall_seconds=wall,
                off_loop_seconds=timer.off_loop_seconds,
                payload_bytes=_ingress_bytes.get(),
            )

    return _measured


__all__: list[str] = [
    "ContractDispatchCostCounter",
    "attach_contract_dispatcher",
    "get_process_contract_cost_counter",
    "measure_contract_dispatch",
    "record_ingress_bytes",
    "run_in_worker_timed",
]
