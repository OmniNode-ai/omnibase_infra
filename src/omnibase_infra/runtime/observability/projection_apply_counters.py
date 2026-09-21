# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-scoped per-projection apply accumulator (OMN-18910).

What this is
------------
A plain in-process integer accumulator keyed by ``(projection, topic)``, closed
once per health-monitor cycle into a ring of windows that same monitor reads. It is the sibling of
:mod:`omnibase_infra.runtime.observability.consumer_flow_counters` and copies
its constraints deliberately: no daemon, no poller, no ``/metrics`` endpoint, no
schedule of its own, and no clock — every timestamp is injected by the caller
that already has one.

Why it is a second accumulator and not a field on the first
-----------------------------------------------------------
``ConsumerFlowCounters`` fills ``ModelConsumerFlowDelta``, which rides the node
heartbeat onto a topic another repository's projection consumes. Adding
``rows_upserted`` there would make a cross-repository wire change, consumer
first, out of a reading that never leaves this process. These counts are read
by ``ServiceRuntimeHealthMonitor`` in the same process that writes them, so
they stay here. The verdict derived from them IS published, on the health event
that already exists.

What a zero row means
---------------------
The same thing it means next door: ``register()`` is called for every
projection dispatch the kernel wires, so a projection that took nothing still
emits a row every window. "Alive, took nothing" and "not observed" are
different facts, and only the first is a zero.

The drop gauge is a gauge
-------------------------
``record_drop_total`` sets a cumulative lifetime value rather than adding to
one, and it SURVIVES the window reset, because the graded fact is whether it
rises across windows. Resetting it every window would turn every reading into a
first reading and the accumulation test could never fire.

Related Tickets:
    - OMN-18910: this module (epic OMN-18906 AC-4)
    - OMN-16777 / OMN-16994: the sibling accumulator and its retained ring
    - OMN-18992: the ordering-guard refusal this counts separately from a write
"""

from __future__ import annotations

import logging
import threading
from collections import deque

from omnibase_infra.models.observability.model_projection_apply_delta import (
    ModelProjectionApplyDelta,
)

logger = logging.getLogger(__name__)

#: How many CLOSED apply windows are retained for the health monitor to read.
#:
#: Derivation, not a magic number. ``ServiceRuntimeHealthMonitor`` closes one
#: window per cycle and evaluates over the ring, so at its 300 s default this
#: is fifty minutes of history. Divergence needs one window; drop accumulation
#: needs ``DELTA_DROP_MIN_RISING_WINDOWS + 1``, so the shortest honest
#: detection is three cycles. Ten leaves headroom for a lane on a shorter
#: interval without letting a single stale reading dominate the series.
#:
#: The monitor is the only closer on purpose. Closing on the heartbeat instead
#: would couple these dimensions to introspection being enabled and to this
#: node holding flow-window carriage, and on a lane where neither holds no
#: window would ever close -- leaving both dimensions permanently on their
#: unobserved branch, which is a check that cannot pass.
RETAINED_APPLY_WINDOW_COUNT: int = 10


class ProjectionApplyCounters:
    """Per-``(projection, topic)`` apply counters for one process.

    Thread-safe. Every mutation is O(1) under a short lock; ``close_window()``
    swaps the accumulator maps and appends an immutable window to the ring.
    """

    def __init__(self, *, maxlen: int = RETAINED_APPLY_WINDOW_COUNT) -> None:
        self._lock = threading.Lock()
        self._registered: set[tuple[str, str]] = set()
        self._consumed: dict[tuple[str, str], int] = {}
        self._upserted: dict[tuple[str, str], int] = {}
        self._refused: dict[tuple[str, str], int] = {}
        # Cumulative, and deliberately NOT reset by ``_reset_unlocked``.
        self._dropped_total: dict[tuple[str, str], int] = {}
        self._windows: deque[tuple[ModelProjectionApplyDelta, ...]] = deque(
            maxlen=maxlen
        )

    # ---------------------------------------------------------------- register

    def register(self, projection: str, topic: str) -> None:
        """Declare a projection dispatch so it emits a row every window."""
        if not projection or not topic:
            return
        with self._lock:
            self._registered.add((projection, topic))

    def registered_projections(self) -> tuple[str, ...]:
        """Every projection wired for dispatch in this process, sorted.

        The health monitor's scope set. Taken from registration rather than
        from observed traffic on purpose: a projection that registered and then
        stopped taking anything must stay in scope, and deriving scope from
        traffic is how a consumer that died drops out of its own dimension.
        """
        with self._lock:
            return tuple(sorted({projection for projection, _ in self._registered}))

    # ------------------------------------------------------------------ record

    def record_apply(
        self,
        projection: str,
        topic: str,
        *,
        consumed: int = 1,
        upserted: int = 0,
        refused_by_guard: int = 0,
    ) -> None:
        """Record one projection dispatch outcome.

        ``consumed`` counts the dispatch, ``upserted`` the rows the handler
        reported persisting, and ``refused_by_guard`` the zero-row writes the
        handler attributed to its ordering guard (OMN-18992) — correct
        behaviour, kept apart from both a write and a silent nothing.
        """
        key = (projection, topic)
        if not projection or not topic:
            return
        with self._lock:
            self._registered.add(key)
            if consumed > 0:
                self._consumed[key] = self._consumed.get(key, 0) + consumed
            if upserted > 0:
                self._upserted[key] = self._upserted.get(key, 0) + upserted
            if refused_by_guard > 0:
                self._refused[key] = self._refused.get(key, 0) + refused_by_guard

    def record_drop_total(self, projection: str, topic: str, total: int) -> None:
        """Set the cumulative discarded-delta gauge for this projection.

        A SET, not an add: callers hand over a lifetime total they already
        hold, and two callers reporting the same total must not double it.
        """
        if not projection or not topic or total < 0:
            return
        key = (projection, topic)
        with self._lock:
            self._registered.add(key)
            self._dropped_total[key] = total

    # ------------------------------------------------------------------- close

    def close_window(self) -> tuple[ModelProjectionApplyDelta, ...]:
        """Close the open window, retain it, and reset the per-window counts.

        Returns the closed window so a caller can assert on it. The cumulative
        drop gauge is carried forward rather than reset; everything else is
        zeroed.
        """
        with self._lock:
            window = tuple(
                ModelProjectionApplyDelta(
                    projection=projection,
                    topic=topic,
                    consumed=self._consumed.get((projection, topic), 0),
                    upserted=self._upserted.get((projection, topic), 0),
                    refused_by_guard=self._refused.get((projection, topic), 0),
                    deltas_dropped_total=self._dropped_total.get(
                        (projection, topic), 0
                    ),
                )
                for projection, topic in sorted(self._registered)
            )
            self._reset_unlocked()
            self._windows.append(window)
            return window

    def retained_windows(self) -> tuple[tuple[ModelProjectionApplyDelta, ...], ...]:
        """The retained closed windows, OLDEST FIRST.

        Order is load-bearing for the drop dimension, which reads monotonicity
        across the series. Empty is a distinct state from "nothing flowed": it
        means no window has closed yet in this process, which the health
        surface must treat as UNKNOWN and never as proven-clean.
        """
        with self._lock:
            return tuple(self._windows)

    def _reset_unlocked(self) -> None:
        """Zero the per-window counts. Registrations and the gauge survive."""
        self._consumed = {}
        self._upserted = {}
        self._refused = {}


# One lazily-built accumulator per process, held in a single-entry mapping so
# the accessor mutates a container rather than rebinding a module global. The
# two seams that share it — the projection dispatch arm and the heartbeat tick —
# MUST observe the same instance, because a double-construction race would split
# a window in half and under-report both halves, which is the failure mode this
# module exists to detect. Hence the lock, not a bare dict read. Mirrors
# ``consumer_flow_counters`` exactly.
_SLOT_KEY = "process"
_SLOT_LOCK = threading.Lock()
_SLOT: dict[str, ProjectionApplyCounters] = {}


def get_projection_apply_counters() -> ProjectionApplyCounters:
    """Return the process-scoped apply accumulator, creating it on first use."""
    with _SLOT_LOCK:
        counters = _SLOT.get(_SLOT_KEY)
        if counters is None:
            counters = ProjectionApplyCounters()
            _SLOT[_SLOT_KEY] = counters
        return counters


def reset_projection_apply_counters_for_test() -> None:
    """Drop the process accumulator. Tests only.

    Named for what it is so no production path reaches for it by accident: a
    production caller that reset these counters would erase the very window the
    health monitor is about to grade.
    """
    with _SLOT_LOCK:
        _SLOT.pop(_SLOT_KEY, None)


__all__ = [
    "RETAINED_APPLY_WINDOW_COUNT",
    "ProjectionApplyCounters",
    "get_projection_apply_counters",
    "reset_projection_apply_counters_for_test",
]
