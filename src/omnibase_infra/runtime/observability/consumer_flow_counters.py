# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-scoped per-consumer throughput accumulator (OMN-16777).

What this is
------------
A plain in-process integer accumulator, keyed by ``(consumer_group, topic)``,
drained once per heartbeat tick into a ``ModelNodeFlowWindow`` that rides the
heartbeat the runtime **already** emits.

What this deliberately is NOT
-----------------------------
Not a daemon, not a poller, not a scraper, not a ``/metrics`` endpoint, not a
``Plugin*`` subclass, and it owns no schedule of its own.  It has no loop and no
clock: every timestamp is injected by the caller that already has one.  Those
are hard constraints from OMN-16777 / epic OMN-16776, not preferences — a
separate poller would keep cheerfully reporting on a runtime that is already
dead, whereas a heartbeat that stops arriving *is itself* the signal.

Why it is process-scoped
------------------------
The two seams that must agree are the auto-wiring subscription callback (which
knows ``consumer_group`` and ``topic``) and the introspection heartbeat tick
(which knows ``node_id`` and the clock).  They share no object.  A process-scoped
accumulator is the honest seam, and it mirrors the module-scoped
``_BOUNDARY_MESSAGE_LOST_COUNTER`` already living in ``handler_wiring``.

Carriage
--------
Several objects in one process may publish heartbeats.  The FIRST node to drain
claims carriage and every later drain from a different ``node_id`` returns
``None`` rather than stealing a partial window.  This is a labelling choice, not
a correctness one: the emitted rows are keyed by ``(consumer_group, topic)``, so
which node's heartbeat carries them does not change what they say.

Zero rows are facts
-------------------
``register()`` is called for every subscription at wiring time, and every
registered key emits a row every window even when nothing moved.  A zero row
means "alive, took nothing" — that is how ``IDLE`` is proven.  A MISSING row
means "we do not know", which is a different fact; the projection materializes a
sequence gap as ``UNKNOWN`` and never as zero traffic (OMN-16777 AC5).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING

from omnibase_infra.models.observability import (
    ModelConsumerFlowDelta,
    ModelNodeFlowWindow,
    ModelTopicProduceDelta,
)

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

# The (consumer_group, topic) of the message currently in flight on THIS task.
# Set by the auto-wiring callback; read by the seams further down the stack that
# know an outcome (published / DLQ'd / raised) but not which subscription they
# are serving.  contextvars are task-local, so concurrent dispatches on the same
# event loop cannot cross-attribute.
_ACTIVE_FLOW_KEY: ContextVar[tuple[str, str] | None] = ContextVar(
    "onex_active_consumer_flow_key",
    default=None,
)

# How many CLOSED windows are kept after ``drain()`` so an off-cycle reader can
# take a ratio over them (OMN-16994).
#
# Derivation, not a magic number: the heartbeat closes a window every 30 s
# (``mixin_node_introspection.initialize_introspection``'s
# ``heartbeat_interval_seconds`` default) and ``ServiceRuntimeHealthMonitor``
# evaluates every 300 s (``_DEFAULT_CHECK_INTERVAL``), so ten closed windows is
# exactly one health-check interval of history. A live read of the OPEN window
# would instead race the heartbeat and usually see whatever fraction of a window
# happened to be accumulated, which for a projection consuming in bursts is a
# ratio taken over a sample too small to mean anything.
RETAINED_FLOW_WINDOW_COUNT: int = 10


class RetainedFlowWindows:
    """Bounded ring of the most recently CLOSED flow windows (OMN-16994).

    Its own object, with its own lock, because it answers a different question
    from the accumulator that fills it: the accumulator owns the OPEN window and
    is reset every drain, while this owns the CLOSED history that survives one.
    A reader on a slower cycle than the heartbeat has nothing else to read.

    Windows are immutable ``ModelNodeFlowWindow`` values, fully built before
    they arrive here, so a reader either sees a whole window or does not see it
    — there is no torn state to guard against beyond the deque mutation itself.
    """

    def __init__(self, maxlen: int = RETAINED_FLOW_WINDOW_COUNT) -> None:
        self._lock = threading.Lock()
        self._windows: deque[ModelNodeFlowWindow] = deque(maxlen=maxlen)

    def append(self, window: ModelNodeFlowWindow) -> None:
        """Retain one closed window, evicting the oldest past the bound."""
        with self._lock:
            self._windows.append(window)

    def snapshot(self) -> tuple[ModelNodeFlowWindow, ...]:
        """Return the retained closed windows, oldest first.

        Empty is a distinct state from "nothing flowed": it means no window has
        closed yet on this process (or this node does not carry the window), so
        a reader must treat it as UNKNOWN and never as proven-idle.
        """
        with self._lock:
            return tuple(self._windows)


class ConsumerFlowCounters:
    """Per-(consumer_group, topic) throughput counters for one process.

    Thread-safe.  Every mutation is O(1) under a short lock; ``drain()`` swaps
    the accumulator maps and returns an immutable window model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registered: set[tuple[str, str]] = set()
        self._messages_in: dict[tuple[str, str], int] = {}
        self._messages_out: dict[tuple[str, str], int] = {}
        self._messages_dlq: dict[tuple[str, str], int] = {}
        self._handler_errors: dict[tuple[str, str], int] = {}
        self._produced: dict[str, int] = {}
        self._window_start: datetime | None = None
        self._window_sequence: int = 0
        self._carrier_node_id: UUID | None = None
        # OMN-16994: the last N CLOSED windows, retained so the runtime health
        # monitor can read a DLQ ratio on its own 300 s cycle without racing the
        # 30 s heartbeat that resets the accumulator. Public: it is the read
        # surface for that monitor, not accumulator-internal state.
        self.retained_windows = RetainedFlowWindows(RETAINED_FLOW_WINDOW_COUNT)

    # ---------------------------------------------------------------- register

    def register(self, consumer_group: str, topic: str) -> None:
        """Declare a subscription so it emits a row every window, traffic or not.

        Without this, a consumer that takes nothing would emit no row, and "took
        nothing" would be indistinguishable from "not observed".
        """
        if not consumer_group or not topic:
            return
        with self._lock:
            self._registered.add((consumer_group, topic))

    # ----------------------------------------------------------------- record

    def record_in(self, consumer_group: str, topic: str, count: int = 1) -> None:
        """Count envelopes handed to the dispatch engine."""
        self._bump(self._messages_in, (consumer_group, topic), count)

    def record_out(self, consumer_group: str, topic: str, count: int = 1) -> None:
        """Count envelopes successfully published as the handler's result."""
        self._bump(self._messages_out, (consumer_group, topic), count)

    def record_dlq(self, consumer_group: str, topic: str, count: int = 1) -> None:
        """Count envelopes routed to a DLQ or the platform quarantine sink."""
        self._bump(self._messages_dlq, (consumer_group, topic), count)

    def record_error(self, consumer_group: str, topic: str, count: int = 1) -> None:
        """Count dispatches whose handler raised."""
        self._bump(self._handler_errors, (consumer_group, topic), count)

    def record_produced(self, topic: str, count: int = 1) -> None:
        """Count envelopes this process published TO ``topic``.

        This is the upstream-production evidence that separates ``STARVED`` from
        ``IDLE`` downstream.  It is taken from our own publish seam, never from a
        broker query.
        """
        if not topic or count <= 0:
            return
        with self._lock:
            self._produced[topic] = self._produced.get(topic, 0) + count

    def _bump(
        self, target: dict[tuple[str, str], int], key: tuple[str, str], count: int
    ) -> None:
        if not key[0] or not key[1] or count <= 0:
            return
        with self._lock:
            self._registered.add(key)
            target[key] = target.get(key, 0) + count

    # ------------------------------------------------------------------ drain

    def drain(self, *, node_id: UUID, now: datetime) -> ModelNodeFlowWindow | None:
        """Close the current window and return it, resetting the counters.

        Args:
            node_id: The node publishing the heartbeat this window will ride.
            now: INJECTED event time. This accumulator never reads a clock.

        Returns:
            The closed window, or ``None`` when this node does not carry the
            window — either because another node already claimed carriage, or
            because this is the priming drain.

        Priming drain (returns ``None``): the first drain has no observed window
        start, so there is no interval it could honestly report.  It records
        ``now`` as the start of window 1 and discards whatever accumulated
        before it.  In practice that is the sub-tick of traffic between wiring
        and the first heartbeat.  Reporting it against a fabricated start would
        be worse than not reporting it.
        """
        with self._lock:
            if self._carrier_node_id is None:
                self._carrier_node_id = node_id
                logger.info(
                    "Consumer-flow window carriage claimed by node_id=%s "
                    "(rows are keyed by consumer_group/topic; carriage is a "
                    "labelling choice, OMN-16777)",
                    node_id,
                )
            if self._carrier_node_id != node_id:
                return None

            if self._window_start is None:
                self._window_start = now
                self._reset_unlocked()
                return None

            window_start = self._window_start
            self._window_sequence += 1
            sequence = self._window_sequence

            consumer_deltas = tuple(
                ModelConsumerFlowDelta(
                    consumer_group=group,
                    topic=topic,
                    node_id=node_id,
                    window_start=window_start,
                    window_end=now,
                    window_sequence=sequence,
                    messages_in=self._messages_in.get((group, topic), 0),
                    messages_out=self._messages_out.get((group, topic), 0),
                    messages_dlq=self._messages_dlq.get((group, topic), 0),
                    handler_errors=self._handler_errors.get((group, topic), 0),
                )
                for group, topic in sorted(self._registered)
            )
            produce_deltas = tuple(
                ModelTopicProduceDelta(
                    topic=topic,
                    node_id=node_id,
                    window_start=window_start,
                    window_end=now,
                    window_sequence=sequence,
                    messages_produced=produced,
                )
                for topic, produced in sorted(self._produced.items())
            )

            self._window_start = now
            self._reset_unlocked()

            window = ModelNodeFlowWindow(
                node_id=node_id,
                window_start=window_start,
                window_end=now,
                window_sequence=sequence,
                consumer_deltas=consumer_deltas,
                produce_deltas=produce_deltas,
            )
        # Retained OUTSIDE this lock, on the ring's own lock: the window is a
        # fully-built immutable value by this point, and taking the ring's lock
        # while holding the accumulator's would nest two locks for no gain
        # (OMN-16994).
        self.retained_windows.append(window)
        return window

    def _reset_unlocked(self) -> None:
        """Zero the per-window counters. Registrations survive; counts do not."""
        self._messages_in = {}
        self._messages_out = {}
        self._messages_dlq = {}
        self._handler_errors = {}
        self._produced = {}


# One lazily-built accumulator per process, held in a single-entry mapping so the
# accessor mutates a container rather than rebinding a module global. The two
# seams that share these counters (the wiring boundary and the heartbeat tick)
# MUST observe the same instance: a double-construction race would split a window
# in half and under-report both halves, which is precisely the failure mode this
# module exists to detect. Hence the lock, not a bare dict read.
_SLOT_KEY = "process"
_SLOT_LOCK = threading.Lock()
_SLOT: dict[str, ConsumerFlowCounters] = {}


def get_consumer_flow_counters() -> ConsumerFlowCounters:
    """Return the process-scoped accumulator, creating it on first use."""
    with _SLOT_LOCK:
        counters = _SLOT.get(_SLOT_KEY)
        if counters is None:
            counters = ConsumerFlowCounters()
            _SLOT[_SLOT_KEY] = counters
        return counters


def reset_consumer_flow_counters() -> None:
    """Drop the process-scoped accumulator. Test seam only."""
    with _SLOT_LOCK:
        _SLOT.clear()


@contextmanager
def active_flow_key(consumer_group: str, topic: str) -> Iterator[None]:
    """Bind the in-flight ``(consumer_group, topic)`` for the current task.

    Seams further down the stack (the result applier's publish loop, the
    projection-arm quarantine router) know an OUTCOME but not which subscription
    produced it.  Rather than thread a parameter through five call sites that do
    not otherwise care, they read this task-local binding.
    """
    token = _ACTIVE_FLOW_KEY.set((consumer_group, topic))
    try:
        yield
    finally:
        _ACTIVE_FLOW_KEY.reset(token)


def record_active_out(count: int = 1) -> None:
    """Record ``messages_out`` against the in-flight subscription, if any."""
    key = _ACTIVE_FLOW_KEY.get()
    if key is None:
        return
    get_consumer_flow_counters().record_out(key[0], key[1], count)


def record_active_dlq(count: int = 1) -> None:
    """Record ``messages_dlq`` against the in-flight subscription, if any."""
    key = _ACTIVE_FLOW_KEY.get()
    if key is None:
        return
    get_consumer_flow_counters().record_dlq(key[0], key[1], count)


def record_active_error(count: int = 1) -> None:
    """Record ``handler_errors`` against the in-flight subscription, if any."""
    key = _ACTIVE_FLOW_KEY.get()
    if key is None:
        return
    get_consumer_flow_counters().record_error(key[0], key[1], count)


def record_produced_topic(topic: str, count: int = 1) -> None:
    """Record an envelope this process published to ``topic``."""
    get_consumer_flow_counters().record_produced(topic, count)


__all__ = [
    "RETAINED_FLOW_WINDOW_COUNT",
    "ConsumerFlowCounters",
    "RetainedFlowWindows",
    "active_flow_key",
    "get_consumer_flow_counters",
    "record_active_dlq",
    "record_active_error",
    "record_active_out",
    "record_produced_topic",
    "reset_consumer_flow_counters",
]
