# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Long-lived terminal-event correlator for auto-wired request/response EFFECT handlers.

Some EFFECT handlers drive a request/response interaction over the bus: they
publish a command, then block until the correlated terminal event arrives on a
reply topic (correlating by ``correlation_id``), and read result fields back
from that terminal payload. ``HandlerContextRoiRunner`` is the motivating case
(OMN-13005): per (task x arm x trial) it publishes ``node-generation-requested.v1``
and must wait for the correlated terminal — ``node-generation-completed.v1`` OR
``node-generation-failed.v1`` (OMN-13038) — before recording an attempt-reduction
row.

Long-lived correlator (OMN-13118 Tier B redesign)
--------------------------------------------------
The earlier implementation built a brand-new OS thread + asyncio event loop +
``AIOKafkaConsumer`` + ``AIOKafkaClient`` (full bootstrap / metadata fetch /
coordination) for EVERY (trial x terminal-topic) — up to 80 churns for a 40-cell
run, 320+ for the 160-battery — and tore each down right after, racing a ~1s
generation. Five offset/subscription patches (#1969 set_topics no-op, #1970
partition-less assign, #1971 seek-pin, #1972 independent consumers) tuned that
per-trial-ephemeral substrate and all failed the live K>=10 battery gate with the
identical wedge signature. The defect class is the ephemerality itself: a
brand-new client per trial races its own bootstrap and teardown against the fast
terminal, so the COMPLETED leg silently never delivers while only the empty-set
FAILED leg times out at the 30s assign cap. (See
``docs/evidence/2026-06-12-weekend-pass/experiments/exp1/B1_M1_M2_PROBE_RESULT.md``.)

This module owns ONE runtime-lifetime consumer subscribed ONCE to BOTH terminal
topics, running a SINGLE poll loop on a SINGLE thread+loop for the whole batch.
It demuxes every terminal to the waiting trial by ``correlation_id`` via a
``correlation_id -> Future`` registry: a trial registers its ``correlation_id``
BEFORE publishing its command (preserving the subscribe-before-publish guarantee
#1971/OMN-13012 established at the consumer level), then awaits its future with
the trial timeout. There is no per-trial consumer / assign / pin / thread / loop /
client to race — the entire timing-bug class is structurally removed, not
re-tuned. This mirrors the already-working long-lived ``node_generation_consumer``
and ``RuntimePatternBBroker.start()`` command subscription.

Architecture (ARCH-002 — "runtime owns all Kafka plumbing"):
  - Nodes/handlers declare the consume requirement in their contract and call the
    injected ``event_consumer``; they never touch the Kafka client.
  - ``TerminalEventConsumer`` is the contract-declared consume dependency
    materialized by the runtime auto-wiring (``handler_wiring.py``) through
    ``make_terminal_event_consumer``. Its ``.open(topic)`` returns a
    ``TerminalConsumerSession`` bound to the shared long-lived correlator;
    ``session.wait(correlation_id, timeout)`` registers the cid and blocks on the
    correlated future. The handler keeps the same two-phase
    open-before-publish / wait-after-publish protocol.

Loop isolation (why a worker thread):
  The auto-wired dispatch callback invokes a sync handler ``handle()`` directly
  on the runtime's running event loop. A blocking Kafka correlate cannot run on
  that same loop (it would deadlock, and ``asyncio.run()`` raises inside a
  running loop). The correlator therefore owns ONE dedicated event loop on ONE
  daemon thread for the whole process lifetime; the sync ``wait`` blocks the
  calling thread on a future resolved by the correlator's poll loop on that loop.
  The runtime dispatch loop stays free to deliver the very terminals the
  correlator reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Coroutine, Iterable
from concurrent.futures import Future
from types import TracebackType
from typing import cast

from aiokafka import AIOKafkaConsumer, TopicPartition

from omnibase_infra.runtime.service_pattern_b_broker import (
    _build_direct_terminal_consumer,
    _extract_direct_terminal_correlation_id,
    _refresh_terminal_topic_metadata,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# (terminal_topic, correlation_id, timeout_seconds) -> deserialized payload | None
EventConsumer = object

# Bound the one-time subscription/partition-assignment phase so a broker that
# never surfaces a terminal topic fails fast instead of hanging the first open.
_SUBSCRIBE_TIMEOUT_CAP_SECONDS = 30.0
_METADATA_POLL_INTERVAL_SECONDS = 0.05

# Grace beyond a submitted coroutine's own timeout so a clean shutdown completes
# before the calling thread gives up on the worker loop.
_LOOP_SUBMIT_GRACE_SECONDS = 5.0

# Per-iteration poll wait: short so the loop notices cancellation/closing
# promptly while still batching broker fetches.
_POLL_GETONE_TIMEOUT_SECONDS = 1.0

# Cap the unmatched-terminal registry so a long-running batch that reads
# terminals for correlation_ids no live trial is waiting on (late deliveries,
# other runs) cannot grow without bound. Mirrors the bounded ``on_terminal``
# guard in ``RuntimePatternBBroker``.
_MAX_PENDING_UNMATCHED = 4096


class LongLivedTerminalCorrelator:
    """ONE consumer, subscribed once to all terminal topics, demuxing by cid.

    Owns a single event loop on a single daemon thread for its whole lifetime.
    Lazily starts ONE ``AIOKafkaConsumer`` on first use, assigns the partitions
    of every requested terminal topic, pins each to its current end offset, and
    runs a single poll loop that resolves the per-correlation futures as
    terminals arrive. Adding a terminal topic re-resolves the assignment to the
    union and pins the new partitions to their current end — done ONCE per topic,
    not per trial.

    Thread-safety: all Kafka state lives on the worker loop. Public methods
    (``register``, ``ensure_topic``, ``wait``) submit coroutines to that loop and
    block the calling thread on the result, so they are safe to call from the
    sync handler thread.
    """

    def __init__(self, *, event_bus: object, handler_name: str) -> None:
        self._event_bus = event_bus
        self._handler_name = handler_name
        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._consumer: AIOKafkaConsumer | None = None
        self._subscribed_topics: set[str] = set()
        # correlation_id -> future resolved with the terminal payload (or None).
        self._pending: dict[str, asyncio.Future[dict[str, object] | None]] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._diag_poll_iters = 0  # DIAG13118: throttle poll-loop assignment logging
        self._start_lock = asyncio.Lock()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"terminal-correlator-{handler_name}",
            daemon=True,
        )
        self._thread.start()

    # -- worker-loop plumbing ------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(
        self,
        coro: Coroutine[object, object, object],
        *,
        timeout: float,
    ) -> object:
        future: Future[object] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # -- one-time consumer lifecycle (on the worker loop) --------------------

    async def _ensure_consumer_started(self) -> None:
        if self._consumer is not None:
            return
        consumer = _build_direct_terminal_consumer(self._event_bus)
        await asyncio.wait_for(consumer.start(), timeout=_SUBSCRIBE_TIMEOUT_CAP_SECONDS)
        self._consumer = consumer
        self._poll_task = self._loop.create_task(self._poll_forever())

    async def _ensure_topic_assigned(self, terminal_topic: str) -> None:
        """Add a terminal topic to the single consumer's assignment ONCE.

        Resolves the partitions for ``terminal_topic`` (a bounded metadata grace
        for a topic that exists but is slow to surface; an empty set for a reply
        topic never produced to, e.g. the FAILED topic in an all-pass run is a
        valid steady state), unions them into the existing assignment, and pins
        the NEW partitions to their current end offset so a terminal published
        after this call is delivered. Idempotent per topic.
        """
        async with self._start_lock:
            await self._ensure_consumer_started()
            if terminal_topic in self._subscribed_topics:
                return
            consumer = self._consumer
            assert consumer is not None
            new_partitions = await self._resolve_partitions(consumer, terminal_topic)
            existing = set(consumer.assignment())
            union = existing.union(new_partitions)
            if union:
                consumer.assign(list(union))
            # Pin only the newly added partitions to their current end so the
            # already-positioned topics keep their continuously-advancing cursor
            # (one consumer, one monotonic position per partition).
            if new_partitions:
                end_offsets = await asyncio.wait_for(
                    consumer.end_offsets(list(new_partitions)),
                    timeout=_SUBSCRIBE_TIMEOUT_CAP_SECONDS,
                )
                for topic_partition, offset in end_offsets.items():
                    consumer.seek(topic_partition, offset)
                logger.warning(
                    "DIAG13118 pin handler=%s topic=%s end_offsets=%s",
                    self._handler_name,
                    terminal_topic,
                    {str(k): v for k, v in end_offsets.items()},
                )
            logger.warning(
                "DIAG13118 ensure_topic handler=%s topic=%s new_partitions=%s "
                "full_assignment=%s",
                self._handler_name,
                terminal_topic,
                [str(p) for p in new_partitions],
                [str(p) for p in consumer.assignment()],
            )
            self._subscribed_topics.add(terminal_topic)

    async def _resolve_partitions(
        self, consumer: AIOKafkaConsumer, terminal_topic: str
    ) -> list[TopicPartition]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _SUBSCRIBE_TIMEOUT_CAP_SECONDS
        client = getattr(consumer, "_client", None)
        set_topics = getattr(client, "set_topics", None) if client else None
        if callable(set_topics):
            result = set_topics([*self._subscribed_topics, terminal_topic])
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        await _refresh_terminal_topic_metadata(consumer)
        while True:
            partitions = consumer.partitions_for_topic(terminal_topic) or set()
            if partitions:
                return [TopicPartition(terminal_topic, p) for p in partitions]
            if loop.time() >= deadline:
                # A reply topic with no partitions has never been produced to —
                # a valid steady state (OMN-13118). No terminal for any live
                # correlation can be on it, so contribute no partitions; the
                # other terminal topic carries the correlated record.
                return []
            await asyncio.sleep(_METADATA_POLL_INTERVAL_SECONDS)
            await _refresh_terminal_topic_metadata(consumer)

    # -- single poll loop: demux terminals to waiting trials -----------------

    async def _poll_forever(self) -> None:
        consumer = self._consumer
        assert consumer is not None
        while not self._closed:
            try:
                if not consumer.assignment():
                    await asyncio.sleep(_METADATA_POLL_INTERVAL_SECONDS)
                    continue
                self._diag_poll_iters += 1
                if self._diag_poll_iters % 50 == 1:
                    logger.warning(
                        "DIAG13118 poll handler=%s assignment=%s pending=%s",
                        self._handler_name,
                        [str(p) for p in consumer.assignment()],
                        list(self._pending.keys())[:10],
                    )
                message = await asyncio.wait_for(
                    consumer.getone(), timeout=_POLL_GETONE_TIMEOUT_SECONDS
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — poll boundary: log + continue
                logger.warning(
                    "terminal-event correlator: poll error for %s: %s",
                    self._handler_name,
                    sanitize_error_message(exc),
                )
                await asyncio.sleep(_METADATA_POLL_INTERVAL_SECONDS)
                continue
            logger.warning(
                "DIAG13118 getone handler=%s topic=%s partition=%s offset=%s",
                self._handler_name,
                getattr(message, "topic", None),
                getattr(message, "partition", None),
                getattr(message, "offset", None),
            )
            self._deliver(message)

    def _deliver(self, message: object) -> None:
        value = getattr(message, "value", None)
        if value is None:
            logger.warning(
                "DIAG13118 deliver handler=%s drop=value_none", self._handler_name
            )
            return
        try:
            body: dict[str, object] = json.loads(value.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            logger.warning(
                "DIAG13118 deliver handler=%s drop=decode_error", self._handler_name
            )
            return
        correlation_id = _extract_direct_terminal_correlation_id(body)
        if correlation_id is None:
            logger.warning(
                "DIAG13118 deliver handler=%s drop=cid_none body_keys=%s",
                self._handler_name,
                sorted(body.keys()),
            )
            return
        future = self._pending.pop(correlation_id, None)
        if future is None:
            # Unmatched terminal (late, or a correlation no live trial awaits).
            # Drop it — the correlator is fire-and-forget for unawaited cids.
            logger.warning(
                "DIAG13118 deliver handler=%s extracted_cid=%s UNMATCHED "
                "pending_keys=%s",
                self._handler_name,
                correlation_id,
                list(self._pending.keys())[:10],
            )
            return
        logger.warning(
            "DIAG13118 deliver handler=%s extracted_cid=%s MATCHED future_done=%s",
            self._handler_name,
            correlation_id,
            future.done(),
        )
        if not future.done():
            future.set_result(body)

    # -- public API (called from the sync handler thread) --------------------

    def ensure_topic(self, terminal_topic: str) -> None:
        """Subscribe the single consumer to ``terminal_topic`` (once, blocking)."""
        if self._closed:
            return
        self._submit(
            self._ensure_topic_assigned(terminal_topic),
            timeout=_SUBSCRIBE_TIMEOUT_CAP_SECONDS + _LOOP_SUBMIT_GRACE_SECONDS,
        )

    def register(self, correlation_id: str) -> None:
        """Register a correlation_id's future BEFORE the caller publishes.

        Subscribe-before-publish at the correlation level: the future exists in
        the registry before the command is on the wire, so a terminal that lands
        immediately after publish is delivered to it by the always-running poll
        loop (no per-trial open/assign/pin to race).
        """
        if self._closed:
            return
        self._submit(self._register(correlation_id), timeout=_LOOP_SUBMIT_GRACE_SECONDS)

    async def _register(self, correlation_id: str) -> None:
        if len(self._pending) >= _MAX_PENDING_UNMATCHED:
            # Bounded registry: evict the oldest unresolved future rather than
            # grow without bound. A live trial re-registers on its next wait.
            stale_cid, stale_future = next(iter(self._pending.items()))
            self._pending.pop(stale_cid, None)
            if not stale_future.done():
                stale_future.set_result(None)
        if correlation_id not in self._pending:
            self._pending[correlation_id] = self._loop.create_future()
        logger.warning(
            "DIAG13118 register handler=%s cid=%s pending_count=%s",
            self._handler_name,
            correlation_id,
            len(self._pending),
        )

    def wait(
        self, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        """Block up to ``timeout_seconds`` for the correlated terminal.

        The future was created by ``register`` (pre-publish). Blocks the calling
        thread on it via the worker loop, returning the matched payload or
        ``None`` on timeout. On timeout the registry entry is dropped so a late
        delivery is not mismatched to a future trial.
        """
        if self._closed:
            return None
        try:
            result = self._submit(
                self._wait(correlation_id, timeout_seconds),
                timeout=timeout_seconds + _LOOP_SUBMIT_GRACE_SECONDS,
            )
        except BaseException as exc:  # noqa: BLE001 — re-raised as None to caller
            logger.warning(
                "terminal-event correlator: wait failed for %s (cid=%s): %s",
                self._handler_name,
                correlation_id,
                sanitize_error_message(exc)
                if isinstance(exc, Exception)
                else type(exc).__name__,
            )
            return None
        return cast("dict[str, object] | None", result)

    async def _wait(
        self, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        future = self._pending.get(correlation_id)
        if future is None:
            # Defensive: caller skipped register(); create it now (race-prone but
            # never worse than the legacy single-call path).
            logger.warning(
                "DIAG13118 wait handler=%s cid=%s NO_PRE_REGISTER",
                self._handler_name,
                correlation_id,
            )
            future = self._loop.create_future()
            self._pending[correlation_id] = future
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout_seconds
            )
            logger.warning(
                "DIAG13118 wait handler=%s cid=%s RESOLVED",
                self._handler_name,
                correlation_id,
            )
            return result
        except TimeoutError:
            logger.warning(
                "DIAG13118 wait handler=%s cid=%s TIMEOUT after %ss",
                self._handler_name,
                correlation_id,
                timeout_seconds,
            )
            self._pending.pop(correlation_id, None)
            return None

    def close(self) -> None:
        """Stop the poll loop, the consumer, and the worker loop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            stop_future: Future[object] = asyncio.run_coroutine_threadsafe(
                self._shutdown(), self._loop
            )
            stop_future.result(timeout=_LOOP_SUBMIT_GRACE_SECONDS)
        except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
            logger.warning(
                "terminal-event correlator: shutdown raced for %s: %s",
                self._handler_name,
                sanitize_error_message(exc),
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=_LOOP_SUBMIT_GRACE_SECONDS)
        try:
            self._loop.close()
        except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
            logger.debug(
                "terminal-event correlator: loop close raced for %s: %s",
                self._handler_name,
                exc,
            )

    async def _shutdown(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
                pass
        for future in self._pending.values():
            if not future.done():
                future.set_result(None)
        self._pending.clear()
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "terminal-event correlator: failed to stop consumer for %s: %s",
                    self._handler_name,
                    sanitize_error_message(exc),
                )
            self._consumer = None


class TerminalConsumerSession:
    """Two-phase handle over ONE terminal topic, backed by the shared correlator.

    ``open`` ensures the shared long-lived consumer is subscribed to the topic
    (a no-op after the first trial) and registers the caller's correlation_id
    BEFORE publish. ``wait`` blocks on the correlated future the always-running
    poll loop resolves. No per-trial thread / loop / consumer / assign / pin.

    Usage (subscribe-before-publish)::

        session = consumer.open(terminal_topic)   # ensure topic subscribed
        session.register(correlation_id)          # register BEFORE publish
        publisher(command_topic, payload)         # publish AFTER positioning
        result = session.wait(correlation_id, timeout)
    """

    def __init__(
        self,
        *,
        correlator: LongLivedTerminalCorrelator,
        terminal_topic: str,
    ) -> None:
        self._correlator = correlator
        self._terminal_topic = terminal_topic
        self._registered: set[str] = set()

    def open(self) -> TerminalConsumerSession:
        """Ensure the shared consumer is subscribed to this terminal topic."""
        self._correlator.ensure_topic(self._terminal_topic)
        return self

    def wait(
        self, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        """Register the cid (if not yet) and block for the correlated terminal."""
        self.register(correlation_id)
        return self._correlator.wait(correlation_id, timeout_seconds)

    def register(self, correlation_id: str) -> None:
        """Register the cid BEFORE publish (explicit subscribe-before-publish)."""
        if correlation_id not in self._registered:
            self._correlator.register(correlation_id)
            self._registered.add(correlation_id)

    def close(self) -> None:
        """No-op: the shared correlator outlives any single session.

        The per-trial session owns no resources of its own — the consumer,
        thread, and loop are the correlator's and live for the whole batch.
        Closing a session must NOT tear the shared correlator down (that is the
        teardown-races-fetch class the redesign eliminates).
        """
        return

    def __enter__(self) -> TerminalConsumerSession:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class TerminalEventConsumer:
    """Callable injected as ``event_consumer`` — backed by ONE long-lived correlator.

    ``.open(topic)`` returns a :class:`TerminalConsumerSession` for
    subscribe-before-publish callers. Calling the instance directly preserves the
    legacy ``(topic, cid, timeout) -> dict | None`` shape (open + register + wait
    in one step). Both route through the single shared correlator — there is no
    per-call consumer, thread, or loop.
    """

    def __init__(self, *, event_bus: object, handler_name: str) -> None:
        self._event_bus = event_bus
        self._handler_name = handler_name
        self._correlator = LongLivedTerminalCorrelator(
            event_bus=event_bus, handler_name=handler_name
        )

    def open(self, terminal_topic: str) -> TerminalConsumerSession:
        """Open a session bound to the shared correlator (ensures topic at open)."""
        session = TerminalConsumerSession(
            correlator=self._correlator,
            terminal_topic=terminal_topic,
        )
        return session.open()

    def __call__(
        self, terminal_topic: str, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        """Single-call form: ensure topic + register + wait in one step."""
        session = self.open(terminal_topic)
        session.register(correlation_id)
        return session.wait(correlation_id, timeout_seconds)

    def topics(self) -> Iterable[str]:
        return tuple(self._correlator._subscribed_topics)

    def close(self) -> None:
        """Tear down the shared correlator (runtime shutdown / test cleanup)."""
        self._correlator.close()


def make_terminal_event_consumer(
    *,
    event_bus: object,
    handler_name: str,
) -> TerminalEventConsumer:
    """Build the ``event_consumer`` injected into request/response handlers.

    The returned object is directly callable with the legacy single-call shape
    ``(terminal_topic, correlation_id, timeout_seconds) -> dict | None`` and also
    exposes ``.open(topic) -> TerminalConsumerSession`` for the two-phase
    (subscribe-before-publish) protocol. Both forms route through ONE long-lived
    correlator (one consumer subscribed once to all terminal topics, single poll
    loop, demux by correlation_id) running on a dedicated event loop in a worker
    thread, so blocking does not deadlock the runtime dispatch loop that delivers
    the awaited terminals — and there is no per-trial consumer/thread/loop to
    race (OMN-13118 Tier B).
    """
    return TerminalEventConsumer(event_bus=event_bus, handler_name=handler_name)


__all__: list[str] = [
    "EventConsumer",
    "LongLivedTerminalCorrelator",
    "TerminalConsumerSession",
    "TerminalEventConsumer",
    "make_terminal_event_consumer",
]
