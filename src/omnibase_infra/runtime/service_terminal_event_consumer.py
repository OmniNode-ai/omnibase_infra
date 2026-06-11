# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Blocking terminal-event consumer for auto-wired request/response EFFECT handlers.

Some EFFECT handlers drive a request/response interaction over the bus: they
publish a command, then block until the correlated terminal event arrives on a
reply topic (correlating by ``correlation_id``), and read result fields back
from that terminal payload. ``HandlerContextRoiRunner`` is the motivating case
(OMN-13005): it publishes ``node-generation-requested.v1`` and must wait for the
correlated ``node-generation-completed.v1`` terminal before recording an
attempt-reduction row.

These handlers declare an injectable ``event_consumer`` materialized by the
runtime auto-wiring (``handler_wiring.py``) through ``make_terminal_event_consumer``
so the handler never imports Kafka. This mirrors the existing ``event_publisher``
injection (``_make_sync_event_publisher``).

Two-phase (subscribe-before-publish) — OMN-13012
------------------------------------------------
The single-call form ``consumer(topic, cid, timeout)`` does
``assign → seek_to_end → poll`` internally, all AFTER the handler has already
published its command. Once the dispatch-loop starvation was removed (OMN-13010),
generation completes in ~1s, so the correlated terminal is published BEFORE the
single-call consumer's post-publish ``seek_to_end`` positions — and ``seek_to_end``
skips PAST the already-emitted terminal, which the runner then never sees. The
runner waits the full timeout on an offset beyond the terminal and times out
(probe3, ``run_id=20260611T2140Z-probe3``).

The fix splits positioning from waiting so the caller can subscribe BEFORE it
publishes:

  - ``session = consumer.open(topic)`` — start the ephemeral consumer, assign the
    reply-topic partitions, and ``seek_to_end`` NOW (records the pre-publish
    high-water-mark as the resume position). Call this BEFORE publishing.
  - ``payload = session.wait(correlation_id, timeout)`` — block up to ``timeout``
    polling from the position captured at ``open``, returning the correlated
    terminal payload or ``None`` on genuine timeout. Call this AFTER publishing.
  - ``session.close()`` — stop the consumer (``wait`` closes automatically;
    ``close`` is also idempotent for explicit/early teardown).

Because the cursor is positioned at ``open`` time (before publish), a terminal
emitted in the publish→wait gap is still on an offset at-or-after the captured
position and is delivered to ``wait``. ``open`` is also used as a context manager.

Backward compatibility
-----------------------
The object returned by ``make_terminal_event_consumer`` is still directly callable
with the legacy single-call shape ``(topic, cid, timeout) -> dict | None`` for any
consumer that does not need subscribe-before-publish. The single-call form simply
opens a session and waits in one step (seek happens at call time, after the
caller's publish — the original, race-prone behavior, retained only for callers
that do not care about the race).

Architecture (ARCH-002 — "runtime owns all Kafka plumbing"):
  - Nodes/handlers declare the consume requirement in their contract and call the
    injected ``event_consumer``; they never touch ``AIOKafkaConsumer``.
  - The correlate-and-wait loop here is the same proven shape used by
    ``RuntimePatternBBroker._dispatch_and_wait_with_direct_kafka_consumer``:
    assign reply-topic partitions, seek to end, then poll until a message whose
    body ``correlation_id`` matches, honoring the caller's timeout.

Loop isolation (why a worker thread):
  The auto-wired dispatch callback invokes a sync handler ``handle()`` directly
  on the runtime's running event loop. A blocking Kafka correlate cannot run on
  that same loop (it would deadlock, and ``asyncio.run()`` raises inside a
  running loop). The session therefore owns a dedicated event loop running in a
  separate daemon thread for the whole ``open``→``wait``→``close`` lifecycle, and
  the sync methods block the calling thread on coroutines submitted to that loop.
  The dispatch loop stays free to deliver the very terminal event the handler is
  waiting for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Future
from types import TracebackType
from typing import cast

from aiokafka import AIOKafkaConsumer, TopicPartition

logger = logging.getLogger(__name__)

# Type alias mirrored from the declaring handler:
# (terminal_topic, correlation_id, timeout_seconds) -> deserialized payload | None
EventConsumer = Callable[[str, str, float], dict[str, object] | None]

# Bound the metadata/partition-assignment phase so a broker that never surfaces
# the reply topic fails fast instead of consuming the whole caller timeout.
_ASSIGN_TIMEOUT_CAP_SECONDS = 30.0
_METADATA_POLL_INTERVAL_SECONDS = 0.05

# aiokafka client tuning — matches the request-response / pattern-B defaults so a
# slow generation does not trigger a session-timeout rebalance mid-wait.
_DEFAULT_SESSION_TIMEOUT_MS = 45000
_DEFAULT_HEARTBEAT_INTERVAL_MS = 15000
_DEFAULT_MAX_POLL_INTERVAL_MS = 300000

# Grace beyond a submitted coroutine's own timeout so a clean Kafka shutdown
# completes before the calling thread gives up on the worker loop.
_LOOP_SUBMIT_GRACE_SECONDS = 5.0


def _bootstrap_servers(event_bus: object) -> str:
    servers = getattr(event_bus, "_bootstrap_servers", None)
    if not isinstance(servers, str) or not servers:
        raise RuntimeError(
            "terminal-event consumer: runtime event_bus exposes no string "
            "_bootstrap_servers; cannot build a Kafka correlate consumer."
        )
    return servers


def _auth_kwargs(event_bus: object) -> dict[str, object]:
    build_auth_kwargs = getattr(event_bus, "_build_auth_kwargs", None)
    if not callable(build_auth_kwargs):
        return {}
    auth = cast("Callable[[], Mapping[str, object] | None]", build_auth_kwargs)()
    return dict(auth or {})


def _extract_correlation_id(payload: dict[str, object]) -> str | None:
    value = payload.get("correlation_id")
    if value is None:
        return None
    return str(value)


def _build_consumer(event_bus: object) -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        bootstrap_servers=_bootstrap_servers(event_bus),
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        session_timeout_ms=_DEFAULT_SESSION_TIMEOUT_MS,
        heartbeat_interval_ms=_DEFAULT_HEARTBEAT_INTERVAL_MS,
        max_poll_interval_ms=_DEFAULT_MAX_POLL_INTERVAL_MS,
        **_auth_kwargs(event_bus),
    )


async def _assign_terminal_partitions(
    consumer: AIOKafkaConsumer,
    terminal_topic: str,
    assign_cap_seconds: float,
) -> None:
    """Wait for the reply topic's partition metadata, then assign all partitions.

    A freshly created topic may not have partition metadata immediately; poll
    until it appears or the cap elapses (raising ``TimeoutError`` so the caller's
    ``try`` returns ``None`` cleanly rather than hanging).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + assign_cap_seconds
    while True:
        client = getattr(consumer, "_client", None)
        set_topics = getattr(client, "set_topics", None)
        if callable(set_topics):
            set_topics([terminal_topic])
        partitions = consumer.partitions_for_topic(terminal_topic) or set()
        if partitions:
            consumer.assign([TopicPartition(terminal_topic, p) for p in partitions])
            return
        if loop.time() >= deadline:
            raise TimeoutError
        await asyncio.sleep(_METADATA_POLL_INTERVAL_SECONDS)


async def _open_positioned_consumer(
    *,
    event_bus: object,
    terminal_topic: str,
) -> AIOKafkaConsumer:
    """Start an ephemeral consumer, assign the reply topic, and seek to end NOW.

    Returns the started, positioned consumer. The caller is responsible for
    stopping it (``_poll_correlated_terminal`` stops it in its ``finally``). The
    ``seek_to_end`` here captures the pre-publish high-water-mark: any terminal
    published at-or-after this point is delivered to a subsequent poll, even if it
    arrives before the poll begins. This is the subscribe-before-publish leg.
    """
    consumer = _build_consumer(event_bus)
    try:
        await asyncio.wait_for(consumer.start(), timeout=_ASSIGN_TIMEOUT_CAP_SECONDS)
        await _assign_terminal_partitions(
            consumer, terminal_topic, _ASSIGN_TIMEOUT_CAP_SECONDS
        )
        await consumer.seek_to_end(*consumer.assignment())
    except BaseException:
        try:
            await consumer.stop()
        except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
            logger.warning(
                "terminal-event consumer: failed to stop consumer during open "
                "cleanup for %s: %s",
                terminal_topic,
                exc,
            )
        raise
    return consumer


async def _poll_correlated_terminal(
    *,
    consumer: AIOKafkaConsumer,
    terminal_topic: str,
    correlation_id: str,
    timeout_seconds: float,
) -> dict[str, object] | None:
    """Poll an already-positioned consumer for the correlated terminal payload.

    Blocks up to ``timeout_seconds`` from the position captured at ``open`` time.
    Returns the deserialized terminal body whose ``correlation_id`` matches, or
    ``None`` on genuine timeout. Stops the consumer on the way out.
    """
    loop = asyncio.get_running_loop()
    try:
        deadline = loop.time() + timeout_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                message = await asyncio.wait_for(consumer.getone(), timeout=remaining)
            except TimeoutError:
                return None
            if message.value is None:
                continue
            try:
                body: dict[str, object] = json.loads(message.value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if _extract_correlation_id(body) == correlation_id:
                return body
    finally:
        try:
            await consumer.stop()
        except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
            logger.warning(
                "terminal-event consumer: failed to stop consumer for %s: %s",
                terminal_topic,
                exc,
            )


class TerminalConsumerSession:
    """Two-phase (seek-now / wait-later) handle over a single terminal topic.

    Owns a dedicated event loop on a daemon worker thread for the whole
    ``open``→``wait``→``close`` lifecycle. Synchronous methods submit coroutines
    to that loop and block the calling thread on the result, so the session is
    safe to drive from a sync handler running on the runtime's own (already
    running) event loop without deadlocking it.

    Usage (subscribe-before-publish)::

        session = consumer.open(terminal_topic)   # assign + seek_to_end NOW
        publisher(command_topic, payload)         # publish AFTER positioning
        result = session.wait(correlation_id, timeout)  # block from the position
        # wait() closes automatically; close() is idempotent for early teardown.
    """

    def __init__(
        self,
        *,
        event_bus: object,
        handler_name: str,
        terminal_topic: str,
    ) -> None:
        self._event_bus = event_bus
        self._handler_name = handler_name
        self._terminal_topic = terminal_topic
        self._loop = asyncio.new_event_loop()
        self._consumer: AIOKafkaConsumer | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"terminal-consumer-{handler_name}",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(
        self, coro: Coroutine[object, object, object], *, timeout: float
    ) -> object:
        """Run ``coro`` on the worker loop and block the caller on its result."""
        future: Future[object] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def open(self) -> TerminalConsumerSession:
        """Start the consumer, assign the topic, and seek to end NOW (pre-publish).

        Idempotent: a second call is a no-op once positioned. Returns ``self`` so
        ``consumer.open(topic)`` reads naturally and supports ``with`` usage.
        """
        if self._consumer is not None or self._closed:
            return self
        self._consumer = cast(
            "AIOKafkaConsumer",
            self._submit(
                _open_positioned_consumer(
                    event_bus=self._event_bus,
                    terminal_topic=self._terminal_topic,
                ),
                timeout=_ASSIGN_TIMEOUT_CAP_SECONDS + _LOOP_SUBMIT_GRACE_SECONDS,
            ),
        )
        return self

    def wait(
        self, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        """Block up to ``timeout_seconds`` for the correlated terminal, then close.

        Polls from the position captured at ``open`` time, so a terminal emitted in
        the publish→wait gap is still delivered. Returns the matched payload or
        ``None`` on timeout/failure. Closes the session on the way out.
        """
        if self._consumer is None:
            # Defensive: caller skipped open() — fall back to seek-at-wait (the
            # race-prone single-call path) rather than failing hard.
            self.open()
        consumer = self._consumer
        if consumer is None:
            self.close()
            return None
        try:
            result = self._submit(
                _poll_correlated_terminal(
                    consumer=consumer,
                    terminal_topic=self._terminal_topic,
                    correlation_id=correlation_id,
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds
                + _ASSIGN_TIMEOUT_CAP_SECONDS
                + _LOOP_SUBMIT_GRACE_SECONDS,
            )
        except BaseException as exc:  # noqa: BLE001 — re-raised as None to caller
            logger.warning(
                "terminal-event consumer: wait failed for %s (topic=%s, cid=%s): %s",
                self._handler_name,
                self._terminal_topic,
                correlation_id,
                exc,
            )
            self.close()
            return None
        # _poll_correlated_terminal stopped the consumer in its finally; drop the
        # reference so close() does not double-stop.
        self._consumer = None
        self.close()
        return cast("dict[str, object] | None", result)

    def close(self) -> None:
        """Stop the worker loop and join its thread. Idempotent."""
        if self._closed:
            return
        self._closed = True
        consumer = self._consumer
        self._consumer = None
        if consumer is not None:
            try:
                stop_future: Future[object] = asyncio.run_coroutine_threadsafe(
                    cast("Coroutine[object, object, object]", consumer.stop()),
                    self._loop,
                )
                stop_future.result(timeout=_LOOP_SUBMIT_GRACE_SECONDS)
            except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "terminal-event consumer: failed to stop consumer during close "
                    "for %s (topic=%s): %s",
                    self._handler_name,
                    self._terminal_topic,
                    exc,
                )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=_LOOP_SUBMIT_GRACE_SECONDS)
        try:
            self._loop.close()
        except Exception as exc:  # noqa: BLE001 — boundary: best-effort cleanup
            logger.debug(
                "terminal-event consumer: loop close raced for %s: %s",
                self._handler_name,
                exc,
            )

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
    """Callable injected as ``event_consumer`` — both single-call and two-phase.

    Calling the instance directly preserves the legacy
    ``(topic, cid, timeout) -> dict | None`` shape. ``.open(topic)`` returns a
    :class:`TerminalConsumerSession` for subscribe-before-publish callers.
    """

    def __init__(self, *, event_bus: object, handler_name: str) -> None:
        self._event_bus = event_bus
        self._handler_name = handler_name

    def open(self, terminal_topic: str) -> TerminalConsumerSession:
        """Open a positioned (assign + seek_to_end NOW) session before publishing."""
        session = TerminalConsumerSession(
            event_bus=self._event_bus,
            handler_name=self._handler_name,
            terminal_topic=terminal_topic,
        )
        return session.open()

    def __call__(
        self, terminal_topic: str, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        """Single-call form: open + wait in one step (seek happens at call time).

        Retained for backward compatibility. This is the race-prone path: the seek
        runs when the call is made, which is after the caller's publish. Callers
        that need subscribe-before-publish must use ``open`` + ``wait`` instead.
        """
        session = self.open(terminal_topic)
        return session.wait(correlation_id, timeout_seconds)


def make_terminal_event_consumer(
    *,
    event_bus: object,
    handler_name: str,
) -> TerminalEventConsumer:
    """Build the ``event_consumer`` injected into request/response handlers.

    The returned object is directly callable with the legacy single-call shape
    ``(terminal_topic, correlation_id, timeout_seconds) -> dict | None`` and also
    exposes ``.open(topic) -> TerminalConsumerSession`` for the two-phase
    (subscribe-before-publish) protocol. Both forms run the Kafka correlate-and-wait
    work on a dedicated event loop in a worker thread, so they are safe to call
    from a sync handler executing on the runtime's own (already running) event
    loop without deadlocking that loop.
    """
    return TerminalEventConsumer(event_bus=event_bus, handler_name=handler_name)


__all__: list[str] = [
    "EventConsumer",
    "TerminalConsumerSession",
    "TerminalEventConsumer",
    "make_terminal_event_consumer",
]
