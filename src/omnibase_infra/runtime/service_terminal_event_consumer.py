# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One-shot blocking terminal-event consumer for auto-wired effect handlers.

Some EFFECT handlers drive a request/response interaction over the bus: they
publish a command, then block until the correlated terminal event arrives on a
reply topic (correlating by ``correlation_id``), and read result fields back
from that terminal payload. ``HandlerContextRoiRunner`` is the motivating case
(OMN-13005): it publishes ``node-generation-requested.v1`` and must wait for the
correlated ``node-generation-completed.v1`` terminal before recording an
attempt-reduction row.

These handlers declare an injectable ``event_consumer`` with the synchronous
shape ``(topic, correlation_id, timeout_seconds) -> dict | None``. The runtime
auto-wiring (``handler_wiring.py``) materializes a concrete consumer through
``make_terminal_event_consumer`` so the handler never imports Kafka. This mirrors
the existing ``event_publisher`` injection (``_make_sync_event_publisher``).

Architecture (ARCH-002 — "runtime owns all Kafka plumbing"):
  - Nodes/handlers declare the consume requirement in their contract and call a
    sync ``event_consumer``; they never touch ``AIOKafkaConsumer``.
  - The correlate-and-wait loop here is the same proven shape used by
    ``RuntimePatternBBroker._dispatch_and_wait_with_direct_kafka_consumer``:
    assign reply-topic partitions, seek to end, then poll until a message whose
    body ``correlation_id`` matches, honoring the caller's timeout.

Loop isolation (why a worker thread):
  The auto-wired dispatch callback invokes a sync handler ``handle()`` directly
  on the runtime's running event loop. A blocking Kafka correlate cannot run on
  that same loop (it would deadlock, and ``asyncio.run()`` raises inside a
  running loop). The sync adapter therefore runs the async correlate coroutine
  on a dedicated event loop in a separate thread and blocks the calling thread
  on the result. The dispatch loop stays free to deliver the very terminal event
  the handler is waiting for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Mapping
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


async def _await_correlated_terminal(
    *,
    event_bus: object,
    terminal_topic: str,
    correlation_id: str,
    timeout_seconds: float,
) -> dict[str, object] | None:
    """Block up to ``timeout_seconds`` for the correlated terminal payload.

    Returns the deserialized terminal event body whose ``correlation_id`` matches
    the request, or ``None`` on genuine timeout. Uses an ephemeral group-less
    consumer (``group_id=None``) assigned directly to the reply topic's
    partitions and seeked to end, so it only sees terminals produced after the
    handler published its command.
    """
    consumer = AIOKafkaConsumer(
        bootstrap_servers=_bootstrap_servers(event_bus),
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        session_timeout_ms=_DEFAULT_SESSION_TIMEOUT_MS,
        heartbeat_interval_ms=_DEFAULT_HEARTBEAT_INTERVAL_MS,
        max_poll_interval_ms=_DEFAULT_MAX_POLL_INTERVAL_MS,
        **_auth_kwargs(event_bus),
    )

    loop = asyncio.get_running_loop()
    try:
        assign_cap = min(_ASSIGN_TIMEOUT_CAP_SECONDS, timeout_seconds)
        await asyncio.wait_for(consumer.start(), timeout=assign_cap)
        await _assign_terminal_partitions(consumer, terminal_topic, assign_cap)
        await consumer.seek_to_end(*consumer.assignment())

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


def make_terminal_event_consumer(
    *,
    event_bus: object,
    handler_name: str,
) -> EventConsumer:
    """Build the sync ``event_consumer`` injected into request/response handlers.

    The returned callable has the handler-declared shape
    ``(terminal_topic, correlation_id, timeout_seconds) -> dict | None``. It runs
    the async correlate-and-wait coroutine on a dedicated event loop in a worker
    thread and blocks the caller on the result, so it is safe to call from a sync
    handler executing on the runtime's own (already-running) event loop without
    deadlocking that loop.
    """

    def _consume(
        terminal_topic: str, correlation_id: str, timeout_seconds: float
    ) -> dict[str, object] | None:
        result_box: dict[str, dict[str, object] | None] = {}
        error_box: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result_box["value"] = asyncio.run(
                    _await_correlated_terminal(
                        event_bus=event_bus,
                        terminal_topic=terminal_topic,
                        correlation_id=correlation_id,
                        timeout_seconds=timeout_seconds,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 — re-raised on caller thread
                error_box["error"] = exc

        thread = threading.Thread(
            target=_runner,
            name=f"terminal-consumer-{handler_name}",
            daemon=True,
        )
        thread.start()
        # Join with a small grace beyond the correlate timeout so a clean Kafka
        # shutdown completes; the coroutine already enforces timeout_seconds.
        thread.join(timeout=timeout_seconds + _ASSIGN_TIMEOUT_CAP_SECONDS + 5.0)

        if thread.is_alive():
            logger.warning(
                "terminal-event consumer: correlate thread for %s did not finish "
                "within grace window (topic=%s, cid=%s)",
                handler_name,
                terminal_topic,
                correlation_id,
            )
            return None
        if "error" in error_box:
            logger.warning(
                "terminal-event consumer: correlate failed for %s (topic=%s, "
                "cid=%s): %s",
                handler_name,
                terminal_topic,
                correlation_id,
                error_box["error"],
            )
            return None
        return result_box.get("value")

    return _consume


__all__: list[str] = [
    "EventConsumer",
    "make_terminal_event_consumer",
]
