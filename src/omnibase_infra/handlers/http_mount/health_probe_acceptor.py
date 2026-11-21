# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Accept health-port connections off the runtime event loop (OMN-26316).

The runtime's HTTP health server is an aiohttp application on the kernel's
asyncio loop. Any synchronous work on that loop (contract YAML parsing and
module imports during auto-wiring, a CPU-bound handler, a retry loop that never
awaits) also stops ``/health`` from answering. A probe then reads no answer at
all. The container healthcheck reports that as ``probe_unreachable``, which
names neither the runtime's state nor the cause.

This acceptor owns the listening socket and accepts on its own thread. A
heartbeat task on the loop records when the loop last ran a task:

* each accepted connection is offered to the loop, which claims it and hands
  it to the aiohttp server, so it is answered exactly as before;
* when the loop has not claimed it within ``wait_budget_seconds``, the acceptor
  withdraws the offer and answers itself with HTTP 503 and a ``degraded`` body
  whose ``degraded_reason`` is
  :attr:`EnumHealthDegradedReason.EVENT_LOOP_UNRESPONSIVE`.

A claim and a withdrawal take the same lock, so exactly one side answers a
connection, including one offered just before the loop stalls.

Each accepted connection is routed on its own short-lived thread. A stalled
loop makes every connection wait out ``wait_budget_seconds``; routing them one
after another on the accept thread served a second probe at about twice the
budget, past the container healthcheck's 5 s timeout, so it read
``probe_unreachable`` for a runtime that was alive (OMN-26417).

The 503 fails ``curl -sf`` and the semantic container healthcheck just as a
silent socket did, so no gate becomes easier to pass. The difference is that
the probe gets an answer within its timeout, and the answer names the cause.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Callable

from omnibase_infra.models.health.enum_health_degraded_reason import (
    EnumHealthDegradedReason,
)
from omnibase_infra.models.health.model_event_loop_stall_details import (
    ModelEventLoopStallDetails,
)
from omnibase_infra.models.health.model_event_loop_stall_health import (
    ModelEventLoopStallHealth,
)

logger = logging.getLogger(__name__)

# The semantic container healthcheck's default HTTP timeout is 5 s and the
# compose probes use 10 s, so a stalled-loop answer must be sent within 5 s.
DEFAULT_WAIT_BUDGET_SECONDS = 4.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 0.25
_ACCEPT_POLL_SECONDS = 0.25
_REQUEST_READ_SECONDS = 0.5
# Connections routed at once. Each waits at most the wait budget, so a stalled
# loop holds at most this many short-lived threads; one past the cap is closed.
_MAX_INFLIGHT_CONNECTIONS = 32


def render_event_loop_stall_response(
    *, version: str, stalled_seconds: float, wait_budget_seconds: float
) -> bytes:
    """Render the HTTP/1.1 503 response for a stalled event loop."""
    body = ModelEventLoopStallHealth(
        version=version,
        details=ModelEventLoopStallDetails(
            degraded_reason=EnumHealthDegradedReason.EVENT_LOOP_UNRESPONSIVE,
            event_loop_stalled_seconds=round(max(stalled_seconds, 0.0), 3),
            wait_budget_seconds=wait_budget_seconds,
        ),
    ).model_dump_json()
    payload = body.encode("utf-8")
    head = (
        "HTTP/1.1 503 Service Unavailable\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode("ascii") + payload


class ConnectionOffer:
    """One connection offered to the loop; claimed by it or withdrawn."""

    __slots__ = ("claimed", "lock", "withdrawn")

    def __init__(self) -> None:
        self.claimed = threading.Event()
        self.lock = threading.Lock()
        self.withdrawn = False


class HealthProbeAcceptor:
    """Owns the health port's socket and accepts connections on a thread."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        version: str,
        protocol_factory: Callable[[], asyncio.BaseProtocol],
        wait_budget_seconds: float = DEFAULT_WAIT_BUDGET_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._host = host
        self._port = port
        self._version = version
        self._protocol_factory = protocol_factory
        self._wait_budget_seconds = wait_budget_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping: threading.Event | None = None
        self._last_beat: float | None = None
        self._beat_task: asyncio.Task[None] | None = None
        self._handoffs: set[asyncio.Task[object]] = set()
        self._inflight = threading.BoundedSemaphore(_MAX_INFLIGHT_CONNECTIONS)

    @property
    def port(self) -> int:
        """The bound port (the OS-assigned one when constructed with 0)."""
        return self._port

    def event_loop_stalled_seconds(self) -> float:
        """Seconds since the loop's heartbeat task last ran."""
        if self._last_beat is None:
            return 0.0
        return max(0.0, time.monotonic() - self._last_beat)

    async def start(self) -> None:
        """Bind the port, start the heartbeat and the accept thread."""
        self._loop = asyncio.get_running_loop()
        sock = socket.create_server((self._host, self._port), backlog=128)
        sock.settimeout(_ACCEPT_POLL_SECONDS)
        self._sock = sock
        self._port = int(sock.getsockname()[1])
        self._stopping = threading.Event()
        self._last_beat = time.monotonic()
        self._beat_task = self._loop.create_task(self._beat())
        self._thread = threading.Thread(
            target=self._accept_loop, name="onex-health-acceptor", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        """Stop accepting, close the socket and end the heartbeat; never raises."""
        if self._stopping is not None:
            self._stopping.set()
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, _ACCEPT_POLL_SECONDS * 8)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._beat_task is not None:
            self._beat_task.cancel()
            try:
                await self._beat_task
            except asyncio.CancelledError:
                pass
            self._beat_task = None
        self._thread = None

    async def _beat(self) -> None:
        while True:
            self._last_beat = time.monotonic()
            await asyncio.sleep(self._heartbeat_interval_seconds)

    def _accept_loop(self) -> None:
        sock = self._sock
        stopping = self._stopping
        if sock is None or stopping is None:
            return
        while not stopping.is_set():
            try:
                conn, _ = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self._dispatch(conn)

    def _dispatch(self, conn: socket.socket) -> None:
        """Route ``conn`` on its own thread so a stalled wait blocks no other probe."""
        if not self._inflight.acquire(blocking=False):
            logger.warning(
                "health acceptor has %d connections in flight; closing one",
                _MAX_INFLIGHT_CONNECTIONS,
            )
            conn.close()
            return
        try:
            threading.Thread(
                target=self._route_connection,
                args=(conn,),
                name="onex-health-route",
                daemon=True,
            ).start()
        except RuntimeError:
            self._inflight.release()
            logger.exception("health acceptor failed to start a routing thread")
            conn.close()

    def _route_connection(self, conn: socket.socket) -> None:
        try:
            self._route(conn)
        except Exception:
            logger.exception("health acceptor failed to route a connection")
            conn.close()
        finally:
            self._inflight.release()

    def _route(self, conn: socket.socket) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            conn.close()
            return
        offer = ConnectionOffer()
        try:
            loop.call_soon_threadsafe(self._hand_off, conn, offer)
        except RuntimeError:
            conn.close()
            return
        if offer.claimed.wait(self._wait_budget_seconds):
            return
        with offer.lock:
            if offer.claimed.is_set():
                return
            offer.withdrawn = True
        self._answer_stalled(conn)

    def _hand_off(self, conn: socket.socket, offer: ConnectionOffer) -> None:
        with offer.lock:
            if offer.withdrawn:
                return
            offer.claimed.set()
        loop = self._loop
        if loop is None:
            conn.close()
            return
        conn.setblocking(False)
        task: asyncio.Task[object] = loop.create_task(
            loop.connect_accepted_socket(self._protocol_factory, conn)
        )
        self._handoffs.add(task)
        task.add_done_callback(self._handoffs.discard)

    def _answer_stalled(self, conn: socket.socket) -> None:
        stalled = self.event_loop_stalled_seconds()
        logger.warning(
            "health probe answered off-loop: event loop ran no task for %.1fs "
            "(degraded_reason=%s)",
            stalled,
            EnumHealthDegradedReason.EVENT_LOOP_UNRESPONSIVE.value,
        )
        try:
            conn.settimeout(_REQUEST_READ_SECONDS)
            try:
                conn.recv(65536)
            except OSError:
                pass
            conn.sendall(
                render_event_loop_stall_response(
                    version=self._version,
                    stalled_seconds=stalled,
                    wait_budget_seconds=self._wait_budget_seconds,
                )
            )
        except OSError:
            pass
        finally:
            conn.close()


__all__ = [
    "DEFAULT_WAIT_BUDGET_SECONDS",
    "HealthProbeAcceptor",
    "render_event_loop_stall_response",
]
