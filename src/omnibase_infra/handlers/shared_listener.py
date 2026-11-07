# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime shared listener: accept on the host socket only while ready (OMN-22377).

Rules A1 to A6 of the OMN-22151 stage-2 design (T1). One ``SharedListener`` per
runtime holds the listener state machine behind one lock, the ``stopping`` and
``revoking`` latches, the drain (A3) and the hold (A6). The broker wire shape is
documented in :mod:`omnibase_infra.handlers.shared_listener_broker_client`.

Opt-in: ``ServiceHealth`` builds a listener only when ``/run/onex-handover`` is
mounted; without the mount none of this code runs.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import weakref
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.typedefs import Handler, Middleware

from omnibase_core.types import JsonType
from omnibase_infra.enums.enum_detach_answer import EnumDetachAnswer
from omnibase_infra.enums.enum_shared_listener_state import EnumSharedListenerState
from omnibase_infra.handlers.shared_listener_broker_client import (
    BrokerSession,
    BrokerUnavailableError,
    fetch_listener,
    unix_connector,
)
from omnibase_infra.models.model_shared_listener_policy import (
    ModelSharedListenerPolicy,
)
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)

HANDOVER_MOUNT = Path("/run/onex-handover")
_READY_PATH = "/ready"

State = EnumSharedListenerState


class AttachAbandonedError(Exception):
    """A latch or a not-ready reading stopped the attach before ``confirmed``."""


class SharedListener:
    """Listener state machine of one runtime over the shared host socket."""

    def __init__(
        self,
        *,
        policy: ModelSharedListenerPolicy,
        build_app: Callable[[Middleware], web.Application],
        is_ready: Callable[[], Awaitable[bool]],
        is_passive: Callable[[], bool],
        epoch: Callable[[], int],
        is_consuming: Callable[[], bool],
        connect: Callable[[], socket.socket] | None = None,
        handover_dir: Path = HANDOVER_MOUNT,
    ) -> None:
        self._policy = policy
        self._build_app = build_app
        self._is_ready = is_ready
        self._is_passive = is_passive
        self._epoch = epoch
        self._is_consuming = is_consuming
        self._connect = connect or unix_connector(handover_dir)
        self._watch_s = policy.passive_ready_watch_ms / 1000

        self._lock = asyncio.Lock()
        self._state = State.DETACHED
        self.stopping = False
        self.revoking = False

        self._runner: web.AppRunner | None = None
        self._site: web.SockSite | None = None
        self._session: BrokerSession | None = None

        self._watch_task: asyncio.Task[None] | None = None
        self._attach_task: asyncio.Task[bool] | None = None
        self._accepting = False
        self._attach_in_phase2 = False
        self._drain_task: asyncio.Task[None] | None = None

        self._active: weakref.WeakKeyDictionary[object, int] = (
            weakref.WeakKeyDictionary()
        )
        self._served: weakref.WeakSet[object] = weakref.WeakSet()

    # ------------------------------------------------------------------ views

    @property
    def state(self) -> EnumSharedListenerState:
        return self._state

    @property
    def accepting(self) -> bool:
        """True while this process holds an open duplicate of the shared socket."""
        return self._accepting

    @property
    def accepted_connections(self) -> int:
        """Connections this process accepted from the shared socket and still holds."""
        server = self._runner.server if self._runner is not None else None
        return len(server.connections) if server is not None else 0

    def ready_fields(self) -> dict[str, JsonType]:
        """The ``/ready`` body fields of rule A4."""
        return {
            "role": "passive" if self._is_passive() else "active",
            "epoch": self._epoch(),
            "listener_attached": self._state in (State.ATTACHED, State.HOLDING),
            "consuming": self._is_consuming(),
        }

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the readiness monitor, which also drives attach (A1, A2)."""
        if self._watch_task is None:
            self._watch_task = asyncio.ensure_future(self._watch())

    def mark_stopping(self) -> None:
        """Set the ``stopping`` latch. A plain flag: safe in a signal handler."""
        self.stopping = True

    async def stop(self) -> None:
        """SIGTERM path (A5): latch, join the attach, detach by A6, drain (A3).

        Returns only after the drain has finished, so the readiness teardown
        that follows can no longer change what an accepted connection reads.
        """
        self.stopping = True
        if self._watch_task is not None:
            self._watch_task.cancel()
            await asyncio.gather(self._watch_task, return_exceptions=True)
            self._watch_task = None
        attach = self._attach_task
        if attach is not None and not attach.done():
            if not self._attach_in_phase2:
                attach.cancel()
            await asyncio.gather(attach, return_exceptions=True)
        async with self._lock:
            if self._state in (State.ATTACHED, State.HOLDING):
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._policy.detach_successor_wait_seconds
                while True:
                    answer = await self._ask_detach()
                    if answer in (EnumDetachAnswer.GRANTED, EnumDetachAnswer.ALONE):
                        break
                    if loop.time() >= deadline:
                        logger.warning(
                            "shared listener: no detach grant within %ss; "
                            "detaching (R1: successor did not attach)",
                            self._policy.detach_successor_wait_seconds,
                        )
                        break
                    await asyncio.sleep(self._watch_s)
                self._begin_drain()
        if self._drain_task is not None:
            await asyncio.gather(self._drain_task, return_exceptions=True)

    # ---------------------------------------------------------------- attach

    def _blocked(self) -> bool:
        return self.stopping or self.revoking

    async def attach(self) -> bool:
        """Try to attach (A1). True when the runtime is a confirmed acceptor."""
        if self._attach_task is not None and not self._attach_task.done():
            return await asyncio.shield(self._attach_task)
        self._attach_task = asyncio.ensure_future(self._attach())
        return await asyncio.shield(self._attach_task)

    async def _attach(self) -> bool:
        async with self._lock:
            if self._state is not State.DETACHED or self._blocked():
                return False
            if not await self._is_ready() or self._blocked():
                return False
            self._state = State.ATTACHING
            self._attach_in_phase2 = False
            session: BrokerSession | None = None
            listener: socket.socket | None = None
            try:
                session, listener = await fetch_listener(
                    self._connect,
                    self._policy.pool,
                    self._policy.probe_client_timeout_seconds,
                )
                if self._blocked() or not await self._is_ready() or self._blocked():
                    raise AttachAbandonedError
                self._attach_in_phase2 = True
                runner = web.AppRunner(
                    self._build_app(self._middleware),
                    shutdown_timeout=self._policy.listener_first_request_seconds,
                )
                await runner.setup()
                self._runner = runner
                self._site = web.SockSite(runner, listener)
                self._session = session
                await self._site.start()
                self._accepting = True
                await session.confirmed()
                self._state = State.ATTACHED
                return True
            except (AttachAbandonedError, BrokerUnavailableError, OSError) as exc:
                logger.debug("shared listener attach abandoned: %r", exc)
                await self._abandon(session, listener)
                return False
            except BaseException:
                await self._abandon(session, listener)
                raise
            finally:
                self._attach_in_phase2 = False

    async def _abandon(
        self, session: BrokerSession | None, listener: socket.socket | None
    ) -> None:
        """Return to ``detached``: close the fd and the provisional session."""
        if self._site is not None:
            # The site had started: drain it like any other.
            self._state = State.DRAINING
            await self._run_drain()
            return
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if listener is not None:
            listener.close()
        if session is not None:
            session.close()
        self._session = None
        self._state = State.DETACHED

    # ----------------------------------------------------- detach permission

    async def _ask_detach(self) -> EnumDetachAnswer | None:
        """Ask B for a detach; None when B is unreachable."""
        if self._session is None:
            return None
        try:
            return await self._session.detach()
        except (BrokerUnavailableError, ValueError):
            return None

    async def _withdraw_or_hold(self) -> None:
        """Ask for a detach; leave only on ``granted``, otherwise hold (A6)."""
        async with self._lock:
            if self._state not in (State.ATTACHED, State.HOLDING):
                return
            if await self._ask_detach() is EnumDetachAnswer.GRANTED:
                self._begin_drain()
            else:
                self._state = State.HOLDING

    async def begin_revocation(self) -> None:
        """``on_partitions_revoked``: latch first, then ask B; return once resolved."""
        self.revoking = True
        await self._withdraw_or_hold()
        if self._drain_task is not None:
            await asyncio.gather(self._drain_task, return_exceptions=True)

    def end_revocation(self) -> None:
        """``on_partitions_assigned`` finished: the attach latch clears."""
        self.revoking = False

    # ------------------------------------------------------------------ drain

    def _begin_drain(self) -> None:
        """Mark ``draining`` and start the drain. Caller holds the lock."""
        self._state = State.DRAINING
        self._drain_task = asyncio.ensure_future(self._run_drain())

    async def _run_drain(self) -> None:
        """A3: stop accepting, serve what was accepted, close the session."""
        loop = asyncio.get_running_loop()
        site, runner, session = self._site, self._runner, self._session
        try:
            if site is not None:
                # Closes this process's duplicate; never shutdown() the socket,
                # so connections still queued stay for the other acceptor.
                await site.stop()
            self._accepting = False
            if runner is not None and runner.server is not None:
                started = loop.time()
                first_deadline = started + self._policy.listener_first_request_seconds
                hard_deadline = (
                    first_deadline + self._policy.probe_client_timeout_seconds
                )
                while True:
                    connections = list(runner.server.connections)
                    if not connections or loop.time() >= hard_deadline:
                        break
                    now = loop.time()
                    for handler in connections:
                        transport = handler.transport
                        if transport is None or self._active.get(transport, 0) > 0:
                            continue
                        if transport in self._served or now >= first_deadline:
                            handler.close()  # idle keep-alive, or never spoke
                    await asyncio.sleep(self._watch_s)
            if runner is not None:
                await runner.cleanup()
        finally:
            if session is not None:
                session.close()
            self._site = self._runner = self._session = None
            self._state = State.DETACHED

    # ---------------------------------------------------------------- monitor

    async def _watch(self) -> None:
        while not self.stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("shared listener monitor tick failed")
            await asyncio.sleep(self._watch_s)

    async def _tick(self) -> None:
        ready = await self._is_ready()
        if self._state is State.DETACHED:
            if ready and not self._blocked():
                await self.attach()
        elif self._state in (State.ATTACHED, State.HOLDING):
            if (not ready and self._is_passive()) or (
                self._state is State.HOLDING and self.revoking
            ):
                await self._withdraw_or_hold()
            elif self._state is State.HOLDING and ready and not self.revoking:
                self._state = State.ATTACHED

    # ------------------------------------------------------------- middleware

    @web.middleware
    async def _middleware(
        self, request: web.Request, handler: Handler
    ) -> web.StreamResponse:
        transport = request.transport
        if transport is not None:
            self._active[transport] = self._active.get(transport, 0) + 1
        try:
            if (
                self._state is State.ATTACHED
                and self._is_passive()
                and not self.stopping
                and not await self._is_ready()
            ):
                await self._withdraw_or_hold()
            if self._state is State.HOLDING and request.path == _READY_PATH:
                await self._hold()
            response = await handler(request)
        finally:
            if transport is not None:
                self._active[transport] = self._active.get(transport, 1) - 1
                self._served.add(transport)
        if self._state is State.DRAINING:
            response.force_close()
        return response

    async def _hold(self) -> None:
        """A6: hold a request until readiness returns or the hold bound passes."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._policy.last_attached_hold_seconds
        while self._state is State.HOLDING and loop.time() < deadline:
            if not self.revoking and await self._is_ready():
                return
            await asyncio.sleep(min(self._watch_s, max(deadline - loop.time(), 0)))


def build_shared_listener(
    *,
    policy: ModelSharedListenerPolicy,
    build_app: Callable[[Middleware], web.Application],
    get_runtime: Callable[[], RuntimeHostProcess | None],
    handover_dir: Path | None = None,
) -> SharedListener:
    """Wire a listener to a runtime: readiness and the A4 identity fields.

    ``role``, ``epoch`` and ``consuming`` come from the runtime when it exposes
    them (the standby lane, T2, supplies them); an active runtime without them
    reads ``active``, epoch 0 and consuming.
    """

    async def is_ready() -> bool:
        runtime = get_runtime()
        if runtime is None:
            return False
        details = await runtime.readiness_check(
            correlation_id=generate_correlation_id()
        )
        return bool(details.get("ready", False))

    def is_passive() -> bool:
        return getattr(get_runtime(), "role", "active") == "passive"

    def is_consuming() -> bool:
        runtime = get_runtime()
        return bool(getattr(runtime, "is_consuming", not is_passive()))

    return SharedListener(
        policy=policy,
        build_app=build_app,
        is_ready=is_ready,
        is_passive=is_passive,
        epoch=lambda: int(getattr(get_runtime(), "epoch", 0)),
        is_consuming=is_consuming,
        handover_dir=handover_dir or HANDOVER_MOUNT,
    )


__all__: list[str] = ["HANDOVER_MOUNT", "SharedListener", "build_shared_listener"]
