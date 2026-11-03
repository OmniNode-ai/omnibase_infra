# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ServiceHealth glue for the shared host listener (OMN-22377).

Kept out of ``services/health_checker.py`` (a baselined file that only shrinks):
the probe application factory, the opt-in start behind ``/run/onex-handover``,
the SIGTERM latch, the drain and the ``/ready`` body fields of rule A4.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.typedefs import Middleware

from omnibase_core.types import JsonType
from omnibase_infra.handlers.shared_listener import (
    HANDOVER_MOUNT,
    SharedListener,
    build_shared_listener,
)
from omnibase_infra.models.model_shared_listener_policy import (
    ModelSharedListenerPolicy,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.http_mount import AsgiMounts
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)


class MixinSharedListener:
    """Probe app factory and shared-listener lifecycle for ``ServiceHealth``."""

    _shared_listener: SharedListener | None = None

    if TYPE_CHECKING:
        _http_mounts: AsgiMounts
        _runtime: RuntimeHostProcess | None

        _handle_health: Callable[[web.Request], Awaitable[web.Response]]
        _handle_readiness: Callable[[web.Request], Awaitable[web.Response]]
        _handle_health_detailed: Callable[[web.Request], Awaitable[web.Response]]
        _handle_skill: Callable[[web.Request], Awaitable[web.Response]]
        _handle_introspection_manifest: Callable[[web.Request], Awaitable[web.Response]]

    def _build_app(self, *extra_middlewares: Middleware) -> web.Application:
        """The probe application: one per listener (in-container, shared host)."""
        app = web.Application(
            middlewares=[self._http_mounts.middleware, *extra_middlewares]
        )
        # Register routes (OMN-1931: /ready has its own handler)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/ready", self._handle_readiness)
        # OMN-519: Detailed diagnostics endpoint
        app.router.add_get("/health/detailed", self._handle_health_detailed)
        # OMN-10860: Skill dispatch endpoint
        app.router.add_post("/skill", self._handle_skill)
        # OMN-11198: Manifest introspection endpoint (internal only)
        app.router.add_get(
            "/v1/introspection/manifest", self._handle_introspection_manifest
        )
        return app

    async def _start_shared_listener(self) -> None:
        """Opt-in: build and start the listener only behind the handover mount."""
        if not HANDOVER_MOUNT.is_dir():
            return
        self._shared_listener = build_shared_listener(
            policy=ModelSharedListenerPolicy.from_environ(os.environ),
            build_app=self._build_app,
            get_runtime=lambda: self._runtime,
            handover_dir=HANDOVER_MOUNT,
        )
        self._shared_listener.start()

    def _with_listener_fields(
        self, details: dict[str, JsonType]
    ) -> dict[str, JsonType]:
        """Add ``role``, ``epoch``, ``listener_attached`` and ``consuming`` (A4)."""
        if self._shared_listener is None:
            return details
        return {**details, **self._shared_listener.ready_fields()}

    @property
    def shared_listener(self) -> SharedListener | None:
        """The shared host listener; None without the handover mount."""
        return self._shared_listener

    def mark_stopping(self) -> None:
        """Set the ``stopping`` latch; a plain flag, safe in a signal handler (A5)."""
        if self._shared_listener is not None:
            self._shared_listener.mark_stopping()

    async def drain_shared_listener(self) -> None:
        """Stop accepting and answer what was accepted, before readiness changes."""
        if self._shared_listener is None:
            return
        try:
            await self._shared_listener.stop()
        except (OSError, RuntimeError) as exc:
            logger.warning("shared listener drain failed: %r", exc)


__all__: list[str] = ["MixinSharedListener"]
