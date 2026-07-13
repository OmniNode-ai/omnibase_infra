# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic intent-effect bridge to a write-effect handler's canonical handle().

One adapter for every intent-emitting audit/projection consumer (OMN-14516).

Replaces the former per-node ``IntentEffect*`` bridge classes
(``IntentEffectLedgerAppend`` / ``IntentEffectBuildLoopAppend`` /
``IntentEffectPrStateUpsert``). Each of those hand-wrote the same shape: take an
intent payload, coerce it to a per-node typed payload, and call a per-node method
(``append`` / ``upsert``). That per-node surface is exactly what the kernel had
to know by NAME, which is why a new audit/projection consumer silently died until
someone remembered to hand-add it to the kernel's result-applier allowlist.

Every write-effect handler already exposes the canonical
``handle(envelope) -> ModelHandlerOutput`` dispatch entrypoint (the same one the
runtime's auto-wiring binds), and that entrypoint already coerces the payload and
resolves the correlation_id. This bridge therefore needs to know nothing about
any specific node — it wraps *any* handler with a canonical ``handle()`` and
adapts the ``ProtocolIntentEffect.execute(payload, *, correlation_id)`` surface to
it. The kernel derives which handler to wrap from the projection contract's
``intent_consumption.intent_routing_table`` — no by-name lookup, no bespoke class.
"""

from __future__ import annotations

import logging
from typing import Protocol
from uuid import UUID

logger = logging.getLogger(__name__)


class DispatchTarget(Protocol):
    """Minimal surface the bridge requires: an async canonical ``handle()``."""

    async def handle(self, envelope: object) -> object: ...


class IntentEffectDispatchBridge:
    """Adapt a write-effect handler's ``handle()`` to ``ProtocolIntentEffect``.

    The dispatch path delivers the emitted intent to
    ``DispatchResultApplier`` -> ``IntentExecutor``, which routes on
    ``payload.intent_type`` and calls ``execute(payload, correlation_id=...)`` on
    the registered effect. This bridge forwards that payload to the write-effect
    handler through the very same dict-shaped envelope the live runtime dispatch
    path materializes (``{"payload": ..., "correlation_id": ...}``), so the
    handler's own ``handle()`` performs the payload coercion and persistence.
    """

    def __init__(self, target: DispatchTarget) -> None:
        self._target = target

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        await self._target.handle(
            {"payload": payload, "correlation_id": correlation_id}
        )
        logger.debug(
            "intent bridged to %s.handle() (correlation_id=%s)",
            type(self._target).__name__,
            str(correlation_id) if correlation_id is not None else None,
        )


__all__ = ["IntentEffectDispatchBridge"]
