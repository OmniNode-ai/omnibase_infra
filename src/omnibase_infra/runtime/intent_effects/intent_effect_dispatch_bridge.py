# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic intent-effect bridge to a write-effect handler's canonical handle().

One adapter for every intent-emitting audit/projection consumer (OMN-14516): it
adapts ``ProtocolIntentEffect.execute(payload, *, correlation_id)`` to the
target's ``handle()`` and owns the conversion to the shape that ``handle()``
declares (OMN-21700):

- ``handle(request: <pydantic model>)`` gets the validated model, with a
  dispatch envelope carrying the intent's correlation_id bound for the call.
- any other ``handle()`` gets ``{"payload": ..., "correlation_id": ...}``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, get_type_hints
from uuid import UUID

from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope


class DispatchTarget(Protocol):
    async def handle(self, envelope: object) -> object: ...


def _request_model(handle: Callable[..., object]) -> type[BaseModel] | None:
    params = list(inspect.signature(handle).parameters)
    try:
        hint = get_type_hints(handle).get(params[0]) if len(params) == 1 else None
    except NameError:  # TYPE_CHECKING-only annotation: not a runtime model
        return None
    return hint if isinstance(hint, type) and issubclass(hint, BaseModel) else None


class IntentEffectDispatchBridge:
    """Adapt a write-effect handler's ``handle()`` to ``ProtocolIntentEffect``."""

    def __init__(self, target: DispatchTarget) -> None:
        self._target = target
        self._model = _request_model(target.handle)

    async def execute(
        self, payload: object, *, correlation_id: UUID | None = None
    ) -> None:
        model = self._model
        if model is None:
            await self._target.handle(
                {"payload": payload, "correlation_id": correlation_id}
            )
            return
        request = (
            payload if isinstance(payload, model) else model.model_validate(payload)
        )
        envelope = (
            ModelEventEnvelope[object](
                payload=request, correlation_id=correlation_id, tenant_id=None
            )
            if correlation_id is not None
            else None
        )
        with bind_dispatch_envelope(envelope):
            await self._target.handle(request)


__all__ = ["IntentEffectDispatchBridge"]
