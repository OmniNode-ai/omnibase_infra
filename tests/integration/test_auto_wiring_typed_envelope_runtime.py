# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration regressions for runtime auto-wiring typed envelope dispatch."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import BaseModel

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef


class RuntimeTypedPayload(BaseModel):
    correlation_id: UUID
    value: str


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_model_dispatch_preserves_envelope_handler_signature() -> None:
    """Declared event_model routes still support ProtocolMessageHandler shape."""
    received: list[ModelEventEnvelope[RuntimeTypedPayload]] = []

    class Handler:
        async def handle(
            self,
            envelope: ModelEventEnvelope[RuntimeTypedPayload],
        ) -> ModelHandlerOutput[object]:
            received.append(envelope)
            return ModelHandlerOutput(
                input_envelope_id=envelope.envelope_id,
                correlation_id=envelope.payload.correlation_id,
                handler_id="handler-runtime-typed-envelope",
                node_kind=EnumNodeKind.ORCHESTRATOR,
                events=(),
                intents=(),
                projections=(),
                result=None,
                processing_time_ms=0.0,
                timestamp=envelope.envelope_timestamp,
            )

    correlation_id = UUID("33333333-3333-4333-8333-333333333333")
    callback = _make_dispatch_callback(
        Handler(),
        ModelHandlerRef(name="RuntimeTypedPayload", module=__name__),
    )

    result = await callback(
        {
            "payload": {
                "correlation_id": str(correlation_id),
                "value": "heartbeat",
            },
            "__debug_trace": {
                "topic": "onex.evt.platform.node-heartbeat.v1",
                "event_type": "platform.node-heartbeat",
                "correlation_id": str(correlation_id),
            },
        }
    )

    assert len(received) == 1
    assert isinstance(received[0], ModelEventEnvelope)
    assert isinstance(received[0].payload, RuntimeTypedPayload)
    assert received[0].payload.value == "heartbeat"
    assert received[0].correlation_id == correlation_id
    assert result is not None
    assert result.correlation_id == correlation_id
