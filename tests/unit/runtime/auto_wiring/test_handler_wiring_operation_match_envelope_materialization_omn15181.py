# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test for OMN-15181 Finding 4.

Live incident: a real, grant-authorized ``onex run-node node_redeploy_orchestrator``
dispatch against the PROD Kafka broker crashed inside the deployed
``omninode-prod-runtime-effects`` container BEFORE the orchestrator ever emitted
the grant-resolve command:

    Dispatcher 'dispatcher.auto.node_redeploy_orchestrator.HandlerRedeployOrchestrator...'
    failed: AttributeError: 'dict' object has no attribute 'event_type'

Root cause: ``node_redeploy_orchestrator/contract.yaml`` wires
``HandlerRedeployOrchestrator`` via ``routing_strategy: operation_match`` and
declares only an ``input_model:`` (not ``event_model:``) on the handler entry
(``discovery.py`` only populates ``ModelHandlerRoutingEntry.event_model`` from an
``event_model:`` key), so ``_make_dispatch_callback`` is built with
``event_model=None``. ``HandlerRedeployOrchestrator.handle(self, envelope:
ModelEventEnvelope[Any])`` accepts an envelope (its sole parameter is literally
named ``envelope``), so ``_handler_accepts_event_envelope`` returns True.

Before this fix, the ``event_model is None`` branch of ``_callback`` handled
that case by leaving ``dispatch_arg`` as the untouched dispatch-engine input —
the RAW ``dict`` built by
``MessageDispatchEngine._materialize_envelope_with_bindings``
(``{"payload": ..., "__bindings": {...}, "__debug_trace": {...}}``), never a
``ModelEventEnvelope`` instance. Only the sibling ``event_model is not None``
(``payload_type_match``) branch called ``_materialize_typed_event_envelope`` /
``_materialize_raw_event_envelope`` before invoking an envelope-accepting
handler. The handler's first line (``envelope.event_type``) then crashed with
exactly the live error.

This test reproduces the crash with the exact materialized-dict shape the
dispatch engine hands the callback, and pins the fix: an envelope-accepting
handler reached via ``operation_match`` (``event_model=None``) must receive a
real ``ModelEventEnvelope`` with ``event_type``/``correlation_id``/``payload``
populated from the materialized dict, regardless of routing strategy.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback


def _materialized_dispatch_dict(
    *, event_type: str, correlation_id: str, payload: dict[str, object]
) -> dict[str, object]:
    """Build the exact shape ``MessageDispatchEngine._materialize_envelope_with_bindings``
    hands every dispatcher — a plain dict, never a ``ModelEventEnvelope`` instance.
    """
    return {
        "payload": payload,
        "__bindings": {},
        "__debug_trace": {
            "event_type": event_type,
            "correlation_id": correlation_id,
            "trace_id": None,
            "causation_id": None,
            "topic": "onex.cmd.omnimarket.redeploy-start.v1",
            "timestamp": None,
            "partition_key": None,
        },
    }


class _EnvelopeAcceptingOrchestratorHandler:
    """Mirrors ``HandlerRedeployOrchestrator.handle`` — sole param named ``envelope``."""

    def __init__(self) -> None:
        self.received: ModelEventEnvelope[object] | None = None

    async def handle(self, envelope: ModelEventEnvelope[object]) -> None:
        # The live crash site: HandlerRedeployOrchestrator.handle line 1 is
        # ``event_type = envelope.event_type or ""`` — first attribute access.
        _ = envelope.event_type
        self.received = envelope


@pytest.mark.unit
class TestOperationMatchEnvelopeAcceptingHandlerMaterialization:
    async def test_raw_materialized_dict_does_not_crash_envelope_handler(
        self,
    ) -> None:
        """RED before the fix: AttributeError: 'dict' object has no attribute
        'event_type'. GREEN after: the handler receives a real ModelEventEnvelope.
        """
        handler = _EnvelopeAcceptingOrchestratorHandler()
        # event_model=None reproduces the operation_match wiring of
        # node_redeploy_orchestrator/contract.yaml (no event_model: key on the
        # handler_routing entry).
        callback = _make_dispatch_callback(handler, event_model=None)

        correlation_id = str(uuid4())
        dispatch_dict = _materialized_dispatch_dict(
            event_type="onex.cmd.omnimarket.redeploy-start.v1",
            correlation_id=correlation_id,
            payload={
                "correlation_id": correlation_id,
                "runtime_lane": "prod",
                "image_digest": "sha256:ddb296f8...",
            },
        )

        # Before the fix this line raised AttributeError inside handler.handle().
        await callback(dispatch_dict)

        assert handler.received is not None
        assert isinstance(handler.received, ModelEventEnvelope)
        assert handler.received.event_type == "onex.cmd.omnimarket.redeploy-start.v1"
        assert str(handler.received.correlation_id) == correlation_id
        assert handler.received.payload["runtime_lane"] == "prod"  # type: ignore[index]
