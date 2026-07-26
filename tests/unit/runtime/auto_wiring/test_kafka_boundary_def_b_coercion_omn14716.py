# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-then-GREEN seam proof for OMN-14716 — the Kafka auto-wiring adapter must
coerce a raw wire dict into a def-B handler's DECLARED input model, and must
SURFACE (not silently swallow) a handler/coercion crash.

Incident (2026-07-17, .201 stability-test, omnimarket candidate 55d82eb2,
correlation 8fbb14b9-31ad-4c6f-aa93-a9fec31137fa): a synthetic delegation
dispatched the RSD canary ``node_finding_aggregator_compute`` (#1781), a
canonical def-B handler ``handle(request: ModelFindingAggregatorInput)`` routed
by ``operation_match`` (no contract ``event_model``). The Kafka auto-wiring
adapter handed the raw materialized wire DICT straight to ``handle()``, so
``request.config`` raised ``AttributeError: 'dict' object has no attribute
'config'``. The engine catch-all converted that crash into a ``HANDLER_ERROR``
``ModelDispatchResult``, which the applier silently skipped — so both terminal
topics stayed HWM=0 and nothing terminalized.

``runtime_local`` already survives this exact ``operation_match`` case by
signature-introspecting the def-B ``handle()`` annotation and validating the
dict into the declared model (``LocalRuntimeBusAdapter._coercion_target_model_type``,
OMN-8724). These tests drive the ACTUAL Kafka path — a raw wire dict through
``_make_event_bus_callback`` → a real, frozen ``MessageDispatchEngine`` → the
``_make_dispatch_callback`` dispatcher → the bare def-B ``handle()`` — and assert
both parity mechanisms:

  1. the handler receives an INSTANCE of its declared pydantic input model, not a
     dict (FIX A: coercion at the adapter boundary — never a ``ModelEventEnvelope``);
  2. when the handler raises, the boundary SURFACES the failure via the existing
     OMN-14507 DLQ path instead of vanishing at HWM=0 (FIX B).

RED proof (verified against the pre-fix tree): test 1 records ``dict`` (the
materialized envelope) and fails the ``issubclass(..., BaseModel)`` assertion;
test 2's crash is converted to a silently-skipped ``HANDLER_ERROR`` result that
never reaches the DLQ, so ``_publish_raw_to_dlq`` is never awaited.

Neither test imports any fix-only symbol, so both run against the pre-fix code and
fail on the BEHAVIOURAL assertion (received type / DLQ await), not on an import.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    _make_dispatch_callback,
    _make_event_bus_callback,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

# Synthetic COMMAND topic; ``.cmd.`` makes EnumMessageCategory.from_topic() resolve
# COMMAND, matching the route registered below.
_TOPIC = "onex.cmd.omninode.finding-aggregate.v1"  # onex-topic-allow: test-only synthetic seam topic


class _ModelSeamConfig(BaseModel):
    """Nested model so the handler can touch ``request.config`` exactly like the
    real ``ModelFindingAggregatorInput`` the canary crashed on."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold: int = 1


class _ModelSeamInput(BaseModel):
    """The def-B handler's DECLARED input model — the coercion target."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    config: _ModelSeamConfig = Field(default_factory=_ModelSeamConfig)


class _ModelSeamOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    ok: bool = True


class _HandlerSeamProbe:
    """Canonical def-B handler: ``handle(self, request: ModelX) -> ModelY``.

    Imports NO ``ModelEventEnvelope`` (canonical shape); records the concrete
    type it was actually handed so the seam can assert coercion happened at the
    adapter boundary rather than inside the handler.
    """

    def __init__(self) -> None:
        self.received_types: list[type] = []

    async def handle(self, request: _ModelSeamInput) -> _ModelSeamOutput:
        self.received_types.append(type(request))
        # Touch the nested attr the real canary crashed on. A raw dict would
        # AttributeError here exactly like the incident; a coerced model will not.
        _ = request.config
        return _ModelSeamOutput(correlation_id=request.correlation_id)


class _HandlerSeamCrash:
    """def-B handler that raises inside ``handle()`` — the surfacing probe."""

    def __init__(self) -> None:
        self.received_types: list[type] = []

    async def handle(self, request: _ModelSeamInput) -> _ModelSeamOutput:
        self.received_types.append(type(request))
        raise RuntimeError(
            "finding-aggregator handler boom (OMN-14716 surfacing probe)"
        )


def _dlq_capable_event_bus() -> MagicMock:
    """A bus mock spec'd to EventBusKafka so only real attributes (including
    ``_publish_raw_to_dlq`` from MixinKafkaDlq) can be set — satisfies the
    transport-mock-lint gate (OMN-13026)."""
    bus = MagicMock(spec=EventBusKafka)
    bus._publish_raw_to_dlq = AsyncMock()
    return bus


def _frozen_engine_for(handler: object) -> MessageDispatchEngine:
    """Register ``handler`` as an ``operation_match`` (``event_model=None``)
    dispatcher on a real, frozen ``MessageDispatchEngine`` — the exact wiring
    shape the incident node used (no per-handler ``event_model``)."""
    engine = MessageDispatchEngine()
    # event_model=None is the operation_match parity case: no contract-declared
    # event model, so the dict→model coercion must come from signature
    # introspection of the def-B handle() annotation (OMN-14716 / OMN-8724).
    dispatcher = _make_dispatch_callback(handler, None)  # type: ignore[arg-type]
    engine.register_dispatcher(
        dispatcher_id="seam-dispatcher",
        dispatcher=dispatcher,
        category=EnumMessageCategory.COMMAND,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            # ONEX 5-segment wildcard (mirrors _derive_topic_pattern_from_topic):
            # onex.cmd.omninode.finding-aggregate.v1 -> *.cmd.omninode.finding-aggregate.*
            route_id="seam-route",
            topic_pattern="*.cmd.omninode.finding-aggregate.*",
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id="seam-dispatcher",
        )
    )
    engine.freeze()
    return engine


def _kafka_message(payload: dict[str, object]) -> MagicMock:
    """A raw Kafka message carrying JSON-encoded wire bytes — what the consumer
    hands ``_make_event_bus_callback`` before any envelope exists."""
    msg = MagicMock()
    msg.value = json.dumps(payload).encode("utf-8")
    return msg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_boundary_coerces_wire_dict_to_declared_def_b_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX A: the def-B handler must receive an INSTANCE of its declared input
    model, not the raw materialized wire dict.

    RED (pre-fix): the ``event_model is None`` branch passes the materialized
    envelope dict straight to ``handle()``, so ``received_types == [dict]`` and
    the handler AttributeErrors on ``request.config`` — the incident.
    """
    monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)
    handler = _HandlerSeamProbe()
    engine = _frozen_engine_for(handler)
    event_bus = _dlq_capable_event_bus()
    callback = _make_event_bus_callback(_TOPIC, engine, event_bus=event_bus)

    correlation_id = uuid4()
    wire = {"correlation_id": str(correlation_id), "config": {"threshold": 3}}
    await callback(_kafka_message(wire))  # drives the ACTUAL Kafka path

    assert handler.received_types, "handler was never invoked through the Kafka path"
    received = handler.received_types[0]
    assert issubclass(received, BaseModel) and received is _ModelSeamInput, (
        f"def-B handler received {received!r} at the Kafka adapter boundary — the "
        "adapter passed the raw wire dict instead of coercing it into the declared "
        "input model (OMN-14716). Expected _ModelSeamInput; a dict here is the "
        "'dict' object has no attribute 'config' incident."
    )
    # Happy path terminalizes normally: nothing routed to the DLQ.
    event_bus._publish_raw_to_dlq.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_boundary_surfaces_crashing_def_b_handler_to_dlq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX B: a def-B handler that raises must be SURFACED, not silently dropped
    at HWM=0.

    The engine catch-all converts the handler crash into a ``HANDLER_ERROR``
    ``ModelDispatchResult`` (it never re-raises), which the applier silently
    skips. With ``ONEX_BOUNDARY_DLQ_ENABLED`` the boundary must route that silent
    failure to the DLQ via the existing OMN-14507 surfacing path.

    RED (pre-fix): the ``HANDLER_ERROR`` result is applier-skipped and never
    reaches the swallow-surfacing path, so ``_publish_raw_to_dlq`` is never
    awaited.
    """
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")
    handler = _HandlerSeamCrash()
    engine = _frozen_engine_for(handler)
    event_bus = _dlq_capable_event_bus()
    callback = _make_event_bus_callback(_TOPIC, engine, event_bus=event_bus)

    correlation_id = uuid4()
    wire = {"correlation_id": str(correlation_id), "config": {"threshold": 1}}
    await callback(_kafka_message(wire))  # must never crash the consumer loop

    # FIX B: the crash is surfaced to the DLQ, not vanished at HWM=0.
    event_bus._publish_raw_to_dlq.assert_awaited_once()
    kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs
    assert kwargs["original_topic"] == _TOPIC
    assert kwargs["failure_type"] == "handler_exception"
    # FIX A still applied on this path: the crash proves the handler was reached
    # WITH a coerced model instance, never the raw dict.
    assert handler.received_types == [_ModelSeamInput]
