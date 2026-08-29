# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16940 Defect 2 — one handler failure, exactly one dead-letter record.

Measured on the .201 dev lane, 2026-08-29:

* ``onex.evt.omnibase-infra.inference-response.v1`` — 143 records
  (LOG-START 35, HIGH-WATERMARK 178).
* ``onex.dlq.omnimarket.projection-delegation-inference-response-malformed.v1``
  — 286 records (LOG-START 70, HIGH-WATERMARK 356). Exactly ``2 x 143``.
* The pairs sit on adjacent offsets with identical payloads and ``retry_count:
  0`` on both, 12-40ms apart. Not a retry.

The mechanism is NOT two routing layers disagreeing; the runtime log shows both
writes coming out of the SAME surface, ``_route_projection_error_to_dlq``, twice
per correlation id::

    [WIRING-CALLBACK] Message received on topic=...inference-response.v1,
        consumer_group=local.runtime_config.delegation-orchestrator.consume.1.0.0
    [WIRING-CALLBACK] Dispatching to engine: ...
    service_kernel: Dispatch started
    service_kernel: Dispatch started               <-- twice, one message
    handler_wiring: Projection handler error ... PermissionError
    handler_wiring: ... routed malformed/erroring event to DLQ ...
    handler_wiring: Projection handler error ... PermissionError
    handler_wiring: ... routed malformed/erroring event to DLQ ...

and ``rpk group describe`` confirms TWO groups consumed all 143 records in the
same ``runtime-main`` instance:

* ``local.omnimarket.node_projection_delegation_inference_response.consume.1.0.0``
  — the projection contract's OWN auto-wired subscription (offset 178/178)
* ``local.runtime_config.delegation-orchestrator.consume.1.0.0`` — a FOREIGN
  subscription belonging to the delegation orchestrator (offset 178/178)

``EventBusSubcontractWiring._create_dispatch_callback`` calls
``dispatch_engine.dispatch(topic, envelope)`` with no dispatcher scope, so it
fans out process-globally and re-executes a dispatcher owned by a different
contract — the exact failure OMN-15474 closed on the auto-wired boundary
("A contract-owned Kafka callback must never fall back to process-global
dispatch") and never closed on this one.

Ownership rule proven here: a dispatcher whose owning contract holds its own
subscription for a topic is dispatched by THAT subscription and by nothing else.
An unscoped dispatch on that topic DEFERS. It is not deduplicated downstream —
AC6 rules that out explicitly, and a dedupe filter would leave the handler
running twice, the projection writing twice, and ``messages_in`` counting twice.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _route_projection_error_to_dlq,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

# onex-topic-allow: the exact topics from the OMN-16940 live capture.
_SOURCE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
_DLQ_TOPIC = "onex.dlq.omnimarket.projection-delegation-inference-response-malformed.v1"
_OWNER_CONTRACT = "node_projection_delegation_inference_response"
_DISPATCHER_ID = "node_projection_delegation_inference_response::projection"
_EVENT_TYPE = "omnibase-infra.inference-response"


class _RecordingBus:
    """Captures every publish so DLQ depth is counted, not inferred."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None]] = []

    async def publish(self, topic: str, key: object, value: bytes) -> None:
        self.published.append((topic, value))

    def dlq_records(self) -> list[bytes | None]:
        return [value for topic, value in self.published if topic == _DLQ_TOPIC]


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope(
        event_type=_EVENT_TYPE,
        payload={
            "llm_call_id": "03f87083f56a48e79c63a44039ba8eb3",
            "content": "alive",
            "latency_ms": 155,
        },
        correlation_id=uuid4(),
    )


def _build_engine(
    dispatcher: Any, *, contract_owned_topics: tuple[str, ...] = ()
) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=dispatcher,
        category=EnumMessageCategory.EVENT,
        owner_contract_name=_OWNER_CONTRACT,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"{_DISPATCHER_ID}::route",
            topic_pattern=_SOURCE_TOPIC,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_DISPATCHER_ID,
        )
    )
    for topic in contract_owned_topics:
        engine.register_contract_owned_subscription(_OWNER_CONTRACT, topic)
    engine.freeze()
    return engine


@pytest.mark.unit
async def test_one_handler_failure_writes_exactly_one_dlq_record() -> None:
    """AC5 — the DLQ high-watermark advances by ONE for one failing message.

    Both live boundaries are driven against one engine, in the live order: the
    contract's own scoped subscription, then the foreign unscoped one.
    """
    bus = _RecordingBus()
    invocations: list[str] = []

    async def failing_projection(envelope: ModelEventEnvelope[object]) -> None:
        invocations.append("called")
        await _route_projection_error_to_dlq(
            bus,
            [_DLQ_TOPIC],
            envelope,
            "HandlerProjectionDelegationInferenceResponse",
            "PermissionError: Projection binding 'tenant_projection' connected "
            "as ('role_omnidash', 'omnidash_analytics')",
        )

    engine = _build_engine(failing_projection, contract_owned_topics=(_SOURCE_TOPIC,))
    envelope = _envelope()

    await engine.dispatch(
        _SOURCE_TOPIC, envelope, allowed_dispatcher_ids={_DISPATCHER_ID}
    )
    await engine.dispatch(_SOURCE_TOPIC, envelope)

    assert len(invocations) == 1, (
        "the projection handler ran once per subscription; the foreign, unscoped "
        "boundary must defer to the contract that owns the dispatcher"
    )
    assert len(bus.dlq_records()) == 1, (
        f"one failure produced {len(bus.dlq_records())} dead-letter records "
        "(the live 286 = 2 x 143 signature)"
    )


@pytest.mark.unit
async def test_an_unscoped_dispatch_defers_to_the_owning_subscription() -> None:
    """AC6 — ownership is decided at the boundary, not by a downstream filter."""
    calls: list[str] = []

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        calls.append("called")

    engine = _build_engine(dispatcher, contract_owned_topics=(_SOURCE_TOPIC,))
    result = await engine.dispatch(_SOURCE_TOPIC, _envelope())

    assert calls == []
    # SKIPPED, not NO_DISPATCHER: the record is being delivered on the owner's
    # own group, so the calling boundary must commit and write NO dead-letter
    # record. Answering NO_DISPATCHER here would move the duplicate from the
    # handler's DLQ to the boundary's rather than removing it.
    assert result.status is EnumDispatchStatus.SKIPPED, result
    assert result.dlq_topic is None, result


@pytest.mark.unit
async def test_the_owning_subscription_still_dispatches() -> None:
    """The deferral must not silence the owner — that would be a new outage."""
    calls: list[str] = []

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        calls.append("called")

    engine = _build_engine(dispatcher, contract_owned_topics=(_SOURCE_TOPIC,))
    await engine.dispatch(
        _SOURCE_TOPIC, _envelope(), allowed_dispatcher_ids={_DISPATCHER_ID}
    )

    assert calls == ["called"]


@pytest.mark.unit
async def test_a_dispatcher_without_its_own_subscription_still_fans_out() -> None:
    """No regression for the core-runtime-owned and manually-registered paths.

    ``handler_wiring`` deliberately SKIPS a contract's own subscription when the
    core RuntimeDispatch owns the topic (OMN-14758 / OMN-14771 single-owner
    invariant). Those dispatchers have no self-owned subscription, so the
    unscoped route is the only one they have and must keep working.
    """
    calls: list[str] = []

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        calls.append("called")

    engine = _build_engine(dispatcher)  # no contract-owned subscription
    await engine.dispatch(_SOURCE_TOPIC, _envelope())

    assert calls == ["called"]


@pytest.mark.unit
async def test_deferral_is_scoped_to_the_topic_the_owner_subscribes() -> None:
    """A self-owned subscription on topic A does not silence topic B."""
    calls: list[str] = []

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        calls.append("called")

    # onex-topic-allow: a second real topic the same contract does not subscribe.
    other = "onex.evt.omnibase-infra.routing-decision.v1"
    engine = _build_engine(dispatcher, contract_owned_topics=(other,))
    await engine.dispatch(_SOURCE_TOPIC, _envelope())

    assert calls == ["called"]


@pytest.mark.unit
def test_registering_a_contract_owned_subscription_requires_canonical_input() -> None:
    """Fail closed on a blank contract name or topic rather than silencing all."""
    engine = MessageDispatchEngine()
    for contract_name, topic in ((" ", _SOURCE_TOPIC), (_OWNER_CONTRACT, "")):
        with pytest.raises(Exception):
            engine.register_contract_owned_subscription(contract_name, topic)


@pytest.mark.unit
async def test_the_auto_wired_subscribe_path_claims_ownership(
    tmp_path: Any,
) -> None:
    """The claim is made by the code that actually attaches the subscription.

    The engine-level tests above prove what a claim DOES. This one proves the
    claim is MADE — on the real ``_subscribe_contract_topics`` path, after
    ``subscribe()`` returns, for every topic the contract attaches. Without it
    the deferral rule would be inert in production and the duplicate would be
    back with no failing test.
    """
    from types import SimpleNamespace

    from omnibase_infra.runtime.auto_wiring import handler_wiring

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: node_projection_delegation_inference_response\n"
        "event_bus:\n"
        f"  subscribe_topics: ['{_SOURCE_TOPIC}']\n"
    )

    calls: list[str] = []

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        calls.append("called")

    engine = _build_engine(dispatcher)
    subscribed: list[str] = []

    class _Bus:
        async def subscribe(
            self, *, topic: str, node_identity: Any, on_message: Any
        ) -> None:
            subscribed.append(topic)

    contract = SimpleNamespace(
        name=_OWNER_CONTRACT,
        package_name="omnimarket",
        contract_version="1.0.0",
        contract_path=contract_path,
        terminal_event=None,
        event_bus=SimpleNamespace(
            subscribe_topics=[_SOURCE_TOPIC],
            publish_topics=[],
            dlq_topics=(),
            tenant_scoped_ingress=False,
            consumer_purpose=None,
        ),
    )

    topics = await handler_wiring._subscribe_contract_topics(
        contract=contract,  # type: ignore[arg-type]
        dispatch_engine=engine,
        event_bus=_Bus(),
        environment="local",
        allowed_dispatcher_ids={_DISPATCHER_ID},
    )

    assert topics == [_SOURCE_TOPIC]
    assert subscribed == [_SOURCE_TOPIC]
    assert engine._contract_owned_subscriptions == {_OWNER_CONTRACT: {_SOURCE_TOPIC}}
