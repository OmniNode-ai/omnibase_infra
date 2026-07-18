# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""S8 cross-boundary seam test: Kafka topic -> typed model -> envelope -> FSM -> emit.

Epic OMN-14717, ticket OMN-14771 (S8 wave-2). This is the ONE required test that drives
the ACTUAL delegation runtime seam END-TO-END, in a SINGLE crossing (the OMN-14208
two-green-no-op guard — NOT two independent unit suites):

    inbound wire bytes on a Kafka topic (InMemoryTransport, S2)
      -> ``build_routing_map`` resolves the topic -> DispatchRoute (§a)
      -> ``RuntimeDispatch`` (S4) decodes the ``ModelEventEnvelope`` boundary
      -> coerces ``envelope.payload`` into the route's typed def-B ``input_model_cls``
      -> the delegation FSM ADVANCES its per-correlation workflow (RECEIVED -> ROUTED)
      -> emits a correlated OUTBOUND ``ModelEventEnvelope`` (correlation propagated,
         causation_id = inbound.envelope_id, deterministic uuid5 envelope id).

The FSM handler is a genuine stateful stand-in for the omnimarket
``HandlerDelegationWorkflow`` seam-2 slice (infra must NOT depend on the market package,
so the FSM cannot be the real class — mirrored exactly, incl. the "Ignoring routing
decision without an active delegation workflow" drop guard). Everything the seam actually
tests is REAL: the ``build_routing_map`` + single-owner gate (S8 §4b), ``RuntimeDispatch``,
the ``InMemoryTransport``, the ``runtime_envelope_router`` boundary, and the INBOUND +
OUTBOUND wire models are the REAL ``omnibase_core`` delegation DTOs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.delegation.wire import (
    ModelDelegationRequest,
    ModelInferenceIntent,
    ModelRoutingIntent,
)
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.runtime.runtime_dispatch import RuntimeDispatch
from omnibase_core.runtime.runtime_envelope_router import CAUSATION_ID_TAG
from omnibase_core.runtime.transport.runtime_in_memory_broker import InMemoryBroker
from omnibase_core.runtime.transport.runtime_in_memory_transport import (
    InMemoryTransport,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.core_runtime.dlq_resolver import (
    build_delegation_dlq_resolver,
)
from omnibase_infra.runtime.core_runtime.routing_map_builder import build_routing_map
from omnibase_infra.runtime.core_runtime.single_owner import assert_single_owner_split

# --- topics (the real delegation command spine) --------------------------------------
REQUEST_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
DECISION_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"
ROUTING_REQUEST_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
INFERENCE_REQUEST_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
INFERENCE_RESPONSE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"  # fan-out
GROUP = "onex.core-runtime.delegation"
FIXED_CLOCK = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)

ORCHESTRATOR = "node_delegation_orchestrator"


class FixtureRoutingDecision(BaseModel):
    """Stand-in for the market ``ModelRoutingDecision`` (infra must not import market).

    Same seam-relevant shape: it carries the ``correlation_id`` the FSM keys on plus the
    selected model. The runtime coerces the enveloped dict into THIS declared model, so
    the typed-deserialization half of the seam is genuinely exercised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    selected_model: str = "local-reasoner"


class _DelegationFsmFixture:
    """Genuine stateful FSM: the ``HandlerDelegationWorkflow`` RECEIVED->ROUTED slice.

    Keyed by ``correlation_id`` in a single in-process dict (one handler instance services
    BOTH topics — the R-7 shared-instance parity that makes the advance possible). Mirrors
    the real orchestrator's drop guard: a routing decision for a correlation with no active
    workflow row is IGNORED (returns nothing), never advanced. def-B: the sole ``handle``
    parameter is named ``request`` so the runtime passes the validated model straight
    through (never the raw envelope, never a dict).
    """

    RECEIVED = "RECEIVED"
    ROUTED = "ROUTED"

    def __init__(self) -> None:
        self.workflows: dict[UUID, str] = {}
        self.received: list[object] = []
        self.dropped: list[UUID] = []

    def handle(self, request: object) -> BaseModel | None:
        self.received.append(request)
        if isinstance(request, ModelDelegationRequest):
            # Workflow ENTRY: the ONLY site that creates the FSM row (RECEIVED).
            self.workflows[request.correlation_id] = self.RECEIVED
            return ModelRoutingIntent(payload=request)
        if isinstance(request, FixtureRoutingDecision):
            corr = request.correlation_id
            state = self.workflows.get(corr)
            if state is None:
                # Seam-2 dead-end guard (verbatim intent of the real handler): no active
                # workflow -> drop the routing decision, advance nothing, emit nothing.
                self.dropped.append(corr)
                return None
            assert state == self.RECEIVED, f"unexpected FSM state {state!r} for {corr}"
            self.workflows[corr] = self.ROUTED
            return ModelInferenceIntent(
                base_url="http://local-inference",
                model=request.selected_model,
                system_prompt="",
                prompt="advance to inference",
                max_tokens=256,
                correlation_id=corr,
            )
        raise ModelOnexError(  # fail closed on an undeclared payload type
            message=f"unexpected payload type {type(request).__name__} at FSM seam",
        )


def _contract(
    name: str,
    *,
    subscribe: str,
    event_model_name: str,
    event_model_module: str,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=6, patch=0),
        contract_path=Path(f"/nonexistent/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(subscribe,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerDelegationWorkflow",
                        module="tests.integration.runtime.test_s8_delegation_fsm_seam",
                    ),
                    event_model=ModelHandlerRef(
                        name=event_model_name, module=event_model_module
                    ),
                    operation=f"delegation.{name}",
                ),
            ),
        ),
    )


def _model_resolver(ref: ModelHandlerRef) -> type[BaseModel]:
    return {
        "ModelDelegationRequest": ModelDelegationRequest,
        "FixtureRoutingDecision": FixtureRoutingDecision,
    }[ref.name]


def _build_seam(handler: _DelegationFsmFixture):
    """Build routing_map (§a) + DLQ resolver (§b) + RuntimeDispatch over InMemory (S2)."""
    request_contract = _contract(
        ORCHESTRATOR,
        subscribe=REQUEST_TOPIC,
        event_model_name="ModelDelegationRequest",
        event_model_module="omnibase_core.models.delegation.wire",
    )
    decision_contract = _contract(
        f"{ORCHESTRATOR}_decision",
        subscribe=DECISION_TOPIC,
        event_model_name="FixtureRoutingDecision",
        event_model_module="tests.integration.runtime.test_s8_delegation_fsm_seam",
    )
    contracts = [request_contract, decision_contract]
    published_by_path = {
        request_contract.contract_path: {"ModelRoutingIntent": ROUTING_REQUEST_TOPIC},
        decision_contract.contract_path: {
            "ModelInferenceIntent": INFERENCE_REQUEST_TOPIC
        },
    }
    routing_map = build_routing_map(
        contracts,
        frozenset({REQUEST_TOPIC, DECISION_TOPIC}),
        handler_resolver=lambda _ref: handler,
        model_resolver=_model_resolver,
        published_events_loader=lambda path: published_by_path[path],
    )
    dlq_resolver, _ = build_delegation_dlq_resolver(
        contracts, frozenset({REQUEST_TOPIC, DECISION_TOPIC})
    )
    broker = InMemoryBroker(num_partitions=1)
    transport = InMemoryTransport(
        broker=broker, group=GROUP, topics=[REQUEST_TOPIC, DECISION_TOPIC]
    )
    dispatch = RuntimeDispatch(
        consumer=transport,
        producer=transport,
        routing_map=routing_map,
        dlq_topic_resolver=dlq_resolver,
        clock=lambda: FIXED_CLOCK,
    )
    return dispatch, transport, broker


def _envelope_bytes(
    payload: BaseModel, *, correlation_id: UUID, envelope_id: UUID
) -> bytes:
    envelope = ModelEventEnvelope(
        envelope_id=envelope_id,
        payload=payload.model_dump(mode="json"),
        correlation_id=correlation_id,
        event_type="seam.inbound",
        payload_type=type(payload).__name__,
    )
    return envelope.model_dump_json().encode("utf-8")


def _envelopes_on(
    broker: InMemoryBroker, topic: str
) -> list[ModelEventEnvelope[object]]:
    return [
        ModelEventEnvelope[object].model_validate_json(msg.value)
        for msg in broker.records(topic, 0)
    ]


@pytest.mark.asyncio
async def test_seam_delegation_fsm_advances_and_emits_correlated_outbound() -> None:
    """THE seam test: one delegation drives request->RECEIVED then decision->ROUTED+emit."""
    handler = _DelegationFsmFixture()
    dispatch, transport, broker = _build_seam(handler)
    corr = uuid4()

    # (1) Front-door command: delegation-request creates the FSM workflow row (RECEIVED).
    await transport.send(
        REQUEST_TOPIC,
        key=b"k",
        value=_envelope_bytes(
            _delegation_request(corr), correlation_id=corr, envelope_id=uuid4()
        ),
        headers={},
    )
    assert await dispatch.drain() == 1
    assert handler.workflows[corr] == _DelegationFsmFixture.RECEIVED
    # Handler saw a VALIDATED ModelDelegationRequest (typed coercion), not a raw dict.
    assert isinstance(handler.received[0], ModelDelegationRequest)
    # It emitted a routing intent on the routing-request topic (fan-out via published_events).
    routing_intents = _envelopes_on(broker, ROUTING_REQUEST_TOPIC)
    assert len(routing_intents) == 1
    assert routing_intents[0].correlation_id == corr

    # (2) routing-decision advances the SAME correlation's workflow RECEIVED -> ROUTED.
    decision_envelope_id = uuid4()
    await transport.send(
        DECISION_TOPIC,
        key=b"k",
        value=_envelope_bytes(
            FixtureRoutingDecision(correlation_id=corr),
            correlation_id=corr,
            envelope_id=decision_envelope_id,
        ),
        headers={},
    )
    assert await dispatch.drain() == 1

    # FSM genuinely advanced (not a vacuous pass).
    assert handler.workflows[corr] == _DelegationFsmFixture.ROUTED
    assert isinstance(handler.received[1], FixtureRoutingDecision)
    assert handler.dropped == []

    # The correlated OUTBOUND event: exactly one ModelInferenceIntent envelope, with the
    # full correlation chain the runtime (not the handler) stamps.
    inference_intents = _envelopes_on(broker, INFERENCE_REQUEST_TOPIC)
    assert len(inference_intents) == 1
    out = inference_intents[0]
    assert out.payload_type == "ModelInferenceIntent"
    # event_type DERIVED from the resolved publish topic (fail-closed, never null).
    assert out.event_type == "omnibase-infra.delegation-inference-request"
    # correlation propagated from the inbound routing-decision envelope.
    assert out.correlation_id == corr
    # causation_id chains to the inbound routing-decision envelope id.
    assert out.metadata.tags[CAUSATION_ID_TAG] == str(decision_envelope_id)
    # deterministic uuid5 envelope id (reproducible across redeliveries).
    assert out.envelope_id == uuid5(corr, "ModelInferenceIntent:0")
    assert out.envelope_timestamp == FIXED_CLOCK


@pytest.mark.asyncio
async def test_seam_routing_decision_without_workflow_is_dropped_not_advanced() -> None:
    """RED-vs-exists-but-wrong: a decision for an unknown correlation drops, emits nothing.

    Proves the ROUTED advance in the happy-path test is REAL (workflow-row-gated), not a
    handler that blindly emits on any routing decision.
    """
    handler = _DelegationFsmFixture()
    dispatch, transport, broker = _build_seam(handler)
    corr = uuid4()  # never sent a delegation-request for this correlation

    await transport.send(
        DECISION_TOPIC,
        key=b"k",
        value=_envelope_bytes(
            FixtureRoutingDecision(correlation_id=corr),
            correlation_id=corr,
            envelope_id=uuid4(),
        ),
        headers={},
    )
    assert await dispatch.drain() == 1

    assert corr not in handler.workflows
    assert handler.dropped == [corr]
    # No outbound inference intent — the decision was dropped, not advanced.
    assert _envelopes_on(broker, INFERENCE_REQUEST_TOPIC) == []


def test_seam_fanout_owner_route_built_and_single_owner_gate_passes() -> None:
    """S8 §4b boot gate over the same builder: one owner MOVES, others stay legacy.

    Ties the single-owner boot gate to the real routing-map machinery: for a genuine
    fan-out topic (orchestrator + two legacy projection consumers), the routing map binds
    ONLY the designated owner's route, and ``assert_single_owner_split`` PASSES because the
    non-owner consumers keep distinct legacy consumer groups.
    """
    owner = _contract(
        ORCHESTRATOR,
        subscribe=INFERENCE_RESPONSE_TOPIC,
        event_model_name="FixtureRoutingDecision",
        event_model_module="tests.integration.runtime.test_s8_delegation_fsm_seam",
    )
    legacy_projection = _contract(
        "node_projection_delegation_inference_response",
        subscribe=INFERENCE_RESPONSE_TOPIC,
        event_model_name="FixtureRoutingDecision",
        event_model_module="tests.integration.runtime.test_s8_delegation_fsm_seam",
    )
    legacy_forwarder = _contract(
        "node_bus_forwarder_effect",
        subscribe=INFERENCE_RESPONSE_TOPIC,
        event_model_name="FixtureRoutingDecision",
        event_model_module="tests.integration.runtime.test_s8_delegation_fsm_seam",
    )
    contracts = [owner, legacy_projection, legacy_forwarder]
    owners = {INFERENCE_RESPONSE_TOPIC: ORCHESTRATOR}

    routing_map = build_routing_map(
        contracts,
        frozenset({INFERENCE_RESPONSE_TOPIC}),
        handler_resolver=lambda _ref: _DelegationFsmFixture(),
        owners=owners,
        model_resolver=_model_resolver,
        published_events_loader=lambda _path: {
            "ModelInferenceIntent": INFERENCE_REQUEST_TOPIC
        },
    )
    # Exactly ONE core-runtime route — the owner's — despite three subscribers.
    assert set(routing_map) == {INFERENCE_RESPONSE_TOPIC}
    assert routing_map[INFERENCE_RESPONSE_TOPIC].name == "HandlerDelegationWorkflow"

    # Gate PASSES: fan-out topic is exempt from the single-owner legacy-overlap check, and
    # the two non-owner legacy consumers derive distinct groups (distinct node names).
    assert_single_owner_split(
        core_runtime_topics=frozenset({INFERENCE_RESPONSE_TOPIC}),
        routing_map=routing_map,
        legacy_subscribed_topics=frozenset({INFERENCE_RESPONSE_TOPIC}),
        contracts=contracts,
        owners=owners,
    )


def _delegation_request(correlation_id: UUID) -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt="classify this",
        task_type="research",
        correlation_id=correlation_id,
        emitted_at=FIXED_CLOCK,
    )
