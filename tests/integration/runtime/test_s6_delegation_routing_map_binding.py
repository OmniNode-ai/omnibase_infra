# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""S6 cross-boundary seam test: topic -> handler -> emit over InMemoryTransport.

Epic OMN-14717, ticket OMN-14758, plan section (e). This drives the REAL seam
end-to-end — the §a routing-map builder, the §b DLQ resolver, ``RuntimeDispatch`` (S4),
the ``InMemoryTransport`` (S2), and the ``runtime_envelope_router`` boundary — with a
FIXED clock. Each assertion is a distinct failure mode:

1. Binding fires: the routing topic dispatches to the handler with a VALIDATED
   ``ModelRoutingIntent`` (not a dict), emits a routing-decision envelope with a
   topic-derived non-null ``event_type``, propagates the correlation_id, and COMMITS.
2. Topic-keyed, not name-keyed: a payload whose ``payload_type`` is wrong for the topic
   still routes by topic.
3. Unmapped topic fails closed: DLQ'd to the ONEX-canonical DLQ (not ``<topic>.dlq``,
   not silent commit).
4. DLQ resolver is real: resolves the routing topic to an ``onex.dlq.*`` topic in the
   provision set.
5. Single-owner build gate: two contracts on one topic raise at build.
6. Fan-out injective: two published classes -> one topic raise at RuntimeDispatch init.

The seam is real; the routing-reducer HANDLER is a fixture standing in for the omnimarket
``HandlerRoutingIntent`` so this infra test does not take a runtime dependency on the
market package (infra must not depend on market). The INPUT model is the REAL
``omnibase_core`` ``ModelRoutingIntent`` (resolved by the builder's default importer), so
the coercion + validation asserted in (1) is genuine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.delegation.wire import ModelRoutingIntent
from omnibase_core.models.delegation.wire.model_delegation_wire_request import (
    ModelDelegationRequest,
)
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.runtime.runtime_dispatch import DispatchRoute, RuntimeDispatch
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

ROUTING_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
DECISION_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"
UNMAPPED_TOPIC = "onex.cmd.omnibase-infra.unmapped-command.v1"
GROUP = "onex.core-runtime.test"
FIXED_CLOCK = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


class RoutingDecision(BaseModel):
    """Fixture emit payload (short-name 'RoutingDecision' matches published_events)."""

    selected_model: str = "local-reasoner"
    correlation_id: UUID


class _FixtureRoutingHandler:
    """Stand-in for omnimarket HandlerRoutingIntent: def-B, returns a RoutingDecision."""

    def __init__(self) -> None:
        self.received: list[object] = []

    def handle(self, intent: ModelRoutingIntent) -> RoutingDecision:
        # Records the runtime-coerced object so the test can assert it is a validated
        # ModelRoutingIntent (not a raw dict) — the non-vacuous coercion proof.
        self.received.append(intent)
        return RoutingDecision(correlation_id=intent.payload.correlation_id)


def _routing_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=3, patch=0),
        contract_path=Path("/nonexistent/routing/contract.yaml"),
        entry_point_name="node_delegation_routing_reducer",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(ROUTING_TOPIC,),
            publish_topics=(DECISION_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerRoutingIntent",
                        module="tests.integration.runtime.test_s6_delegation_routing_map_binding",
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelRoutingIntent",
                        module="omnibase_core.models.delegation.wire",
                    ),
                    operation="delegation_routing",
                ),
            ),
        ),
    )


def _published_events_loader(_path: Path) -> dict[str, str]:
    return {"RoutingDecision": DECISION_TOPIC}


def _build_seam(handler: _FixtureRoutingHandler, *, assigned_topics: list[str]):
    """Build routing_map (§a) + DLQ resolver (§b) + RuntimeDispatch over InMemory (S2)."""
    contract = _routing_contract()
    routing_map = build_routing_map(
        [contract],
        frozenset({ROUTING_TOPIC}),
        handler_resolver=lambda _ref: handler,
        published_events_loader=_published_events_loader,
    )
    dlq_resolver, provision = build_delegation_dlq_resolver(
        [contract], frozenset({ROUTING_TOPIC})
    )
    broker = InMemoryBroker(num_partitions=1)
    transport = InMemoryTransport(broker=broker, group=GROUP, topics=assigned_topics)
    dispatch = RuntimeDispatch(
        consumer=transport,
        producer=transport,
        routing_map=routing_map,
        dlq_topic_resolver=dlq_resolver,
        clock=lambda: FIXED_CLOCK,
    )
    return dispatch, transport, broker, dlq_resolver, provision


def _routing_intent(correlation_id: UUID) -> ModelRoutingIntent:
    return ModelRoutingIntent(
        payload=ModelDelegationRequest(
            prompt="classify this",
            task_type="research",
            correlation_id=correlation_id,
            emitted_at=FIXED_CLOCK,
        )
    )


def _inbound_envelope_bytes(
    *, correlation_id: UUID, payload_type: str, intent: ModelRoutingIntent
) -> bytes:
    envelope = ModelEventEnvelope(
        payload=intent.model_dump(mode="json"),
        correlation_id=correlation_id,
        event_type="omnibase-infra.delegation-routing-request",
        payload_type=payload_type,
    )
    return envelope.model_dump_json().encode("utf-8")


def _decision_envelopes(broker: InMemoryBroker) -> list[ModelEventEnvelope[object]]:
    return [
        ModelEventEnvelope[object].model_validate_json(msg.value)
        for msg in broker.records(DECISION_TOPIC, 0)
    ]


@pytest.mark.asyncio
async def test_assertion_1_binding_fires_and_commits() -> None:
    handler = _FixtureRoutingHandler()
    dispatch, transport, broker, _, _ = _build_seam(
        handler, assigned_topics=[ROUTING_TOPIC]
    )
    corr = uuid4()
    await transport.send(
        ROUTING_TOPIC,
        key=b"k",
        value=_inbound_envelope_bytes(
            correlation_id=corr,
            payload_type="ModelRoutingIntent",
            intent=_routing_intent(corr),
        ),
        headers={},
    )
    processed = await dispatch.drain()
    assert processed == 1

    # Handler reached with a VALIDATED ModelRoutingIntent (not a dict).
    assert len(handler.received) == 1
    assert isinstance(handler.received[0], ModelRoutingIntent)

    # Envelope landed on the decision topic with a topic-derived non-null event_type.
    decisions = _decision_envelopes(broker)
    assert len(decisions) == 1
    out = decisions[0]
    assert out.event_type == "omnibase-infra.routing-decision"
    assert out.correlation_id == corr
    assert out.envelope_timestamp == FIXED_CLOCK  # fixed clock propagated

    # Source committed (partition high-water mark advanced to the consumed offset).
    assert broker.committed_offset(ROUTING_TOPIC, GROUP, 0) == 0


@pytest.mark.asyncio
async def test_assertion_2_topic_keyed_not_name_keyed() -> None:
    handler = _FixtureRoutingHandler()
    dispatch, transport, broker, _, _ = _build_seam(
        handler, assigned_topics=[ROUTING_TOPIC]
    )
    corr = uuid4()
    # payload_type is deliberately WRONG for the topic; the dict still validates into
    # ModelRoutingIntent. Routing must key on TOPIC, not type(payload).__name__.
    await transport.send(
        ROUTING_TOPIC,
        key=b"k",
        value=_inbound_envelope_bytes(
            correlation_id=corr,
            payload_type="TotallyWrongPayloadClassName",
            intent=_routing_intent(corr),
        ),
        headers={},
    )
    processed = await dispatch.drain()
    assert processed == 1
    assert len(handler.received) == 1
    assert isinstance(handler.received[0], ModelRoutingIntent)
    assert len(_decision_envelopes(broker)) == 1


@pytest.mark.asyncio
async def test_assertion_3_unmapped_topic_fails_closed_to_canonical_dlq() -> None:
    handler = _FixtureRoutingHandler()
    dispatch, transport, broker, dlq_resolver, _ = _build_seam(
        handler, assigned_topics=[UNMAPPED_TOPIC]
    )
    corr = uuid4()
    await transport.send(
        UNMAPPED_TOPIC,
        key=b"k",
        value=_inbound_envelope_bytes(
            correlation_id=corr, payload_type="X", intent=_routing_intent(corr)
        ),
        headers={},
    )
    await dispatch.drain()

    expected_dlq = dlq_resolver(UNMAPPED_TOPIC)
    assert expected_dlq == "onex.dlq.omnibase-infra.unmapped-command.v1"
    assert not expected_dlq.endswith(".v1.dlq")  # NOT the placeholder <topic>.dlq
    # Message dead-lettered (not silently dropped), no decision emitted.
    dlq_records = broker.records(expected_dlq, 0)
    assert len(dlq_records) == 1
    assert handler.received == []
    # The source WAS committed past (progress), but only after the DLQ copy exists.
    assert broker.committed_offset(UNMAPPED_TOPIC, GROUP, 0) == 0


def test_assertion_4_dlq_resolver_is_real_and_in_provision_set() -> None:
    _, provision = build_delegation_dlq_resolver(
        [_routing_contract()], frozenset({ROUTING_TOPIC})
    )
    resolver, _ = build_delegation_dlq_resolver(
        [_routing_contract()], frozenset({ROUTING_TOPIC})
    )
    dlq = resolver(ROUTING_TOPIC)
    assert dlq.startswith("onex.dlq.")
    assert not dlq.endswith(".v1.dlq")
    assert dlq in provision


def test_assertion_5_single_owner_build_gate_raises() -> None:
    c1 = _routing_contract()
    c2 = _routing_contract().model_copy(update={"name": "duplicate_owner"})
    with pytest.raises(ModelOnexError, match="single-owner"):
        build_routing_map(
            [c1, c2],
            frozenset({ROUTING_TOPIC}),
            handler_resolver=lambda _ref: _FixtureRoutingHandler(),
            published_events_loader=_published_events_loader,
        )


def test_assertion_6_fanout_injective_raises_at_dispatch_init() -> None:
    broker = InMemoryBroker(num_partitions=1)
    transport = InMemoryTransport(broker=broker, group=GROUP, topics=[ROUTING_TOPIC])
    # Two distinct classes mapped to the SAME topic — non-injective published_events.
    bad_route = DispatchRoute(
        name="HandlerRoutingIntent",
        handler=_FixtureRoutingHandler(),
        published_events={
            "RoutingDecision": DECISION_TOPIC,
            "OtherEvent": DECISION_TOPIC,
        },
    )
    with pytest.raises(ModelOnexError):
        RuntimeDispatch(
            consumer=transport,
            producer=transport,
            routing_map={ROUTING_TOPIC: bad_route},
        )
