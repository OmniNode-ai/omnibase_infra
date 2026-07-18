# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S6 routing-map builder (OMN-14758, §a)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
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
from omnibase_infra.runtime.core_runtime.routing_map_builder import build_routing_map

ROUTING_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
DECISION_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"


class _FakeRoutingIntent(BaseModel):
    value: str = "x"


class _FakeHandler:
    def handle(self, request: object) -> object:  # pragma: no cover - resolver stub
        return request


def _make_contract(
    *,
    name: str,
    subscribe_topics: tuple[str, ...],
    entries: tuple[ModelHandlerRoutingEntry, ...],
    routing_strategy: str = "operation_match",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=Path(f"/nonexistent/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=(DECISION_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy=routing_strategy,
            handlers=entries,
        ),
    )


def _resolver(_ref: ModelHandlerRef) -> object:
    return _FakeHandler()


def _model_resolver(_ref: ModelHandlerRef) -> type[BaseModel]:
    return _FakeRoutingIntent


def _pe_loader(_path: Path) -> dict[str, str]:
    return {"RoutingDecision": DECISION_TOPIC}


def test_builds_topic_keyed_route_with_event_model() -> None:
    contract = _make_contract(
        name="node_delegation_routing_reducer",
        subscribe_topics=(ROUTING_TOPIC,),
        entries=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name="HandlerRoutingIntent", module="m"),
                event_model=ModelHandlerRef(name="ModelRoutingIntent", module="m2"),
                operation="delegation_routing",
            ),
        ),
    )
    routing_map = build_routing_map(
        [contract],
        frozenset({ROUTING_TOPIC}),
        handler_resolver=_resolver,
        model_resolver=_model_resolver,
        published_events_loader=_pe_loader,
    )
    assert set(routing_map) == {ROUTING_TOPIC}
    route = routing_map[ROUTING_TOPIC]
    assert route.name == "HandlerRoutingIntent"
    assert route.input_model_cls is _FakeRoutingIntent  # operation_match + event_model
    assert route.published_events == {"RoutingDecision": DECISION_TOPIC}


def test_single_owner_build_gate_raises_on_two_contracts_same_topic() -> None:
    entry = ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name="HandlerA", module="m"),
    )
    c1 = _make_contract(name="c1", subscribe_topics=(ROUTING_TOPIC,), entries=(entry,))
    c2 = _make_contract(name="c2", subscribe_topics=(ROUTING_TOPIC,), entries=(entry,))
    with pytest.raises(ModelOnexError, match="single-owner"):
        build_routing_map(
            [c1, c2],
            frozenset({ROUTING_TOPIC}),
            handler_resolver=_resolver,
            model_resolver=_model_resolver,
            published_events_loader=_pe_loader,
        )


def test_missing_allowlist_topic_raises() -> None:
    with pytest.raises(ModelOnexError, match=r"not.*declared|not.*subscribe"):
        build_routing_map(
            [],
            frozenset({ROUTING_TOPIC}),
            handler_resolver=_resolver,
            model_resolver=_model_resolver,
            published_events_loader=_pe_loader,
        )


def test_multi_handler_same_topic_fails_closed() -> None:
    entries = (
        ModelHandlerRoutingEntry(handler=ModelHandlerRef(name="H1", module="m")),
        ModelHandlerRoutingEntry(handler=ModelHandlerRef(name="H2", module="m")),
    )
    contract = _make_contract(
        name="multi", subscribe_topics=(ROUTING_TOPIC,), entries=entries
    )
    with pytest.raises(ModelOnexError, match=r"single.*route|multiplexed|ambiguous"):
        build_routing_map(
            [contract],
            frozenset({ROUTING_TOPIC}),
            handler_resolver=_resolver,
            model_resolver=_model_resolver,
            published_events_loader=_pe_loader,
        )


def test_fanout_owner_designation_builds_only_owner_route() -> None:
    """S8 §4b: a designated owner binds the ONLY route; other subscribers are skipped."""
    entry = ModelHandlerRoutingEntry(handler=ModelHandlerRef(name="HOwner", module="m"))
    owner = _make_contract(
        name="node_delegation_orchestrator",
        subscribe_topics=(ROUTING_TOPIC,),
        entries=(entry,),
    )
    # A second + third subscriber on the same topic that MUST NOT produce a route.
    legacy_a = _make_contract(
        name="node_projection_a", subscribe_topics=(ROUTING_TOPIC,), entries=(entry,)
    )
    legacy_b = _make_contract(
        name="node_projection_b", subscribe_topics=(ROUTING_TOPIC,), entries=(entry,)
    )
    routing_map = build_routing_map(
        [owner, legacy_a, legacy_b],
        frozenset({ROUTING_TOPIC}),
        handler_resolver=_resolver,
        owners={ROUTING_TOPIC: "node_delegation_orchestrator"},
        model_resolver=_model_resolver,
        published_events_loader=_pe_loader,
    )
    assert set(routing_map) == {ROUTING_TOPIC}
    assert routing_map[ROUTING_TOPIC].name == "HOwner"


def test_non_allowlisted_topic_is_not_built() -> None:
    other = "onex.cmd.omnibase-infra.other-command.v1"
    entry = ModelHandlerRoutingEntry(handler=ModelHandlerRef(name="H", module="m"))
    contract = _make_contract(
        name="c",
        subscribe_topics=(ROUTING_TOPIC, other),
        entries=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name="H", module="m"), topic=ROUTING_TOPIC
            ),
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name="H2", module="m"), topic=other
            ),
        ),
    )
    routing_map = build_routing_map(
        [contract],
        frozenset({ROUTING_TOPIC}),
        handler_resolver=_resolver,
        model_resolver=_model_resolver,
        published_events_loader=_pe_loader,
    )
    assert set(routing_map) == {ROUTING_TOPIC}
    assert other not in routing_map
