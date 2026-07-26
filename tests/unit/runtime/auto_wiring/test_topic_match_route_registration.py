# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for contract-declared per-handler topic assignment [OMN-13825].

Regression: a ``topic_match`` reducer contract (e.g. ``node_projection_swarm``)
declares TWO handler entries, each carrying an explicit ``topic:`` naming one of
the two subscribe topics. The runtime discarded that ``topic:`` key, so each
entry fell through the multi-handler ambiguity guard in
``_topics_for_handler_entry`` (``return ()``) and registered ZERO dispatch
routes. The reducer's dispatcher was orphaned and every swarm event was DLQ'd
("No dispatcher found for category 'event' and message type
'omnimarket.swarm-dispatch-completed'").

The fix: ``ModelHandlerRoutingEntry`` gains an optional ``topic`` field;
``discovery._parse_handler_routing`` threads the contract's ``topic:`` onto it;
and ``_topics_for_handler_entry`` honors a declared topic as the FIRST
disambiguator — the entry deterministically owns exactly that topic (or ``()``
if the declared topic is not an actual subscribe topic, surfacing a real
contract error rather than silently mis-routing).

Live ``swarm_runs`` materialization is deferred to OMN-13772 (.201 stability
rebuild) and is NOT exercised here — this is the wiring-level route-registration
fix only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _prepare_handler_wiring,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)

# The two subscribe topics owned by node_projection_swarm's topic_match reducer.
_SWARM_COMPLETED = "onex.evt.omnimarket.swarm-dispatch-completed.v1"
_SWARM_FAILED = "onex.evt.omnimarket.swarm-dispatch-failed.v1"
_SWARM_TOPICS = (_SWARM_COMPLETED, _SWARM_FAILED)


def _make_zero_arg_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


def _swarm_entry(*, topic: str, operation: str) -> ModelHandlerRoutingEntry:
    """A single node_projection_swarm-style topic_match handler entry.

    Both entries share the same handler class and event_model (like the real
    contract) and are disambiguated purely by their declared ``topic``.
    """
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(
            name="HandlerProjectionSwarm",
            module="fake.handlers.handler_projection_swarm",
        ),
        event_model=ModelHandlerRef(
            name="ModelSwarmDispatchEvent",
            module="fake.handlers.handler_projection_swarm",
        ),
        operation=operation,
        topic=topic,
    )


def _swarm_contract() -> ModelDiscoveredContract:
    entries = (
        _swarm_entry(topic=_SWARM_COMPLETED, operation="swarm_dispatch_completed"),
        _swarm_entry(topic=_SWARM_FAILED, operation="swarm_dispatch_failed"),
    )
    return ModelDiscoveredContract(
        name="node_projection_swarm",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/node_projection_swarm/contract.yaml"),
        entry_point_name="node_projection_swarm",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=_SWARM_TOPICS,
            publish_topics=("onex.evt.omnimarket.projection-swarm-applied.v1",),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=entries,
        ),
    )


class TestTopicMatchTopicAssignment:
    """`_topics_for_handler_entry` honors the contract-declared per-entry topic."""

    @pytest.mark.unit
    def test_each_entry_owns_its_declared_topic_not_empty(self) -> None:
        """The core defect: each topic_match entry resolves to its ONE declared
        topic. Before the fix each returned ``()`` (multi-handler ambiguity
        guard), registering zero routes and orphaning the dispatcher."""
        contract = _swarm_contract()
        completed, failed = contract.handler_routing.handlers  # type: ignore[union-attr]

        assert _topics_for_handler_entry(contract, completed) == (_SWARM_COMPLETED,)
        assert _topics_for_handler_entry(contract, failed) == (_SWARM_FAILED,)

    @pytest.mark.unit
    def test_declared_topic_not_in_subscribe_topics_returns_empty(self) -> None:
        """A declared topic that is not an actual subscribe topic returns ``()``
        so a real contract error surfaces rather than silently mis-routing."""
        contract = _swarm_contract()
        bogus_entry = _swarm_entry(
            topic="onex.evt.omnimarket.not-a-subscribe-topic.v1",
            operation="swarm_dispatch_bogus",
        )

        assert _topics_for_handler_entry(contract, bogus_entry) == ()


class TestTopicMatchRouteRegistration:
    """`_prepare_handler_wiring` registers one dispatch route per topic_match
    entry, so the swarm reducer registers TWO routes and its dispatcher is
    resolvable for message_type ``omnimarket.swarm-dispatch-completed``."""

    @pytest.mark.unit
    def test_two_entries_register_two_distinct_dispatch_routes(self) -> None:
        contract = _swarm_contract()
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()

        prepared_by_entry = {}
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            for entry in contract.handler_routing.handlers:  # type: ignore[union-attr]
                prepared_by_entry[entry.operation] = _prepare_handler_wiring(
                    contract=contract,
                    entry=entry,
                    dispatch_engine=None,
                    resolver=resolver,
                    ownership_query=ownership,
                    event_bus=None,
                    container=None,
                )

        completed = prepared_by_entry["swarm_dispatch_completed"]
        failed = prepared_by_entry["swarm_dispatch_failed"]

        # Each entry registers exactly one route for its declared topic.
        assert len(completed.routes) == 1
        assert len(failed.routes) == 1

        all_route_ids = set(completed.route_ids) | set(failed.route_ids)
        assert len(all_route_ids) == 2, (
            "the two entries must register 2 distinct routes"
        )

        completed_topics = {r.topic_pattern for r in completed.routes}
        failed_topics = {r.topic_pattern for r in failed.routes}
        assert completed_topics != failed_topics
        assert completed.dispatcher_id != failed.dispatcher_id

    @pytest.mark.unit
    def test_dispatcher_resolvable_for_swarm_completed_event_type(self) -> None:
        """The dispatcher is indexed under the topic-derived event_type alias so
        an envelope with event_type ``omnimarket.swarm-dispatch-completed``
        resolves to the handler instead of DLQ'ing."""
        contract = _swarm_contract()
        completed_entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=completed_entry,
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership,
                event_bus=None,
                container=None,
            )

        assert prepared.message_types is not None
        assert "omnimarket.swarm-dispatch-completed" in prepared.message_types


class TestNonRegression:
    """Existing operation_match / single-handler behavior is unchanged."""

    @pytest.mark.unit
    def test_operation_match_single_entry_topic_none_returns_all_topics(self) -> None:
        """A dep_health-style operation_match single-entry contract with
        ``topic=None`` keeps the sole-handler all-topics assignment."""
        topic = "onex.evt.omnimarket.dep-health-checked.v1"
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerDepHealth", module="fake.module"),
            event_model=ModelHandlerRef(name="ModelDepHealth", module="fake.models"),
            operation="dep_health_checked",
        )
        assert entry.topic is None
        contract = ModelDiscoveredContract(
            name="node_projection_dep_health",
            node_type="REDUCER_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=Path("/fake/contract.yaml"),
            entry_point_name="node_projection_dep_health",
            package_name="omnimarket",
            event_bus=ModelEventBusWiring(
                subscribe_topics=(topic,),
                publish_topics=(),
            ),
            handler_routing=ModelHandlerRouting(
                routing_strategy="operation_match",
                handlers=(entry,),
            ),
        )

        assert _topics_for_handler_entry(contract, entry) == (topic,)

    @pytest.mark.unit
    def test_single_handler_multi_topic_topic_none_still_returns_all(self) -> None:
        """The OMN-12848 sole-handler-multi-topic case (topic=None) is unchanged:
        the single entry still owns every subscribe topic."""
        topics = (
            "onex.cmd.omnimarket.node-generation-requested.v1",
            "onex.cmd.omnimarket.node-deploy.v1",
        )
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerGen", module="fake.module"),
            event_model=ModelHandlerRef(name="ModelGen", module="fake.models"),
        )
        assert entry.topic is None
        contract = ModelDiscoveredContract(
            name="node_generation_consumer",
            node_type="ORCHESTRATOR_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=Path("/fake/contract.yaml"),
            entry_point_name="node_generation_consumer",
            package_name="omnimarket",
            event_bus=ModelEventBusWiring(
                subscribe_topics=topics,
                publish_topics=(),
            ),
            handler_routing=ModelHandlerRouting(
                routing_strategy="operation_match",
                handlers=(entry,),
            ),
        )

        assert _topics_for_handler_entry(contract, entry) == topics


class TestDiscoveryThreadsTopic:
    """`_parse_handler_routing` threads the contract's ``topic:`` onto the model."""

    @pytest.mark.unit
    def test_parse_handler_routing_threads_topic(self) -> None:
        hr_raw = {
            "routing_strategy": "topic_match",
            "handlers": [
                {
                    "topic": _SWARM_COMPLETED,
                    "operation": "swarm_dispatch_completed",
                    "handler": {
                        "name": "HandlerProjectionSwarm",
                        "module": "fake.handlers",
                    },
                    "event_model": {
                        "name": "ModelSwarmDispatchEvent",
                        "module": "fake.handlers",
                    },
                },
            ],
        }

        parsed = _parse_handler_routing(hr_raw)

        assert len(parsed.handlers) == 1
        assert parsed.handlers[0].topic == _SWARM_COMPLETED

    @pytest.mark.unit
    def test_parse_handler_routing_defaults_topic_to_none(self) -> None:
        hr_raw = {
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "operation": "dep_health_checked",
                    "handler": {"name": "HandlerDepHealth", "module": "fake.handlers"},
                    "event_model": {"name": "ModelDepHealth", "module": "fake.models"},
                },
            ],
        }

        parsed = _parse_handler_routing(hr_raw)

        assert parsed.handlers[0].topic is None
