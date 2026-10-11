# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: auto-wiring registers over-long route ids bounded (OMN-20767).

omnimarket 0.4.308 ships two contracts whose natural route id is 206 characters,
over the 200-character ``max_length`` on ``ModelDispatchRoute.route_id``. The
route constructor raised during ``wire_from_manifest`` and the runtime failed at
auto-wiring. This drives the wiring phase for those two contract shapes against a
real ``MessageDispatchEngine`` and asserts both wire, with ids that fit the
route model and are registered on the engine, and that a second boot derives the
same ids.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_ROUTE_ID_MAX = 200

# The two omnimarket 0.4.308 contracts that failed the dev-lane runtime (name,
# handler class, operation, subscribed command topic), from their
# contract.yaml handler_routing.
_OMNIMARKET_0_4_308_JUDGED = (
    (
        "node_delegation_acceptance_judged_replay_compute",
        "HandlerDelegationAcceptanceJudgedReplay",
        "delegation_acceptance_judged_replay",
        "onex.cmd.omnimarket.delegation-acceptance-replay-requested.v1",
    ),
    (
        "node_delegation_acceptance_judged_publish_effect",
        "HandlerDelegationAcceptanceJudgedPublish",
        "delegation_acceptance_judged_publish",
        "onex.cmd.omnimarket.delegation-acceptance-judged-publish.v1",
    ),
)


class _HandlerProbe:
    async def handle(self, envelope: object) -> None:
        return None


def _contract(
    name: str, handler: str, operation: str, topic: str
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(name) / "contract.yaml",
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler, module="omnimarket.fake.handlers"
                    ),
                    event_model=None,
                    operation=operation,
                ),
            ),
        ),
    )


async def _wire() -> tuple[MessageDispatchEngine, tuple[str, ...], tuple[str, ...]]:
    manifest = ModelAutoWiringManifest(
        contracts=tuple(_contract(*shape) for shape in _OMNIMARKET_0_4_308_JUDGED),
        errors=(),
    )
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerProbe,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert report.total_failed == 0, [r.reason for r in report.results]
    assert report.total_wired == len(_OMNIMARKET_0_4_308_JUDGED)
    routes = tuple(rid for r in report.results for rid in r.routes_registered)
    dispatchers = tuple(did for r in report.results for did in r.dispatchers_registered)
    return dispatch_engine, routes, dispatchers


@pytest.mark.asyncio
async def test_over_long_route_ids_wire_bounded_and_register_on_the_engine() -> None:
    dispatch_engine, routes, dispatchers = await _wire()

    assert len(routes) == len(_OMNIMARKET_0_4_308_JUDGED)
    assert len(set(routes)) == len(routes)
    assert dispatch_engine.route_count == len(routes)
    for route_id in routes:
        assert len(route_id) <= _ROUTE_ID_MAX
    for dispatcher_id in dispatchers:
        assert len(dispatcher_id) <= _ROUTE_ID_MAX


@pytest.mark.asyncio
async def test_route_ids_are_stable_across_boots() -> None:
    _, first_routes, first_dispatchers = await _wire()
    _, second_routes, second_dispatchers = await _wire()

    assert first_routes == second_routes
    assert first_dispatchers == second_dispatchers
