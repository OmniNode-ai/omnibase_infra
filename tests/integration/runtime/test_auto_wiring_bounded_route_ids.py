# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression for OMN-20767: over-long auto-wired route ids must still wire.

omnimarket 0.4.308's ``node_delegation_acceptance_judged_replay_compute``
derived a 206-character route id, over the 200-character ``max_length`` on
``ModelDispatchRoute.route_id``, so route construction raised and
``wire_from_manifest`` failed the contract. This drives the production path
(``wire_from_manifest`` against a real ``MessageDispatchEngine``) with that
contract's real name, handler and topic.
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

_CONTRACT_NAME = "node_delegation_acceptance_judged_replay_compute"
_TOPIC = "onex.cmd.omnimarket.delegation-acceptance-replay-requested.v1"
_ROUTE_ID_MAX = 200


class HandlerDelegationAcceptanceJudgedReplayProbe:
    """Zero-arg stand-in for the real handler."""

    async def handle(self, envelope: object) -> None:
        return None


@pytest.mark.asyncio
async def test_long_contract_name_wires_with_bounded_route_id() -> None:
    contract = ModelDiscoveredContract(
        name=_CONTRACT_NAME,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("omn-20767/contract.yaml"),
        entry_point_name=_CONTRACT_NAME,
        package_name="test-package",
        event_bus=ModelEventBusWiring(subscribe_topics=(_TOPIC,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    topic=_TOPIC,
                    operation="delegation_acceptance_judged_replay",
                    event_model=ModelHandlerRef(
                        name="ModelReplayRequested", module="omnimarket.models.fake"
                    ),
                    handler=ModelHandlerRef(
                        name="HandlerDelegationAcceptanceJudgedReplay",
                        module="omnimarket.fake.handlers",
                    ),
                ),
            ),
        ),
    )
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerDelegationAcceptanceJudgedReplayProbe,
    ):
        report = await wire_from_manifest(
            manifest=ModelAutoWiringManifest(contracts=(contract,), errors=()),
            dispatch_engine=MessageDispatchEngine(),
            event_bus=event_bus,
            environment="local",
        )

    assert report.total_failed == 0, report.results
    result = report.results[0]
    assert len(result.routes_registered) == 1
    assert len(result.routes_registered[0]) <= _ROUTE_ID_MAX
    assert len(result.dispatchers_registered) == 1
    assert len(result.dispatchers_registered[0]) <= _ROUTE_ID_MAX
