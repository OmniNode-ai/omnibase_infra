# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S6 single-owner boot assertions (OMN-14758, §c.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.runtime.runtime_dispatch import DispatchRoute
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.core_runtime.single_owner import assert_single_owner_split

TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"


class _H:
    def handle(self, request: object) -> object:  # pragma: no cover
        return request


def _route() -> DispatchRoute:
    return DispatchRoute(name="H", handler=_H(), published_events={})


def _contract(name: str, subscribe: tuple[str, ...]) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=Path(f"/nonexistent/{name}.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=subscribe),
    )


def test_passes_disjoint_single_owner() -> None:
    assert_single_owner_split(
        core_runtime_topics=frozenset({TOPIC}),
        routing_map={TOPIC: _route()},
        legacy_subscribed_topics=frozenset({"onex.cmd.omnibase-infra.other.v1"}),
        contracts=[_contract("owner", (TOPIC,))],
    )


def test_raises_when_legacy_still_owns_topic() -> None:
    with pytest.raises(ModelOnexError, match="legacy"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={TOPIC: _route()},
            legacy_subscribed_topics=frozenset({TOPIC}),  # overlap!
            contracts=[_contract("owner", (TOPIC,))],
        )


def test_raises_when_routing_map_incomplete() -> None:
    with pytest.raises(ModelOnexError, match="do not exactly match"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={},  # missing route
            legacy_subscribed_topics=frozenset(),
            contracts=[_contract("owner", (TOPIC,))],
        )


def test_raises_on_multi_consumer_fanout_topic() -> None:
    with pytest.raises(ModelOnexError, match="multiple subscribing"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={TOPIC: _route()},
            legacy_subscribed_topics=frozenset(),
            contracts=[
                _contract("owner", (TOPIC,)),
                _contract("second_consumer", (TOPIC,)),  # fan-out — reject move
            ],
        )
