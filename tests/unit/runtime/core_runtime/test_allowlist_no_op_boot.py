# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Prove the S6 allowlist defaults to a strict no-op (OMN-14758, §c.0/§c.4/§f).

Default-empty ``ONEX_CORE_RUNTIME_TOPICS`` ⇒ the legacy subscribe path is unchanged
(every topic still subscribed) and no RuntimeDispatch is constructed — the rollback state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import _subscribe_contract_topics
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.core_runtime.composition import (
    build_core_runtime,
    parse_core_runtime_topics,
)

ROUTING_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"


class _RecordingBus:
    def __init__(self) -> None:
        self.subscribed: list[str] = []

    async def subscribe(
        self, *, topic: str, node_identity: object, on_message: object
    ) -> None:
        self.subscribed.append(topic)


def _contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=Path("/nonexistent/contract.yaml"),
        entry_point_name="node_delegation_routing_reducer",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(ROUTING_TOPIC,),
            publish_topics=(),
        ),
    )


def test_parse_defaults_empty() -> None:
    assert parse_core_runtime_topics({}) == frozenset()
    assert parse_core_runtime_topics({"ONEX_CORE_RUNTIME_TOPICS": ""}) == frozenset()
    assert parse_core_runtime_topics({"ONEX_CORE_RUNTIME_TOPICS": "   "}) == frozenset()


@pytest.mark.asyncio
async def test_empty_allowlist_subscribes_topic_unchanged() -> None:
    bus = _RecordingBus()
    subscribed = await _subscribe_contract_topics(
        contract=_contract(),
        dispatch_engine=object(),
        event_bus=bus,
        environment="dev",
        core_runtime_topics=frozenset(),  # default no-op
    )
    assert subscribed == [ROUTING_TOPIC]
    assert bus.subscribed == [ROUTING_TOPIC]


@pytest.mark.asyncio
async def test_allowlisted_topic_is_skipped_by_legacy_path() -> None:
    bus = _RecordingBus()
    subscribed = await _subscribe_contract_topics(
        contract=_contract(),
        dispatch_engine=object(),
        event_bus=bus,
        environment="dev",
        core_runtime_topics=frozenset({ROUTING_TOPIC}),
    )
    # Legacy path must NOT subscribe an allowlisted topic (ownership=core-runtime).
    assert subscribed == []
    assert bus.subscribed == []


def test_build_core_runtime_guards_empty_allowlist() -> None:
    with pytest.raises(ValueError, match="empty allowlist"):
        build_core_runtime(
            core_runtime_topics=frozenset(),
            contracts=[],
            transport=object(),
            handler_resolver=lambda ref: object(),
            legacy_subscribed_topics=frozenset(),
        )
