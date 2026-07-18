# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Prove the S6 allowlist defaults to a strict no-op (OMN-14758, §c.0/§c.4/§f).

Default-empty ``ONEX_CORE_RUNTIME_TOPICS`` ⇒ the legacy subscribe path is unchanged
(every topic still subscribed) and no RuntimeDispatch is constructed — the rollback state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.errors.model_onex_error import ModelOnexError
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
    parse_core_runtime_topic_owners,
    parse_core_runtime_topics,
    resolve_core_runtime_owners,
)

ROUTING_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
FANOUT_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"


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


# --- S8 §D1=4b: topic-owner parse + resolution + fan-out legacy skip ------------------


def _fanout_contract(
    name: str, *, plugin_managed: bool = False
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=Path(f"/nonexistent/{name}.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(FANOUT_TOPIC,),
            plugin_managed=plugin_managed,
        ),
    )


def test_parse_topic_owners_default_empty() -> None:
    assert parse_core_runtime_topic_owners({}) == {}
    assert parse_core_runtime_topic_owners({"ONEX_CORE_RUNTIME_TOPIC_OWNERS": ""}) == {}


def test_parse_topic_owners_valid() -> None:
    parsed = parse_core_runtime_topic_owners(
        {"ONEX_CORE_RUNTIME_TOPIC_OWNERS": f" {FANOUT_TOPIC}=node_orch , "}
    )
    assert parsed == {FANOUT_TOPIC: "node_orch"}


def test_parse_topic_owners_rejects_malformed() -> None:
    with pytest.raises(ModelOnexError, match="malformed entry"):
        parse_core_runtime_topic_owners({"ONEX_CORE_RUNTIME_TOPIC_OWNERS": "no_equals"})


def test_parse_topic_owners_rejects_conflicting_duplicate() -> None:
    with pytest.raises(ModelOnexError, match="conflicting owner"):
        parse_core_runtime_topic_owners(
            {"ONEX_CORE_RUNTIME_TOPIC_OWNERS": f"{FANOUT_TOPIC}=a,{FANOUT_TOPIC}=b"}
        )


def test_resolve_owners_single_subscriber_auto() -> None:
    owners = resolve_core_runtime_owners(
        [_contract()], frozenset({ROUTING_TOPIC}), designated_owners={}
    )
    assert owners == {ROUTING_TOPIC: "node_delegation_routing_reducer"}


def test_resolve_owners_single_subscriber_rejects_wrong_designation() -> None:
    with pytest.raises(ModelOnexError, match="does not match its sole subscriber"):
        resolve_core_runtime_owners(
            [_contract()],
            frozenset({ROUTING_TOPIC}),
            designated_owners={ROUTING_TOPIC: "not_the_subscriber"},
        )


def test_resolve_owners_rejects_plugin_managed_allowlist_subscriber() -> None:
    with pytest.raises(ModelOnexError, match="plugin-managed subscribers"):
        resolve_core_runtime_owners(
            [_fanout_contract("plugin_owner", plugin_managed=True)],
            frozenset({FANOUT_TOPIC}),
            designated_owners={FANOUT_TOPIC: "plugin_owner"},
        )


def test_resolve_owners_fanout_requires_designation() -> None:
    with pytest.raises(ModelOnexError, match="no owner is designated"):
        resolve_core_runtime_owners(
            [_fanout_contract("orch"), _fanout_contract("projection")],
            frozenset({FANOUT_TOPIC}),
            designated_owners={},
        )


def test_resolve_owners_fanout_with_designation() -> None:
    owners = resolve_core_runtime_owners(
        [_fanout_contract("orch"), _fanout_contract("projection")],
        frozenset({FANOUT_TOPIC}),
        designated_owners={FANOUT_TOPIC: "orch"},
    )
    assert owners == {FANOUT_TOPIC: "orch"}


def test_resolve_owners_designated_owner_must_subscribe() -> None:
    with pytest.raises(ModelOnexError, match="does not subscribe"):
        resolve_core_runtime_owners(
            [_fanout_contract("orch"), _fanout_contract("projection")],
            frozenset({FANOUT_TOPIC}),
            designated_owners={FANOUT_TOPIC: "not_a_subscriber"},
        )


@pytest.mark.asyncio
async def test_fanout_owner_subscription_skipped_nonowner_kept() -> None:
    """§4b legacy skip: only the OWNER's subscription is skipped; non-owners stay legacy."""
    owners = {FANOUT_TOPIC: "orch"}

    owner_bus = _RecordingBus()
    owner_subscribed = await _subscribe_contract_topics(
        contract=_fanout_contract("orch"),
        dispatch_engine=object(),
        event_bus=owner_bus,
        environment="dev",
        core_runtime_topics=frozenset({FANOUT_TOPIC}),
        core_runtime_owners=owners,
    )
    assert owner_subscribed == []  # owner moved to core runtime -> legacy skips it
    assert owner_bus.subscribed == []

    legacy_bus = _RecordingBus()
    legacy_subscribed = await _subscribe_contract_topics(
        contract=_fanout_contract("projection"),
        dispatch_engine=object(),
        event_bus=legacy_bus,
        environment="dev",
        core_runtime_topics=frozenset({FANOUT_TOPIC}),
        core_runtime_owners=owners,
    )
    # Non-owner fan-out consumer KEEPS its legacy subscription (its own distinct group).
    assert legacy_subscribed == [FANOUT_TOPIC]
    assert legacy_bus.subscribed == [FANOUT_TOPIC]
