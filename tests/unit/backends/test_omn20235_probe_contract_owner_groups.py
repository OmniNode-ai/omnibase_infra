# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20235: only the contract owner's groups answer its liveness question.

Synthetic groups reproduce the measured 385-group listing without touching
a broker. Per-run groups and another node must neither exhaust the describe
cap nor stand in for the orchestrator when it is down.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupLivenessUnknownError,
    live_consumer_groups,
)
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner

pytestmark = pytest.mark.unit

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_BROKER = "broker.invalid:19092"
_OWNER = ModelConsumerGroupOwner(
    service="omnimarket", node="node_delegate_skill_orchestrator"
)
_OWN_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    f".__i.runtime-effects.__t.{_TOPIC}"
)
_LEDGER_GROUP = (
    "local.omnibase_infra.node_ledger_projection_compute.consume.1.6.0"
    f".__i.runtime-main.__t.{_TOPIC}"
)


class _Response:
    def __init__(self, groups: list[tuple[object, ...]]) -> None:
        self.groups = groups


class _Admin:
    """Admin fake recording every serial describe and its exact candidate."""

    listed: list[str] = []
    states: dict[str, str] = {}
    describes: list[list[str]] = []

    def __init__(self, **kwargs: object) -> None:
        assert kwargs["bootstrap_servers"] == _BROKER

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def describe_cluster(self) -> dict[str, object]:
        return {"brokers": [{"node_id": 1}]}

    async def _send_request(
        self, request: Any, node_id: int | None = None
    ) -> SimpleNamespace:
        struct = request.prepare({16: (0, 4)})
        assert struct.states_filter == ["Stable"]
        return SimpleNamespace(
            error_code=0,
            groups=[
                (group_id, "consumer", state, {})
                for group_id in self.listed
                if (state := self.states.get(group_id, "Stable")) == "Stable"
            ],
        )

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        return [(group, "consumer") for group in _Admin.listed]

    async def describe_consumer_groups(self, group_ids: list[str]) -> list[_Response]:
        _Admin.describes.append(list(group_ids))
        (group_id,) = group_ids
        state = _Admin.states[group_id]
        return [_Response([(0, group_id, state, "consumer", "", [])])]


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> type[_Admin]:
    import aiokafka.admin

    from omnibase_infra.event_bus import kafka_auth

    per_run = [
        f"local.omnibase_core.handlerdelegateskill_run_{index:012x}.consume.v1"
        f".__i.runtime-main.__t.{_TOPIC}"
        for index in range(383)
    ]
    _Admin.listed = [*per_run, _LEDGER_GROUP, _OWN_GROUP]
    _Admin.states = dict.fromkeys(per_run, "Empty")
    _Admin.states.update({_LEDGER_GROUP: "Stable", _OWN_GROUP: "Stable"})
    _Admin.describes = []
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _Admin)
    monkeypatch.setattr(kafka_auth, "build_aiokafka_auth_kwargs_for", lambda _: {})
    return _Admin


def test_contract_owner_describes_only_the_orchestrator_among_385_groups(
    admin: type[_Admin],
) -> None:
    assert len(admin.listed) == 385
    assert live_consumer_groups(
        topic=_TOPIC, bootstrap_servers=_BROKER, owner=_OWNER
    ) == (_OWN_GROUP,)
    assert admin.describes == [[_OWN_GROUP]]


def test_contract_owner_none_keeps_the_unfiltered_cap(admin: type[_Admin]) -> None:
    # OMN-20646: only Stable candidates count toward the serial describe cap.
    admin.states = dict.fromkeys(admin.listed, "Stable")
    with pytest.raises(
        ConsumerGroupLivenessUnknownError,
        match="refusing to run unbounded serial DescribeGroups probes",
    ):
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER, owner=None)
    assert admin.describes == []


def test_contract_owner_other_node_cannot_prove_an_empty_orchestrator_live(
    admin: type[_Admin],
) -> None:
    admin.listed = [_LEDGER_GROUP, _OWN_GROUP]
    admin.states[_OWN_GROUP] = "Empty"
    assert (
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER, owner=_OWNER)
        == ()
    )
    # OMN-20646: the Empty owner group is filtered before DescribeGroups.
    assert admin.describes == []


@pytest.mark.parametrize("env", ["local", "onex-dev", "prepr1"])
@pytest.mark.parametrize("version", ["1.3.0", "1.4.0"])
def test_contract_owner_matches_across_environments_and_versions(
    env: str, version: str
) -> None:
    assert _OWNER.matches(
        f"{env}.omnimarket.node_delegate_skill_orchestrator.consume.{version}"
        f".__i.runtime-effects.__t.{_TOPIC}"
    )


@pytest.mark.parametrize(
    "base",
    [
        "local.omnibase_core.handlerdelegateskill_run_0008facc39fd.consume.v1",
        "local.omnibase_infra.node_ledger_projection_compute.consume.1.6.0",
        "local.omnimarket.node_delegate_skill_orchestrator_shadow.consume.1.3.0",
    ],
)
def test_contract_owner_rejects_other_nodes_and_per_run_groups(base: str) -> None:
    assert not _OWNER.matches(f"{base}.__i.runtime-main.__t.{_TOPIC}")


@pytest.mark.parametrize(
    "scopes",
    [
        "",
        f".__t.{_TOPIC}",
        ".__i.runtime-effects",
        f".__t.{_TOPIC}.__i.runtime-effects",
        f".__i.runtime-effects.__t.{_TOPIC}",
    ],
)
def test_contract_owner_matches_base_before_either_scope(scopes: str) -> None:
    assert _OWNER.matches(
        f"local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0{scopes}"
    )
    assert not _OWNER.matches(
        "local.other.node.consume.1.3.0"
        f"{scopes}.__t.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    )


def test_contract_owner_normalizes_contract_identity() -> None:
    owner = ModelConsumerGroupOwner(
        service="OmniMarket", node="Node_Delegate_Skill_Orchestrator"
    )
    assert owner.matches(_OWN_GROUP)


def test_contract_owner_filtered_cap_names_the_owner(admin: type[_Admin]) -> None:
    admin.listed = [
        f"local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.{index}"
        f".__i.runtime-effects.__t.{_TOPIC}"
        for index in range(17)
    ]
    with pytest.raises(ConsumerGroupLivenessUnknownError) as caught:
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER, owner=_OWNER)
    message = str(caught.value)
    assert (
        "17 candidate groups of omnimarket.node_delegate_skill_orchestrator" in message
    )
    assert "refusing to run unbounded serial DescribeGroups probes" in message
    assert admin.describes == []
