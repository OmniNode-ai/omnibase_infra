# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20209: identify a chain consumer by its complete subscription footprint."""

from __future__ import annotations

from typing import Any

import pytest
from aiokafka.errors import GroupAuthorizationFailedError

from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX
from omnibase_infra.backends import backend_probe

pytestmark = pytest.mark.unit

_COMMAND = "onex.cmd.synthetic.chain-request.v1"
_SUBSCRIBE = (
    _COMMAND,
    "onex.evt.synthetic.routing-decision.v1",
    "onex.evt.synthetic.inference-response.v1",
)
_BROKER = "broker.invalid:19092"
_BASE_A = "local.runtime_config.delegation-orchestrator.consume.1.0.0.__i.runtime-main"
_BASE_B = "local.synthetic.ledger-projection.consume.1.0.0.__i.runtime-main"
_BASE_C = "prepr.runtime_config.delegation-orchestrator.consume.1.0.0.__i.runtime-main"
_A = _BASE_A + TOPIC_SCOPE_INFIX + _COMMAND
_B = _BASE_B + TOPIC_SCOPE_INFIX + _COMMAND
_C = _BASE_C + TOPIC_SCOPE_INFIX + _COMMAND


class _Response:
    def __init__(self, groups: list[tuple[Any, ...]]) -> None:
        self.groups = groups


class _Admin:
    """aiokafka admin fake, mirroring the OMN-19914 denial fixture."""

    listed: list[str] = []
    denied_raise: set[str] = set()
    denied_code: set[str] = set()
    states: dict[str, str] = {}
    described: list[str] = []
    calls: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def start(self) -> None:
        self.calls.append("start")

    async def close(self) -> None:
        self.calls.append("close")

    async def describe_cluster(self) -> dict[str, Any]:
        self.calls.append("cluster")
        return {"brokers": [{"node_id": 1}]}

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        self.calls.append("list")
        return [(group, "consumer") for group in self.listed]

    async def describe_consumer_groups(self, group_ids: list[str]) -> list[_Response]:
        (group_id,) = group_ids
        self.described.append(group_id)
        if group_id in self.denied_raise:
            raise GroupAuthorizationFailedError(f"Unable to get coordinator {group_id}")
        code = 30 if group_id in self.denied_code else 0
        return [
            _Response([(code, group_id, self.states[group_id], "consumer", "", [])])
        ]


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> type[_Admin]:
    import aiokafka.admin

    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    _Admin.listed = [
        base + TOPIC_SCOPE_INFIX + topic
        for base in (_BASE_A, _BASE_C)
        for topic in _SUBSCRIBE
    ] + [_B, "unscoped-group"]
    _Admin.denied_raise = set()
    _Admin.denied_code = set()
    _Admin.states = {_A: "Stable", _B: "Stable", _C: "Empty"}
    _Admin.described = []
    _Admin.calls = []
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _Admin)
    return _Admin


def test_only_complete_footprints_are_described_and_only_stable_is_returned(
    admin: type[_Admin],
) -> None:
    assert backend_probe.live_chain_consumer_groups(
        command_topic=_COMMAND, subscribe_topics=_SUBSCRIBE, bootstrap_servers=_BROKER
    ) == (_A,)
    assert admin.described == sorted([_A, _C])
    assert admin.calls == ["start", "cluster", "list", "close"]


@pytest.mark.parametrize("response_code", [False, True])
def test_every_candidate_denied_is_a_typed_refusal(
    admin: type[_Admin], response_code: bool
) -> None:
    if response_code:
        admin.denied_code = {_A, _C}
    else:
        admin.denied_raise = {_A, _C}
    with pytest.raises(backend_probe.ConsumerGroupDescribeDeniedError) as caught:
        backend_probe.live_chain_consumer_groups(
            command_topic=_COMMAND,
            subscribe_topics=_SUBSCRIBE,
            bootstrap_servers=_BROKER,
        )
    assert caught.value.group_ids == tuple(sorted((_A, _C)))
    assert admin.described == sorted([_A, _C])


def test_subscriptions_must_include_the_command_topic(admin: type[_Admin]) -> None:
    with pytest.raises(ValueError, match="command_topic"):
        backend_probe.live_chain_consumer_groups(
            command_topic=_COMMAND,
            subscribe_topics=_SUBSCRIBE[1:],
            bootstrap_servers=_BROKER,
        )
    assert admin.calls == []
