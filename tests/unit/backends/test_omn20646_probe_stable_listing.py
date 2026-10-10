# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20646: ask the smaller Stable question and classify transport failures."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiokafka.errors import (
    IncompatibleBrokerVersion,
    KafkaConnectionError,
    KafkaTimeoutError,
    NodeNotReadyError,
    RequestTimedOutError,
)

from omnibase_infra.backends import backend_probe
from omnibase_infra.backends.backend_probe import (
    ConsumerGroupLivenessTransientError,
    ConsumerGroupLivenessUnknownError,
)

pytestmark = pytest.mark.unit

ListGroupsRequestV4, ListGroupsResponseV4 = backend_probe._list_groups_v4_structs()
StableListGroupsRequest = backend_probe.StableListGroupsRequest

_TOPIC = "onex.cmd.synthetic.delegate-skill.v1"
_BROKER = "broker.invalid:19092"
_STABLE = f"local.synthetic.stable.__t.{_TOPIC}"
_EMPTY = f"local.synthetic.empty.__t.{_TOPIC}"


def test_request_encodes_the_stable_filter_exactly() -> None:
    request = StableListGroupsRequest().prepare({16: (0, 4)})
    assert isinstance(request, ListGroupsRequestV4)
    assert request.FLEXIBLE_VERSION is True
    assert request.encode() == b"\x02\x07Stable\x00"


def test_request_requires_v4_and_a_known_api_key() -> None:
    with pytest.raises(NotImplementedError):
        StableListGroupsRequest().prepare({16: (0, 2)})
    with pytest.raises(IncompatibleBrokerVersion):
        StableListGroupsRequest().prepare({})


def test_response_decodes_ids_states_and_tags() -> None:
    groups = [(_STABLE, "consumer", "Stable", {}), (_EMPTY, "consumer", "Empty", {})]
    response = ListGroupsResponseV4(
        throttle_time_ms=12, error_code=0, groups=groups, tags={}
    )
    decoded = ListGroupsResponseV4.decode(response.encode())
    assert decoded.throttle_time_ms == 12
    assert decoded.error_code == 0
    assert decoded.groups == groups
    assert decoded.tags == {}


class _Admin:
    def __init__(self, **_: Any) -> None:
        self.states = {_STABLE: "Stable", _EMPTY: "Empty"}
        self.described: list[str] = []
        self.sent: list[int | None] = []
        self.list_calls = 0
        self.closed = False
        self.unsupported: BaseException | None = None
        self.error_code = 0
        self.rebalance_after_listing = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def describe_cluster(self) -> dict[str, Any]:
        return {"brokers": [{"node_id": 1}, {"node_id": 2}]}

    async def _send_request(self, request: Any, node_id: int | None = None) -> Any:
        self.sent.append(node_id)
        request_struct = request.prepare({16: (0, 4)})
        assert request_struct.states_filter == ["Stable"]
        if self.unsupported is not None:
            raise self.unsupported
        groups = [
            (group_id, "consumer", state, {})
            for group_id, state in self.states.items()
            if state == "Stable"
        ]
        return ListGroupsResponseV4(0, self.error_code, groups, {})

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        self.list_calls += 1
        return [(group_id, "consumer") for group_id in self.states]

    async def describe_consumer_groups(self, group_ids: list[str]) -> list[Any]:
        from types import SimpleNamespace

        self.described.extend(group_ids)
        return [
            SimpleNamespace(
                groups=[
                    (
                        0,
                        group_id,
                        "PreparingRebalance"
                        if self.rebalance_after_listing
                        else self.states[group_id],
                        "consumer",
                        "",
                        [],
                    )
                    for group_id in group_ids
                ]
            )
        ]


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> _Admin:
    import aiokafka.admin

    from omnibase_infra.event_bus import kafka_auth

    fake = _Admin()
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", lambda **_: fake)
    monkeypatch.setattr(kafka_auth, "build_aiokafka_auth_kwargs_for", lambda _: {})
    return fake


def test_only_stable_candidates_are_described_without_unfiltered_listing(
    admin: _Admin,
) -> None:
    assert backend_probe.live_consumer_groups(
        topic=_TOPIC, bootstrap_servers=_BROKER
    ) == (_STABLE,)
    assert admin.described == [_STABLE]
    assert admin.sent == [1, 2]
    assert admin.list_calls == 0
    assert admin.closed


@pytest.mark.parametrize(
    "unsupported", [IncompatibleBrokerVersion(), NotImplementedError()]
)
def test_old_broker_falls_back_to_the_same_answer(
    admin: _Admin, unsupported: BaseException
) -> None:
    admin.unsupported = unsupported
    assert backend_probe.live_consumer_groups(
        topic=_TOPIC, bootstrap_servers=_BROKER
    ) == (_STABLE,)
    assert admin.list_calls == 1
    assert admin.described == sorted([_STABLE, _EMPTY])
    assert admin.closed


def test_describe_still_decides_liveness_after_stable_listing(admin: _Admin) -> None:
    admin.rebalance_after_listing = True
    assert (
        backend_probe.live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)
        == ()
    )
    assert admin.described == [_STABLE]


def test_list_response_error_is_unknown_and_closes_the_client(admin: _Admin) -> None:
    admin.error_code = 31  # CLUSTER_AUTHORIZATION_FAILED
    with pytest.raises(ConsumerGroupLivenessUnknownError) as caught:
        backend_probe.live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)
    assert not isinstance(caught.value, ConsumerGroupLivenessTransientError)
    assert admin.described == []
    assert admin.list_calls == 0
    assert admin.closed


def test_candidate_ids_are_sorted_and_unique_across_brokers(admin: _Admin) -> None:
    admin.states = {"z": "Stable", "a": "Stable", "empty": "Empty"}
    ids = asyncio.run(backend_probe._list_candidate_group_ids(admin, [1, 2]))
    assert ids == ["a", "z"]


def test_chain_requires_stable_groups_on_every_footprint_topic(admin: _Admin) -> None:
    footprint = (_TOPIC, "onex.evt.synthetic.routing.v1")
    other_group = _STABLE.partition(".__t.")[0] + ".__t." + footprint[1]
    admin.states[other_group] = "Empty"
    assert (
        backend_probe.live_chain_consumer_groups(
            command_topic=_TOPIC, subscribe_topics=footprint, bootstrap_servers=_BROKER
        )
        == ()
    )
    assert admin.described == []
    admin.states[other_group] = "Stable"
    assert backend_probe.live_chain_consumer_groups(
        command_topic=_TOPIC, subscribe_topics=footprint, bootstrap_servers=_BROKER
    ) == (_STABLE,)
    assert admin.described == [_STABLE]


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize(
    ("failure", "transient"),
    [
        (RequestTimedOutError("slow"), True),
        (KafkaConnectionError("dropped"), True),
        (KafkaTimeoutError("slow"), True),
        (NodeNotReadyError("unready"), True),
        (TimeoutError("slow"), True),
        (ValueError("decode"), False),
    ],
)
def test_transport_classification(
    monkeypatch: pytest.MonkeyPatch, chain: bool, failure: Exception, transient: bool
) -> None:
    async def fail(**_: Any) -> tuple[str, ...]:
        raise failure

    async_name = (
        "_live_chain_consumer_groups_async" if chain else "_live_consumer_groups_async"
    )
    monkeypatch.setattr(backend_probe, async_name, fail)
    with pytest.raises(ConsumerGroupLivenessUnknownError) as caught:
        if chain:
            backend_probe.live_chain_consumer_groups(
                command_topic=_TOPIC,
                subscribe_topics=(_TOPIC,),
                bootstrap_servers=_BROKER,
            )
        else:
            backend_probe.live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)
    assert isinstance(caught.value, ConsumerGroupLivenessTransientError) is transient
    assert caught.value.__cause__ is failure
