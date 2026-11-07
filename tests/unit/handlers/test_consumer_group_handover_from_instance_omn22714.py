# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22714 (T8 part 2): a handover from a retired INSTANCE of the same contract.

The effects containers consume every effects contract under the contract's own
group id carrying their ``KAFKA_INSTANCE_ID`` and the per-topic scope
(``{base}.__i.runtime-effects.__t.{topic}``). A static ``from_group_id`` cannot
name that group for ~175 contracts. ``from_instance_id`` names the retired
process instead, and the retired group is derived by the same bus resolution the
successor's group goes through, so it is the group the old container committed
to, per topic.

The bus is a real ``EventBusKafka`` (never started); only the consumer factory
is replaced, as in ``test_consumer_group_handover_kafka_path_omn21753.py``.
Golden chain: the effects group's commits reach the runtime group. Error chain:
a declaration that names the bus's own instance, or an in-flight source, is
refused with nothing committed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch
from uuid import uuid4

import pytest
from aiokafka.protocol.admin import DescribeGroupsRequest, DescribeGroupsRequest_v0
from aiokafka.structs import TopicPartition
from pydantic import ValidationError

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    ConsumerGroupFenceError,
    HandlerContractSubscribePreflight,
    resolve_retired_group,
)
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_request import (
    ModelContractSubscribePreflightRequest,
)
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    ModelConsumeConcurrency,
)

TOPIC = "onex.evt.omnimarket.effect-demo.v1"
BASE_GROUP = "dev.omnimarket.node_effect_demo.consume.1.0.0"
MAIN_INSTANCE = "runtime-main"
EFFECTS_INSTANCE = "runtime-effects"
FACTORY = (
    "omnibase_infra.handlers.handler_contract_subscribe_preflight."
    "create_offset_consumer"
)

pytestmark = pytest.mark.unit


class _Cluster:
    def __init__(self, partitions: set[int]) -> None:
        self._partitions = partitions

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        return self._partitions


class _Client:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker

    async def fetch_all_metadata(self) -> _Cluster:
        return _Cluster(self._broker.partitions)

    async def coordinator_lookup(self, coordinator_type: int, group: str) -> int:
        return 1

    async def send(self, node_id: int, request: object) -> SimpleNamespace:
        assert isinstance(request, DescribeGroupsRequest)
        (group,) = request.build(DescribeGroupsRequest_v0).groups
        members = self._broker.members.get(group, [])
        return SimpleNamespace(
            groups=[
                (0, group, "Stable" if members else "Empty", "consumer", "", members)
            ]
        )


class _Broker:
    def __init__(self, partitions: set[int]) -> None:
        self.partitions = partitions
        self.committed: dict[str, dict[TopicPartition, int]] = {}
        self.members: dict[str, list[str]] = {}
        self.commit_calls: list[tuple[str, dict[TopicPartition, int]]] = []
        self.read_groups: list[str] = []


class _Consumer:
    def __init__(self, broker: _Broker, group_id: str) -> None:
        self._broker = broker
        self._group = group_id
        self._client = _Client(broker)

    async def start(self) -> None:
        self._broker.read_groups.append(self._group)

    async def stop(self) -> None:
        return None

    def assign(self, partitions: list[TopicPartition]) -> None:
        return None

    async def committed(self, tp: TopicPartition) -> int | None:
        return self._broker.committed.get(self._group, {}).get(tp)

    async def beginning_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return dict.fromkeys(partitions, 0)

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return dict.fromkeys(partitions, 100)

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self._broker.commit_calls.append((self._group, dict(offsets)))
        self._broker.committed.setdefault(self._group, {}).update(offsets)


def _bus(instance_id: str | None = MAIN_INSTANCE) -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="dev",
            instance_id=instance_id,
        )
    )


@contextmanager
def _patched(broker: _Broker) -> Iterator[None]:
    def factory(config: ModelKafkaEventBusConfig, group_id: str) -> _Consumer:
        return _Consumer(broker, group_id)

    with patch(FACTORY, side_effect=factory):
        yield


def _handover(
    rule: Literal["copy_if_uncommitted", "last_ran"] = "last_ran",
    instance: str = EFFECTS_INSTANCE,
) -> ModelConsumerGroupHandover:
    return ModelConsumerGroupHandover(
        from_instance_id=instance,
        from_auto_offset_reset="latest",
        handover_rule=rule,
    )


def _preflight(bus: EventBusKafka, handover: ModelConsumerGroupHandover):
    return HandlerContractSubscribePreflight(
        contract_name="node_effect_demo",
        concurrency=ModelConsumeConcurrency(),
        handover=handover,
        event_bus=bus,
        environment="dev",
        consumer_group=BASE_GROUP,
    )


def _retired_group(bus: EventBusKafka) -> str:
    return resolve_retired_group(
        bus, _handover(), environment="dev", to_group=BASE_GROUP, topic=TOPIC
    )


def test_the_retired_group_is_the_old_containers_per_topic_group() -> None:
    """``{base}.__i.runtime-effects.__t.{topic}``, what the effects container joined."""
    retired = _retired_group(_bus())
    effects_bus = _bus(EFFECTS_INSTANCE)
    joined = effects_bus._resolve_effective_group_id(
        BASE_GROUP, TOPIC, uuid4(), (TOPIC, BASE_GROUP)
    )

    assert retired == joined
    assert retired == f"{BASE_GROUP}.__i.{EFFECTS_INSTANCE}.__t.{TOPIC}"


def test_the_instance_declaration_is_exclusive_with_the_other_two_forms() -> None:
    with pytest.raises(ValidationError, match="from_instance_id"):
        ModelConsumerGroupHandover(
            from_instance_id="runtime-effects",
            from_group_id="a-group",
            from_auto_offset_reset="latest",
        )
    with pytest.raises(ValidationError, match="from_instance_id"):
        ModelConsumerGroupHandover(
            from_instance_id="runtime-effects",
            from_service="svc",
            from_node="node",
            from_version="v1",
            from_auto_offset_reset="latest",
        )
    with pytest.raises(ValidationError, match="pattern"):
        ModelConsumerGroupHandover(
            from_instance_id="bad instance",
            from_auto_offset_reset="latest",
        )


def test_the_static_source_group_refuses_an_instance_declaration() -> None:
    with pytest.raises(ValueError, match="per topic"):
        _handover().source_group("dev")


@pytest.mark.asyncio
async def test_golden_the_runtime_group_resumes_from_the_effects_groups_commits() -> (
    None
):
    """Golden chain: effects committed 40 and 7; main's group holds 10 and none."""
    broker = _Broker({0, 1})
    bus = _bus()
    physical = bus.resolve_bus_binding(TOPIC).physical_topic
    retired = _retired_group(bus)
    successor = bus._resolve_effective_group_id(
        BASE_GROUP, TOPIC, uuid4(), (TOPIC, BASE_GROUP)
    )
    broker.committed[retired] = {
        TopicPartition(physical, 0): 40,
        TopicPartition(physical, 1): 7,
    }
    broker.committed[successor] = {TopicPartition(physical, 0): 10}
    preflight = _preflight(bus, _handover())

    with _patched(broker):
        result = await preflight.handle(
            ModelContractSubscribePreflightRequest(topic=TOPIC)
        )

    assert result.handover is not None
    assert result.handover.from_group == retired
    assert result.handover.committed == {0: 40, 1: 7}
    assert broker.committed[successor] == {
        TopicPartition(physical, 0): 40,
        TopicPartition(physical, 1): 7,
    }
    assert broker.committed[retired] == {
        TopicPartition(physical, 0): 40,
        TopicPartition(physical, 1): 7,
    }, "the retired group is read, never written"


@pytest.mark.asyncio
async def test_error_a_still_running_effects_container_is_refused() -> None:
    """Error chain: the old container still has a member, so nothing is copied."""
    broker = _Broker({0})
    bus = _bus()
    retired = _retired_group(bus)
    physical = bus.resolve_bus_binding(TOPIC).physical_topic
    broker.committed[retired] = {TopicPartition(physical, 0): 40}
    broker.members[retired] = ["member-of-the-running-effects-container"]
    handover = ModelConsumerGroupHandover(
        from_instance_id=EFFECTS_INSTANCE,
        from_auto_offset_reset="latest",
        handover_rule="last_ran",
        fence={"wait_seconds": 0.0, "poll_seconds": 0.01},
    )

    with _patched(broker), pytest.raises(ConsumerGroupFenceError):
        await _preflight(bus, handover).handle(
            ModelContractSubscribePreflightRequest(topic=TOPIC)
        )

    assert broker.commit_calls == []


def test_error_a_declaration_naming_the_buss_own_instance_is_refused() -> None:
    """Handing a group over to itself would read the successor's own commits."""
    with pytest.raises(ValueError, match="own instance"):
        resolve_retired_group(
            _bus(MAIN_INSTANCE),
            _handover(instance=MAIN_INSTANCE),
            environment="dev",
            to_group=BASE_GROUP,
            topic=TOPIC,
        )


def test_error_an_instance_declaration_on_a_bus_that_cannot_derive_it_is_refused() -> (
    None
):
    with pytest.raises(TypeError, match="cannot derive"):
        resolve_retired_group(
            object(),
            _handover(),
            environment="dev",
            to_group=BASE_GROUP,
            topic=TOPIC,
        )
