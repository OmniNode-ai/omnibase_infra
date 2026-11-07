# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22714: the handover fences its source group before it reads an offset.

The one-runtime plan's rule for T8 and every fold after it: nothing takes a
consumer group's position over until the retired group is confirmed to have no
members, so a fold never copies a position its old consumer is still moving.
These tests drive the Kafka path against the fake broker of the OMN-21753 file,
whose fake consumer client answers DescribeGroups from a member list per group.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pytest

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    ConsumerGroupFenceError,
    HandlerContractSubscribePreflight,
    hand_over_committed_offsets,
)
from omnibase_infra.models.event_bus.model_consumer_group_fence import (
    ModelConsumerGroupFence,
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
from tests.unit.handlers.test_consumer_group_handover_kafka_path_omn21753 import (
    RETIRED_GROUP,
    SUCCESSOR_GROUP,
    TOPIC,
    _broker,
    _Client,
    _kafka_bus,
    _new_bus,
    _tp,
)

pytestmark = pytest.mark.unit

NO_WAIT = ModelConsumerGroupFence(wait_seconds=0.0, poll_seconds=0.01)
Rule = Literal["copy_if_uncommitted", "last_ran"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rule", ["copy_if_uncommitted", "last_ran"])
async def test_a_retired_group_with_a_member_is_refused_and_nothing_is_read(
    rule: Rule,
) -> None:
    """The defect the fence removes: a running consumer's position is copied."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0, 1})
    broker.members[RETIRED_GROUP] = [["effects-container-1"]]
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 35}
        with pytest.raises(ConsumerGroupFenceError, match="still has 1 member"):
            await hand_over_committed_offsets(
                bus,
                topic=TOPIC,
                from_group=RETIRED_GROUP,
                to_group=SUCCESSOR_GROUP,
                from_auto_offset_reset="earliest",
                handover_rule=rule,
                fence=NO_WAIT,
            )

    assert broker.commit_calls == []
    assert broker.assigned == {}, "no partition is assigned, so no offset is read"
    assert broker.describes == [RETIRED_GROUP]
    assert sorted(broker.started) == sorted(broker.stopped)


@pytest.mark.asyncio
async def test_a_member_that_leaves_inside_the_wait_lets_the_handover_run() -> None:
    """A member listed until its session times out is waited for, not refused."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})
    broker.members[RETIRED_GROUP] = [["leaving"], ["leaving"], []]
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 12}
        result = await hand_over_committed_offsets(
            bus,
            topic=TOPIC,
            from_group=RETIRED_GROUP,
            to_group=SUCCESSOR_GROUP,
            from_auto_offset_reset="earliest",
            handover_rule="last_ran",
            fence=ModelConsumerGroupFence(wait_seconds=5.0, poll_seconds=0.01),
        )

    assert broker.describes == [RETIRED_GROUP] * 3
    assert result.committed == {0: 12}
    assert result.fenced_group == RETIRED_GROUP


@pytest.mark.asyncio
async def test_an_empty_retired_group_is_fenced_and_named_in_the_result() -> None:
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 7}
        result = await hand_over_committed_offsets(
            bus,
            topic=TOPIC,
            from_group=RETIRED_GROUP,
            to_group=SUCCESSOR_GROUP,
            from_auto_offset_reset="earliest",
        )

    assert result.fenced_group == RETIRED_GROUP
    assert result.committed == {0: 7}
    assert broker.describes == [RETIRED_GROUP]


@pytest.mark.asyncio
async def test_a_group_whose_describe_errors_is_refused_never_read_as_empty() -> None:
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})
    broker.describe_error[RETIRED_GROUP] = 30  # GROUP_AUTHORIZATION_FAILED
    with _kafka_bus(broker) as bus:
        with pytest.raises(ConsumerGroupFenceError, match="error code 30"):
            await hand_over_committed_offsets(
                bus,
                topic=TOPIC,
                from_group=RETIRED_GROUP,
                to_group=SUCCESSOR_GROUP,
                from_auto_offset_reset="earliest",
                fence=NO_WAIT,
            )

    assert broker.commit_calls == []
    assert broker.assigned == {}


@pytest.mark.asyncio
async def test_a_describe_with_no_answer_for_the_group_is_refused() -> None:
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})

    async def silent(self: object, node_id: int, request: object) -> object:
        return SimpleNamespace(groups=[])

    with _kafka_bus(broker) as bus:
        with (
            patch.object(_Client, "send", silent),
            pytest.raises(ConsumerGroupFenceError, match="no answer"),
        ):
            await hand_over_committed_offsets(
                bus,
                topic=TOPIC,
                from_group=RETIRED_GROUP,
                to_group=SUCCESSOR_GROUP,
                from_auto_offset_reset="earliest",
                fence=NO_WAIT,
            )

    assert broker.commit_calls == []
    assert broker.assigned == {}
    assert sorted(broker.started) == sorted(broker.stopped)


def test_the_declaration_carries_a_fence_with_defaults_and_refuses_unknown_keys() -> (
    None
):
    base = {
        "from_group_id": "omninode-runtime-effects",
        "from_auto_offset_reset": "earliest",
        "handover_rule": "last_ran",
    }
    default = ModelConsumerGroupHandover.model_validate(base)
    assert default.fence == ModelConsumerGroupFence()
    declared = ModelConsumerGroupHandover.model_validate(
        {**base, "fence": {"wait_seconds": 90, "poll_seconds": 5}}
    )
    assert declared.fence.wait_seconds == 90.0
    with pytest.raises(ValueError, match="wait"):
        ModelConsumerGroupHandover.model_validate({**base, "fence": {"wait": 90}})
    with pytest.raises(ValueError, match="wait_seconds"):
        ModelConsumerGroupHandover.model_validate(
            {**base, "fence": {"wait_seconds": -1}}
        )


@pytest.mark.asyncio
async def test_the_preflight_passes_the_declared_fence_to_the_kafka_path() -> None:
    """A contract's preflight refuses to subscribe over a running consumer."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})
    broker.members[RETIRED_GROUP] = [["still-running"]]
    with _kafka_bus(broker) as bus:
        preflight = HandlerContractSubscribePreflight(
            contract_name="effects_fixture",
            concurrency=ModelConsumeConcurrency(),
            handover=ModelConsumerGroupHandover.model_validate(
                {
                    "from_group_id": RETIRED_GROUP,
                    "from_auto_offset_reset": "earliest",
                    "handover_rule": "last_ran",
                    "fence": {"wait_seconds": 0, "poll_seconds": 0.01},
                }
            ),
            event_bus=bus,
            environment="dev",
            consumer_group=SUCCESSOR_GROUP,
        )
        with pytest.raises(ConsumerGroupFenceError):
            await preflight.handle(ModelContractSubscribePreflightRequest(topic=TOPIC))

    assert broker.commit_calls == []
    assert broker.assigned == {}
