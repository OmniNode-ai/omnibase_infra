# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the runtime joins at boot only the topics in use.

Measured on the dev lane on 2026-10-28: ``omninode-runtime`` joined
522 consumer groups at boot across 378 contracts, and 408 of them were on
topics with no record in the last 7 days (385 on topics that never held one).
A contract whose subscribe topics hold no record is now started when its first
request arrives, through the same provision -> ready -> attach interleave.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from unittest.mock import patch

import pytest

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.unit.runtime.auto_wiring.test_per_contract_boot_interleave import (
    RecordingProvisioner,
    _contract,
    _fake_handler_cls,
)

pytestmark = pytest.mark.unit

ACTIVE = "onex.cmd.test.delegate-in-use.v1"
IDLE = "onex.cmd.test.never-requested.v1"


class RecordingKafkaBus(EventBusKafka):
    """A Kafka bus whose subscribe records the join instead of joining."""

    def __init__(self) -> None:
        super().__init__(
            config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
        )
        self.joins: list[tuple[str, str | None]] = []
        self.joined = asyncio.Event()

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
        auto_offset_reset: str | None = None,
    ) -> Callable[[], Awaitable[None]]:
        self.joins.append((topic, auto_offset_reset))
        self.joined.set()

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


class FakeDemandSource:
    """Reports which topics hold records and hands back the first-record hook."""

    def __init__(self, in_use: frozenset[str]) -> None:
        self._in_use = in_use
        self.watched: frozenset[str] = frozenset()
        self.on_first_record: Callable[[str], Awaitable[None]] | None = None
        self.stopped = False

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        self.watched = frozenset(topics) - self._in_use
        self.on_first_record = on_first_record

        async def stop() -> None:
            self.stopped = True

        return frozenset(topics) & self._in_use, stop


async def _boot(
    demand: FakeDemandSource | None,
) -> tuple[
    RecordingKafkaBus,
    dict[str, tuple[str, ...]],
    list[ModelContractAttachExclusion],
    list[ModelContractAttachResult],
    list[Callable[[], Awaitable[None]]],
]:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_in_use", (ACTIVE,)),
            _contract("node_idle", (IDLE,)),
        )
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    exclusions: list[ModelContractAttachExclusion] = []
    results: list[ModelContractAttachResult] = []
    stops: list[Callable[[], Awaitable[None]]] = []
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_fake_handler_cls(),
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        subscribed = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=RecordingProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=results,
            exclusions_out=exclusions,
            demand_source=demand,
            demand_watch_stops_out=stops,
        )
    return bus, subscribed, exclusions, results, stops


@pytest.mark.asyncio
async def test_an_idle_contracts_topics_are_not_joined_at_boot() -> None:
    demand = FakeDemandSource(in_use=frozenset({ACTIVE}))
    bus, subscribed, exclusions, results, stops = await _boot(demand)

    assert [topic for topic, _ in bus.joins] == [ACTIVE]
    assert set(subscribed) == {"node_in_use"}
    assert demand.watched == {IDLE}
    deferred = [e for e in exclusions if e.contract_name == "node_idle"]
    assert [e.reason for e in deferred] == [
        EnumContractAttachExclusionReason.DEMAND_DEFERRED
    ]
    assert [r.contract_name for r in results] == ["node_in_use"]
    assert len(stops) == 1


@pytest.mark.asyncio
async def test_a_first_request_starts_the_idle_contract_from_earliest() -> None:
    """The record that started the contract is the first one its group reads."""
    demand = FakeDemandSource(in_use=frozenset({ACTIVE}))
    bus, _subscribed, _exclusions, _results, _stops = await _boot(demand)
    assert demand.on_first_record is not None
    bus.joined.clear()

    await demand.on_first_record(IDLE)
    await asyncio.wait_for(bus.joined.wait(), timeout=2)

    assert (IDLE, "earliest") in bus.joins
    # A second record on the same topic starts nothing new.
    await demand.on_first_record(IDLE)
    await asyncio.sleep(0)
    assert [join for join in bus.joins if join[0] == IDLE] == [(IDLE, "earliest")]


@pytest.mark.asyncio
async def test_without_a_demand_source_every_contract_joins_at_boot() -> None:
    bus, subscribed, exclusions, _results, _stops = await _boot(None)

    assert sorted(topic for topic, _ in bus.joins) == sorted([ACTIVE, IDLE])
    assert set(subscribed) == {"node_in_use", "node_idle"}
    assert not [
        e
        for e in exclusions
        if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
    ]
