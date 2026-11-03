# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the boot interleave attaches contracts concurrently, up to the declared limit.

The declared ``max_concurrent_contract_attach`` is both a floor and a ceiling on
how many contracts are inside the broker join at once:

* with a limit above one, that many contracts are in the join together (a loop
  that attaches one contract at a time never reaches it), and
* no more than the limit are ever in it (unbounded fan-out never stays under it).

The tests gate every join on an event the test releases, so they count joins in
flight and never read a clock. Priority contracts are the first to enter the
join, so a priority list longer than the limit still orders the first wave.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.unit.runtime.auto_wiring.test_omn18087_boot_subscribes_only_topics_in_use import (
    RecordingKafkaBus,
)
from tests.unit.runtime.auto_wiring.test_per_contract_boot_interleave import (
    RecordingProvisioner,
    _contract,
    _fake_handler_cls,
)

pytestmark = pytest.mark.unit

_TOPICS = tuple(f"onex.cmd.test.bounded-{index}.v1" for index in range(6))
_YIELDS = 400


class GatedKafkaBus(RecordingKafkaBus):
    """A bus whose join stays open until the test releases it, counting joins in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.peak = 0
        self.entered: list[str] = []
        self.release = asyncio.Event()

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
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.entered.append(topic)
        try:
            await self.release.wait()
        finally:
            self.in_flight -= 1
        self.joins.append((topic, None))

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


async def _yield_to_loop() -> None:
    for _ in range(_YIELDS):
        await asyncio.sleep(0)


async def _boot_gated(
    names_and_topics: tuple[tuple[str, str], ...],
    config: ModelTopicReadinessConfig,
) -> tuple[GatedKafkaBus, asyncio.Task[None]]:
    manifest = ModelAutoWiringManifest(
        contracts=tuple(_contract(name, (topic,)) for name, topic in names_and_topics)
    )
    engine = MessageDispatchEngine()
    bus = GatedKafkaBus()
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

    async def _attach() -> None:
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=RecordingProvisioner(),
            readiness_config=config,
        )

    return bus, asyncio.create_task(_attach())


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_joins_in_flight_reach_the_declared_limit_and_never_pass_it(
    limit: int,
) -> None:
    pairs = tuple((f"node_bounded_{i}", topic) for i, topic in enumerate(_TOPICS))
    bus, task = await _boot_gated(
        pairs, ModelTopicReadinessConfig(max_concurrent_contract_attach=limit)
    )

    await _yield_to_loop()
    assert bus.in_flight == limit
    assert len(bus.entered) == limit

    bus.release.set()
    await asyncio.wait_for(task, timeout=30)

    assert bus.peak == limit
    assert {topic for topic, _ in bus.joins} == set(_TOPICS)
    assert len(bus.joins) == len(_TOPICS)


@pytest.mark.asyncio
async def test_the_first_wave_of_joins_is_the_priority_contracts() -> None:
    pairs = (
        *((f"node_bounded_{i}", topic) for i, topic in enumerate(_TOPICS[:4])),
        ("projection_claude_hook_events", _TOPICS[4]),
        ("node_delegation_orchestrator", _TOPICS[5]),
    )
    bus, task = await _boot_gated(
        pairs,
        ModelTopicReadinessConfig(
            max_concurrent_contract_attach=2,
            priority_contracts=(
                "node_delegation_orchestrator",
                "projection_claude_hook_events",
            ),
        ),
    )

    await _yield_to_loop()
    assert bus.entered == [_TOPICS[5], _TOPICS[4]]

    bus.release.set()
    await asyncio.wait_for(task, timeout=30)
    assert len(bus.joins) == len(_TOPICS)
