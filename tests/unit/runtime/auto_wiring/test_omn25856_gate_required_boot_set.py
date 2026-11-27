# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856: boot attaches only what the readiness gate requires.

Measured on dev (build bda7e177): 115 s to healthy, 46 s of it attaching 70
contracts while the contract-attach readiness gate required 26. The other 44 hold
a retained record on a non-command topic, so the ``in_use`` policy kept them on the
boot path. The ``gate_required`` policy boots the priority contracts, the command
contracts that are in use, and any contract whose topic cannot be watched; every
other contract starts on the first record that reaches one of its topics, from
``earliest``, so that record is not lost.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from unittest.mock import patch

import pytest

from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.models.model_boot_subscriptions_config import (
    ModelBootSubscriptionsConfig,
)
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

COMMAND = "onex.cmd.test.gate-required.v1"
PROJECTION = "onex.evt.test.retained-projection.v1"
UNREADABLE = "onex.evt.test.not-yet-provisioned.v1"
PRIORITY = "onex.evt.test.boot-first.v1"


class FakeDemandSource:
    """Holds a record on COMMAND and PROJECTION; UNREADABLE cannot be read."""

    retained = frozenset({COMMAND, PROJECTION, PRIORITY})

    def __init__(self) -> None:
        self.watched: frozenset[str] = frozenset()
        self.on_first_record: Callable[[str], Awaitable[None]] | None = None

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
        *,
        watch_retained: Collection[str] = (),
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        asked = set(topics)
        deferred_retained = self.retained & set(watch_retained)
        idle = asked - self.retained - {UNREADABLE}
        self.watched = frozenset(idle | deferred_retained)
        self.on_first_record = on_first_record
        in_use = ((asked & self.retained) - deferred_retained) | (asked & {UNREADABLE})

        async def stop() -> None:
            return None

        return frozenset(in_use), stop


async def _boot(
    policy: str,
) -> tuple[RecordingKafkaBus, FakeDemandSource]:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_command", (COMMAND,)),
            _contract("node_projection", (PROJECTION,)),
            _contract("node_unreadable", (UNREADABLE,)),
            _contract("node_priority", (PRIORITY,)),
        )
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    demand = FakeDemandSource()
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
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=RecordingProvisioner(),
            readiness_config=ModelTopicReadinessConfig(
                priority_contracts=("node_priority",)
            ),
            demand_source=demand,
            defer_retained=ModelBootSubscriptionsConfig(policy=policy).defer_retained,
        )
    return bus, demand


def test_policy_literal_admits_gate_required() -> None:
    assert ModelBootSubscriptionsConfig(policy="gate_required").policy == (
        "gate_required"
    )


@pytest.mark.asyncio
async def test_in_use_policy_still_boots_a_retained_projection() -> None:
    bus, _ = await _boot("in_use")
    assert PROJECTION in {topic for topic, _ in bus.joins}


@pytest.mark.asyncio
async def test_gate_required_boots_only_command_priority_and_unwatchable() -> None:
    bus, demand = await _boot("gate_required")
    assert {topic for topic, _ in bus.joins} == {COMMAND, PRIORITY, UNREADABLE}
    assert demand.watched == {PROJECTION}


@pytest.mark.asyncio
async def test_a_deferred_contract_starts_from_earliest_on_its_first_record() -> None:
    bus, demand = await _boot("gate_required")
    assert demand.on_first_record is not None
    bus.joined.clear()
    await demand.on_first_record(PROJECTION)
    await asyncio.wait_for(bus.joined.wait(), timeout=2)
    assert (PROJECTION, "earliest") in bus.joins
