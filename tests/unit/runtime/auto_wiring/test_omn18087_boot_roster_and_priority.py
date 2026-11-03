# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the boot roster partitions exactly, and priority contracts join first.

Two properties of the in-use boot split, each with a falsifier:

* Every contract in the auto-wiring report is attempted at boot or reported
  excluded (``DEMAND_DEFERRED`` among them), exactly once. A test that drops a
  contract from the deferred report, or reports one in both sets, fails the
  check and names the contract.
* The contracts named in the runtime policy contract's priority list subscribe
  before every other contract, whatever order discovery returned them in.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest

from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers.boot_readiness_checks import verify_roster_partition
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.unit.runtime.auto_wiring.test_omn18087_boot_subscribes_only_topics_in_use import (
    FakeDemandSource,
    RecordingKafkaBus,
)
from tests.unit.runtime.auto_wiring.test_per_contract_boot_interleave import (
    RecordingProvisioner,
    _contract,
    _fake_handler_cls,
)

pytestmark = pytest.mark.unit

HOOK_TOPIC = "onex.evt.test.hook-event.v1"
DELEGATION_TOPIC = "onex.cmd.test.delegation-request.v1"
IDLE_TOPIC = "onex.cmd.test.never-requested.v1"
OTHER_TOPICS = tuple(f"onex.cmd.test.other-{index}.v1" for index in range(6))


async def _boot(
    contracts: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    in_use: frozenset[str] | None,
    readiness_config: ModelTopicReadinessConfig,
) -> tuple[
    RecordingKafkaBus,
    ModelAutoWiringManifest,
    list[ModelContractAttachExclusion],
    list[ModelContractAttachResult],
    list[str],
]:
    manifest = ModelAutoWiringManifest(
        contracts=tuple(_contract(name, topics) for name, topics in contracts)
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
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=RecordingProvisioner(),
            readiness_config=readiness_config,
            attach_results_out=results,
            exclusions_out=exclusions,
            demand_source=None if in_use is None else FakeDemandSource(in_use),
            demand_watch_stops_out=stops,
        )
    roster = [result.contract_name for result in report.results]
    return bus, manifest, exclusions, results, roster


@pytest.mark.asyncio
async def test_started_and_deferred_sets_partition_the_roster() -> None:
    _bus, _manifest, exclusions, results, roster = await _boot(
        (
            ("node_in_use", (OTHER_TOPICS[0],)),
            ("node_in_use_too", (OTHER_TOPICS[1],)),
            ("node_idle", (IDLE_TOPIC,)),
            ("node_idle_too", (OTHER_TOPICS[2],)),
        ),
        in_use=frozenset({OTHER_TOPICS[0], OTHER_TOPICS[1]}),
        readiness_config=ModelTopicReadinessConfig(),
    )

    partition = verify_roster_partition(roster, results, exclusions)

    assert partition.is_partition, partition.describe_violations()
    assert partition.attempted == ("node_in_use", "node_in_use_too")
    assert partition.excluded == ("node_idle", "node_idle_too")
    deferred = {
        e.contract_name
        for e in exclusions
        if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
    }
    assert deferred == {"node_idle", "node_idle_too"}


@pytest.mark.asyncio
async def test_dropping_a_deferred_contract_from_the_report_fails_and_names_it() -> (
    None
):
    _bus, _manifest, exclusions, results, roster = await _boot(
        (("node_in_use", (OTHER_TOPICS[0],)), ("node_idle", (IDLE_TOPIC,))),
        in_use=frozenset({OTHER_TOPICS[0]}),
        readiness_config=ModelTopicReadinessConfig(),
    )
    without_idle = [e for e in exclusions if e.contract_name != "node_idle"]

    partition = verify_roster_partition(roster, results, without_idle)

    assert not partition.is_partition
    assert partition.unaccounted == ("node_idle",)
    assert "node_idle" in partition.describe_violations()


@pytest.mark.asyncio
async def test_a_contract_started_and_deferred_fails_and_names_it() -> None:
    _bus, _manifest, exclusions, results, roster = await _boot(
        (("node_in_use", (OTHER_TOPICS[0],)), ("node_idle", (IDLE_TOPIC,))),
        in_use=frozenset({OTHER_TOPICS[0]}),
        readiness_config=ModelTopicReadinessConfig(),
    )
    deferred_too = [
        *exclusions,
        ModelContractAttachExclusion(
            contract_name="node_in_use",
            reason=EnumContractAttachExclusionReason.DEMAND_DEFERRED,
        ),
    ]

    partition = verify_roster_partition(roster, results, deferred_too)

    assert not partition.is_partition
    assert partition.in_both == ("node_in_use",)


def test_a_reported_contract_outside_the_roster_is_named() -> None:
    partition = verify_roster_partition(
        ["node_a"],
        [],
        [
            ModelContractAttachExclusion(
                contract_name="node_a",
                reason=EnumContractAttachExclusionReason.DEMAND_DEFERRED,
            ),
            ModelContractAttachExclusion(
                contract_name="node_ghost",
                reason=EnumContractAttachExclusionReason.DEMAND_DEFERRED,
            ),
        ],
    )

    assert partition.foreign == ("node_ghost",)


@pytest.mark.asyncio
async def test_priority_contracts_subscribe_before_every_other_contract() -> None:
    """Discovery lists the priority contracts last; they still join first."""
    contracts = (
        *(
            (f"node_other_{index}", (topic,))
            for index, topic in enumerate(OTHER_TOPICS)
        ),
        ("projection_claude_hook_events", (HOOK_TOPIC,)),
        ("node_delegation_orchestrator", (DELEGATION_TOPIC,)),
    )
    bus, _manifest, _exclusions, _results, _roster = await _boot(
        contracts,
        in_use=None,
        readiness_config=ModelTopicReadinessConfig(
            max_concurrent_contract_attach=1,
            priority_contracts=(
                "node_delegation_orchestrator",
                "projection_claude_hook_events",
            ),
        ),
    )

    joined = [topic for topic, _ in bus.joins]
    assert joined[:2] == [DELEGATION_TOPIC, HOOK_TOPIC]
    assert set(joined[2:]) == set(OTHER_TOPICS)


@pytest.mark.asyncio
async def test_a_contract_off_the_priority_list_never_precedes_a_listed_one() -> None:
    contracts = (
        ("node_planted_unlisted", (IDLE_TOPIC,)),
        ("projection_claude_hook_events", (HOOK_TOPIC,)),
    )
    bus, _manifest, _exclusions, _results, _roster = await _boot(
        contracts,
        in_use=None,
        readiness_config=ModelTopicReadinessConfig(
            max_concurrent_contract_attach=1,
            priority_contracts=("projection_claude_hook_events",),
        ),
    )

    joined = [topic for topic, _ in bus.joins]
    assert joined.index(HOOK_TOPIC) < joined.index(IDLE_TOPIC)


@pytest.mark.asyncio
async def test_without_a_priority_list_discovery_order_is_kept() -> None:
    contracts = (
        ("node_planted_unlisted", (IDLE_TOPIC,)),
        ("projection_claude_hook_events", (HOOK_TOPIC,)),
    )
    bus, _manifest, _exclusions, _results, _roster = await _boot(
        contracts,
        in_use=None,
        readiness_config=ModelTopicReadinessConfig(max_concurrent_contract_attach=1),
    )

    assert [topic for topic, _ in bus.joins] == [IDLE_TOPIC, HOOK_TOPIC]
