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

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest

from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import delegate_locus
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
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
from omnibase_infra.handlers.boot_readiness_checks import (
    parse_ready_bound_seconds,
    verify_roster_partition,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_kernel import resolve_topic_readiness_config
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
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
ROOT = Path(__file__).resolve().parents[4]
DELEGATE_SKILL = "node_delegate_skill_orchestrator"
# The first-hop contract's subscriptions, kept local so this regression runs
# without installing the domain package or requiring a sibling checkout.
DELEGATE_SKILL_TOPICS = (
    "onex.cmd.omnimarket.delegate-skill.v1",
    "onex.intent.platform.runtime-tick.v1",
    "onex.evt.omnibase-infra.delegation-completed.v1",
    "onex.evt.omnibase-infra.delegation-failed.v1",
)
DELEGATE_SKILL_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0."
    "__i.runtime-effects.__t.onex.cmd.omnimarket.delegate-skill.v1"
)


async def _boot(
    contracts: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    in_use: frozenset[str] | None,
    readiness_config: ModelTopicReadinessConfig,
    demand: FakeDemandSource | None = None,
    provisioner: RecordingProvisioner | None = None,
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
        engine.freeze()
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=provisioner or RecordingProvisioner(),
            readiness_config=readiness_config,
            attach_results_out=results,
            exclusions_out=exclusions,
            demand_source=demand
            or (None if in_use is None else FakeDemandSource(in_use)),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("prioritized", [False, True])
async def test_empty_delegation_topics_and_real_client_probe(
    prioritized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First publication needs a boot consumer; unlisted idle work stays deferred."""
    demand = FakeDemandSource(frozenset())
    provisioner = RecordingProvisioner()
    bus, _manifest, exclusions, results, roster = await _boot(
        (
            ("node_unrelated_idle", (IDLE_TOPIC,)),
            (DELEGATE_SKILL, DELEGATE_SKILL_TOPICS),
        ),
        in_use=frozenset(),
        readiness_config=ModelTopicReadinessConfig(
            priority_contracts=(DELEGATE_SKILL,) if prioritized else (),
        ),
        demand=demand,
        provisioner=provisioner,
    )
    assert verify_roster_partition(roster, results, exclusions).is_partition
    expected_topics = list(DELEGATE_SKILL_TOPICS) if prioritized else []
    assert [topic for topic, _ in bus.joins] == expected_topics
    assert [r.contract_name for r in results] == (
        [DELEGATE_SKILL] if prioritized else []
    )
    assert {e.contract_name for e in exclusions} == (
        {"node_unrelated_idle"}
        if prioritized
        else {"node_unrelated_idle", DELEGATE_SKILL}
    )
    assert all(
        e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
        for e in exclusions
    )
    assert not any(topic == IDLE_TOPIC for _action, topic in provisioner.calls)
    if prioritized:
        for topic in DELEGATE_SKILL_TOPICS:
            assert ("provision", topic) in provisioner.calls
            assert provisioner.calls.index(
                ("provision", topic)
            ) < provisioner.calls.index(("ready", topic))

    clock = [0.0]
    monkeypatch.setattr(delegate_locus, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(
        delegate_locus,
        "_sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        delegate_locus,
        "live_consumer_groups",
        lambda **_kwargs: (DELEGATE_SKILL_GROUP,) if bus.joins else (),
    )
    owner = ModelConsumerGroupOwner(service="omnimarket", node=DELEGATE_SKILL)
    if prioritized:
        groups, _downstream, _waited = delegate_locus._assert_dispatch_viable(
            command_topic=DELEGATE_SKILL_TOPICS[0],
            kafka_bootstrap="localhost:9092",
            owner=owner,
            rebind_wait_seconds=1,
        )
        assert groups == (DELEGATE_SKILL_GROUP,)
        # A later first record must not attach the already booted contract again.
        assert demand.on_first_record is not None
        assert demand.watched == {IDLE_TOPIC}
        await demand.on_first_record(DELEGATE_SKILL_TOPICS[0])
        await asyncio.sleep(0)
        assert [topic for topic, _ in bus.joins] == expected_topics
    else:
        with pytest.raises(delegate_locus.DelegateNoConsumerRefusedError):
            delegate_locus._assert_dispatch_viable(
                command_topic=DELEGATE_SKILL_TOPICS[0],
                kafka_bootstrap="localhost:9092",
                owner=owner,
                rebind_wait_seconds=1,
            )


@pytest.mark.asyncio
async def test_empty_priority_topics_keep_policy_order_ahead_of_active_work() -> None:
    bus, _manifest, exclusions, results, roster = await _boot(
        (
            ("node_active", (OTHER_TOPICS[0],)),
            ("projection_claude_hook_events", (HOOK_TOPIC,)),
            (DELEGATE_SKILL, DELEGATE_SKILL_TOPICS),
        ),
        in_use=frozenset({OTHER_TOPICS[0]}),
        readiness_config=ModelTopicReadinessConfig(
            max_concurrent_contract_attach=1,
            priority_contracts=(DELEGATE_SKILL, "projection_claude_hook_events"),
        ),
    )
    assert [topic for topic, _ in bus.joins] == [
        *DELEGATE_SKILL_TOPICS,
        HOOK_TOPIC,
        OTHER_TOPICS[0],
    ]
    assert verify_roster_partition(roster, results, exclusions).is_partition
    assert exclusions == []


@pytest.mark.asyncio
async def test_priority_does_not_bypass_topic_readiness() -> None:
    bus, _manifest, exclusions, results, roster = await _boot(
        ((DELEGATE_SKILL, DELEGATE_SKILL_TOPICS),),
        in_use=frozenset(),
        readiness_config=ModelTopicReadinessConfig(
            priority_contracts=(DELEGATE_SKILL,)
        ),
        provisioner=RecordingProvisioner(
            not_ready_topics=frozenset({DELEGATE_SKILL_TOPICS[0]})
        ),
    )
    assert bus.joins == []
    assert [(r.contract_name, r.status) for r in results] == [
        (DELEGATE_SKILL, EnumContractAttachStatus.NOT_READY)
    ]
    assert exclusions == []
    assert verify_roster_partition(roster, results, exclusions).is_partition


@pytest.mark.parametrize(
    ("process", "service", "compose"),
    [
        ("main", "omninode-runtime", "docker-compose.infra.yml"),
        ("effects", "runtime-effects", "docker-compose.dev-lane.yml"),
    ],
)
def test_dev_delegation_priority_reaches_both_carriers(
    process: Literal["main", "effects"],
    service: str,
    compose: str,
) -> None:
    policy = load_contract(ROOT / "contracts/services/runtime_policy.contract.yaml")
    boot = policy.profiles["dev"].processes[process].boot
    assert boot is not None
    assert DELEGATE_SKILL in boot.priority_contracts
    key = f"DEV_RUNTIME_{process.upper()}_BOOT_PRIORITY_CONTRACTS"
    assert DELEGATE_SKILL in render_env(policy)[key].split(",")
    environment = load_compose(ROOT / "docker" / compose)["services"][service][
        "environment"
    ]
    assert (
        environment["ONEX_BOOT_PRIORITY_CONTRACTS"]
        == f"${{{key}:?runtime policy contract must set {key}}}"
    )


def test_dev_effects_boot_policy_reaches_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_contract(ROOT / "contracts/services/runtime_policy.contract.yaml")
    effects = policy.profiles["dev"].processes["effects"]
    assert effects.boot is not None
    # Distinct values catch a carrier silently using the kernel or base defaults.
    policy.profiles["dev"].processes["effects"] = effects.model_copy(
        update={
            "boot": effects.boot.model_copy(
                update={
                    "ready_bound_seconds": 321,
                    "max_concurrent_contract_attach": 7,
                    "memory_limit": "2048M",
                }
            )
        }
    )
    rendered = render_env(policy)
    carrier = load_compose(ROOT / "docker/docker-compose.dev-lane.yml")["services"][
        "runtime-effects"
    ]
    boot_env: dict[str, str] = {}
    for variable, suffix in (
        ("ONEX_BOOT_READY_BOUND_SECONDS", "READY_BOUND_SECONDS"),
        ("ONEX_MAX_CONCURRENT_CONTRACT_ATTACH", "MAX_CONCURRENT_CONTRACT_ATTACH"),
        ("ONEX_BOOT_PRIORITY_CONTRACTS", "BOOT_PRIORITY_CONTRACTS"),
    ):
        key = f"DEV_RUNTIME_EFFECTS_{suffix}"
        assert carrier["environment"][variable] == (
            f"${{{key}:?runtime policy contract must set {key}}}"
        )
        boot_env[variable] = rendered[key]
        monkeypatch.setenv(variable, rendered[key])
    assert parse_ready_bound_seconds(boot_env) == 321
    assert resolve_topic_readiness_config().max_concurrent_contract_attach == 7
    key = "DEV_RUNTIME_EFFECTS_MEMORY"
    assert carrier["deploy"]["resources"]["limits"]["memory"] == (
        f"${{{key}:?runtime policy contract must set {key}}}"
    )
    assert rendered[key] == "2048M"
