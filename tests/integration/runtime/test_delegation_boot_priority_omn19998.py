# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connect shipped boot policy and the demand split to the boot path.

Broker I/O is recorded at subscribe, so these tests do not claim that a live
Kafka group joined. Both carriers use the kernel's configuration resolver and
the production wire -> freeze -> subscribe path exercised by the boot helper.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.runtime.service_kernel import resolve_topic_readiness_config
from scripts import lab_alarm
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.runtime.auto_wiring.test_omn18087_boot_roster_and_priority import (
    DELEGATE_SKILL,
    DELEGATE_SKILL_GROUP,
    DELEGATE_SKILL_TOPICS,
    IDLE_TOPIC,
    ROOT,
    _boot,
)
from tests.unit.runtime.auto_wiring.test_per_contract_boot_interleave import (
    RecordingProvisioner,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("process", ["main", "effects"])
@pytest.mark.parametrize("control", ["ready", "unlisted", "not_ready"])
async def test_shipped_priority_boots_empty_first_hop_(
    process: Literal["main", "effects"], control: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = load_contract(ROOT / "contracts/services/runtime_policy.contract.yaml")
    rendered = render_env(policy)
    service, compose = (
        ("omninode-runtime", "docker-compose.infra.yml")
        if process == "main"
        else ("runtime-effects", "docker-compose.dev-lane.yml")
    )
    carrier = load_compose(ROOT / "docker" / compose)["services"][service]
    for variable, suffix in (
        ("ONEX_BOOT_READY_BOUND_SECONDS", "READY_BOUND_SECONDS"),
        ("ONEX_MAX_CONCURRENT_CONTRACT_ATTACH", "MAX_CONCURRENT_CONTRACT_ATTACH"),
    ):
        policy_key = f"DEV_RUNTIME_{process.upper()}_{suffix}"
        assert carrier["environment"][variable] == (
            f"${{{policy_key}:?runtime policy contract must set {policy_key}}}"
        )
        monkeypatch.setenv(variable, rendered[policy_key])
    memory_key = f"DEV_RUNTIME_{process.upper()}_MEMORY"
    assert carrier["deploy"]["resources"]["limits"]["memory"] == (
        f"${{{memory_key}:?runtime policy contract must set {memory_key}}}"
    )
    key = f"DEV_RUNTIME_{process.upper()}_BOOT_PRIORITY_CONTRACTS"
    monkeypatch.setenv(
        "ONEX_BOOT_PRIORITY_CONTRACTS",
        "" if control == "unlisted" else rendered[key],
    )
    knobs = resolve_topic_readiness_config()
    boot = policy.profiles["dev"].processes[process].boot
    assert boot is not None
    assert knobs.max_concurrent_contract_attach == boot.max_concurrent_contract_attach
    assert (DELEGATE_SKILL in knobs.priority_contracts) == (control != "unlisted")
    provisioner = RecordingProvisioner(
        not_ready_topics=(
            frozenset({DELEGATE_SKILL_TOPICS[0]})
            if control == "not_ready"
            else frozenset()
        )
    )
    bus, _manifest, exclusions, results, _roster = await _boot(
        ((DELEGATE_SKILL, DELEGATE_SKILL_TOPICS), ("node_idle", (IDLE_TOPIC,))),
        in_use=frozenset(),
        readiness_config=knobs,
        provisioner=provisioner,
    )
    attached = any(
        result.status is EnumContractAttachStatus.ATTACHED for result in results
    )
    assert attached == (control == "ready")
    assert bool(bus.joins) == attached
    assert {item.contract_name for item in exclusions} == (
        {"node_idle", DELEGATE_SKILL} if control == "unlisted" else {"node_idle"}
    )
    assert all(topic != IDLE_TOPIC for topic, _offset in bus.joins)
    if control == "not_ready":
        assert [result.status for result in results] == [
            EnumContractAttachStatus.NOT_READY
        ]


@pytest.mark.parametrize(
    ("group_state", "members"), [("Empty", 0), ("Stable", 0), ("PreparingRebalance", 1)]
)
def test_alarm_threshold_uses_an_independent_clock(
    group_state: str, members: int
) -> None:
    since = datetime.fromisoformat("2026-10-01T00:00:00+00:00")
    state = lab_alarm.ModelAlarmState(
        effects_held_since={"runtime-effects": since.isoformat()}
    )

    def reader() -> tuple[tuple[str, str, int], ...]:
        return ((DELEGATE_SKILL_GROUP, group_state, members),)

    before, recovery = lab_alarm.evaluate_effects_held_behind_runtime(
        reader, state=state, now=(since + timedelta(seconds=59)).isoformat()
    )
    assert before.outcome is lab_alarm.EnumConditionOutcome.OK
    assert recovery is None
    at_bound, recovery = lab_alarm.evaluate_effects_held_behind_runtime(
        reader, state=state, now=(since + timedelta(seconds=60)).isoformat()
    )
    assert at_bound.outcome is lab_alarm.EnumConditionOutcome.ALARM
    assert recovery is None
    assert "instances for 0:01:00 " in at_bound.evidence
