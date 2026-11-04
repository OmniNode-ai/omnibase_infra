# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connect shipped boot policy, the demand split and effects-held observations.

Broker I/O is recorded at subscribe, so these tests do not claim that a live
Kafka group joined. Both carriers use the kernel's configuration resolver and
the production wire -> freeze -> subscribe path exercised by the boot helper.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.runtime.service_kernel import resolve_topic_readiness_config
from scripts import lab_alarm
from scripts.render_runtime_policy_env import load_contract, render_env
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

RECORDED_ROWS = json.loads(
    (ROOT / "tests/fixtures/delegate_skill_effects_held.json").read_text()
)


@pytest.mark.asyncio
@pytest.mark.parametrize("process", ["main", "effects"])
@pytest.mark.parametrize("control", ["ready", "unlisted", "not_ready"])
async def test_shipped_priority_boots_empty_first_hop_before_alarm_readback(
    process: str, control: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = load_contract(ROOT / "contracts/services/runtime_policy.contract.yaml")
    rendered = render_env(policy)
    key = f"DEV_RUNTIME_{process.upper()}_BOOT_PRIORITY_CONTRACTS"
    monkeypatch.setenv(
        "ONEX_BOOT_PRIORITY_CONTRACTS",
        "" if control == "unlisted" else rendered[key],
    )
    knobs = resolve_topic_readiness_config()
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

    # A subscriber recorded above stands in for the broker's Stable/member
    # observation. Missing priority or failed readiness must leave the alarm up.
    for row in RECORDED_ROWS:
        since = re.search(r"\(since ([^)]+)\)", row["evidence"])
        assert since is not None
        state = lab_alarm.ModelAlarmState(
            effects_held_since={"runtime-effects": since[1]}
        )
        report, recovery = lab_alarm.evaluate_effects_held_behind_runtime(
            lambda: (
                (
                    DELEGATE_SKILL_GROUP,
                    "Stable" if attached else "Empty",
                    int(attached),
                ),
            ),
            state=state,
            now=row["raised_at"],
        )
        assert report.outcome is (
            lab_alarm.EnumConditionOutcome.OK
            if attached
            else lab_alarm.EnumConditionOutcome.ALARM
        )
        assert (recovery is not None) == attached


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
