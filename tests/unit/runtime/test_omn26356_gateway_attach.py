# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26356: the GitHub repo gateway consumer attaches in the boot priority wave.

Lane gwroot-0335 recorded the gateway consumer group joining 234 s and 247 s
after two dev runtime replacements on 2026-11-19, so `onex scm` reads and pull
request creation failed in that window. The fixture holds those recorded
timings. A node whose recorded attach is past the budget must be declared in the
dev main boot priority set, and with the real policy it must subscribe inside
the priority waves, ahead of the backlog.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.unit.runtime.auto_wiring.test_omn18087_boot_roster_and_priority import (
    GATEWAY_TOPIC,
    _boot,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "omn26356" / "gwroot_0335_gateway_attach.json"
POLICY = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"


def _recorded() -> dict[str, object]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _seconds(start: str, end: str) -> int:
    def parse(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)

    return int((parse(end) - parse(start)).total_seconds())


def test_gateway_attach_fixture_timings_are_self_consistent_and_past_the_budget() -> (
    None
):
    recorded = _recorded()
    budget = recorded["attach_budget_seconds"]
    rolls = recorded["rolls"]
    assert isinstance(budget, int)
    assert isinstance(rolls, list)
    assert len(rolls) == 2
    for roll in rolls:
        assert roll["attach_seconds"] == _seconds(
            roll["container_start"], roll["group_joined"]
        )
        assert roll["attach_seconds"] > budget


def test_gateway_attach_is_declared_priority_when_the_recorded_attach_missed_the_budget() -> (
    None
):
    recorded = _recorded()
    node = str(recorded["node"])
    policy = load_contract(POLICY)
    boot = policy.profiles["dev"].processes["main"].boot
    assert boot is not None
    rolls = recorded["rolls"]
    assert isinstance(rolls, list)
    assert all(roll["attach_seconds"] > 60 for roll in rolls)

    assert node in boot.priority_contracts
    assert node in render_env(policy)["DEV_RUNTIME_MAIN_BOOT_PRIORITY_CONTRACTS"].split(
        ","
    )


@pytest.mark.asyncio
async def test_gateway_attach_joins_inside_the_priority_waves_behind_a_backlog() -> (
    None
):
    node = str(_recorded()["node"])
    boot = load_contract(POLICY).profiles["dev"].processes["main"].boot
    assert boot is not None
    backlog = tuple(
        (f"node_backlog_{i:02d}", (f"onex.cmd.test.backlog-{i}.v1",)) for i in range(40)
    )
    roster = (*backlog, (node, (GATEWAY_TOPIC,)))

    bus, _manifest, _exclusions, _results, _names = await _boot(
        roster,
        in_use=None,
        readiness_config=ModelTopicReadinessConfig(
            max_concurrent_contract_attach=boot.max_concurrent_contract_attach,
            priority_contracts=boot.priority_contracts,
        ),
    )

    joined = [topic for topic, _ in bus.joins]
    # Only the gateway is a priority contract in this roster, so it is the first join.
    assert joined[0] == GATEWAY_TOPIC
