# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20410: C16 waits for a busy deploy even while its gateway answers."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import (
    IDLE,
    LANE_A,
    POLL_SECONDS,
    World,
    resolver,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-lab-lane"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
C16 = "chain-canary-c16-receipt-identity.yml"


@pytest.fixture
def recorded_world() -> World:
    fixture = json.loads(
        (REPO_ROOT / "tests/fixtures/omn20410/c16_inflight_dev_deploy.json").read_text(
            encoding="utf-8"
        )
    )
    observation = fixture["timeline"][0]
    remaining = (
        datetime.fromisoformat(fixture["completed_at"])
        - datetime.fromisoformat(observation["observed_at"])
    ).total_seconds()
    assert remaining == 1895
    return World(
        ready_at=0,
        busy_until=remaining,
        busy_payload={
            "health": observation["health"],
            "queue": {"commands_ahead": 0, "control_topic_lag_age_seconds": 1.0},
        },
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    *,
    flag: str | None = "true",
    budget: float = 2400,
    lane: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str]]:
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    environ = {
        "LAB_LANES_JSON": json.dumps([LANE_A if lane is None else lane]),
        "LANE_REQUIRE": "gateway_url",
        "LANE_DEPLOY_WAIT_SECONDS": str(budget),
        "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
        "GITHUB_ENV": str(env_file),
    }
    if flag is not None:
        environ["LANE_DEPLOY_WAIT_WHILE_BUSY"] = flag
    code = resolver.main(environ)
    written = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    return code, written


@pytest.mark.parametrize("flag", ["true", "TRUE", "1", "yes", "YeS"])
def test_recorded_deploy_is_waited_out_while_gateway_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recorded_world: World,
    flag: str,
) -> None:
    code, written = _run(tmp_path, monkeypatch, recorded_world, flag=flag)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert 1895 <= int(written["LANE_WAITED_SECONDS"]) <= 1895 + POLL_SECONDS
    assert recorded_world.sleeps
    out = capsys.readouterr().out
    assert "lab lane deploy wait started -- " in out
    assert "lab lane deploy wait ended -- " in out


@pytest.mark.parametrize("flag", [None, "false", "0", "no", "", "anything"])
def test_flag_off_preserves_the_recorded_no_wait_defect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_world: World,
    flag: str | None,
) -> None:
    code, written = _run(tmp_path, monkeypatch, recorded_world, flag=flag)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert "LANE_WAITED_SECONDS" not in written
    assert recorded_world.sleeps == []


def test_zero_budget_keeps_single_shot_even_with_flag_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_world: World
) -> None:
    code, written = _run(tmp_path, monkeypatch, recorded_world, budget=0)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert "LANE_WAITED_SECONDS" not in written
    assert recorded_world.sleeps == []
    assert recorded_world.agent_reads == []


def test_stuck_deploy_is_red_at_the_budget_even_when_gateway_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(ready_at=0, busy_until=10**6)
    code, written = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1
    assert "LANE_NAME" not in written
    out = capsys.readouterr().out
    for detail in ("still reads busy", "120s", "state=deploying", "RED, not a pass"):
        assert detail in out
    assert world.sleeps
    assert world.now <= 120


def test_agent_going_unreadable_mid_deploy_is_not_a_finished_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The agent re-execs during a deploy; one unreadable read after a busy one
    must not release the probe into it."""
    world = World(ready_at=0, busy_until=10**6, unreadable_from=50)
    code, written = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1
    assert "LANE_NAME" not in written
    assert world.now <= 120
    out = capsys.readouterr().out
    assert "agent unreadable on the last read" in out
    assert "RED, not a pass" in out


def test_idle_agent_needs_no_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=0, busy_until=0, busy_payload=IDLE)
    code, written = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert "LANE_WAITED_SECONDS" not in written
    assert world.sleeps == []


def test_unreadable_agent_warns_and_keeps_the_answering_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(ready_at=0, busy_until=10**6, agent_readable=False)
    code, written = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert world.sleeps == []
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "unreadable" in out
    assert "deploy state unknown" in out


def test_undeclared_agent_warns_without_reading_an_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(ready_at=0, busy_until=10**6)
    lane = {key: value for key, value in LANE_A.items() if key != "deploy_agent_url"}
    code, written = _run(tmp_path, monkeypatch, world, lane=lane)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert world.sleeps == []
    assert world.agent_reads == []
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "no deploy_agent_url" in out
    assert "deploy state unknown" in out


class PostDeployRecoveryWorld(World):
    """Gateway answers during the deploy, then needs twenty seconds to recover."""

    def health(self, url: str, timeout: float) -> str | None:
        if self.busy_until <= self.now < self.busy_until + 20:
            return "no answer (Connection refused)"
        return super().health(url, timeout)


def test_lane_must_answer_again_after_deploy_with_one_shared_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = PostDeployRecoveryWorld(ready_at=0, busy_until=30)
    code, written = _run(tmp_path, monkeypatch, world, budget=60)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert int(written["LANE_WAITED_SECONDS"]) >= world.busy_until + 20
    assert world.now <= 60


def test_recovery_cannot_spend_a_second_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = PostDeployRecoveryWorld(ready_at=0, busy_until=30)
    code, written = _run(tmp_path, monkeypatch, world, budget=42)
    assert code == 1
    assert "LANE_NAME" not in written
    assert world.now == 42
    assert world.sleeps[-1] == 2


def _resolve_step(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    found = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses") == "./.github/actions/resolve-lab-lane"
    ]
    assert len(found) == 1
    step: dict[str, Any] = found[0]
    return step


def test_c16_opts_in_with_its_existing_budget_and_action_env() -> None:
    with_block = _resolve_step(C16)["with"]
    assert with_block["wait-while-busy"] == "true"
    assert (
        with_block["deploy-wait-seconds"]
        == "${{ github.event.inputs.deploy_wait_seconds || '2400' }}"
    )
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["wait-while-busy"]["default"] == "false"
    assert action["inputs"]["wait-while-busy"]["required"] is False
    assert (
        action["runs"]["steps"][0]["env"]["LANE_DEPLOY_WAIT_WHILE_BUSY"]
        == "${{ inputs.wait-while-busy }}"
    )


@pytest.mark.parametrize(
    "name",
    [
        "chain-canary-c11-negative-paths.yml",
        "chain-canary-c12-provider-catalogue.yml",
        "chain-canary-c28-consumer-flow.yml",
    ],
)
def test_other_board_probes_keep_their_existing_mode(name: str) -> None:
    assert "wait-while-busy" not in _resolve_step(name)["with"]
