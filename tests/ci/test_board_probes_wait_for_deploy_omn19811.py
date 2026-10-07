# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19811: the board's lane probes wait out a redeploy as the chain canary does.

The M4 C12 probe went RED at 19:33Z on 2026-10-07 (run 37675385944, job "Resolve
the lab lane") because the dev-201 lab lane was redeploying when it asked once;
a re-run passed. OMN-19811 had already taught the chain canary (C15) that a
deploy is in progress, through the resolver action's opt-in
``deploy-wait-seconds``. C11, C12, C16 and C28 resolve the same lane through the
same action and never opted in.

WHAT THIS PINS
  1. Wiring: each of the four probes passes the chain canary's budget (default
     2400 s, overridable by a ``deploy_wait_seconds`` dispatch input) to the
     resolve step, and the job that runs the step has a timeout that covers the
     budget plus the five minutes resolution always had.
  2. Behaviour, per state, over each probe's OWN ``require``/``match`` shape
     read from its workflow (so a probe that changes what it requires is still
     driven as it is wired):
       * no deploy, lane answers          -> resolved at once, no wait;
       * deploy in progress, lane returns -> waited, then resolved;
       * deploy outlasts the budget       -> RED, and the log says it waited;
       * lane silent behind an idle agent -> RED (a lane that is down);
       * lane silent, agent unreadable    -> RED (unreadable is no evidence).
     The last three are the criterion's strength: waiting is granted only on
     the resolver's own busy evidence, so nothing a probe checks is weakened.

WHAT THIS CANNOT PIN. That the live deploy agent reports busy during a real
redeploy; the resolver's own tests and the lab readback cover that.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import (
    DEPLOYING,
    IDLE,
    POLL_SECONDS,
    World,
    resolver,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
RESOLVE_ACTION = "./.github/actions/resolve-lab-lane"
DEFAULT_BUDGET_SECONDS = 2400
RESOLUTION_MINUTES = 5

PROBES = (
    pytest.param("chain-canary-c11-negative-paths.yml", id="C11"),
    pytest.param("chain-canary-c12-provider-catalogue.yml", id="C12"),
    pytest.param("chain-canary-c16-receipt-identity.yml", id="C16"),
    pytest.param("chain-canary-c28-consumer-flow.yml", id="C28"),
)


def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (WORKFLOWS / name).read_text(encoding="utf-8")
    )
    return loaded


def _resolve_job_and_step(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    found = [
        (job, step)
        for job in _workflow(name)["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses") == RESOLVE_ACTION
    ]
    assert len(found) == 1, f"{name}: expected exactly one resolve step"
    return found[0]


@pytest.mark.parametrize("name", PROBES)
def test_the_probe_grants_lane_resolution_the_chain_canary_budget(name: str) -> None:
    _, step = _resolve_job_and_step(name)
    granted = step["with"]["deploy-wait-seconds"]
    assert "inputs.deploy_wait_seconds" in granted
    assert f"'{DEFAULT_BUDGET_SECONDS}'" in granted
    # YAML 1.1 reads `on:` as the boolean True.
    declared = _workflow(name)[True]["workflow_dispatch"]["inputs"][
        "deploy_wait_seconds"
    ]
    assert declared["default"] == str(DEFAULT_BUDGET_SECONDS)
    assert declared["required"] is False


@pytest.mark.parametrize("name", PROBES)
def test_the_job_that_waits_has_a_timeout_that_covers_the_budget(name: str) -> None:
    job, _ = _resolve_job_and_step(name)
    assert job["timeout-minutes"] >= RESOLUTION_MINUTES + DEFAULT_BUDGET_SECONDS // 60


def _lane_for(name: str, step: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """A lane declaring what this probe requires and matches, and the env the
    action would hand the resolver for it."""
    require = step["with"]["require"].split()
    match_pairs = step["with"].get("match", "").split()
    lane: dict[str, Any] = {
        "name": "lane-a",
        "deploy_agent_url": "http://10.0.0.1:8098",
    }
    for field in require:
        if field.endswith("_url"):
            lane[field] = "http://10.0.0.1:8085"
        elif field.endswith("_runs_on"):
            lane[field] = ["runner-a"]
        else:
            lane[field] = "declared"
    for pair in match_pairs:
        field, _, want = pair.partition("=")
        lane[field] = want
    env = {
        "LAB_LANES_JSON": json.dumps([lane]),
        "LANE_REQUIRE": step["with"]["require"],
        "LANE_MATCH": step["with"].get("match", ""),
        "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
        "LANE_DEPLOY_WAIT_SECONDS": str(DEFAULT_BUDGET_SECONDS),
    }
    return lane, env


def _resolve(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
) -> tuple[int, dict[str, str]]:
    _, step = _resolve_job_and_step(name)
    _, environ = _lane_for(name, step)
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    environ["GITHUB_ENV"] = str(env_file)
    code = resolver.main(environ)
    written = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    return code, written


@pytest.mark.parametrize("name", PROBES)
def test_no_deploy_and_a_lane_that_answers_resolves_without_waiting(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=0, busy_until=0)
    code, written = _resolve(name, tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == "lane-a"
    assert "LANE_WAITED_SECONDS" not in written
    assert world.sleeps == []


@pytest.mark.parametrize("name", PROBES)
def test_a_deploy_in_progress_is_waited_out_not_graded(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 19:33Z shape: the lane refuses while its agent reports a deploy. The
    lane comes back 987 s later (the agent's mean service time)."""
    world = World(ready_at=987, busy_until=987, busy_payload=DEPLOYING)
    code, written = _resolve(name, tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == "lane-a"
    assert 987 <= float(written["LANE_WAITED_SECONDS"]) <= 987 + POLL_SECONDS


@pytest.mark.parametrize("name", PROBES)
def test_a_deploy_that_outlasts_the_budget_is_red_and_says_it_waited(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(ready_at=10**6, busy_until=10**6, busy_payload=DEPLOYING)
    code, written = _resolve(name, tmp_path, monkeypatch, world)
    out = capsys.readouterr().out
    assert code == 1
    assert "LANE_NAME" not in written
    assert re.search(r"waited \d+(\.\d+)?s of 2400s budget", out)


@pytest.mark.parametrize("name", PROBES)
def test_a_silent_lane_behind_an_idle_agent_is_still_red(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=10**6, busy_until=0, busy_payload=IDLE)
    code, written = _resolve(name, tmp_path, monkeypatch, world)
    assert code == 1
    assert "LANE_NAME" not in written
    assert world.sleeps == []


@pytest.mark.parametrize("name", PROBES)
def test_a_silent_lane_with_an_unreadable_agent_is_still_red(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=10**6, busy_until=10**6, agent_readable=False)
    code, written = _resolve(name, tmp_path, monkeypatch, world)
    assert code == 1
    assert "LANE_NAME" not in written
    assert world.sleeps == []
