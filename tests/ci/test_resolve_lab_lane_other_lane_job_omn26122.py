# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26122: a probe does not wait for another lane's deploy job.

One deploy agent rolls every lane single-flight. On 2026-11-21 its job store held
fourteen jobs from the 20th 15:21Z to the 21st 17:42Z, all fourteen of the dev
lane and all ending ``failed``; ten were ``agent/idle-restore`` (one every two
hours, each between 43 and 79 minutes) and the agent's ``/queue`` reported a mean
service time of 4325 s. The resolver of the C15 and C16 proofs read the agent as
busy for the whole of each, so C16 run 44919897557 (2026-11-21 15:34Z) spent its
2400 s budget behind the 15:00Z job and ended RED without submitting anything,
while the stability lane itself answered its health route 200 and its workflow
route 401 without a key.

OMN-21485 (operator ruling 2026-10-27) moved those proofs onto the stability
lane so that a dev redeploy no longer turns them red. These tests pin the rest
of that ruling: the busy test knows which lane it is grading (the lane's own
``bus_lane``, which the agent's ``runtime_lane`` is the name of). A job the agent
names as rolling another lane is not this lane's deploy; whatever it does not
name stays a reason to wait.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import (
    LANE_A,
    POLL_SECONDS,
    World,
    resolver,
)

pytestmark = pytest.mark.unit

OWN = "lane-under-test"
OTHER = "some-other-lane"
GRADED = {**LANE_A, "bus_lane": OWN}


def _payload(
    *, job_lane: str | None, state: str = "deploying", settling: bool = False
) -> dict[str, Any]:
    """The agent's /health as read on 2026-11-21 at 17:43Z, with the job's lane."""
    named = {} if job_lane is None else {"runtime_lane": job_lane}
    active = (
        None
        if settling
        else {
            "correlation_id": "c80367de-9020-4d9a-a14b-a002daad3012",
            "current_phase": "runtime",
            "started_at": "2026-11-21T17:00:03.082105+00:00",
            **named,
        }
    )
    last = {
        "correlation_id": "c8fd5fda-5f5b-4af1-8a84-77e1fc8c2a64",
        "status": "failed",
        "settling": settling,
        "settling_stage": "boot_recovery" if settling else None,
        **(named if settling else {}),
    }
    return {
        "health": {"state": state, "active_job": active, "last_result": last},
        "queue": {"commands_ahead": None, "control_topic_lag_age_seconds": None},
    }


@pytest.mark.parametrize(
    ("snapshot", "lane", "busy"),
    [
        pytest.param(_payload(job_lane=OTHER), OWN, False, id="other-lane-active"),
        pytest.param(_payload(job_lane=OWN), OWN, True, id="own-lane-active"),
        pytest.param(
            _payload(job_lane=OTHER, state="settling", settling=True),
            OWN,
            False,
            id="other-lane-settling",
        ),
        pytest.param(
            _payload(job_lane=OWN, state="settling", settling=True),
            OWN,
            True,
            id="own-lane-settling",
        ),
        # What the agent does not say, the resolver does not guess.
        pytest.param(_payload(job_lane=None), OWN, True, id="older-agent-no-lane"),
        pytest.param(_payload(job_lane=""), OWN, True, id="empty-lane-field"),
        # A lane that declares no bus_lane is graded as before: any job is busy.
        pytest.param(_payload(job_lane=OTHER), "", True, id="lane-declares-none"),
        pytest.param(_payload(job_lane=OTHER), None, True, id="no-lane-given"),
    ],
)
def test_busy_is_decided_by_the_graded_lanes_own_job(
    snapshot: dict[str, Any], lane: str | None, busy: bool
) -> None:
    reason = (
        resolver.deploy_agent_busy_reason(snapshot)
        if lane is None
        else resolver.deploy_agent_busy_reason(snapshot, lane)
    )
    assert (reason is not None) is busy


def test_a_queued_command_still_reads_busy_whatever_lane_runs() -> None:
    # The queue names no lane, so a command ahead of ours stays a reason to wait.
    snapshot = {
        "health": {
            "state": "idle",
            "active_job": None,
            "last_result": {"settling": False},
        },
        "queue": {"commands_ahead": 2, "control_topic_lag_age_seconds": 1.0},
    }
    assert resolver.deploy_agent_busy_reason(snapshot, OWN) == "2 command(s) queued"


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    *,
    budget: float = 2400,
) -> tuple[int, dict[str, str]]:
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    code = resolver.main(
        {
            "LAB_LANES_JSON": json.dumps([GRADED]),
            "LANE_REQUIRE": "ingress_url bus_lane",
            "LANE_DEPLOY_WAIT_SECONDS": str(budget),
            "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
            "LANE_DEPLOY_WAIT_WHILE_BUSY": "true",
            "GITHUB_ENV": str(env_file),
        }
    )
    written = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    return code, written


def test_run_44919897557_no_longer_waits_out_another_lanes_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The agent reads deploying for longer than the whole budget, as it did from
    # 15:35Z to 16:15Z, but the job it is on rolls another lane.
    world = World(ready_at=0, busy_until=10**9, busy_payload=_payload(job_lane=OTHER))
    code, written = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == GRADED["name"]
    assert written["LANE_INGRESS_URL"] == GRADED["ingress_url"]
    assert world.sleeps == []
    assert "deploy wait started" not in capsys.readouterr().out


def test_the_same_run_still_goes_red_behind_its_own_lanes_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Positive control: identical world, but the job rolls the graded lane. The
    # wait is spent in full and the run is RED, as run 44919897557 was.
    world = World(ready_at=0, busy_until=10**9, busy_payload=_payload(job_lane=OWN))
    code, _ = _run(tmp_path, monkeypatch, world)
    assert code == 1
    assert world.sleeps
    assert "no probe may submit into a deploy" in capsys.readouterr().out


def test_a_job_that_names_no_lane_is_still_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=0, busy_until=300, busy_payload=_payload(job_lane=None))
    code, written = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert 300 <= int(written["LANE_WAITED_SECONDS"]) <= 300 + POLL_SECONDS
