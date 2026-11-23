# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26511: every resolve pass prints the wait set it considered.

omnibase_infra#6144 made the stability lane wait only for its own lane's deploy
jobs (OMN-26122). The resolver printed a line only when it waited, so with the
deploy agent idle no log line told the new rule from the old one: a scheduled C16
run read the same either way. These tests pin one line per pass, printed even when
nothing waits, that names the lane being resolved and every deploy job the agent
showed, each marked ``counted`` toward the wait or ``ignored`` as another lane's,
with the lane id and the job id and nothing secret.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import (
    IDLE,
    LANE_A,
    POLL_SECONDS,
    World,
    resolver,
)
from tests.ci.test_resolve_lab_lane_other_lane_job_omn26122 import (
    OTHER,
    OWN,
    _payload,
)

pytestmark = pytest.mark.unit

GRADED = {**LANE_A, "bus_lane": OWN}
OTHER_JOB_ID = "c80367de-9020-4d9a-a14b-a002daad3012"
LINE_MARK = "lab lane wait set"


def _wait_set_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(LINE_MARK)]


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    *,
    budget: float = 2400,
) -> int:
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    return resolver.main(
        {
            "LAB_LANES_JSON": json.dumps([GRADED]),
            "LANE_REQUIRE": "ingress_url bus_lane",
            "LANE_DEPLOY_WAIT_SECONDS": str(budget),
            "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
            "LANE_DEPLOY_WAIT_WHILE_BUSY": "true",
            "GITHUB_ENV": str(env_file),
        }
    )


def test_an_idle_pass_prints_one_wait_set_line_naming_the_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The scheduled C16 case: the agent is idle, nothing waits, and the log must
    # still say which lane was resolved and that no deploy job was seen.
    world = World(ready_at=0, busy_until=0)
    assert _run(tmp_path, monkeypatch, world) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert GRADED["name"] in line
    assert OWN in line
    assert "no deploy job seen" in line
    assert "counted" not in line
    assert "ignored" not in line
    assert world.sleeps == []


def test_another_lanes_job_is_named_as_ignored_with_lane_and_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=10**9, busy_payload=_payload(job_lane=OTHER))
    assert _run(tmp_path, monkeypatch, world) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert f"job {OTHER_JOB_ID} lane {OTHER} ignored" in line
    assert "counted" not in line


def test_the_graded_lanes_own_job_is_named_as_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=300, busy_payload=_payload(job_lane=OWN))
    assert _run(tmp_path, monkeypatch, world) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert f"job {OTHER_JOB_ID} lane {OWN} counted" in line
    assert "ignored" not in line


def test_a_job_naming_no_lane_is_counted_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=300, busy_payload=_payload(job_lane=None))
    assert _run(tmp_path, monkeypatch, world) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert f"job {OTHER_JOB_ID} lane unnamed counted" in line


def test_an_unreadable_agent_is_named_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=0, agent_readable=False)
    assert _run(tmp_path, monkeypatch, world) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert GRADED["name"] in line
    assert "agent unreadable" in line


def test_a_single_shot_pass_says_the_wait_set_was_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=0)
    assert _run(tmp_path, monkeypatch, world, budget=0) == 0
    (line,) = _wait_set_lines(capsys.readouterr().out)
    assert "not read" in line
    assert world.agent_reads == []


def test_a_red_pass_still_prints_its_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Positive control: the lane never answers and its own job outlasts the
    # budget. The pass is RED, and the one line is there before the error.
    world = World(ready_at=10**9, busy_until=10**9, busy_payload=_payload(job_lane=OWN))
    assert _run(tmp_path, monkeypatch, world, budget=60) == 1
    out = capsys.readouterr().out
    assert len(_wait_set_lines(out)) == 1
    assert out.index(LINE_MARK) < out.index("::error::")


@pytest.mark.parametrize(
    ("snapshot", "lane", "verdict"),
    [
        pytest.param(_payload(job_lane=OTHER), OWN, "ignored", id="other-lane-active"),
        pytest.param(_payload(job_lane=OWN), OWN, "counted", id="own-lane-active"),
        pytest.param(_payload(job_lane=None), OWN, "counted", id="unnamed-lane"),
        pytest.param(_payload(job_lane=OTHER), "", "counted", id="lane-declares-none"),
        pytest.param(
            _payload(job_lane=OTHER, state="settling", settling=True),
            OWN,
            "ignored",
            id="other-lane-settling",
        ),
    ],
)
def test_the_mark_agrees_with_the_busy_test(
    snapshot: dict[str, Any], lane: str, verdict: str
) -> None:
    text = resolver._wait_set_text({"name": "n", "bus_lane": lane}, snapshot)
    assert text.endswith(verdict)
    # The line never disagrees with the busy test that decides the wait.
    busy = resolver.deploy_agent_busy_reason(snapshot, lane) is not None
    assert busy is (verdict == "counted")


def test_an_idle_agent_and_a_queued_command_read_as_the_busy_test_does() -> None:
    candidate = {"name": "n", "bus_lane": OWN}
    assert resolver._wait_set_text(candidate, IDLE).endswith("no deploy job seen")
    queued = {
        "health": {"state": "idle", "active_job": None, "last_result": {}},
        "queue": {"commands_ahead": 2, "control_topic_lag_age_seconds": 1.0},
    }
    assert resolver._wait_set_text(candidate, queued).endswith(
        "agent 2 command(s) queued counted"
    )
