# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev runtime that runs but stays unhealthy reads as down, so it is restored (OMN-25106).

Measured on the lab dev lane, 2026-11-16 to 2026-11-17: ``omninode-runtime``
started at 23:03:13Z, its event bus consumers exited at 23:34:34Z, and from then
on ``/health`` and ``/ready`` did not answer. Docker marked the container
``unhealthy`` (FailingStreak 72 at 00:13Z, a probe every 30 s), but its state
stayed ``running``. The deploy agent's idle restore read the lane through
``read_lane_presence``, which looked at ``State.Status`` alone, so the lane read
``running`` and was never restored. Every bus path that could have redeployed it
(redeploy-start, the orchestrator, the deploy effect) runs inside that runtime.

Control: a running container that is healthy, still starting, or unhealthy for
fewer probes than the wedge bound reads ``running``, so a slow boot or one failed
probe does not restore a lane.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod

pytestmark = pytest.mark.unit

MAIN = "omninode-runtime"
EFFECTS = "omninode-runtime-effects"
TARGETS = ((MAIN, 8085), (EFFECTS, 8086))
MAIN_ONLY = ((MAIN, 8085), (MAIN, 8086))


def _docker(answers: dict[str, str]) -> Callable[..., Any]:
    """A ``docker inspect`` fake answering with the template's fields for a name."""

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in answers:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{answers[name]}\n", "")

    return run


def _presence(
    answers: dict[str, str], targets: tuple[tuple[str, int], ...] = MAIN_ONLY
) -> tuple[str, str]:
    return guard_mod.read_lane_presence(
        targets,
        frozenset(),
        run=_docker(answers),
        unreadable=(OSError, subprocess.TimeoutExpired),
    )


def test_the_recorded_wedge_reads_down() -> None:
    state, detail = _presence({MAIN: "running unhealthy 72"})

    assert state == "down"
    assert MAIN in detail
    assert "unhealthy" in detail
    assert "72" in detail


def test_the_wedge_bound_is_ten_minutes_of_thirty_second_probes() -> None:
    assert guard_mod.WEDGED_FAILING_STREAK == 20
    bound = guard_mod.WEDGED_FAILING_STREAK

    assert _presence({MAIN: f"running unhealthy {bound}"})[0] == "down"
    assert _presence({MAIN: f"running unhealthy {bound - 1}"})[0] == "running"


@pytest.mark.parametrize(
    "answer",
    [
        "running healthy 0",
        "running starting 3",
        "running unhealthy 5",
        "running",
        "running ",
    ],
)
def test_a_running_container_that_is_not_wedged_reads_running(answer: str) -> None:
    assert _presence({MAIN: answer})[0] == "running"


def test_a_wedged_companion_reads_companion_down() -> None:
    state, detail = _presence(
        {MAIN: "running healthy 0", EFFECTS: "running unhealthy 40"}, TARGETS
    )

    assert state == "companion_down"
    assert EFFECTS in detail


def test_the_read_asks_docker_for_health() -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "running healthy 0\n", "")

    guard_mod.read_lane_presence(
        MAIN_ONLY,
        frozenset(),
        run=run,
        unreadable=(OSError, subprocess.TimeoutExpired),
    )

    assert seen
    assert "FailingStreak" in seen[0][3]


@pytest.mark.parametrize("stopped", ["created", "exited", "dead"])
def test_a_stopped_container_still_reads_down(stopped: str) -> None:
    assert _presence({MAIN: stopped})[0] == "down"
