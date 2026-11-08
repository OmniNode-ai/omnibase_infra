# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev boot guard reads a lane's effects container, not main alone (OMN-23106).

Replays the recorded outage of 2026-11-07 through the handler's own lane
presence read, with docker answered by a fake host execution seam: after deploy
job 3715d9e9 the dev lane ran ``omninode-runtime`` and had no
``omninode-runtime-effects`` for seven hours, and the only consumer of
``onex.cmd.omnimarket.delegate-skill.v1`` was the missing container. The read
used to name main alone, so the lane read as healthy.

Control: the same read with both containers running, with effects folded into
main and with the service retired reads running, so the change does not restore
a lane that needs no companion. A read docker cannot answer stays ``unknown``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod

MAIN = "omninode-runtime"
EFFECTS = "omninode-runtime-effects"
TARGETS = ((MAIN, 8085), (EFFECTS, 8086))
FOLDED_TARGETS = ((MAIN, 8085), (MAIN, 8086))
RETIRED = frozenset({"runtime-effects"})


def _docker(states: dict[str, str]) -> Callable[..., Any]:
    """A ``docker inspect`` fake: a container's state, or docker's no-such-object."""

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in states:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{states[name]}\n", "")

    return run


def _presence(
    states: dict[str, str],
    targets: tuple[tuple[str, int], ...] = TARGETS,
    disabled: frozenset[str] = frozenset(),
) -> tuple[str, str]:
    return guard_mod.read_lane_presence(
        targets,
        disabled,
        run=_docker(states),
        unreadable=(OSError, subprocess.TimeoutExpired),
    )


@pytest.mark.integration
def test_recorded_outage_main_running_effects_absent_reads_companion_down() -> None:
    state, detail = _presence({MAIN: "running"})

    assert state == "companion_down"
    assert EFFECTS in detail


@pytest.mark.integration
@pytest.mark.parametrize("effects_state", ["created", "exited", "dead"])
def test_effects_container_that_is_not_running_reads_companion_down(
    effects_state: str,
) -> None:
    assert _presence({MAIN: "running", EFFECTS: effects_state})[0] == "companion_down"


@pytest.mark.integration
def test_both_containers_running_reads_running() -> None:
    assert _presence({MAIN: "running", EFFECTS: "running"})[0] == "running"


@pytest.mark.integration
def test_main_down_reads_down_whatever_effects_does() -> None:
    assert _presence({EFFECTS: "running"})[0] == "down"
    assert _presence({})[0] == "down"


@pytest.mark.integration
def test_lane_that_folds_effects_into_main_needs_main_alone() -> None:
    assert guard_mod.lane_runtime_containers(FOLDED_TARGETS, frozenset()) == (MAIN,)
    assert _presence({MAIN: "running"}, targets=FOLDED_TARGETS)[0] == "running"


@pytest.mark.integration
def test_lane_that_retires_effects_needs_main_alone() -> None:
    assert guard_mod.lane_runtime_containers(TARGETS, RETIRED) == (MAIN,)
    assert _presence({MAIN: "running"}, disabled=RETIRED)[0] == "running"


@pytest.mark.integration
def test_effects_container_docker_cannot_read_stays_unknown() -> None:
    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        if cmd[-1] == EFFECTS:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, "running\n", "")

    state, _detail = guard_mod.read_lane_presence(
        TARGETS,
        frozenset(),
        run=run,
        unreadable=(OSError, subprocess.TimeoutExpired),
    )

    assert state == "unknown"
