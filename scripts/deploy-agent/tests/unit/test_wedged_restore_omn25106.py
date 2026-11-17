# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent restores a dev runtime that runs but stays unhealthy (OMN-25106).

Measured on the lab dev lane, 2026-11-16 to 2026-11-17: ``omninode-runtime``
stayed ``running`` and docker ``unhealthy`` (FailingStreak 72 at 00:13Z) after
its event bus consumers exited at 23:34:34Z. The agent's own idle restore read
it as RUNNING and never restored it, and the bus chain that could redeploy it
runs inside that runtime. The idle restore is the agent's local path: it needs
no bus to start.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import lane_config_for
from deploy_agent.idle_converge import (
    RESTORE_AFTER,
    EnumIdleRestoreVerdict,
    EnumRuntimePresence,
    ModelIdleRestoreInputs,
    decide_restore,
    read_lane_presence,
)

pytestmark = pytest.mark.unit

DEV = lane_config_for(EnumRuntimeLane.DEV)
MAIN_ONLY = DEV.model_copy(update={"disabled_services": frozenset({"runtime-effects"})})
MAIN = "omninode-runtime"
NOW = datetime(2026, 11, 17, 0, 13, tzinfo=UTC)


def _docker(answers: dict[str, str]) -> Any:
    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in answers:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{answers[name]}\n", "")

    return run


def _decide(presence: EnumRuntimePresence, detail: str) -> EnumIdleRestoreVerdict:
    return decide_restore(
        ModelIdleRestoreInputs(
            now=NOW,
            presence=presence,
            presence_detail=detail,
            job_active=False,
            last_activity=NOW - RESTORE_AFTER - timedelta(minutes=1),
            last_attempt=None,
        )
    ).verdict


def test_the_recorded_wedge_is_restored() -> None:
    presence, detail = read_lane_presence(
        MAIN_ONLY, run=_docker({MAIN: "running unhealthy 72"})
    )

    assert presence is EnumRuntimePresence.DOWN
    assert "unhealthy" in detail
    assert _decide(presence, detail) is EnumIdleRestoreVerdict.RESTORE


@pytest.mark.parametrize(
    "answer", ["running healthy 0", "running starting 4", "running unhealthy 3"]
)
def test_a_runtime_that_is_not_wedged_is_left_alone(answer: str) -> None:
    presence, detail = read_lane_presence(MAIN_ONLY, run=_docker({MAIN: answer}))

    assert presence is EnumRuntimePresence.RUNNING
    assert _decide(presence, detail) is EnumIdleRestoreVerdict.NOT_DOWN
