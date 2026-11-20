# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev lane whose effects container is gone is restored (OMN-22845).

MEASURED on the lab dev lane, 2026-11-07 to 2026-11-08 (dev broker, deploy agent
job records, the host's docker journal):

* ``onex.cmd.omnimarket.delegate-skill.v1`` has one partition and one consuming
  group, the runtime-effects group of ``node_delegate_skill_orchestrator``.
* Deploy job 3715d9e9 (an install proof of an open branch, 19:39:43Z to
  20:09:55Z) left ``omninode-runtime`` running and ``omninode-runtime-effects``
  absent; its forward redeploy read ``verified=False``.
* The host's capture-silence watch logged ``No such container:
  omninode-runtime-effects`` on 85 consecutive five-minute ticks, 20:01:47Z to
  03:02:15Z, and the container joined its network again at 03:04:20Z.
* 45 delegation requests published 20:01:22Z to 02:48Z sat in the topic and
  were answered at 03:05Z to 03:06Z, each refused ``queue_budget_exhausted``.
* The restore rule (OMN-20864) read only the main runtime container, so a lane
  with main up and effects absent read NOT_DOWN for seven hours. What restored
  it was ``agent/idle-converge`` (job c9df5b56, 02:47Z) firing for another
  reason.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    DEFAULT_DEV_INSTANCE,
    DEV_INSTANCE_LANE_CONFIGS,
    lane_config_for,
    load_dev_boot_guard,
)
from deploy_agent.idle_converge import (
    COMPANION_RESTORE_AFTER,
    IDLE_RESTORE_REQUESTER,
    RESTORE_AFTER,
    RESTORE_RETRY_AFTER,
    EnumIdleRestoreVerdict,
    EnumRuntimePresence,
    ModelIdleRestoreInputs,
    decide_restore,
    read_lane_presence,
)
from deploy_agent.routing import ModelLaneFlags, ModelLaneFreeze

from .test_dev_lane_outage_omn20864 import _agent, _jobs

pytestmark = pytest.mark.unit

# OMN-22930 folded the dev effects role into the main runtime, so the dev lane the
# agent runs on the lab needs ``omninode-runtime`` alone. The companion rule
# (OMN-22845) still applies to a lane that runs a separate effects container. That
# lane is built from the dev composition BEFORE any fold, straight from the table
# the agent ships, so no host policy file decides it. Never ``lane_config_for(DEV)``
# at import: that reads ``docker/runtime-policy.env`` of the host the suite runs on
# (the lab hosts carry one, hosted CI does not), and the autouse fixture that hides
# the file only applies once a test is running.
SEPARATE = DEV_INSTANCE_LANE_CONFIGS[DEFAULT_DEV_INSTANCE]
MAIN_ONLY = SEPARATE.model_copy(
    update={"disabled_services": frozenset({"runtime-effects"})}
)
MAIN = "omninode-runtime"
EFFECTS = "omninode-runtime-effects"
NOW = datetime(2026, 11, 7, 20, 30, tzinfo=UTC)
TRACKED_RUNTIME_POLICY = (
    Path(__file__).resolve().parents[4] / "docker" / ("runtime-policy.env")
)


@pytest.fixture
def folded_dev(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The dev lane as the checked-in runtime policy folds it (OMN-22930).

    The policy file is this checkout's tracked one, never the host's copy, so the
    lane reads the same on a lab host and on a hosted runner.
    """
    monkeypatch.setattr(executor_mod, "RUNTIME_POLICY_ENV_FILE", TRACKED_RUNTIME_POLICY)
    return lane_config_for(EnumRuntimeLane.DEV)


def _containers(lane: Any) -> tuple[str, ...]:
    return load_dev_boot_guard().lane_runtime_containers(
        lane.runtime_health_targets, lane.disabled_services
    )


def _docker(states: dict[str, str]) -> Any:
    """A ``docker inspect`` fake: a name's state, or docker's no-such-object."""

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in states:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{states[name]}\n", "")

    return run


# --- which containers the lane needs ------------------------------------------


def test_the_default_dev_lane_needs_main_alone(folded_dev: Any) -> None:
    # The fold put the effects role inside main: its health target is main's container.
    assert folded_dev.runtime_health_targets[1][0] == MAIN
    assert _containers(folded_dev) == (MAIN,)


def test_a_lane_with_a_separate_effects_container_needs_main_and_effects() -> None:
    assert SEPARATE.runtime_health_targets[1][0] != MAIN
    assert _containers(SEPARATE) == (MAIN, EFFECTS)


def test_the_prod_lane_needs_its_own_main_and_effects() -> None:
    # Prod keeps a separate effects container; its names carry the lane.
    assert _containers(lane_config_for(EnumRuntimeLane.PROD)) == (
        "omninode-prod-runtime",
        "omninode-prod-runtime-effects",
    )


def test_a_lane_that_folds_effects_into_main_needs_main_alone() -> None:
    # lane_config_for points a folded effects target at the main container.
    (main, main_port), (_, effects_port) = SEPARATE.runtime_health_targets
    folded = SEPARATE.model_copy(
        update={"runtime_health_targets": ((main, main_port), (main, effects_port))}
    )

    assert _containers(folded) == (MAIN,)


def test_a_lane_that_switches_effects_off_needs_main_alone() -> None:
    off = SEPARATE.model_copy(
        update={"disabled_services": frozenset({"runtime-effects"})}
    )

    assert _containers(off) == (MAIN,)


# --- what the lane's presence reads -------------------------------------------


@pytest.mark.parametrize(
    ("states", "presence"),
    [
        ({MAIN: "running", EFFECTS: "running"}, EnumRuntimePresence.RUNNING),
        ({MAIN: "running"}, EnumRuntimePresence.COMPANION_DOWN),
        ({MAIN: "running", EFFECTS: "created"}, EnumRuntimePresence.COMPANION_DOWN),
        ({MAIN: "running", EFFECTS: "exited"}, EnumRuntimePresence.COMPANION_DOWN),
        ({EFFECTS: "running"}, EnumRuntimePresence.DOWN),
        ({}, EnumRuntimePresence.DOWN),
    ],
)
def test_read_lane_presence(
    states: dict[str, str], presence: EnumRuntimePresence
) -> None:
    assert read_lane_presence(SEPARATE, run=_docker(states))[0] is presence


@pytest.mark.parametrize(
    ("states", "presence"),
    [
        ({MAIN: "running"}, EnumRuntimePresence.RUNNING),
        ({MAIN: "running", EFFECTS: "running"}, EnumRuntimePresence.RUNNING),
        ({MAIN: "running", EFFECTS: "exited"}, EnumRuntimePresence.RUNNING),
        ({MAIN: "exited"}, EnumRuntimePresence.DOWN),
        ({MAIN: "created"}, EnumRuntimePresence.DOWN),
        ({EFFECTS: "running"}, EnumRuntimePresence.DOWN),
        ({}, EnumRuntimePresence.DOWN),
    ],
)
def test_read_lane_presence_on_the_folded_dev_lane(
    folded_dev: Any, states: dict[str, str], presence: EnumRuntimePresence
) -> None:
    assert read_lane_presence(folded_dev, run=_docker(states))[0] is presence


def test_a_separate_effects_lane_reads_its_own_containers() -> None:
    presence, detail = read_lane_presence(SEPARATE, run=_docker({MAIN: "running"}))

    assert presence is EnumRuntimePresence.COMPANION_DOWN
    assert EFFECTS in detail


def test_the_down_container_is_named_in_the_detail() -> None:
    _, detail = read_lane_presence(SEPARATE, run=_docker({MAIN: "running"}))

    assert EFFECTS in detail


def test_the_folded_dev_lane_never_names_or_reads_an_effects_container(
    folded_dev: Any,
) -> None:
    asked: list[str] = []
    fake = _docker({MAIN: "running"})

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        asked.append(cmd[-1])
        return fake(cmd, timeout)

    up = read_lane_presence(folded_dev, run=run)
    down = read_lane_presence(folded_dev, run=_docker({}))

    assert asked == [MAIN]
    assert EFFECTS not in up[1]
    assert down[0] is EnumRuntimePresence.DOWN
    assert EFFECTS not in down[1]


def _effects_unreadable(cmd: list[str], timeout: int, **_kw: object) -> Any:
    if cmd[-1] == EFFECTS:
        return subprocess.CompletedProcess(
            cmd, 1, "", "Cannot connect to the Docker daemon"
        )
    return subprocess.CompletedProcess(cmd, 0, "running\n", "")


def test_an_unreadable_effects_container_never_restores() -> None:
    presence, _ = read_lane_presence(SEPARATE, run=_effects_unreadable)

    assert presence is EnumRuntimePresence.UNKNOWN


def test_an_unreadable_main_container_on_the_folded_dev_lane_never_restores(
    folded_dev: Any,
) -> None:
    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        return subprocess.CompletedProcess(
            cmd, 1, "", "Cannot connect to the Docker daemon"
        )

    assert read_lane_presence(folded_dev, run=run)[0] is EnumRuntimePresence.UNKNOWN
    # The effects container is not read there, so its answer cannot matter.
    assert (
        read_lane_presence(folded_dev, run=_effects_unreadable)[0]
        is EnumRuntimePresence.RUNNING
    )


def test_a_main_only_lane_reads_as_it_always_did() -> None:
    assert (
        read_lane_presence(MAIN_ONLY, run=_docker({MAIN: "running"}))[0]
        is EnumRuntimePresence.RUNNING
    )
    assert read_lane_presence(MAIN_ONLY, run=_docker({}))[0] is EnumRuntimePresence.DOWN


# --- the decision ---------------------------------------------------------------


def _inputs(**overrides: Any) -> ModelIdleRestoreInputs:
    values: dict[str, Any] = {
        "now": NOW,
        "presence": EnumRuntimePresence.COMPANION_DOWN,
        "presence_detail": f"no container named {EFFECTS}",
        "job_active": False,
        "last_activity": NOW - COMPANION_RESTORE_AFTER,
        "last_attempt": None,
    }
    values.update(overrides)
    return ModelIdleRestoreInputs(**values)


@pytest.mark.parametrize(
    ("overrides", "verdict"),
    [
        ({}, EnumIdleRestoreVerdict.RESTORE),
        (
            {"last_activity": NOW - COMPANION_RESTORE_AFTER + timedelta(seconds=1)},
            EnumIdleRestoreVerdict.RECENTLY_ACTIVE,
        ),
        ({"job_active": True}, EnumIdleRestoreVerdict.JOB_ACTIVE),
        ({"frozen_reason": "FROZEN until x"}, EnumIdleRestoreVerdict.FROZEN),
        (
            {"last_attempt": NOW - RESTORE_RETRY_AFTER + timedelta(seconds=1)},
            EnumIdleRestoreVerdict.RECENTLY_ATTEMPTED,
        ),
    ],
)
def test_decide_restore_for_a_lane_missing_its_effects_container(
    overrides: dict[str, Any], verdict: EnumIdleRestoreVerdict
) -> None:
    assert decide_restore(_inputs(**overrides)).verdict is verdict


def test_a_missing_effects_container_waits_longer_than_a_missing_main_runtime() -> None:
    # An install proof of a branch is not replaced sooner than an idle converge
    # would have replaced it; a lane with no main runtime still restores at once.
    quiet = NOW - RESTORE_AFTER
    assert COMPANION_RESTORE_AFTER > RESTORE_AFTER
    assert (
        decide_restore(_inputs(last_activity=quiet)).verdict
        is EnumIdleRestoreVerdict.RECENTLY_ACTIVE
    )
    assert (
        decide_restore(
            _inputs(presence=EnumRuntimePresence.DOWN, last_activity=quiet)
        ).verdict
        is EnumIdleRestoreVerdict.RESTORE
    )


# --- the agent, as the 2026-11-07 outage ran it ---------------------------------


def _agent_reading_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, states: dict[str, str]
) -> tuple[Any, Any]:
    """The agent with its own presence seam, over a faked docker inspect."""
    agent = _agent(tmp_path, monkeypatch, EnumRuntimePresence.RUNNING)
    default = type(agent)(skip_self_update=True)._idle_read_runtime_presence
    agent._idle_read_runtime_presence = default
    agent._idle_converge_started_at = NOW - timedelta(hours=2)
    agent._idle_converge_now = lambda: NOW

    fake = _docker(states)

    def run(argv: list[str], **kw: Any) -> Any:
        return fake(argv, kw.get("timeout", 0))

    return agent, patch("deploy_agent.idle_converge.subprocess.run", side_effect=run)


def test_replay_20_10z_main_up_effects_gone_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, docker = _agent_reading_docker(tmp_path, monkeypatch, {MAIN: "running"})

    with docker:
        agent._maybe_idle_converge()

    jobs = _jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].command["requested_by"] == IDLE_RESTORE_REQUESTER
    assert jobs[0].command["runtime_lane"] == "dev"

    # One restore per RESTORE_RETRY_AFTER, not one per idle interval.
    agent._idle_converge_last_check = None
    with docker:
        agent._maybe_idle_converge()
    assert len(_jobs(tmp_path)) == 1


def test_a_lane_with_both_containers_running_is_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, docker = _agent_reading_docker(
        tmp_path, monkeypatch, {MAIN: "running", EFFECTS: "running"}
    )
    agent._idle_read_running_ref = lambda repo: "b" * 40  # up to date

    with docker:
        agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []


def test_a_frozen_lane_missing_effects_is_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, docker = _agent_reading_docker(tmp_path, monkeypatch, {MAIN: "running"})
    agent._lane_flags = ModelLaneFlags(
        freeze=ModelLaneFreeze(until=NOW + timedelta(days=1), reason="test")
    )

    with docker:
        agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []


def test_a_folded_dev_lane_with_main_up_is_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folded_dev: Any
) -> None:
    # The fold left no effects container to miss: main alone running is a healthy lane.
    agent, docker = _agent_reading_docker(tmp_path, monkeypatch, {MAIN: "running"})
    agent._idle_read_running_ref = lambda repo: "b" * 40  # up to date

    with docker:
        agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []


def test_a_folded_dev_lane_with_main_gone_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folded_dev: Any
) -> None:
    agent, docker = _agent_reading_docker(tmp_path, monkeypatch, {})

    with docker:
        agent._maybe_idle_converge()

    jobs = _jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].command["requested_by"] == IDLE_RESTORE_REQUESTER
    assert jobs[0].command["runtime_lane"] == "dev"
