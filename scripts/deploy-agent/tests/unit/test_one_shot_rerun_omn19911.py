# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A changed completed one-shot is re-run by a no-deps runtime rebuild (OMN-19911).

Measured on a lab dev lane on 2026-09-28. OMN-19543 moved creation of the control-plane
topic into the ``redpanda-partition-cap`` one-shot. That container had last run
at 2026-09-25T10:54Z and its compose config hash (``acb4200c``) differed from
the rendered one (``ee9e8e27``), but the agent recreates runtime services with
``up -d --no-deps``, and compose never revisits a completed one-shot it was not
asked to start. The tenant-registry writer was recreated against a broker with
no such topic and logged 3212 missing-topic lines in 10 minutes.

What is pinned here:

1. A control-plane one-shot whose running config hash differs from the rendered
   one is in the recreate plan, naming both hashes.
2. One whose hash is unchanged is not, and one that never ran is.
3. The re-run is a forced, ``--no-deps`` recreate that finishes BEFORE the
   runtime services are started.
4. Each re-run is recorded with its exit code, on the executor, the job record
   (``/job/{cid}``) and the terminal event; a one-shot that did not exit 0
   still leaves its record and fails the phase.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import (
    EnumOneShotRerunReason,
    EnumRuntimeLane,
    ModelOneShotRerun,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    CONTROL_PLANE_ONESHOT_SERVICES,
    DeployExecutor,
)
from deploy_agent.job_state import JobStore
from deploy_agent.publisher import build_completion_payload

pytestmark = pytest.mark.unit

ONESHOT = "redpanda-partition-cap"
OLD_HASH = "acb4200c" + "0" * 56
NEW_HASH = "ee9e8e27" + "1" * 56


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _done(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class _Lane:
    """A fake lane: the hash compose renders and the hash the container carries."""

    def __init__(self, *, rendered: str, running: str) -> None:
        self.rendered = rendered
        self.running = running
        self.calls: list[list[str]] = []

    def run(
        self, cmd: list[str], timeout: int = 0, **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if "config" in cmd and "--hash" in cmd:
            return _done(f"{ONESHOT} {self.rendered}\n")
        if cmd[:2] == ["docker", "ps"]:
            return _done(f"{self.running}\n" if self.running else "")
        return _done()

    def up_commands(self) -> list[list[str]]:
        return [c for c in self.calls if "up" in c]


def _plan(lane: _Lane) -> list[ModelOneShotRerun]:
    with patch("deploy_agent.executor._run", side_effect=lane.run):
        return DeployExecutor().plan_oneshot_reruns(EnumRuntimeLane.DEV)


def test_partition_cap_is_a_declared_control_plane_one_shot() -> None:
    assert ONESHOT in CONTROL_PLANE_ONESHOT_SERVICES


def test_one_shot_whose_command_changed_is_in_the_recreate_plan() -> None:
    plan = _plan(_Lane(rendered=NEW_HASH, running=OLD_HASH))

    assert [item.service for item in plan] == [ONESHOT]
    assert plan[0].reason is EnumOneShotRerunReason.CONFIG_CHANGED
    assert plan[0].running_config_hash == OLD_HASH
    assert plan[0].rendered_config_hash == NEW_HASH


def test_one_shot_with_an_unchanged_hash_is_left_alone() -> None:
    assert _plan(_Lane(rendered=NEW_HASH, running=NEW_HASH)) == []


def test_one_shot_that_never_ran_is_in_the_recreate_plan() -> None:
    plan = _plan(_Lane(rendered=NEW_HASH, running=""))

    assert [item.service for item in plan] == [ONESHOT]
    assert plan[0].reason is EnumOneShotRerunReason.NEVER_RAN


def test_one_shot_unreadable_render_fails_loudly_instead_of_skipping() -> None:
    def run(cmd: list[str], timeout: int = 0, **_: object):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="no such service"
        )

    with (
        patch("deploy_agent.executor._run", side_effect=run),
        pytest.raises(RuntimeError, match="config --hash"),
    ):
        DeployExecutor().plan_oneshot_reruns(EnumRuntimeLane.DEV)


def _rerun(
    lane: _Lane, *, exit_code: int | None, state: str = "exited"
) -> DeployExecutor:
    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=lane.run),
        patch(
            "deploy_agent.executor._compose_service_states",
            return_value={ONESHOT: (state, exit_code)},
        ),
        patch(
            "deploy_agent.executor.verify_oneshots_completed",
            side_effect=lambda services, **_: (
                (True, []) if exit_code == 0 else (False, list(services))
            ),
        ),
    ):
        executor.rerun_changed_oneshots(lane=EnumRuntimeLane.DEV, timeout=60)
    return executor


def test_one_shot_rerun_is_a_forced_no_deps_recreate_of_that_service() -> None:
    lane = _Lane(rendered=NEW_HASH, running=OLD_HASH)
    _rerun(lane, exit_code=0)

    (cmd,) = lane.up_commands()
    assert cmd[-1] == ONESHOT
    assert "--no-deps" in cmd
    assert "--force-recreate" in cmd


def test_one_shot_unchanged_issues_no_compose_up() -> None:
    lane = _Lane(rendered=NEW_HASH, running=NEW_HASH)
    executor = _rerun(lane, exit_code=0)

    assert lane.up_commands() == []
    assert executor.oneshot_reruns == []


def test_one_shot_rerun_finishes_before_the_runtime_services_start() -> None:
    lane = _Lane(rendered=NEW_HASH, running=OLD_HASH)
    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=lane.run),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch(
            "deploy_agent.executor._compose_service_states",
            return_value={ONESHOT: ("exited", 0)},
        ),
        patch(
            "deploy_agent.executor.verify_oneshots_completed",
            return_value=(True, []),
        ),
        patch(
            "deploy_agent.executor.verify_containers_up",
            return_value=(True, []),
        ),
    ):
        executor._compose_up(Phase.RUNTIME, Scope.RUNTIME, [], _noop)

    started = [c[-1] if c[-1] == ONESHOT else "runtime" for c in lane.up_commands()]
    assert started == [ONESHOT, "runtime"]


def test_one_shot_job_record_lists_each_rerun_with_its_exit_code() -> None:
    executor = _rerun(_Lane(rendered=NEW_HASH, running=OLD_HASH), exit_code=0)

    (record,) = executor.oneshot_reruns
    assert record.service == ONESHOT
    assert record.exit_code == 0
    assert record.completed is True
    assert record.reason is EnumOneShotRerunReason.CONFIG_CHANGED
    assert record.running_config_hash == OLD_HASH
    assert record.rendered_config_hash == NEW_HASH
    assert ONESHOT in record.describe()


def test_one_shot_job_record_keeps_a_failed_rerun_and_fails_the_phase() -> None:
    lane = _Lane(rendered=NEW_HASH, running=OLD_HASH)
    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=lane.run),
        patch(
            "deploy_agent.executor._compose_service_states",
            return_value={ONESHOT: ("exited", 1)},
        ),
        patch(
            "deploy_agent.executor.verify_oneshots_completed",
            return_value=(False, [ONESHOT]),
        ),
        pytest.raises(RuntimeError, match=ONESHOT),
    ):
        executor.rerun_changed_oneshots(lane=EnumRuntimeLane.DEV, timeout=60)

    (record,) = executor.oneshot_reruns
    assert record.exit_code == 1
    assert record.completed is False


def test_one_shot_job_record_is_cleared_for_the_next_job() -> None:
    executor = _rerun(_Lane(rendered=NEW_HASH, running=OLD_HASH), exit_code=0)
    executor.reset_deploy_observations()

    assert executor.oneshot_reruns == []


def _record() -> ModelOneShotRerun:
    return ModelOneShotRerun(
        service=ONESHOT,
        lane=EnumRuntimeLane.DEV,
        compose_project="omnibase-infra",
        reason=EnumOneShotRerunReason.CONFIG_CHANGED,
        running_config_hash=OLD_HASH,
        rendered_config_hash=NEW_HASH,
        exit_code=0,
        completed=True,
    )


def test_one_shot_job_record_is_persisted_and_served(tmp_path: Path) -> None:
    from deploy_agent.health import _job_handler

    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime", "runtime_lane": "dev"})
    store.complete(cid, status="success", oneshot_reruns=[_record()])

    reloaded = store.load(cid)
    assert reloaded is not None
    assert [r.service for r in reloaded.oneshot_reruns] == [ONESHOT]

    request = SimpleNamespace(
        app={"job_store": store}, match_info={"correlation_id": str(cid)}
    )
    body = json.loads(asyncio.run(_job_handler(cast("Any", request))).text)
    assert body["oneshot_reruns"][0]["service"] == ONESHOT
    assert body["oneshot_reruns"][0]["exit_code"] == 0


def test_one_shot_job_record_rides_the_terminal_event(tmp_path: Path) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime", "runtime_lane": "dev"})
    job = store.complete(cid, status="success")

    payload = build_completion_payload(job, "a" * 40, oneshot_reruns=[_record()])

    assert payload["oneshot_reruns"][0]["service"] == ONESHOT
    assert payload["oneshot_reruns"][0]["exit_code"] == 0


def test_one_shot_job_record_is_handed_to_both_terminal_writes() -> None:
    """Read from the source: driving a whole job needs a broker and a lane."""
    agent_source = (
        Path(__file__).resolve().parents[2] / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert agent_source.count("oneshot_reruns=self.executor.oneshot_reruns,") == 2, (
        "the success write and the failure write must both carry the re-runs"
    )
