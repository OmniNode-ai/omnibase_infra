# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registered dev-lane branch installs load: one store, applied before the gate (OMN-26300).

Five whys of 2026-11-21: the dev deploy agent kept registered branch installs in two stores
and read one, applied them only after the health gate had judged the build, and let its own
idle restore take the recreate window ahead of a waiting install. A registered fix that makes
the runtime healthy could therefore never load.

Each test name carries the criterion it proves: ``single_store`` (AC1),
``apply_before_health_gate`` (AC2), ``idle_restore_yields`` (AC3), ``held_reason_visible``
(AC4), ``gate_not_weakened`` (AC5), ``recorded_devwindow`` (AC6). Run each from the
repository root::

    uv run --project scripts/deploy-agent --extra dev pytest \\
        scripts/deploy-agent/tests/unit/test_deploy_agent_registered_installs.py -k <name> -q
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelHealthCheck,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import VerificationFailedError
from deploy_agent.idle_converge import EnumRuntimePresence

from .test_branch_installs_agent_omn17427 import (
    HEAD,
    MISSING_LINE,
    REAPPLY_ENTRY,
    REPO_SRC,
    SERVICE,
    WRITER,
    _agent,
    _Lock,
)

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "devwindow_recorded_rows_omn26300.json"
LEGACY_ENV = "DEPLOY_AGENT_LEGACY_BRANCH_INSTALL_STORE"
DEV_UNIT = Path(__file__).resolve().parents[2] / "deploy" / "deploy-agent-dev.service"


@pytest.fixture
def lock(monkeypatch: pytest.MonkeyPatch) -> type[_Lock]:
    _Lock.events = []
    _Lock.contended = False
    monkeypatch.setattr(agent_mod, "lane_lock", _Lock)
    return _Lock


def _held(agent: agent_mod.DeployAgent) -> list[dict[str, str]]:
    listing: dict[str, Any] = agent._latest_branch_installs()
    return list(listing["held"])


def _refusal() -> VerificationFailedError:
    """What the health gate raises for a runtime that answers HTTP 503."""
    return VerificationFailedError(
        [
            ModelHealthCheck(
                service=SERVICE,
                endpoint="http://localhost:8085/health",
                status="fail",
                detail="HTTP 503",
            )
        ]
    )


def _register(
    state: Path,
    tmp_path: Path,
    *,
    install_id: str = "registered-omnimarket-6510",
    pr: int = 6510,
    head: str = HEAD,
    restart: bool = False,
) -> None:
    """Register one install through the handler's own command, as a lane does."""
    source = tmp_path / "branch" / f"{install_id}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(f"# {install_id}\n".encode())
    argv = [
        sys.executable,
        str(REPO_SRC / REAPPLY_ENTRY),
        "register",
        "--state-dir",
        str(state),
        "--lane",
        "dev",
        "--id",
        install_id,
        "--repo",
        "omnimarket",
        "--pr",
        str(pr),
        "--branch",
        f"omn-26152-{install_id}",
        "--head",
        head,
        "--ticket",
        "OMN-26152",
        "--registered-by",
        "devwindow-lane",
        "--service",
        SERVICE,
        "--expires-in-hours",
        "48",
        "--file",
        f"{WRITER}={source}",
    ]
    if not restart:
        argv.append("--no-restart")
    done = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
    )
    assert done.returncode == 0, done.stdout + done.stderr


def _write_status(state: Path, readbacks: list[dict[str, str]]) -> None:
    (state / "branch-installs" / "status.json").write_text(
        json.dumps({"readbacks": readbacks})
    )


def _jobs_state(agent: agent_mod.DeployAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane whose main runtime is up and whose companion is down, idle for a day."""
    agent._idle_converge_started_at = datetime.now(UTC) - timedelta(days=1)
    monkeypatch.setattr(agent.job_store, "has_active_job", lambda: False)
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.COMPANION_DOWN,
        "effects container not running",
    )


def _spy_restore(
    agent: agent_mod.DeployAgent, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    ran: list[str] = []
    monkeypatch.setattr(
        agent, "_execute_command", lambda cmd: ran.append(cmd.requested_by)
    )
    return ran


# --------------------------------------------------------------------------- #
# AC1: one store                                                              #
# --------------------------------------------------------------------------- #
def test_single_store_an_install_in_the_old_location_is_read_and_the_location_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    old = tmp_path / "jobs"
    _register(old, tmp_path, install_id="in-the-old-store", pr=6510)
    _register(tmp_path / "state", tmp_path, install_id="in-the-agent-store", pr=6496)
    monkeypatch.setenv(LEGACY_ENV, str(old / "branch-installs"))

    listing: dict[str, Any] = agent._latest_branch_installs()

    assert sorted(i["id"] for i in listing["declared"]) == [
        "in-the-agent-store",
        "in-the-old-store",
    ]
    assert not (old / "branch-installs").exists(), "the old location is retired"
    store = tmp_path / "state" / "branch-installs"
    assert len(list((store / "content").iterdir())) == 2, "the file bytes moved with it"
    # the apply step reads the same single store
    runner.answer("check", code=4, err=MISSING_LINE)
    agent._maybe_reapply_branch_installs()
    assert runner.commands() == ["release-due", "check", "apply"]
    argv = runner.calls[0][0]
    assert argv[argv.index("--state-dir") + 1] == str(tmp_path / "state")
    assert agent_mod._registered_installs().declared_repos() == ["omnimarket"]


def test_single_store_an_unreadable_old_location_is_kept_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    old = tmp_path / "jobs" / "branch-installs"
    old.mkdir(parents=True)
    (old / "records.json").write_text("{not json")
    monkeypatch.setenv(LEGACY_ENV, str(old))

    agent._latest_branch_installs()

    assert (old / "records.json").read_text() == "{not json"


def test_single_store_the_dev_unit_names_the_old_location_to_retire() -> None:
    text = DEV_UNIT.read_text(encoding="utf-8")
    assert re.search(
        r"^Environment=DEPLOY_AGENT_STATE_DIR=.*/state/jobs-dev$", text, re.M
    )
    assert re.search(
        rf"^Environment={LEGACY_ENV}=\S*/state/jobs/branch-installs$", text, re.M
    )


# --------------------------------------------------------------------------- #
# AC2: applied in the build, before the gates judge it                        #
# --------------------------------------------------------------------------- #
def _boot_guard_timeline(timeline: list[str]) -> None:
    guard = agent_mod.DevBootGuard.return_value
    guard.wait.side_effect = lambda **kw: timeline.append("boot guard wait")


def test_apply_before_health_gate_the_install_is_in_the_build_the_gates_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, jobs, runner, timeline = _agent(tmp_path, monkeypatch)
    _boot_guard_timeline(timeline)
    _register(tmp_path / "state", tmp_path)

    agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"
    assert timeline == ["recreate", "handler apply", "boot guard wait", "verify"]
    assert "--in-job" in runner.calls[0][0], "the job's own recreate took the window"


def test_apply_before_health_gate_the_entry_does_not_hold_an_in_job_apply_for_the_window(
    tmp_path: Path,
) -> None:
    """The runtime entry: a job's apply is not held; the idle tick's apply still is."""
    code = "\n".join(
        [
            "import sys",
            "from omnibase_infra.nodes.node_dev_roll_events_effect"
            ".runtime_dev_branch_install_reapply import main",
            "seen = []",
            "def handler(argv):",
            "    seen.append(list(argv))",
            "    return 0",
            "kw = dict(handler_main=handler, announce=lambda *a, **k: False,",
            "          window=lambda *a, **k: True)",
            "print('IN-JOB', main(['apply', '--state-dir', sys.argv[1], '--in-job'], **kw), seen)",
            "print('IDLE', main(['apply', '--state-dir', sys.argv[1]], **kw), len(seen))",
        ]
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("IN-JOB 0 [['apply', '--state-dir'"), lines
    assert "--in-job" not in lines[0], "the flag is the entry's, never the handler's"
    assert "RECREATE-WINDOW-HOLD" in done.stdout
    assert lines[-1] == "IDLE 0 1", (
        "the idle tick's apply is held, no second handler run"
    )


# --------------------------------------------------------------------------- #
# AC3: the idle restore yields                                                #
# --------------------------------------------------------------------------- #
def test_idle_restore_yields_the_window_to_a_waiting_install_and_takes_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    state = tmp_path / "state"
    _register(state, tmp_path)
    _jobs_state(agent, monkeypatch)
    ran = _spy_restore(agent, monkeypatch)

    assert agent._maybe_idle_restore(datetime.now(UTC)) is True
    assert ran == [], "a waiting install keeps the restore off the window"

    _write_status(
        state, [{"install_key": "registered-omnimarket-6510", "state": "present"}]
    )
    assert agent._maybe_idle_restore(datetime.now(UTC)) is True
    assert ran == ["agent/idle-restore"], "none waiting: the restore takes the window"


def test_idle_restore_yields_never_leaves_a_down_lane_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A down lane has no container for the install's apply; the restore builds it and applies."""
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    _register(tmp_path / "state", tmp_path)
    _jobs_state(agent, monkeypatch)
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.DOWN,
        "no container",
    )
    ran = _spy_restore(agent, monkeypatch)

    agent._maybe_idle_restore(datetime.now(UTC))

    assert ran == ["agent/idle-restore"]


# --------------------------------------------------------------------------- #
# AC4: a reason for every held install                                        #
# --------------------------------------------------------------------------- #
def test_held_reason_visible_for_each_hold_in_the_listing_and_in_status_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    state = tmp_path / "state"
    _register(state, tmp_path)

    # registered, nothing read back yet
    [held] = _held(agent)
    assert held["code"] == "not_checked" and held["reason"].strip()

    # a deploy holds the lane: the idle tick's apply waits, and says so
    runner.answer("check", code=4, err=MISSING_LINE)
    lock.contended = True
    agent._maybe_reapply_branch_installs()
    [held] = _held(agent)
    assert held["code"] == "lane_lock_held_by_deploy" and held["reason"].strip()
    written = json.loads((state / "branch-installs" / "status.json").read_text())
    assert [h["reason"] for h in written["held"]] == [held["reason"]]

    # the recreate window holds the reapply restart
    agent_mod._registered_installs().record_hold(
        "waiting_recreate_window", "the reapply restart waits for the window"
    )
    [held] = _held(agent)
    assert held["code"] == "waiting_recreate_window"


def test_held_reason_visible_every_held_install_has_one_or_the_listing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    state = tmp_path / "state"
    for n, pr in enumerate((6510, 6496)):
        _register(state, tmp_path, install_id=f"held-install-{n}", pr=pr)
    _write_status(state, [{"install_key": "held-install-0", "state": "missing"}])

    listing: dict[str, Any] = agent._latest_branch_installs()

    held = listing["held"]
    assert sorted(h["id"] for h in held) == ["held-install-0", "held-install-1"]
    assert all(h["reason"].strip() for h in held)
    with pytest.raises(ValueError):
        agent_mod._registered_installs().record_hold("not_applied", "  ")


def test_held_reason_visible_the_receipt_writes_the_reason_into_status_json(
    tmp_path: Path,
) -> None:
    """The handler's own receipt (what ``status.json`` is) names each install it holds."""
    code = (
        "import json; from datetime import UTC, datetime; "
        "from omnibase_infra.models.model_dev_lane_branch_install_receipt import "
        "ModelDevLaneBranchInstallReceipt as R; "
        "from omnibase_infra.models.model_dev_lane_branch_install_readback import "
        "ModelDevLaneBranchInstallReadback as B; "
        "b = B(install_key='k', repo='omnimarket', pr=1, head='a'*40, ticket='OMN-1', "
        "expires_at=datetime.now(UTC), state='missing', files=()); "
        "print(R(checked_at=datetime.now(UTC), compose_project='p', readbacks=(b,))"
        ".model_dump_json())"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    [held] = json.loads(done.stdout)["held"]
    assert held["id"] == "k" and held["code"] == "not_applied" and held["reason"]


# --------------------------------------------------------------------------- #
# AC5: the gates are not weakened                                             #
# --------------------------------------------------------------------------- #
def test_gate_not_weakened_a_build_whose_health_check_fails_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, jobs, runner, timeline = _agent(tmp_path, monkeypatch)
    _boot_guard_timeline(timeline)
    _register(tmp_path / "state", tmp_path)

    def refuse(**kwargs: object) -> list[object]:
        timeline.append("verify")
        raise _refusal()

    monkeypatch.setattr(agent.executor, "verify", refuse)

    agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "failed", (
        "the install did not bypass the gate"
    )
    assert timeline == ["recreate", "handler apply", "boot guard wait", "verify"]
    assert timeline.count("verify") == 1
    assert not any("skip" in a or "bypass" in a for a in runner.calls[0][0])


def test_gate_not_weakened_a_failed_apply_does_not_stand_in_for_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, jobs, runner, timeline = _agent(tmp_path, monkeypatch)
    _boot_guard_timeline(timeline)
    _register(tmp_path / "state", tmp_path)
    runner.answer("apply", code=4, err="BRANCH-INSTALL-MISSING")

    agent._run_deploy(cmd)

    # the gates still ran on whatever is running; a failed apply is logged, not a verdict
    assert timeline == ["recreate", "handler apply", "boot guard wait", "verify"]
    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"


def test_gate_not_weakened_no_install_is_applied_after_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An apply after the verdict would restart a container nothing judged."""
    agent, cmd, _, runner, timeline = _agent(tmp_path, monkeypatch)
    _register(tmp_path / "state", tmp_path)

    agent._run_deploy(cmd)

    assert timeline.index("verify") > timeline.index("handler apply")
    assert timeline[timeline.index("verify") :] == ["verify"]
    assert runner.commands() == ["apply"]


# --------------------------------------------------------------------------- #
# AC6: the recorded devwindow rows                                            #
# --------------------------------------------------------------------------- #
def _recorded() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_recorded_devwindow_both_registered_installs_are_seen_in_one_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _recorded()
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    old = tmp_path / "jobs"
    for install in recorded["registered_installs"]:
        where = old if install["store"] == "old" else tmp_path / "state"
        _register(
            where,
            tmp_path,
            install_id=install["id"],
            pr=install["pr"],
            head=install["head"],
            restart=install["restart"],
        )
    monkeypatch.setenv(LEGACY_ENV, str(old / "branch-installs"))

    listing: dict[str, Any] = agent._latest_branch_installs()

    assert sorted((i["pr"], i["head"]) for i in listing["declared"]) == [
        (6496, "3d2e1873910ce384bd281e3f9d28fdc0d9701df2"),
        (6510, "828262734ff36e1b57ceb8c7149085af843c953c"),
    ]
    assert all(h["reason"].strip() for h in listing["held"])
    assert not (old / "branch-installs").exists()


@pytest.mark.parametrize(
    "requested_by",
    ["agent/idle-restore", "operator/pinned-fix/devwindow3-2340"],
)
def test_recorded_devwindow_the_restore_and_the_pinned_fix_builds_carry_the_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested_by: str
) -> None:
    """Jobs 5623655f, 8648de6f and f79001c5 recreated the runtime and the gate read a build
    without the registered fixes; each now applies them first."""
    recorded = _recorded()
    agent, _, jobs, _, timeline = _agent(tmp_path, monkeypatch)
    _boot_guard_timeline(timeline)
    for install in recorded["registered_installs"]:
        _register(
            tmp_path / "state",
            tmp_path,
            install_id=install["id"],
            pr=install["pr"],
            head=install["head"],
            restart=install["restart"],
        )
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by=requested_by,
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    jobs.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    agent._run_deploy(cmd)

    assert timeline == ["recreate", "handler apply", "boot guard wait", "verify"]


def test_recorded_devwindow_the_failed_pinned_fix_job_a9ecbe18_is_still_refused_at_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job a9ecbe18 failed at the runtime health gate (HTTP 503); the gate still refuses a
    build it reads unhealthy, now after the installs went in."""
    recorded = _recorded()
    job_id = next(
        j["job"] for j in recorded["pinned_fix_jobs"] if j["job"].startswith("a9ecbe18")
    )
    agent, _, jobs, _, timeline = _agent(tmp_path, monkeypatch)
    _register(tmp_path / "state", tmp_path)
    cmd = ModelRebuildRequested(
        correlation_id=UUID(job_id),
        requested_by="operator/pinned-fix/devwindow2-2215",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    jobs.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    def refuse(**kwargs: object) -> list[object]:
        timeline.append("verify")
        raise _refusal()

    monkeypatch.setattr(agent.executor, "verify", refuse)

    agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert "HTTP 503" in str(job.errors)
    assert timeline == ["recreate", "handler apply", "verify"]


def test_recorded_devwindow_the_idle_restore_windows_of_19_00_and_21_00_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _recorded()
    assert [j["job"] for j in recorded["idle_restore_jobs"]] == [
        "5623655f",
        "8648de6f",
    ]
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    for install in recorded["registered_installs"]:
        _register(
            tmp_path / "state",
            tmp_path,
            install_id=install["id"],
            pr=install["pr"],
            head=install["head"],
        )
    _jobs_state(agent, monkeypatch)
    ran = _spy_restore(agent, monkeypatch)

    agent._maybe_idle_restore(datetime.now(UTC))

    assert ran == [], "the restore leaves the window to the two waiting installs"
