# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-systemd harness for the self-update unit restart (OMN-22683).

Run it on a host with a systemd user manager, from ``scripts/deploy-agent``, with the deploy
agent's own environment (``uv run --project scripts/deploy-agent python ...``). It starts a
stand-in agent as a transient ``systemd-run --user`` service whose stop job times out after
``STOP_TIMEOUT`` seconds. The stand-in calls the REAL ``DeployExecutor.self_update`` of the code
tree under test, against a git clone that moved under the process and a unit sync that reports
its own unit, so the real code restarts the unit it runs in. It then takes the next job, a child
process of ``JOB_SECONDS`` seconds, and the harness reads the marker files::

    python3 systemd_restart_harness.py                      # the code at this checkout
    python3 systemd_restart_harness.py --control-ref REF    # deploy_agent as of REF

Two things of the real agent are mirrored: a SIGTERM only sets a flag
(``DeployAgent._handle_shutdown``), and the re-exec ``self_update`` falls back to is replaced by
a recorder, after which the same pid, and so the same cgroup, takes the next job. A job found
started but not done by the next process is failed, not run again, as the job store does
(``failed: interrupted between phases``).

Expected: the checkout's code ends with the job done, the control ref (the code before
OMN-22683, ``9e91ea759^``) ends with the job interrupted. Exit 0 when the observation matches
the expectation, 1 when it does not.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

STOP_TIMEOUT = 5
JOB_SECONDS = 12
OBSERVE_SECONDS = 60
AGENT_DIR = Path(__file__).resolve().parents[2]
BRANCH = "dev"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=True)


def _moved_clone(root: Path) -> Path:
    """A clone of a real origin whose HEAD moved after the process recorded its loaded code."""
    from deploy_agent import loaded_code

    origin, clone = root / "origin", root / "clone"
    origin.mkdir()
    _git(origin, "init", "--initial-branch", BRANCH)
    for repo in (origin,):
        _git(repo, "config", "user.email", "agent@example.invalid")
        _git(repo, "config", "user.name", "harness")
    (origin / "agent.py").write_text("print('v1')\n", encoding="utf-8")
    _git(origin, "add", "agent.py")
    _git(origin, "commit", "-m", "v1")
    _git(root, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "harness")
    os.environ["DEPLOY_AGENT_DIR"] = str(clone)
    os.environ["DEPLOY_AGENT_TRACKING_REF"] = BRANCH
    loaded_code.record_loaded_code_sha(str(clone))
    (origin / "agent.py").write_text("print('v2')\n", encoding="utf-8")
    _git(origin, "add", "agent.py")
    _git(origin, "commit", "-m", "v2")
    _git(clone, "fetch", "origin", BRANCH)
    _git(clone, "reset", "--hard", f"origin/{BRANCH}")
    return clone


def run_agent(state: Path, unit: str) -> int:
    """The stand-in agent: one self-update that restarts its own unit, then the next job."""
    stopping: list[bool] = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    (state / f"start-{len(list(state.glob('start-*'))) + 1}-{os.getpid()}").write_text(
        str(time.time())
    )
    if (state / "job-started").exists():
        # The job store's crash recovery: a job running when the process died is failed.
        (state / "job-interrupted").write_text(str(time.time()))
        return 0
    if not (state / "self-update-ran").exists():
        (state / "self-update-ran").write_text(str(time.time()))
        from deploy_agent.events import EnumSelfUpdateBoundary
        from deploy_agent.executor import DeployExecutor

        _moved_clone(state)

        def reexec(*_: object) -> None:
            (state / "reexec").write_text(str(os.getpid()))

        with (
            patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=unit),
            patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
            patch("os.execv", reexec),
        ):
            DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
        if not (state / "reexec").exists():
            # The update ended with the unit restart under way: wait for the stop and exit.
            (state / "restart-under-way").write_text(str(time.time()))
            deadline = time.time() + 30
            while not stopping and time.time() < deadline:
                time.sleep(0.2)
            return 0
    (state / "job-started").write_text(f"pid={os.getpid()}")
    subprocess.run(["sleep", str(JOB_SECONDS)], check=True)
    (state / "job-done").write_text(str(time.time()))
    return 0


def run_harness(control_ref: str | None) -> int:
    label = "control" if control_ref else "module"
    unit = f"omn22683-harness-{label}-{os.getpid()}.service"
    with tempfile.TemporaryDirectory(prefix="omn22683-harness-") as tmp:
        root = Path(tmp)
        state = root / "state"
        state.mkdir()
        code = AGENT_DIR
        if control_ref:
            top = Path(
                subprocess.run(
                    ["git", "-C", str(AGENT_DIR), "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            code = root / "control" / "scripts" / "deploy-agent"
            code.mkdir(parents=True)
            archive = subprocess.run(
                [
                    "git",
                    "-C",
                    str(top),
                    "archive",
                    control_ref,
                    "scripts/deploy-agent",
                    "config",
                ],
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["tar", "-x", "-C", str(root / "control")],
                input=archive.stdout,
                check=True,
            )
        subprocess.run(
            [
                "systemd-run", "--user", f"--unit={unit}",
                f"--property=TimeoutStopSec={STOP_TIMEOUT}",
                "--property=Restart=on-failure", "--property=RestartSec=1",
                "--property=RestartForceExitStatus=SIGTERM",
                f"--setenv=PYTHONPATH={code}",
                sys.executable, str(Path(__file__).resolve()),
                "--agent", "--state", str(state), "--unit", unit,
            ],
            check=True,
        )  # fmt: skip
        deadline = time.time() + OBSERVE_SECONDS
        while time.time() < deadline and not (
            (state / "job-done").exists() or (state / "job-interrupted").exists()
        ):
            time.sleep(1)
        done = (state / "job-done").exists()
        reached_outcome = (state / "restart-under-way").exists() or (
            state / "reexec"
        ).exists()
        starts = sorted(p.name for p in state.glob("start-*"))
        print(
            f"run={label} control_ref={control_ref} unit={unit} job_done={done} "
            f"job_interrupted={(state / 'job-interrupted').exists()} "
            f"restart_under_way={(state / 'restart-under-way').exists()} "
            f"reexec={(state / 'reexec').exists()}"
        )
        print(f"agent starts: {starts}")
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], check=False)
    expected = control_ref is None
    if not (reached_outcome):
        print("INVALID: self_update ended in neither a restart under way nor a re-exec")
        return 1
    verdict = "OK" if done == expected else "MISMATCH"
    print(f"observed job_done={done}, expected job_done={expected}: {verdict}")
    return 0 if done == expected else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-ref", help="run the deploy_agent package as of this git ref"
    )
    parser.add_argument("--agent", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--state", help=argparse.SUPPRESS)
    parser.add_argument("--unit", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.agent:
        return run_agent(Path(args.state), args.unit)
    return run_harness(args.control_ref)


if __name__ == "__main__":
    sys.exit(main())
