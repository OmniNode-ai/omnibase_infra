# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22683 -- a self-update restarts its own unit from outside the unit.

Measured on the dev deploy agent on 2026-11-06: after the deploy of 5e69171dd the agent synced its unit and ran
``systemctl --user restart deploy-agent-dev.service`` from inside that unit. systemd stops the unit by sending SIGTERM to
every process of its cgroup, the restart command included, so the command died with SIGTERM and ``self_update`` logged
``unit sync/restart failed ... died with <Signals.SIGTERM: 15>`` and went on to re-exec in the SAME pid. The process the
stop job waits for never exited, the stop job ran into its 90 s timeout (09:12:50 to 09:14:21 local) and systemd killed
the whole cgroup with SIGKILL, including the docker build of the next job, which the journal recorded as ``Recovered 1
crashed job(s)`` and the job store as ``failed: interrupted between phases``.

The restart now goes through a transient ``systemd-run --user`` unit, which lives in its own cgroup, and a restart that
ends 0 or -SIGTERM ends the update without a re-exec.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import loaded_code
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import SELF_RESTART_COMMAND, DeployExecutor

from .test_self_update_loaded_code_omn18200 import (
    TRACKING_BRANCH,
    _advance_origin,
    _git,
    _make_origin_and_clone,
)

pytestmark = pytest.mark.unit

UNIT = "deploy-agent-dev.service"
SYNC = "deploy_agent.executor._sync_own_unit_after_pull"


def _clone_moved_under_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, clone = _make_origin_and_clone(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    loaded_code.record_loaded_code_sha(str(clone))
    _advance_origin(origin, "print('v2')\n")
    _git(clone, "fetch", "origin", TRACKING_BRANCH)
    _git(clone, "reset", "--hard", f"origin/{TRACKING_BRANCH}")


def _restart_ends_with(returncode: int, seen: list[list[str]] | None = None):
    def run(cmd, *args, **kwargs):
        is_restart = cmd[0] == "systemd-run" or cmd[:3] == [
            "systemctl",
            "--user",
            "restart",
        ]
        if seen is not None and is_restart:
            seen.append(cmd)
        code = returncode if is_restart else 0
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr="")

    return run


def test_restart_command_runs_in_a_transient_unit_outside_the_agent_unit() -> None:
    command = SELF_RESTART_COMMAND.split()
    assert command[:2] == ["systemd-run", "--user"]
    assert "--wait" in command
    assert "--collect" in command
    assert command[-3:] == ["systemctl", "--user", "restart"]


def test_golden_sigterm_restart_killed_by_the_unit_stop_does_not_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    seen: list[list[str]] = []
    with (
        caplog.at_level("INFO", logger="deploy_agent.executor"),
        patch(SYNC, return_value=UNIT),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("subprocess.run", side_effect=_restart_ends_with(-signal.SIGTERM, seen)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()
    assert "restarting via systemd" in caplog.text
    assert len(seen) == 1
    assert seen[0][-4:] == ["systemctl", "--user", "restart", UNIT]
    assert seen[0][:2] == ["systemd-run", "--user"]


def test_clean_restart_ends_the_update_without_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    with (
        patch(SYNC, return_value=UNIT),
        patch("subprocess.run", side_effect=_restart_ends_with(0)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()


def test_error_restart_that_fails_for_another_reason_falls_back_to_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    with (
        patch(SYNC, return_value=UNIT),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("subprocess.run", side_effect=_restart_ends_with(1)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_called_once()


def test_error_host_without_systemd_run_falls_back_to_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)

    def no_systemd_run(cmd, *args, **kwargs):
        if cmd[0] == "systemd-run":
            raise FileNotFoundError(2, "No such file or directory", "systemd-run")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch(SYNC, return_value=UNIT),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("subprocess.run", side_effect=no_systemd_run),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_called_once()
