# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22478 -- a self-update that restarts its own unit does not re-exec the process systemd is stopping.

Measured on the lab dev deploy agent on 2026-11-06: after the deploy of 5e69171dd the agent synced its unit and ran
``systemctl --user restart deploy-agent-dev.service`` from inside that unit. systemd stops the unit by sending SIGTERM to
every process of its cgroup, the restart command included, so the command died with SIGTERM and ``self_update`` logged
``unit sync/restart failed ... died with <Signals.SIGTERM: 15>`` and went on to re-exec in the SAME pid. The process the
stop job waits for never exited, the stop job ran into its 90 s timeout (09:12:50 to 09:14:21 local) and systemd killed
the whole cgroup with SIGKILL -- including the docker build of the next job, which the journal recorded as ``Recovered 1
crashed job(s)`` and the job store as ``failed: interrupted between phases``. Every self-update that changes the unit
file fails the job that follows it.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import loaded_code
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor

from .test_self_update_loaded_code_omn18200 import (
    TRACKING_BRANCH,
    _advance_origin,
    _git,
    _make_origin_and_clone,
)

pytestmark = pytest.mark.unit

UNIT = "deploy-agent-dev.service"


def _clone_moved_under_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, clone = _make_origin_and_clone(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    loaded_code.record_loaded_code_sha(str(clone))
    _advance_origin(origin, "print('v2')\n")
    _git(clone, "fetch", "origin", TRACKING_BRANCH)
    _git(clone, "reset", "--hard", f"origin/{TRACKING_BRANCH}")


def _restart_ends_with(returncode: int):
    """``subprocess.run(check=True)`` as the executor calls it: a non-zero return code raises."""

    def run(cmd, *args, **kwargs):
        if cmd[:3] == ["systemctl", "--user", "restart"] and returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_golden_a_restart_command_killed_by_the_unit_stop_does_not_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    with (
        caplog.at_level("INFO", logger="deploy_agent.executor"),
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=UNIT),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("subprocess.run", side_effect=_restart_ends_with(-signal.SIGTERM)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()
    assert "unit restart failed" in caplog.text


def test_error_a_restart_that_failed_for_another_reason_still_falls_back_to_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    with (
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=UNIT),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("subprocess.run", side_effect=_restart_ends_with(1)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_called_once()


def test_control_a_restart_that_returned_cleanly_ends_the_update_without_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clone_moved_under_the_process(tmp_path, monkeypatch)
    with (
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=UNIT),
        patch("subprocess.run", side_effect=_restart_ends_with(0)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()
