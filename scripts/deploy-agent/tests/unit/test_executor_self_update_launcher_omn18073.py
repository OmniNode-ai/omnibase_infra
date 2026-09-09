# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""self_update must re-exec THROUGH the launcher, not the bare interpreter.

OMN-18073. ``os.execv`` replaces the process image but inherits the caller's
environment verbatim. Re-execing ``sys.executable`` therefore carries the
environment the process STARTED with forward forever, however many times it
updates.

That is not a theoretical property. On 2026-09-09 the dev agent (pid 2171795,
started 01:38:54Z) re-execed four times -- 01:55:03Z, 09:05:01Z, 12:41:12Z and
12:54:36Z -- advancing its code from ``b0b46c18`` to ``c65d8a8b`` exactly as
designed, with ``NRestarts=0`` and an unchanged ``ExecMainStartTimestamp``
throughout (``os.execv`` keeps the pid, so systemd sees no restart). The
mangled ``ONEXBOT_OCC_PRIVATE_KEY`` it had inherited from systemd's
``EnvironmentFile=`` at 01:38:54Z survived every one of them. Only a systemd
restart re-read the file.

Re-execing ``deploy/deploy-agent-launch.sh`` re-``source``s the operator env
store, so a repaired or rotated value is picked up at the next job boundary.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor

_BOUNDARY = EnumSelfUpdateBoundary.POST_TERMINAL

_SHA_LOCAL = "aaaaaaaabbbbbbbb"
_SHA_REMOTE = "ccccccccdddddddd"

_LAUNCHER = (
    "/data/omninode/omnibase_infra/scripts/deploy-agent/deploy/deploy-agent-launch.sh"
)


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _behind_git(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
    """Clean tree, fetch ok, HEAD behind the tracking ref, pull + uv sync ok."""
    if "status" in cmd and "--porcelain" in cmd:
        return _ok("")
    if "rev-parse" in cmd:
        if any(part.startswith("origin/") for part in cmd):
            return _ok(_SHA_REMOTE)
        return _ok(_SHA_LOCAL)
    return _ok()


def test_re_execs_the_launcher_when_the_launcher_started_the_process(
    monkeypatch,
) -> None:
    """GREEN: DEPLOY_AGENT_LAUNCHER is set, so the launcher is what gets exec'd."""
    monkeypatch.setenv("DEPLOY_AGENT_LAUNCHER", _LAUNCHER)
    monkeypatch.delenv("DEPLOY_AGENT_NO_SELF_UPDATE", raising=False)
    monkeypatch.setattr(sys, "argv", ["/path/to/deploy_agent/__main__.py", "--verbose"])

    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=_behind_git),
        patch("deploy_agent.executor.os.execv") as mock_execv,
    ):
        executor.self_update(boundary=_BOUNDARY)

    mock_execv.assert_called_once_with(_LAUNCHER, [_LAUNCHER, "--verbose"])
    # argv[0] is the module path systemd never passed; forwarding it to the
    # launcher would hand `python -m deploy_agent` a stray positional argument.
    assert "__main__.py" not in mock_execv.call_args.args[1]


def test_re_execs_the_interpreter_when_no_launcher_is_declared(monkeypatch) -> None:
    """Negative control: the pre-existing path is unchanged off the unit.

    A developer running ``python -m deploy_agent`` directly, and the container
    mode's supervisor respawn, both arrive here with no launcher. Without this
    control the test above would pass on a build that simply always re-execs
    that one hardcoded path.
    """
    monkeypatch.delenv("DEPLOY_AGENT_LAUNCHER", raising=False)
    monkeypatch.delenv("DEPLOY_AGENT_NO_SELF_UPDATE", raising=False)
    monkeypatch.setattr(sys, "argv", ["/path/to/deploy_agent/__main__.py", "--verbose"])

    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=_behind_git),
        patch("deploy_agent.executor.os.execv") as mock_execv,
    ):
        executor.self_update(boundary=_BOUNDARY)

    mock_execv.assert_called_once_with(
        sys.executable,
        [sys.executable, "/path/to/deploy_agent/__main__.py", "--verbose"],
    )


def test_up_to_date_never_re_execs_the_launcher(monkeypatch) -> None:
    """The boundary contract is unchanged: no head change, no re-exec.

    Re-execing on every boundary would re-read the env store more eagerly but
    would also restart the process on every polled command, which is the
    mid-deploy re-exec failure OMN-16442 removed.
    """
    monkeypatch.setenv("DEPLOY_AGENT_LAUNCHER", _LAUNCHER)
    monkeypatch.delenv("DEPLOY_AGENT_NO_SELF_UPDATE", raising=False)

    def _current(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        if "status" in cmd and "--porcelain" in cmd:
            return _ok("")
        if "rev-parse" in cmd:
            return _ok(_SHA_LOCAL)
        return _ok()

    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=_current),
        patch("deploy_agent.executor.os.execv") as mock_execv,
    ):
        executor.self_update(boundary=_BOUNDARY)

    mock_execv.assert_not_called()
