# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Two runs of the reconciler never share the dispatch venv staging path (OMN-17427).

`<dispatch venv>.rebuilding` is a fixed name. An orphaned run was copying the
live venv into it while the tick's run was deleting it, the venv the tick then
built had no interpreter, and `venv:omnimarket` read DID_NOT_MOVE, which failed
the whole reconcile. The staging pass now takes a lock beside the path whose
target is the holder record, and a second run refuses by name instead of racing
the first.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import (
    _EXIT_FAILED,
    _EXIT_OK,
    _drift_the_interpreter,
    _Workspace,
)

pytestmark = pytest.mark.unit

_VENV_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "reconcile-workspace-venvs.sh"
)


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "workspace")


def _lock_for(staging: Path) -> Path:
    return Path(str(staging) + ".lock")


def _record(pid: int, host: str) -> str:
    return f"pid={pid};host={host};at=2026-10-22T02:48:05Z"


def test_a_live_holder_makes_a_second_run_refuse_without_touching_staging(
    workspace: _Workspace,
) -> None:
    staging = _drift_the_interpreter(workspace)
    staging.mkdir()
    sentinel = staging / "half-copied-by-the-other-run"
    sentinel.write_text("x", encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        lock = _lock_for(staging)
        record = _record(holder.pid, socket.gethostname())
        lock.symlink_to(record)

        result = workspace.run()

        assert result.returncode == _EXIT_FAILED, result.stdout + result.stderr
        assert f"pid {holder.pid}" in result.stdout, result.stdout
        assert sentinel.exists(), "the refused run deleted the live holder's staging"
        assert str(lock.readlink()) == record, (
            "the refused run released a lock it did not hold"
        )
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()


def test_a_holder_on_another_host_is_never_reclaimed(workspace: _Workspace) -> None:
    staging = _drift_the_interpreter(workspace)
    lock = _lock_for(staging)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait()
    record = _record(gone.pid, "some-other-host.invalid")
    lock.symlink_to(record)

    result = workspace.run()

    assert result.returncode == _EXIT_FAILED, result.stdout + result.stderr
    assert str(lock.readlink()) == record


def test_a_dead_holder_is_reclaimed_and_the_run_proceeds(
    workspace: _Workspace,
) -> None:
    staging = _drift_the_interpreter(workspace)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait()
    _lock_for(staging).symlink_to(_record(gone.pid, socket.gethostname()))

    result = workspace.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr


def test_the_lock_is_released_when_the_run_ends(workspace: _Workspace) -> None:
    staging = _drift_the_interpreter(workspace)

    result = workspace.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert not _lock_for(staging).is_symlink(), (
        "the staging lock survived the run, so every later tick would be refused"
    )


def _staging_lock_field_source() -> str:
    source = _VENV_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^staging_lock_field\(\) \{\n.*?^\}\n", source, re.M | re.S)
    assert match, "reconcile-workspace-venvs.sh defines no staging_lock_field()"
    return match.group(0)


def test_the_holder_record_is_parsed_through_process_substitution() -> None:
    function = _staging_lock_field_source()
    assert "< <(" in function, function
    assert "<<<" not in function, (
        "staging_lock_field feeds read through a here-string, which can deadlock "
        "on a full pipe (OMN-19709)"
    )
    record = _record(4242, "lab-host") + ";note=" + "x" * 200 * 1024
    # The record goes in on stdin: Linux refuses any single argv string over
    # 128 KiB (MAX_ARG_STRLEN) with E2BIG, so a 200 KiB argument never runs.
    fields = subprocess.run(
        [
            "bash",
            "-c",
            function + 'record="$(cat)"; for key in pid host at; do '
            'staging_lock_field "$record" "$key"; printf \'\\n\'; done',
        ],
        input=record,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.splitlines()
    assert fields == ["4242", "lab-host", "2026-10-22T02:48:05Z"]
