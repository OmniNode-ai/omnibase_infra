# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the entrypoint's root phase; mount setup must fail before registration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
WORKSPACE_VARS = ("RUNNER_HOME", "RUNNER_WORK_DIR", "RUNNER_USER_HOME")
ENTRYPOINT = Path(__file__).resolve().parents[3] / "docker/runners/entrypoint.sh"


def _run(
    tmp_path: Path,
    *,
    uid: int = 0,
    chown_status: int = 0,
    mkdir_fails: bool = False,
    default_workspace: bool = False,
) -> subprocess.CompletedProcess[str]:
    # Redirect the production paths into the temporary tree. Only privileged
    # commands are substituted; bash executes the actual startup control flow.
    marker = "# Required environment variables"
    text = ENTRYPOINT.read_text()
    assert text.count(marker) == 1, "entrypoint marker moved; tested prefix is stale"
    prefix = text.split(marker, 1)[0]
    assert 'mkdir -p "${RUNNER_USER_HOME}/.cache"' in prefix, (
        "mount setup left the prefix"
    )
    assert 'RUNNER_USER_HOME="${RUNNER_USER_HOME:-' in prefix, "home default moved"
    commands = (
        'id() { printf "%s\\n" "$TEST_UID"; }\n'
        'chown() { printf "%s\\n" "$@" > "$CHOWN_CALLS"; '
        'if [[ "$CHOWN_STATUS" != 0 ]]; then '
        'echo "chown: read-only mount" >&2; return "$CHOWN_STATUS"; fi; }\n'
    )
    if mkdir_fails:
        commands += 'mkdir() { echo "mkdir: read-only mount" >&2; return 74; }\n'
    env = {k: v for k, v in os.environ.items() if k not in WORKSPACE_VARS}
    env.update(RUNNER_USER_HOME=str(tmp_path / "home"))
    if not default_workspace:
        env.update(
            RUNNER_HOME=str(tmp_path / "runner home"), RUNNER_WORK_DIR="custom work"
        )
    return subprocess.run(
        ["bash", "-c", commands + prefix + '\necho "registration reached"\n'],
        env={
            **env,
            "TEST_UID": str(uid),
            "CHOWN_STATUS": str(chown_status),
            "CHOWN_CALLS": str(tmp_path / "chown.calls"),
            "DOCKER_SOCKET_PATH": str(tmp_path / "no-socket"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_missing_mount_directories_are_created_before_chown(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "home/.cache").is_dir()
    assert (tmp_path / "runner home/custom work").is_dir()
    assert (tmp_path / "chown.calls").read_text().splitlines() == [
        "runner:runner",
        str(tmp_path / "home/.cache"),
        str(tmp_path / "runner home/custom work"),
    ]
    assert "registration reached" in result.stdout


def test_chown_failure_stops_startup_and_preserves_diagnostic(tmp_path: Path) -> None:
    result = _run(tmp_path, chown_status=73)
    assert result.returncode == 73
    assert "chown: read-only mount" in result.stderr
    assert "registration reached" not in result.stdout


def test_mkdir_failure_stops_before_chown_and_registration(tmp_path: Path) -> None:
    result = _run(tmp_path, mkdir_fails=True)
    assert result.returncode == 74
    assert "mkdir: read-only mount" in result.stderr
    assert not (tmp_path / "chown.calls").exists()
    assert "registration reached" not in result.stdout


def test_nonroot_startup_does_not_try_to_change_mount_ownership(tmp_path: Path) -> None:
    result = _run(tmp_path, uid=1000, chown_status=73, mkdir_fails=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "chown.calls").exists()
    assert not (tmp_path / "home/.cache").exists()
    assert "registration reached" in result.stdout


def test_default_workspace_is_created_and_chowned(tmp_path: Path) -> None:
    result = _run(tmp_path, default_workspace=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "home/actions-runner/_work").is_dir()
    assert (tmp_path / "chown.calls").read_text().splitlines() == [
        "runner:runner",
        str(tmp_path / "home/.cache"),
        str(tmp_path / "home/actions-runner/_work"),
    ]
