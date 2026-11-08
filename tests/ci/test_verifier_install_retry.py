# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned DoD verifier install retries a download that a provider answers with an error.

On 2026-11-08 at 10:23:38Z, ``Tests (Split 2/15)`` and eight other splits of
omnibase_infra#5702 failed in the step "Install the pinned DoD verifier before pytest
times it (OMN-19998)" before any pytest ran: ``uv venv --python 3.13`` could not
download cpython from GitHub (``HTTP status server error (504 Gateway Timeout)``) and
the install script stopped on the first failure. The install script now runs each of
its three network commands (the tag fetch, the venv, the lockfile sync) up to three
times, waiting 10 s and then 20 s, and still refuses a moved tag at once.

These tests run the real install script with ``git``, ``uv`` and ``sleep`` replaced by
recording stand-ins on ``PATH``; the 504 text is the one the CI job printed.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.ci import check_deploy_scope_dod as hook

pytestmark = pytest.mark.unit

_TAG = "dev-20261101000200-949993edf3b7"
_COMMIT = "949993edf3b7" + "0" * 28
_UV_504 = (
    "error: Failed to download\n"
    "  | https://github.com/astral-sh/python-build-standalone/releases/download/"
    "20250409/cpython-3.13.3%2B20250409-x86_64-unknown-linux-gnu-install_only_stripped"
    ".tar.gz\n"
    "  `-> HTTP status server error (504 Gateway Timeout) for url\n"
)
_GIT_503 = "fatal: The requested URL returned error: 503\n"

_GIT = """#!/usr/bin/env bash
echo "git $*" >> "$STUB_LOG"
count() { n=$(cat "$STUB_STATE/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$STUB_STATE/$1"; echo "$n"; }
case "$*" in
  init*) mkdir -p "${@: -1}" ;;
  *" fetch "*)
    n=$(count fetch)
    if (( n < ${STUB_FETCH_FAILS:-0} )); then echo "$STUB_GIT_ERROR" >&2; exit 128; fi ;;
  *rev-parse*) echo "$STUB_COMMIT" ;;
esac
"""
_UV = """#!/usr/bin/env bash
echo "uv $*" >> "$STUB_LOG"
count() { n=$(cat "$STUB_STATE/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$STUB_STATE/$1"; echo "$n"; }
if [[ "$1" == venv ]]; then
  n=$(count venv)
  if (( n < ${STUB_VENV_FAILS:-0} )); then printf '%s' "$STUB_UV_ERROR" >&2; exit 2; fi
  mkdir -p "${@: -1}/bin"
fi
"""
_SLEEP = """#!/usr/bin/env bash
echo "sleep $*" >> "$STUB_LOG"
"""


class _Run:
    def __init__(self, tmp_path: Path) -> None:
        self.entry = tmp_path / "cache" / _TAG
        self.state = tmp_path / "state"
        self.log = tmp_path / "calls.log"
        self.state.mkdir()
        self.log.write_text("")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (("git", _GIT), ("uv", _UV), ("sleep", _SLEEP)):
            path = bin_dir / name
            path.write_text(body, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        self.env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "VERIFIER_REPO": hook._VERIFIER_REPO,
            "STUB_LOG": str(self.log),
            "STUB_STATE": str(self.state),
            "STUB_COMMIT": _COMMIT,
            "STUB_GIT_ERROR": _GIT_503,
            "STUB_UV_ERROR": _UV_504,
        }

    def run(self, **env: object) -> subprocess.CompletedProcess[str]:
        self.entry.parent.mkdir(parents=True, exist_ok=True)
        merged = {**self.env, **{k: str(v) for k, v in env.items()}}
        return subprocess.run(
            [
                "bash",
                "-c",
                hook._VERIFIER_INSTALL_SCRIPT,
                "install",
                _TAG,
                str(self.entry),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=merged,
            cwd=os.fspath(self.state),
        )

    def calls(self, prefix: str) -> list[str]:
        return [
            line
            for line in self.log.read_text().splitlines()
            if line.startswith(prefix)
        ]

    @property
    def installed(self) -> bool:
        return (self.entry / "installed").is_file()


@pytest.fixture
def stubs(tmp_path: Path) -> _Run:
    return _Run(tmp_path)


def test_a_uv_download_504_is_retried_until_the_venv_is_made(stubs: _Run) -> None:
    done = stubs.run(STUB_VENV_FAILS=2)

    assert done.returncode == 0, done.stderr
    assert stubs.installed
    assert len(stubs.calls("uv venv")) == 3
    assert stubs.calls("sleep") == ["sleep 10", "sleep 20"]


def test_a_git_fetch_503_is_retried_once_and_the_install_succeeds(
    stubs: _Run,
) -> None:
    done = stubs.run(STUB_FETCH_FAILS=1)

    assert done.returncode == 0, done.stderr
    assert stubs.installed
    assert len([c for c in stubs.calls("git") if " fetch " in c]) == 2
    assert stubs.calls("sleep") == ["sleep 10"]


def test_a_504_that_persists_fails_after_three_attempts_with_the_error(
    stubs: _Run,
) -> None:
    done = stubs.run(STUB_VENV_FAILS=99)

    assert done.returncode != 0
    assert "504 Gateway Timeout" in done.stderr
    assert not stubs.installed
    assert len(stubs.calls("uv venv")) == 3
    assert stubs.calls("sleep") == ["sleep 10", "sleep 20"]


def test_a_moved_tag_is_refused_at_once_and_never_retried(stubs: _Run) -> None:
    done = stubs.run(STUB_COMMIT="f" * 40)

    assert done.returncode == 1
    assert "the tag was moved" in done.stderr
    assert not stubs.installed
    assert len([c for c in stubs.calls("git") if " fetch " in c]) == 1
    assert stubs.calls("sleep") == []
    assert stubs.calls("uv") == []


def test_a_finished_install_is_reused_without_any_network_command(
    stubs: _Run,
) -> None:
    stubs.entry.mkdir(parents=True)
    (stubs.entry / "installed").write_text("tag\n", encoding="utf-8")

    done = stubs.run()

    assert done.returncode == 0
    assert stubs.calls("git") == []
    assert stubs.calls("uv") == []
