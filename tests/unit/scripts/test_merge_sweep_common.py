# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/lib/merge_sweep_common.sh (OMN-14761 / F-22).

The shared bash library replaces hand-typed zsh-fragile probe loops with a
canonical, shellcheck-clean surface: the REPOS array, a for_each_repo iterator,
an opt-in strict-mode switch, and a heredoc'd merge-queue probe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "scripts" / "lib" / "merge_sweep_common.sh"


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_lib_exists() -> None:
    assert LIB.is_file()


def test_lib_is_shellcheck_style_clean() -> None:
    """The canonical helper lib must meet the strict style bar it establishes."""
    result = subprocess.run(
        ["shellcheck", "--severity=style", str(LIB)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_repos_array_is_nonempty_and_canonical() -> None:
    result = _bash(
        'source scripts/lib/merge_sweep_common.sh; printf "%s\\n" "${REPOS[@]}"'
    )
    assert result.returncode == 0, result.stderr
    repos = result.stdout.split()
    assert len(repos) >= 10
    for expected in ("omnibase_infra", "omnimarket", "onex_change_control"):
        assert expected in repos, f"{expected} missing from REPOS"


def test_for_each_repo_visits_every_repo_in_order() -> None:
    result = _bash(
        "source scripts/lib/merge_sweep_common.sh; "
        '_v() { printf "visit:%s\\n" "$1"; }; '
        "for_each_repo _v"
    )
    assert result.returncode == 0, result.stderr
    visited = [
        ln.split(":", 1)[1]
        for ln in result.stdout.splitlines()
        if ln.startswith("visit:")
    ]
    assert visited[0] == "omniclaude"
    assert "onex_change_control" in visited
    assert len(visited) >= 10


def test_for_each_repo_fails_fast_by_default() -> None:
    """A failing command stops the iteration and returns non-zero by default."""
    result = _bash(
        "source scripts/lib/merge_sweep_common.sh; "
        '_fail() { [ "$1" = omniclaude ] && return 7; printf "%s\\n" "$1"; }; '
        "for_each_repo _fail; echo rc=$?"
    )
    assert "rc=7" in result.stdout, result.stdout + result.stderr
    # omniclaude is first and fails, so no other repo is printed.
    assert "omnibase_infra" not in result.stdout


def test_for_each_repo_continue_on_error_visits_all() -> None:
    result = _bash(
        "source scripts/lib/merge_sweep_common.sh; "
        "export MERGE_SWEEP_CONTINUE_ON_ERROR=1; "
        '_maybe() { [ "$1" = omniclaude ] && return 1; printf "%s\\n" "$1"; }; '
        "for_each_repo _maybe; echo rc=$?"
    )
    assert "omnibase_infra" in result.stdout
    assert "onex_change_control" in result.stdout
    assert "rc=1" in result.stdout


def test_msc_strict_is_a_function() -> None:
    result = _bash("source scripts/lib/merge_sweep_common.sh; type -t msc_strict")
    assert result.stdout.strip() == "function"


def test_direct_execution_is_refused() -> None:
    result = subprocess.run(
        ["bash", str(LIB)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "library" in (result.stdout + result.stderr).lower()


def test_no_hardcoded_user_paths() -> None:
    text = LIB.read_text()
    assert "/Users/" not in text
    assert "/Volumes/" not in text
