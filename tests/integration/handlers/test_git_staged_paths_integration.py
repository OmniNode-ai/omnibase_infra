# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: staged_paths reads a real git repository (OMN-22867)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.git_staged_paths import GitDiscoveryError, staged_paths

pytestmark = pytest.mark.integration

_TIMEOUT = 30.0


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
    )


def test_staged_paths_reads_real_repository(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.sql").write_text("select 1;\n")
    (tmp_path / "b.txt").write_text("unstaged\n")
    _git(tmp_path, "add", "a.sql")
    assert staged_paths(tmp_path, _TIMEOUT) == ("a.sql",)


def test_staged_paths_outside_repository_raises_discovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(GitDiscoveryError) as excinfo:
        staged_paths(tmp_path, _TIMEOUT)
    assert "not a git repository" in excinfo.value.detail
