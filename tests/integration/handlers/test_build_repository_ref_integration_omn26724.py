# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build-ref check over a real git clone refuses a missing commit (OMN-26724)."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.integration

ENTRY = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/handlers/build_repository_ref/agent_entry.py"
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return done.stdout.strip()


def _check(repo: Path) -> Any:
    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=scrub_git_location_env(),
        )

    return runpy.run_path(str(ENTRY))["BuildRepositoryRefCheck"](str(repo), run)


def test_real_clone_holds_one_ref_and_lacks_another(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "dev")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    (origin / "f").write_text("1")
    _git(origin, "add", "f")
    _git(origin, "commit", "-q", "-m", "one")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(),
    )
    real = _git(clone, "rev-parse", "HEAD")

    check = _check(clone)
    assert check(real).refuses is False
    missing = check("a" * 40)
    assert missing.refuses is True
    assert missing.verdict.value == "missing"
