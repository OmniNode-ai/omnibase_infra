# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22635: one live-contact exception file per pull request.

The local live-contact hook read its exception from the one shared file
``.github/live-contact-exception.md``, so every change to CI tooling appended a
line to it and any two conflicted. A staged file under
``.github/live-contact-exception/`` now carries the exception, one file per
change, so two changes add different files.

* AC1: a staged carrier under the directory with a stated reason admits a change.
* AC2: no carrier, an unstaged one, or a reason that is too short is refused.
* AC3: two branches that each add a different carrier file merge without a
  conflict, and two appends to the shared file conflict.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_ci_live_contact import HandlerCILiveContact
from omnibase_infra.models.model_ci_live_contact_request import (
    ModelCILiveContactRequest,
)

pytestmark = pytest.mark.unit

CARRIER_DIR = ".github/live-contact-exception"
SHARED = ".github/live-contact-exception.md"
REASON = (
    "Live-contact exception: This changes a workflow display label; "
    "there is no API or runner behaviour to replay.\n"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )


@pytest.fixture
def change(tmp_path: Path) -> Path:
    """A staged change to CI tooling with no recorded contact."""
    assert _git(tmp_path, "init", "--quiet").returncode == 0
    assert _git(tmp_path, "commit", "--allow-empty", "-m", "base").returncode == 0
    (tmp_path / "scripts/ci").mkdir(parents=True)
    (tmp_path / "scripts/ci/check.py").write_text("print('gate')\n")
    assert _git(tmp_path, "add", ".").returncode == 0
    return tmp_path


async def _admitted(change: Path) -> bool:
    result = await HandlerCILiveContact().handle(
        ModelCILiveContactRequest(repository=str(change))
    )
    return result.success


async def test_a_staged_carrier_under_the_directory_admits_the_change(
    change: Path,
) -> None:
    carrier = change / CARRIER_DIR / "OMN-22635.md"
    carrier.parent.mkdir(parents=True)
    carrier.write_text(REASON)
    assert _git(change, "add", str(carrier)).returncode == 0
    assert await _admitted(change)


async def test_no_carrier_is_refused(change: Path) -> None:
    assert not await _admitted(change)


async def test_an_unstaged_carrier_is_refused(change: Path) -> None:
    carrier = change / CARRIER_DIR / "OMN-22635.md"
    carrier.parent.mkdir(parents=True)
    carrier.write_text(REASON)
    assert not await _admitted(change)


async def test_a_carrier_with_too_short_a_reason_is_refused(change: Path) -> None:
    carrier = change / CARRIER_DIR / "OMN-22635.md"
    carrier.parent.mkdir(parents=True)
    carrier.write_text("Live-contact exception: impossible\n")
    assert _git(change, "add", str(carrier)).returncode == 0
    assert not await _admitted(change)


async def test_a_file_outside_the_directory_does_not_count(change: Path) -> None:
    other = change / ".github" / "live-contact-exceptions-extra" / "OMN-22635.md"
    other.parent.mkdir(parents=True)
    other.write_text(REASON)
    assert _git(change, "add", str(other)).returncode == 0
    assert not await _admitted(change)


def _two_branches(tmp_path: Path, *, shared: bool) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "--quiet", "-b", "dev").returncode == 0
    (repo / ".github").mkdir()
    (repo / SHARED).write_text(
        "Live-contact exception: an older reason for a change.\n"
    )
    assert _git(repo, "add", ".").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "base").returncode == 0
    for branch in ("a", "b"):
        assert _git(repo, "checkout", "-q", "-b", branch, "dev").returncode == 0
        if shared:
            with (repo / SHARED).open("a") as handle:
                handle.write(f"Live-contact exception: reason for branch {branch}.\n")
        else:
            carrier = repo / CARRIER_DIR / f"OMN-{branch}.md"
            carrier.parent.mkdir(parents=True, exist_ok=True)
            carrier.write_text(REASON)
        assert _git(repo, "add", ".").returncode == 0
        assert _git(repo, "commit", "-q", "-m", branch).returncode == 0
    assert _git(repo, "checkout", "-q", "a").returncode == 0
    return _git(repo, "merge", "--no-edit", "b")


def test_two_branches_adding_different_carrier_files_merge_cleanly(
    tmp_path: Path,
) -> None:
    merged = _two_branches(tmp_path, shared=False)
    assert merged.returncode == 0, merged.stdout + merged.stderr


def test_two_branches_appending_to_the_shared_file_conflict(tmp_path: Path) -> None:
    """Failing control: the old habit conflicts."""
    merged = _two_branches(tmp_path, shared=True)
    assert merged.returncode != 0
    assert "CONFLICT" in merged.stdout
