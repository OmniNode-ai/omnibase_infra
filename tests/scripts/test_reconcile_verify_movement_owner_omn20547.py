# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The owner of a refused clone's dirty paths comes from the ledger reader (OMN-20547).

``clone-refusal`` names the lane that owns the dirty tracked paths of a refused
canonical clone. It used to open the ledger file and take the newest CLAIM row
that named the clone and a path. A file copy is a second ledger, so it now asks
the reader node (reader ``reconcile-verify-movement``, query ``open-claims``) and
never opens the file. A reader that cannot answer (exit 3 or 69) is an unknown
owner, never a guess and never a file read.

Each test drives the REAL script over a throwaway clone; only the bus read is
replaced, by the stub interpreter in ``ledger_reader_stub``. The ledger file the
old code read is planted with a DIFFERENT owner, so the old reader fails the
assertions on the reader's rows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.scripts.ledger_reader_stub import LedgerReaderStub

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "reconcile_verify_movement.py"
REPO = "omnibase_fixture"
DIRTY = "docs/plan.md"


def _git(clone: Path, *args: str) -> None:
    env = dict(os.environ)
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    subprocess.run(
        ["git", "-C", str(clone), *args],
        env=scrub_git_location_env(env),
        check=True,
        capture_output=True,
    )


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, LedgerReaderStub]:
    clone = tmp_path / REPO
    (clone / "docs").mkdir(parents=True)
    _git(clone, "init", "-q", "-b", "main")
    (clone / DIRTY).write_text("one\n", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-q", "-m", "seed")
    (clone / DIRTY).write_text("two\n", encoding="utf-8")
    return clone, LedgerReaderStub(tmp_path / "ledger-reader")


def _refusal(
    tmp_path: Path,
    clone: Path,
    reader: LedgerReaderStub,
    *extra: str,
    reader_command: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("ONEX_LEDGER_PATH", None)
    env.update(reader.env())
    if not reader_command:
        env.pop("ONEX_LEDGER_READER_COMMAND")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "clone-refusal",
            "--clone",
            str(clone),
            "--repo",
            REPO,
            "--state-dir",
            str(tmp_path / "state"),
            "--lock-age-s",
            "600",
            *extra,
        ],
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _file_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ROLLING_WORK_LEDGER.md"
    path.write_text(
        f"2026-09-24T10:00:00Z | CLAIM | lane=file-lane | {REPO} {DIRTY}\n",
        encoding="utf-8",
    )
    return path


def test_owner_is_the_newest_open_claim_the_reader_returns(
    world: tuple[Path, LedgerReaderStub], tmp_path: Path
) -> None:
    clone, reader = world
    reader.set_claims(
        ("2026-11-20T10:00:00Z", "older-lane", "OMN-1", f"{REPO} {DIRTY}", True),
        ("2026-11-21T10:00:00Z", "reader-lane", "OMN-2", f"{REPO} {DIRTY}", True),
        (
            "2026-11-21T11:00:00Z",
            "other-repo-lane",
            "OMN-3",
            f"elsewhere {DIRTY}",
            True,
        ),
    )
    result = _refusal(tmp_path, clone, reader, "--ledger", str(_file_ledger(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert "owner\towner: reader-lane" in result.stdout, result.stdout
    assert "file-lane" not in result.stdout, result.stdout
    calls = reader.calls()
    assert any(
        "--reader reconcile-verify-movement --query open-claims --window-hours 72" in c
        for c in calls
    ), calls


def test_a_closed_claim_does_not_own_the_paths(
    world: tuple[Path, LedgerReaderStub], tmp_path: Path
) -> None:
    clone, reader = world
    reader.set_claims(
        ("2026-11-21T10:00:00Z", "done-lane", "OMN-2", f"{REPO} {DIRTY}", False),
    )
    result = _refusal(tmp_path, clone, reader, "--ledger", str(_file_ledger(tmp_path)))
    assert "owner\towner: unowned" in result.stdout, result.stdout


def test_the_owner_needs_no_ledger_file(
    world: tuple[Path, LedgerReaderStub], tmp_path: Path
) -> None:
    clone, reader = world
    reader.set_claims(
        ("2026-11-21T10:00:00Z", "reader-lane", "OMN-2", f"{REPO} {DIRTY}", True),
    )
    result = _refusal(tmp_path, clone, reader)
    assert "owner\towner: reader-lane" in result.stdout, result.stdout


@pytest.mark.parametrize("code", [3, 69])
def test_a_reader_that_cannot_answer_leaves_the_owner_unknown(
    world: tuple[Path, LedgerReaderStub], tmp_path: Path, code: int
) -> None:
    clone, reader = world
    reader.refuse_with(code)
    result = _refusal(tmp_path, clone, reader, "--ledger", str(_file_ledger(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert "owner\towner: unknown" in result.stdout, result.stdout
    assert f"reader exit {code}" in result.stdout, result.stdout
    assert "file-lane" not in result.stdout, result.stdout
    assert "unowned" not in result.stdout, result.stdout


def test_no_reader_command_leaves_the_owner_unknown(
    world: tuple[Path, LedgerReaderStub], tmp_path: Path
) -> None:
    clone, reader = world
    result = _refusal(
        tmp_path,
        clone,
        reader,
        "--ledger",
        str(_file_ledger(tmp_path)),
        reader_command=False,
    )
    assert "owner\towner: unknown" in result.stdout, result.stdout
    assert "ONEX_LEDGER_READER_COMMAND is not set" in result.stdout, result.stdout
    assert "file-lane" not in result.stdout, result.stdout
