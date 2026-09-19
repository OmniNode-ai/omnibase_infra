# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16729 -- a ledger may not sit directly in the root of a git repository.

The defect. A lane brief spelled the documented append recipe with an
un-substituted placeholder where the ledger path belongs. A worker passed that
placeholder through literally, so ``ledger_lock.py`` was handed a bare name and
did exactly as it was told: it CREATED a file of that name at the repository
root and appended to it, reporting success. Three rows landed there across five
days -- two on 2026-09-14 and one on 2026-09-19 -- and every one of them was
invisible to every reader of the real ledger, because nothing reads a file at
the repository root.

What is asserted here is the shape of the refusal, in both directions:

* RED -- the literal placeholder name, and an arbitrary other bare name, are
  both refused at a repository root, with no file created and nothing written;
* POSITIVE CONTROL -- a tracking-shaped ledger and an archive-shaped ledger
  inside the same repository still append normally.

The positive control is the half that makes the red meaningful. A guard that
refused everything would produce the same two red results, so a zero here is
only evidence when the same harness is shown returning rows.

The guard is deliberately POSITIONAL rather than nominal: it refuses by where
the file sits, never by what it is called. Naming the accepted ledger would
hardcode a tracking path into a tool whose claim is that it is told which
ledger to protect and resolves none itself (OMN-17235), so
``test_the_guard_names_no_particular_ledger`` pins that too.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ledger_lock.py"

# The exit code the refusal owns. A distinct code matters: a caller that
# already treats 75 as "locked" and 76 as "bad row" must be able to tell a
# misplaced ledger apart from both without parsing prose.
EXIT_LEDGER_PATH = 77


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _row(kind: str = "NOTE") -> str:
    """A row stamped from the live clock.

    The wall-clock guard (OMN-17427) refuses a row frozen at a date literal,
    and the row-shape guard (OMN-18801) refuses one that does not open with a
    date. Neither is what this module is about, so its rows must be ones the
    tool admits anywhere.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp} | {kind} | lane=omn16729-test | actor=agent | placement probe"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway tree with a ``.git`` marker.

    A marker is enough: the guard asks where the repository root is, which is
    a filesystem question the tool answers without invoking git -- the same
    way it answers inside a worktree, where ``.git`` is a file rather than a
    directory.
    """
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- #
# RED -- the misplaced ledger
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("ledger", id="the-literal-placeholder-name"),
        pytest.param("SOME_OTHER_FILE.md", id="an-arbitrary-repo-root-path"),
    ],
)
def test_a_repo_root_ledger_is_refused_and_no_file_is_created(
    repo: Path, name: str
) -> None:
    target = repo / name
    result = _run(str(target), "--append", _row())

    assert result.returncode == EXIT_LEDGER_PATH, (
        "a ledger at the repository root must be refused with the placement "
        f"exit code\n{result.stdout}{result.stderr}"
    )
    assert not target.exists(), (
        "the refusal created the very file it refuses -- a refusal that writes "
        "is not a refusal"
    )
    assert "LEDGER PATH REFUSED" in result.stderr, result.stderr


@pytest.mark.unit
def test_an_existing_repo_root_ledger_is_refused_without_being_appended_to(
    repo: Path,
) -> None:
    """The second half of the live defect.

    The first bad append CREATED the file; every later one merely extended a
    file that by then existed. A guard scoped to creation would have caught the
    first and waved the rest through, which is the case that actually recurred.
    """
    target = repo / "ledger"
    target.write_text("seeded\n", encoding="utf-8")

    result = _run(str(target), "--append", _row())

    assert result.returncode == EXIT_LEDGER_PATH, (
        f"an EXISTING repo-root ledger must be refused too\n"
        f"{result.stdout}{result.stderr}"
    )
    assert target.read_text(encoding="utf-8") == "seeded\n", (
        "the refusal appended to the misplaced ledger anyway"
    )


@pytest.mark.unit
def test_the_refusal_precedes_the_lock(repo: Path) -> None:
    """Nothing beside the ledger is created either.

    ``LedgerLock`` writes its lock directory next to the ledger it protects, so
    a refusal evaluated after the lock was taken would leave that directory
    behind at the repository root -- a second stray artifact produced by the
    guard against stray artifacts.
    """
    target = repo / "ledger"
    before = sorted(p.name for p in repo.iterdir())

    result = _run(str(target), "--append", _row())

    assert result.returncode == EXIT_LEDGER_PATH, result.stderr
    assert sorted(p.name for p in repo.iterdir()) == before, (
        "the refusal left something behind at the repository root"
    )


@pytest.mark.unit
def test_a_repo_root_ledger_is_refused_for_a_wrapped_command(repo: Path) -> None:
    """Every action is judged, not only ``--append``.

    The lock can be held around an arbitrary command, and the roll writes an
    archive. Both resolve the same ledger path, so a guard wired only into the
    append path would leave two ways to reach the same misplacement.
    """
    result = _run(str(repo / "ledger"), "--", "true")

    assert result.returncode == EXIT_LEDGER_PATH, (
        f"the wrapped-command path must be judged too\n{result.stdout}{result.stderr}"
    )


# --------------------------------------------------------------------------- #
# POSITIVE CONTROL -- the ledgers that must keep working
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_a_tracking_shaped_ledger_inside_the_repo_still_appends(repo: Path) -> None:
    ledger = repo / "docs" / "tracking" / "ROLLING_WORK_LEDGER.md"
    ledger.parent.mkdir(parents=True)
    row = _row("CLAIM_UPDATE")

    result = _run(str(ledger), "--append", row)

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert row in ledger.read_text(encoding="utf-8")


@pytest.mark.unit
def test_an_archive_shaped_ledger_inside_the_repo_still_appends(repo: Path) -> None:
    archive = repo / "docs" / "tracking" / "archive" / "2026-09-19-ROLL.md"
    archive.parent.mkdir(parents=True)
    row = _row()

    result = _run(str(archive), "--append", row)

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert row in archive.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_ledger_outside_any_repository_is_not_judged(tmp_path: Path) -> None:
    """No ``.git`` above it means no repository layout to be misplaced within.

    This is the scratch-directory and test-fixture case, and it is also what
    keeps the existing path-parametrization suite green: those tests append to
    kb-internal-shaped paths under a bare temporary directory.
    """
    ledger = tmp_path / "loose" / "LEDGER.md"
    ledger.parent.mkdir(parents=True)
    row = _row()

    result = _run(str(ledger), "--append", row)

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert row in ledger.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The guard must not become the thing it protects against
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_the_guard_names_no_particular_ledger() -> None:
    """The tool is TOLD which ledger to protect; it resolves none.

    OMN-17235 rests on that property, and a placement guard is exactly the
    change most likely to break it by reaching for an allowlist of known
    ledger paths. The refusal is positional instead, so this holds.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "docs/tracking" not in source, (
        "ledger_lock.py names a particular tracking path -- the ledger move "
        "would then be a code change, not a path change"
    )
