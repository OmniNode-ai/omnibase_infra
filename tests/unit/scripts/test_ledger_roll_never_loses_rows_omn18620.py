# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18620: a roll must not drop rows out of git, or move citations silently.

Measured on ``omni_home`` on 2026-09-17. A cap-crossing append rolled 928 rows
into a new 1.5 MB archive file that it created **untracked**. The lane then
committed exactly the one path rule 19 tells it to commit -- the ledger -- and
produced ``1 file changed, 8 insertions(+), 928 deletions(-)``: a commit
recording the removal of 928 rows with no record of where they went. The rows
existed only as an untracked file, one clean-untracked away from being gone.
Nothing in the tool's output said a roll had happened; it was caught by reading
the commit stat and noticing that 928 deletions is an odd result for a four-row
append. Repaired as omni_home ``79f7ead2d1``.

The same roll removed 926 lines from the top of the live file, so every
``<path>:<line>`` citation written before it now resolves to a different row.
Two gates resolve those citations by line number, so a roll can make a pending
consent unresolvable or point it at the wrong row.

Every test here drives the SHIPPED script in a throwaway tree and reads the
bytes and the git index it actually produced.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ledger_lock.py"

SECTION_HEADING = "## section (append-only)"
_SHIFT_MARKER = "ROLL: lines above"


def _scrub(env: Mapping[str, str]) -> dict[str, str]:
    """Drop git's location variables (OMN-14891/OMN-18434).

    These override both ``cwd=`` and ``git -C``, so a fixture that shells out to
    git under a pre-push hook would otherwise operate on the real invoking
    worktree instead of ``tmp_path``.
    """
    scrubbed = dict(env)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        scrubbed.pop(key, None)
    return scrubbed


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_scrub(os.environ),
    ).stdout


def _ledger_text(rows: int) -> str:
    body = "".join(
        f"2026-09-17T00:00:{i % 60:02d}Z | ROW | lane=l{i} | body {i}\n"
        for i in range(rows)
    )
    return "# Ledger\n\nintro\n\n" + SECTION_HEADING + "\n\n" + body


def _run_roll(
    ledger: Path, archive_dir: Path, *, keep: int, max_rows: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION_HEADING,
            "--on-cap",
            "roll",
            "--archive-dir",
            str(archive_dir),
            "--roll-keep-entries",
            str(keep),
            "--max-section-rows",
            str(max_rows),
            "--timeout",
            "30",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_scrub(os.environ),
    )


def _real_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A tree with a REAL git repository, not a bare ``.git`` directory.

    The sibling OMN-17403 suite only needs a ``.git`` entry to exist, because it
    asks where the repo root is. This suite asks what the INDEX contains after a
    roll, which only a real repository can answer.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    ledger_dir = repo / "docs" / "tracking"
    ledger_dir.mkdir(parents=True)
    ledger = ledger_dir / "LEDGER.md"
    ledger.write_text(_ledger_text(30), encoding="utf-8")
    _git(repo, "add", "--", str(ledger))
    _git(repo, "commit", "--quiet", "-m", "seed")
    return repo, ledger, ledger_dir / "archive"


# --------------------------------------------------------------------------- #
# AC1 -- a roll never leaves rows outside git
# --------------------------------------------------------------------------- #
def test_a_roll_leaves_no_untracked_rows(tmp_path: Path) -> None:
    """The defect, stated as the property it violated.

    The archive holds the only copy of the rolled rows. Untracked, it is outside
    git entirely, and a commit of the ledger alone then records their deletion
    with no record of where they went.
    """
    repo, ledger, archive_dir = _real_repo(tmp_path)

    proc = _run_roll(ledger, archive_dir, keep=5, max_rows=20)
    assert proc.returncode == 0, proc.stderr

    archives = list(archive_dir.glob("*.md"))
    assert archives, "no archive was written, so this test proved nothing"

    porcelain = _git(repo, "status", "--porcelain")
    untracked = [line for line in porcelain.splitlines() if line.startswith("??")]
    assert not untracked, (
        "the roll left rows outside git; a commit of the ledger alone would "
        f"record their deletion with no record of where they went: {untracked!r}"
    )


def test_the_archive_is_staged_alongside_the_rewritten_ledger(
    tmp_path: Path,
) -> None:
    """Both paths, not just the one that stopped being untracked.

    Staging the archive alone would still let a lane commit it without the
    ledger rewrite that removed the rows, which is the same split one step over.
    """
    repo, ledger, archive_dir = _real_repo(tmp_path)

    proc = _run_roll(ledger, archive_dir, keep=5, max_rows=20)
    assert proc.returncode == 0, proc.stderr

    staged = _git(repo, "diff", "--cached", "--name-only").split()
    archive = next(archive_dir.glob("*.md"))
    rel_archive = str(archive.relative_to(repo))
    rel_ledger = str(ledger.relative_to(repo))
    assert rel_archive in staged, f"archive not staged: {staged!r}"
    assert rel_ledger in staged, f"ledger rewrite not staged: {staged!r}"


def test_a_roll_that_cannot_be_staged_says_so(tmp_path: Path) -> None:
    """Outside a repository there is nothing to stage, so it must warn.

    'Or refuse to roll and say so' -- the bytes are already durable by the time
    staging is attempted, so failing here would leave the operator with a
    completed roll and a traceback instead of a completed roll and an
    instruction. Saying so is the useful half.
    """
    ledger_dir = tmp_path / "loose" / "tracking"
    ledger_dir.mkdir(parents=True)
    ledger = ledger_dir / "LEDGER.md"
    ledger.write_text(_ledger_text(30), encoding="utf-8")

    proc = _run_roll(ledger, ledger_dir / "archive", keep=5, max_rows=20)

    assert proc.returncode == 0, proc.stderr
    assert "NOT staged" in proc.stderr
    assert "not inside a git work tree" in proc.stderr
    archive = next((ledger_dir / "archive").glob("*.md"))
    assert str(archive) in proc.stderr, (
        f"the warning does not name the file to commit: {proc.stderr!r}"
    )


# --------------------------------------------------------------------------- #
# AC2 -- a roll cannot happen silently
# --------------------------------------------------------------------------- #
def test_the_roll_announces_itself_in_words(tmp_path: Path) -> None:
    """A JSON receipt line is not an announcement.

    The pre-change output carried one, and the lane that hit the roll still did
    not know it had happened. What was missing was a sentence.
    """
    _, ledger, archive_dir = _real_repo(tmp_path)

    proc = _run_roll(ledger, archive_dir, keep=5, max_rows=20)

    assert "ROLLED 25 row(s)" in proc.stderr, proc.stderr
    archive = next(archive_dir.glob("*.md"))
    assert str(archive) in proc.stderr
    assert "citation" in proc.stderr, (
        "the announcement does not mention that citations moved, which is the "
        f"half a reader cannot infer: {proc.stderr!r}"
    )


def test_a_roll_that_rolls_nothing_announces_nothing(tmp_path: Path) -> None:
    """Control: the announcement must not fire on a no-op.

    ``plan_roll`` returns a plan that rolled nothing whenever the section is
    shorter than ``--roll-keep-entries``, and its receipt carries no offset to
    report. An unguarded announcement crashed on exactly that path.
    """
    _, ledger, archive_dir = _real_repo(tmp_path)

    proc = _run_roll(ledger, archive_dir, keep=500, max_rows=20)

    # The cap is still crossed after a roll that moved nothing, so the tool
    # refuses -- which is the pre-existing behaviour and is correct. What is
    # asserted here is only that nothing claimed to have been rolled.
    assert "ROLL REFUSED" in proc.stderr
    assert "ROLLED" not in proc.stderr


# --------------------------------------------------------------------------- #
# AC3 -- the recorded offset is measured, not asserted
# --------------------------------------------------------------------------- #
def test_the_pointer_line_records_the_offset_the_roll_actually_introduced(
    tmp_path: Path,
) -> None:
    """Checked against the files' own line numbers, never against a constant.

    A figure the roll simply declares would drift from the bytes the moment
    anything else in the pointer block changed length. This recomputes the shift
    from where the first kept row sits before and after, which is the same thing
    a reader resolving an old citation would have to do by hand.
    """
    _, ledger, archive_dir = _real_repo(tmp_path)
    before_text = ledger.read_text(encoding="utf-8")

    proc = _run_roll(ledger, archive_dir, keep=5, max_rows=20)
    assert proc.returncode == 0, proc.stderr
    after_text = ledger.read_text(encoding="utf-8")

    # The first row still present after the roll, found in both files. Matched
    # on the leading timestamp, not on " | ROW | ": the pointer block's own JSON
    # marker embeds the first kept row's text as `first_kept_heading`, so a
    # substring match finds that comment line first and measures the wrong thing.
    first_kept = next(
        line
        for line in after_text.splitlines(keepends=True)
        if line.startswith("2026-")
    )
    line_before = before_text[: before_text.index(first_kept)].count("\n") + 1
    line_after = after_text[: after_text.index(first_kept)].count("\n") + 1
    shift = line_before - line_after

    banner = next(line for line in after_text.splitlines() if _SHIFT_MARKER in line)
    assert f"{_SHIFT_MARKER} {line_before}" in banner, banner
    assert f"shifted by -{shift}" in banner, banner
    assert shift > 0, "the roll removed no lines, so this test proved nothing"


def test_the_offset_sentence_costs_the_section_no_extra_line(
    tmp_path: Path,
) -> None:
    """Why it is a sentence on the pointer line and not a line of its own.

    ``parse_section`` counts the pointer block toward the section's line budget,
    so one extra line changes the roll arithmetic for every caller: an existing
    cap test went from a roll that fit in 10 lines to one refused at 11. Roll
    metadata must not consume the budget meant for rows.
    """
    _, ledger, archive_dir = _real_repo(tmp_path)

    proc = _run_roll(ledger, archive_dir, keep=5, max_rows=20)
    assert proc.returncode == 0, proc.stderr

    after = ledger.read_text(encoding="utf-8").splitlines()
    marker_lines = [line for line in after if _SHIFT_MARKER in line]
    assert len(marker_lines) == 1
    assert marker_lines[0].lstrip().startswith("> Older rows live in"), (
        "the offset is on a line of its own, which consumes the section's line "
        f"budget: {marker_lines[0]!r}"
    )


def test_a_second_roll_replaces_the_offset_rather_than_stacking_them(
    tmp_path: Path,
) -> None:
    """One banner, not a chain, and the bound is proven rather than intended.

    Accumulating one per roll would grow the preamble without limit, and each
    older figure would itself be invalidated by the newer roll. The durable fix
    for an older citation is the timestamp form; the roll chain stays recoverable
    from the archive headers.
    """
    _, ledger, archive_dir = _real_repo(tmp_path)

    assert _run_roll(ledger, archive_dir, keep=5, max_rows=20).returncode == 0
    with ledger.open("a", encoding="utf-8") as handle:
        for i in range(30, 60):
            handle.write(f"2026-09-17T01:00:{i % 60:02d}Z | ROW | lane=l{i} | b\n")
    assert _run_roll(ledger, archive_dir, keep=5, max_rows=20).returncode == 0

    after = ledger.read_text(encoding="utf-8")
    assert after.count(_SHIFT_MARKER) == 1, (
        f"offsets stacked across rolls: {after.count(_SHIFT_MARKER)} present"
    )
