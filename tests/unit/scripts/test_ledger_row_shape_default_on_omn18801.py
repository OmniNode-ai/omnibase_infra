# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18801: the row-shape guard governs the append the fleet actually makes.

The defect, found by the ledger-integrity audit on 2026-09-19 while working
OMN-18798. ``enforce_row_shape`` (OMN-17403) returned ``None`` immediately
whenever no ``--section-heading`` was passed. The documented append recipe --
CLAUDE.md rule 19 and the ledger's own section 0.8, both spelling
``ledger_lock.py <ledger> --append '<row>'`` -- names no section heading. The
guard was therefore OFF on every append the fleet makes, and had been since it
shipped: it governed a calling convention nobody used.

A second, independent hole sat behind it. ``append_text`` normalized the
trailing newline of the PAYLOAD and never looked at the FILE. A ledger whose
last byte is not a newline welded the next append onto the previous row.

Both fired together on the live rolling work ledger: the audit lane's CLAIM row
was accepted, minted a claim token, and landed welded onto a peer lane's
TERMINAL row with no newline between them, so two rows parsed as one until the
repair.

The corrective contract asserted here:

* the shape guard is scoped by the FILE, not by a flag -- a markdown ledger is
  judged with or without a heading;
* a non-markdown ledger is not judged by the markdown row model, because
  ``ROW_START_PATTERN``'s whole vocabulary is markdown and the fleet's one
  non-markdown caller appends JSON Lines;
* an append always begins on a fresh line, whatever state the file is in, and
  that half is NOT scoped to markdown;
* the claim token's offset still points at the row it names.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_LOCK = _REPO / "scripts" / "ledger_lock.py"

EXIT_ROW_SHAPE = 76

#: The commit this fix branched from -- the last revision carrying the defect.
#: AC3 reads the pre-fix module out of the object store at this sha rather than
#: reconstructing it, because a reconstruction proves only that the test author
#: understood the bug.
PRE_FIX_SHA = "e7f65bb8e50a8b8949499fd16fa38c63d3226ab9"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _now() -> str:
    """A stamp the OMN-17427 wall-clock guard accepts: a fixture row must be
    honest about when it was written, so it is stamped at call time."""
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(note: str) -> str:
    """A row in the shape lanes append, deliberately NOT a TERMINAL or CLAIM
    one: those carry their own field requirements (OMN-18274 friction,
    OMN-18554 cost, OMN-18766 executor) which are not what these tests are
    about, and a fixture that has to satisfy them tests those guards instead."""
    return f"{_now()} | NOTE | lane=omn18801 | {note}"


def _ledger(tmp_path: Path, *, trailing_newline: bool, name: str = "L.md") -> Path:
    """A ledger whose last row is present, with or without the newline that
    closes it. The no-newline variant is the live 2026-09-19 shape."""
    text = (
        "# Rolling work ledger\n\n## Action Log (append-only)\n\n"
        "Append-only. Newest at the bottom.\n\n"
        f"{_row('the row already here')}"
    )
    if trailing_newline:
        text += "\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# AC1 -- the guard runs with no section heading.
# --------------------------------------------------------------------------


def test_append_with_no_section_heading_refuses_a_payload_that_opens_no_row(
    tmp_path: Path,
) -> None:
    """The documented recipe passes no heading. Before OMN-18801 this exited 0
    and wrote the payload into the previous row's body."""
    ledger = _ledger(tmp_path, trailing_newline=True)
    before = ledger.read_text(encoding="utf-8")

    result = _run(_LOCK, [str(ledger), "--append", "free-form prose, no row"])

    assert result.returncode == EXIT_ROW_SHAPE, result.stderr
    assert ledger.read_text(encoding="utf-8") == before, "a refusal wrote bytes"


def test_the_refusal_names_a_reason_and_the_offending_line(tmp_path: Path) -> None:
    """A refusal a lane cannot act on gets routed around. The message must say
    what was wrong, which line was wrong, and what a row looks like."""
    ledger = _ledger(tmp_path, trailing_newline=True)

    result = _run(_LOCK, [str(ledger), "--append", "free-form prose, no row"])

    assert "ROW SHAPE REFUSED" in result.stderr
    assert "does not open a row" in result.stderr
    assert "free-form prose, no row" in result.stderr
    assert "Nothing was written." in result.stderr


def test_a_well_formed_row_is_still_accepted_with_no_heading(tmp_path: Path) -> None:
    """The guard must not be a tax on the recipe it now governs."""
    ledger = _ledger(tmp_path, trailing_newline=True)

    row = _row("the new row")
    result = _run(_LOCK, [str(ledger), "--append", row])

    assert result.returncode == 0, result.stderr
    assert row in ledger.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("<NOW> | a bare timestamp", id="bare-timestamp"),
        pytest.param("- 2026-01-02 a bulleted date", id="bulleted-date"),
        pytest.param("| 2026-01-02 | a table row |", id="table-row"),
        pytest.param("### 2026-01-02 a heading", id="markdown-heading"),
    ],
)
def test_every_live_row_shape_is_admitted_without_a_heading(
    tmp_path: Path, payload: str
) -> None:
    """The four shapes counted on the live ledger by OMN-17403 all still open a
    row when no section heading is passed -- widening the SCOPE of the guard
    must not narrow its VOCABULARY."""
    ledger = _ledger(tmp_path, trailing_newline=True)
    payload = payload.replace("<NOW>", _now())

    result = _run(_LOCK, [str(ledger), "--append", payload])

    assert result.returncode == 0, result.stderr
    assert payload in ledger.read_text(encoding="utf-8")


def test_a_non_markdown_ledger_is_not_judged_by_the_markdown_row_model(
    tmp_path: Path,
) -> None:
    """Scope is the file, and ROW_START_PATTERN is markdown vocabulary.

    The fleet's one non-markdown caller appends JSON Lines to
    docs/marketing/linkedin-engagement/events.jsonl, where '{' IS the start of a
    record. Refusing it would be a false positive, not a catch.
    """
    ledger = tmp_path / "events.jsonl"
    ledger.write_text('{"event":"first"}\n', encoding="utf-8")

    result = _run(_LOCK, [str(ledger), "--append", '{"event":"second"}'])

    assert result.returncode == 0, result.stderr
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        '{"event":"first"}',
        '{"event":"second"}',
    ]


# --------------------------------------------------------------------------
# AC2 -- an append always begins on a fresh line.
# --------------------------------------------------------------------------


def test_a_row_is_never_welded_onto_a_file_that_lacks_a_trailing_newline(
    tmp_path: Path,
) -> None:
    """The live 2026-09-19 reproduction, as a committed gate."""
    ledger = _ledger(tmp_path, trailing_newline=False)
    last_before = ledger.read_text(encoding="utf-8").splitlines()[-1]

    row = _row("the welded row that must not weld")
    result = _run(_LOCK, [str(ledger), "--append", row])

    assert result.returncode == 0, result.stderr
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == row, "the appended row does not occupy its own line"
    assert lines[-2] == last_before, "the previous row was rewritten"


def test_the_separator_is_written_for_a_non_markdown_ledger_too(
    tmp_path: Path,
) -> None:
    """Welding corrupts a .jsonl exactly as it corrupts a .md, so the newline
    half of this fix is deliberately NOT scoped the way the shape guard is."""
    ledger = tmp_path / "events.jsonl"
    ledger.write_text('{"event":"first"}', encoding="utf-8")

    result = _run(_LOCK, [str(ledger), "--append", '{"event":"second"}'])

    assert result.returncode == 0, result.stderr
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        '{"event":"first"}',
        '{"event":"second"}',
    ]


def test_repeated_appends_never_weld_and_never_double_space(tmp_path: Path) -> None:
    """Idempotence: the separator fires only when it is needed, so a healthy
    ledger does not grow a blank line per append."""
    ledger = _ledger(tmp_path, trailing_newline=False)

    rows = [_row(f"row {i}") for i in range(3)]
    for row in rows:
        result = _run(_LOCK, [str(ledger), "--append", row])
        assert result.returncode == 0, result.stderr

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines[-3:] == rows
    assert ledger.read_text(encoding="utf-8").endswith("\n")


def test_the_claim_token_offset_points_at_the_row_it_names(tmp_path: Path) -> None:
    """The separator lands before the offset is read, so a claim token minted
    on a file that lacked its trailing newline still addresses its own row.

    An off-by-one here would be worse than the weld: the token would verify
    against the wrong bytes and the claim protocol would silently pass.
    """
    ledger = _ledger(tmp_path, trailing_newline=False)

    row = (
        f"{_now()} | CLAIM | lane=omn18801 | actor=claude:opus5:subagent | "
        "ticket=OMN-18801 | proving the offset | est ~0.1 lane-hours; "
        "displaces nothing"
    )
    result = _run(_LOCK, [str(ledger), "--append", row])
    assert result.returncode == 0, result.stderr

    match = re.search(r"CLAIM-TOKEN (\S+)", result.stdout)
    assert match is not None, result.stdout
    offset = int(match.group(1).split("-")[1])

    raw = ledger.read_bytes()
    assert raw[offset : offset + len(row.encode())].decode("utf-8") == row


# --------------------------------------------------------------------------
# AC3 -- the pre-fix module, read from the object store, reproduces both.
# --------------------------------------------------------------------------


def _pre_fix_module(tmp_path: Path) -> Path:
    """``scripts/ledger_lock.py`` as of PRE_FIX_SHA, written to a scratch path.

    Read with ``git show`` rather than reconstructed, and never through
    ``git stash`` -- the stash list is shared across every worktree of a clone,
    so a scoped stash here would pop into a peer lane's tree (OMN-17334).

    Skips rather than fails when the object is unreachable: a shallow CI
    checkout legitimately does not carry it, and a skip that names the sha is
    honest where a green assertion over a missing pre-image would not be.
    """
    probe = subprocess.run(
        ["git", "-C", str(_REPO), "cat-file", "-e", f"{PRE_FIX_SHA}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )
    if probe.returncode != 0:
        pytest.skip(
            f"pre-fix commit {PRE_FIX_SHA} is not in this checkout "
            f"(git cat-file -e exited {probe.returncode}); a shallow clone "
            "cannot read the pre-image this test compares against"
        )
    shown = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{PRE_FIX_SHA}:scripts/ledger_lock.py"],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )
    assert shown.returncode == 0, shown.stderr
    path = tmp_path / "pre_fix_ledger_lock.py"
    path.write_text(shown.stdout, encoding="utf-8")
    return path


def test_the_pre_fix_module_welds_the_row_and_the_fixed_one_does_not(
    tmp_path: Path,
) -> None:
    """The audit's observation, run against both revisions from one input."""
    pre_fix = _pre_fix_module(tmp_path)
    row = _row("the second row")

    red = _ledger(tmp_path, trailing_newline=False, name="red.md")
    red_last = red.read_text(encoding="utf-8").splitlines()[-1]
    red_result = _run(pre_fix, [str(red), "--append", row])
    assert red_result.returncode == 0, red_result.stderr
    red_lines = red.read_text(encoding="utf-8").splitlines()
    assert red_lines[-1] == f"{red_last}{row}", (
        "the pre-fix module did not weld -- this test no longer reproduces "
        "the defect it was written for"
    )

    green = _ledger(tmp_path, trailing_newline=False, name="green.md")
    green_last = green.read_text(encoding="utf-8").splitlines()[-1]
    green_result = _run(_LOCK, [str(green), "--append", row])
    assert green_result.returncode == 0, green_result.stderr
    green_lines = green.read_text(encoding="utf-8").splitlines()
    assert green_lines[-1] == row
    assert green_lines[-2] == green_last


def test_the_pre_fix_module_accepted_a_payload_that_opens_no_row(
    tmp_path: Path,
) -> None:
    """The other half: with no section heading the shape guard did not run at
    all, so prose landed inside the previous row's body and exited 0."""
    pre_fix = _pre_fix_module(tmp_path)
    prose = "free-form prose, no row"

    red = _ledger(tmp_path, trailing_newline=True, name="red.md")
    red_result = _run(pre_fix, [str(red), "--append", prose])
    assert red_result.returncode == 0, red_result.stderr
    assert prose in red.read_text(encoding="utf-8")

    green = _ledger(tmp_path, trailing_newline=True, name="green.md")
    before = green.read_text(encoding="utf-8")
    green_result = _run(_LOCK, [str(green), "--append", prose])
    assert green_result.returncode == EXIT_ROW_SHAPE, green_result.stderr
    assert green.read_text(encoding="utf-8") == before
