# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17023: section row/byte caps and the archive roll in scripts/ledger_lock.py.

The defect these cover: an append-only ledger section (the rolling work
ledger's "§5 Action Log") had no cap of any kind, so it grew to 21,752 lines
before a human noticed and split it by hand. A cap that is only advisory is
the same defect, so every test here asserts the FILE, not just the exit code:
a blocked append must leave the ledger byte-identical, and a rolling append
must leave the live section under the cap with every rolled row still
readable, verbatim, in the archive.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ledger_lock.py"

EXIT_SECTION_CAP = 74
SECTION = "## §5 Action Log (append-only)"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("ledger_lock_caps", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _entry(n: int, body_lines: int = 2) -> str:
    body = "\n".join(f"row {n} line {i}" for i in range(body_lines))
    return f"## 2026-08-{n:02d} — entry {n}\n\n{body}\n"


def _ledger(tmp_path: Path, entries: int, *, preamble: bool = True) -> Path:
    head = [
        "# Rolling work ledger",
        "",
        "## §2 Work Claims (LIVE)",
        "",
        "| ts | lane |",
        "",
        SECTION,
        "",
    ]
    if preamble:
        head += ["Append-only. Newest at the bottom.", ""]
    text = "\n".join(head) + "\n".join(_entry(i) + "\n" for i in range(1, entries + 1))
    path = tmp_path / "ROLLING_WORK_LEDGER.md"
    path.write_text(text, encoding="utf-8")
    return path


def _receipt(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("ledger_lock: ROLL "):
            return json.loads(line[len("ledger_lock: ROLL ") :])
    raise AssertionError(f"no ROLL receipt on stdout:\n{stdout}")


# --------------------------------------------------------------------------
# argument contract -- a cap with no policy, or a policy with no cap, is a
# usage error rather than a silent no-op.
# --------------------------------------------------------------------------


def test_section_heading_without_a_cap_is_a_usage_error(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    result = _run([str(ledger), "--append", "- x", "--section-heading", SECTION])
    assert result.returncode == 2
    assert "--max-section-rows" in result.stderr


def test_cap_without_on_cap_policy_is_a_usage_error(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    result = _run(
        [
            str(ledger),
            "--append",
            "- x",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "10",
        ]
    )
    assert result.returncode == 2
    assert "--on-cap" in result.stderr


def test_roll_policy_without_archive_dir_is_a_usage_error(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    result = _run(
        [
            str(ledger),
            "--append",
            "- x",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "10",
            "--on-cap",
            "roll",
        ]
    )
    assert result.returncode == 2
    assert "--archive-dir" in result.stderr


def test_absent_section_heading_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--append",
            "- x",
            "--section-heading",
            "## §9 Not Present",
            "--max-section-rows",
            "10",
            "--on-cap",
            "block",
        ]
    )
    assert result.returncode == 2
    assert ledger.read_bytes() == before


def test_duplicated_section_heading_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 2)
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + f"\n{SECTION}\n", encoding="utf-8"
    )
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--append",
            "- x",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "10",
            "--on-cap",
            "block",
        ]
    )
    assert result.returncode == 2
    assert ledger.read_bytes() == before


# --------------------------------------------------------------------------
# block policy -- DoD 1 / done-proof 1, "blocks pending a roll"
# --------------------------------------------------------------------------


def test_block_refuses_the_append_that_would_cross_the_row_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 6)
    before = ledger.read_bytes()
    section_rows = MOD.section_line_count(ledger, SECTION)
    result = _run(
        [
            str(ledger),
            "--append",
            "## 2026-08-31 — one more\n\nbody\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            str(section_rows),
            "--on-cap",
            "block",
        ]
    )
    assert result.returncode == EXIT_SECTION_CAP
    assert ledger.read_bytes() == before, (
        "a blocked append must not write a single byte"
    )
    assert "--roll-section" in result.stderr


def test_block_refuses_the_append_that_would_cross_the_byte_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 6)
    before = ledger.read_bytes()
    section_bytes = MOD.section_byte_count(ledger, SECTION)
    result = _run(
        [
            str(ledger),
            "--append",
            "## 2026-08-31 — one more\n\nbody\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--max-section-bytes",
            str(section_bytes),
            "--on-cap",
            "block",
        ]
    )
    assert result.returncode == EXIT_SECTION_CAP
    assert ledger.read_bytes() == before


def test_an_append_that_stays_under_the_cap_is_written(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    result = _run(
        [
            str(ledger),
            "--append",
            "## 2026-08-31 — one more\n\nbody\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--max-section-bytes",
            "1000000",
            "--on-cap",
            "block",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "one more" in ledger.read_text(encoding="utf-8")


def test_oversized_single_payload_is_refused_even_under_the_roll_policy(
    tmp_path: Path,
) -> None:
    """A roll cannot rescue a row that is itself larger than the cap, so the
    per-append cap refuses rather than rolling the whole section away trying."""
    ledger = _ledger(tmp_path, 3)
    archive = tmp_path / "archive"
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--append",
            "## huge\n" + ("x" * 5000) + "\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(archive),
            "--roll-keep-entries",
            "2",
            "--max-append-bytes",
            "500",
        ]
    )
    assert result.returncode == EXIT_SECTION_CAP
    assert ledger.read_bytes() == before
    assert not archive.exists()


# --------------------------------------------------------------------------
# roll policy -- DoD 1 / done-proof 1, "the roll fires"
# --------------------------------------------------------------------------


def test_append_over_the_cap_rolls_then_appends_and_stays_under_the_cap(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, 8)
    archive_dir = tmp_path / "archive"
    original_entries = MOD.section_entries(ledger, SECTION)
    assert len(original_entries) == 8
    cap = MOD.section_line_count(ledger, SECTION)

    result = _run(
        [
            str(ledger),
            "--append",
            "## 2026-08-31 — one more\n\nbody\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            str(cap),
            "--on-cap",
            "roll",
            "--archive-dir",
            str(archive_dir),
            "--roll-keep-entries",
            "3",
        ]
    )
    assert result.returncode == 0, result.stderr

    receipt = _receipt(result.stdout)
    assert receipt["entries_rolled"] == 5
    assert receipt["entries_kept"] == 3

    # the cap actually held
    assert MOD.section_line_count(ledger, SECTION) <= cap

    # the new row landed
    live = ledger.read_text(encoding="utf-8")
    assert "one more" in live

    # nothing above the section was disturbed
    assert "## §2 Work Claims (LIVE)" in live
    assert live.index("## §2 Work Claims (LIVE)") < live.index(SECTION)

    # every rolled row is readable, verbatim, in the archive
    archive_path = Path(receipt["archive"])
    assert archive_path.exists()
    archived = archive_path.read_text(encoding="utf-8")
    for entry in original_entries[:5]:
        assert entry.text.strip() in archived
    # ...and no longer in the live file
    assert "entry 1" not in live
    # ...while the kept ones still are
    assert "entry 8" in live


def test_roll_loses_no_row(tmp_path: Path) -> None:
    """Union of live+archive headings must equal the pre-roll heading set."""
    ledger = _ledger(tmp_path, 10)
    archive_dir = tmp_path / "archive"
    before = [e.heading for e in MOD.section_entries(ledger, SECTION)]

    result = _run(
        [
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "1",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(archive_dir),
            "--roll-keep-entries",
            "2",
        ]
    )
    assert result.returncode == 0, result.stderr
    receipt = _receipt(result.stdout)
    live_headings = [e.heading for e in MOD.section_entries(ledger, SECTION)]
    archived = Path(receipt["archive"]).read_text(encoding="utf-8")
    archived_headings = [h for h in before if h not in live_headings]
    for heading in archived_headings:
        assert heading in archived
    assert set(live_headings) | set(archived_headings) == set(before)
    assert len(live_headings) == 2


def test_roll_section_under_the_cap_is_a_no_op(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "2",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(result.stdout)["entries_rolled"] == 0
    assert ledger.read_bytes() == before


def test_force_roll_rolls_even_under_the_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 5)
    result = _run(
        [
            str(ledger),
            "--roll-section",
            "--force-roll",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "2",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(result.stdout)["entries_rolled"] == 3


def test_second_roll_updates_the_pointer_block_instead_of_stacking_one(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, 8)
    archive_dir = tmp_path / "archive"
    args = [
        str(ledger),
        "--roll-section",
        "--force-roll",
        "--section-heading",
        SECTION,
        "--max-section-rows",
        "100000",
        "--on-cap",
        "roll",
        "--archive-dir",
        str(archive_dir),
        "--roll-keep-entries",
        "3",
    ]
    assert _run(args).returncode == 0
    _run(
        [
            str(ledger),
            "--append",
            _entry(9),
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "block",
        ]
    )
    assert _run(args).returncode == 0
    live = ledger.read_text(encoding="utf-8")
    assert live.count(MOD.ROLL_POINTER_MARKER) == 1


def test_the_roll_receipt_names_the_oldest_surviving_row(tmp_path: Path) -> None:
    """The receipt is what a watermark holder re-anchors against, so it must
    name the boundary row rather than merely a count."""
    ledger = _ledger(tmp_path, 6)
    result = _run(
        [
            str(ledger),
            "--roll-section",
            "--force-roll",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "2",
        ]
    )
    receipt = _receipt(result.stdout)
    assert receipt["schema"] == MOD.ROLL_RECEIPT_SCHEMA
    assert receipt["first_kept_heading"] == "## 2026-08-05 — entry 5"
    assert receipt["last_rolled_heading"] == "## 2026-08-04 — entry 4"
    assert receipt["section_lines_after"] < receipt["section_lines_before"]


def test_roll_keep_entries_larger_than_the_section_rolls_nothing(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, 2)
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--roll-section",
            "--force-roll",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "99",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(result.stdout)["entries_rolled"] == 0
    assert ledger.read_bytes() == before


def test_cap_still_exceeded_after_a_roll_blocks_rather_than_growing(
    tmp_path: Path,
) -> None:
    """Keeping N entries can itself exceed the cap. That must refuse, not
    silently write past the cap the operator configured."""
    ledger = _ledger(tmp_path, 8)
    before = ledger.read_bytes()
    result = _run(
        [
            str(ledger),
            "--append",
            "## 2026-08-31 — one more\n",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "3",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "8",
        ]
    )
    assert result.returncode == EXIT_SECTION_CAP
    assert ledger.read_bytes() == before


def test_caps_are_opt_in_and_absent_by_default(tmp_path: Path) -> None:
    """No --section-heading means the pre-OMN-17023 behaviour, unchanged."""
    ledger = _ledger(tmp_path, 3)
    result = _run([str(ledger), "--append", "- plain row"])
    assert result.returncode == 0, result.stderr
    assert "- plain row" in ledger.read_text(encoding="utf-8")
