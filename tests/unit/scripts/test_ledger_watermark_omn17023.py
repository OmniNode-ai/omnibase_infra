# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17023: heading-anchored ledger watermark (scripts/ledger_watermark.py).

The defect: `morning-friction-sweep` stored its "how far have I read" mark as
a LINE NUMBER into a file that is not time-ordered and gets rewritten. When
the 2026-08-27 split moved 19,744 lines out, the stored line number silently
addressed different content -- and a stale line watermark after a rewrite
skips MORE rows, not fewer, with no signal that anything was skipped.

The contract asserted here: the mark is a per-row identity (the row's heading
plus a digest of its body), it survives a rewrite/split by finding the row in
the archive, and every failure mode it cannot resolve FAILS CLOSED rather than
advancing past unread rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ledger_watermark.py"
_LOCK = _REPO / "scripts" / "ledger_lock.py"

SECTION = "## §5 Action Log (append-only)"
SOURCE = "rolling_work_ledger"

EXIT_UNRESOLVED = 3
EXIT_SCHEMA = 4


def _run(args: list[str], script: Path = _SCRIPT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _entry(n: int) -> str:
    return f"## 2026-08-{n:02d} — entry {n}\n\nrow {n} body line a\nrow {n} body line b\n\n"


def _ledger(tmp_path: Path, entries: int) -> Path:
    text = (
        "# Rolling work ledger\n\n## §2 Work Claims (LIVE)\n\n| ts | lane |\n\n"
        + SECTION
        + "\n\nAppend-only. Newest at the bottom.\n\n"
        + "".join(_entry(i) for i in range(1, entries + 1))
    )
    path = tmp_path / "ROLLING_WORK_LEDGER.md"
    path.write_text(text, encoding="utf-8")
    return path


def _state(tmp_path: Path, body: dict[str, Any]) -> Path:
    path = tmp_path / "friction-sweep-state.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _v2_state(tmp_path: Path, ledger: Path, anchor: dict[str, Any]) -> Path:
    return _state(
        tmp_path,
        {
            "watermark_schema_version": 2,
            "process": "morning-friction-sweep",
            "watermarks": {
                SOURCE: {
                    "path": str(ledger),
                    "section_heading": SECTION,
                    "archive_dir": str(tmp_path / "archive"),
                    **anchor,
                }
            },
        },
    )


def _out(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, f"rc={result.returncode}\n{result.stderr}"
    return json.loads(result.stdout)


# --------------------------------------------------------------------------
# schema versioning -- DoD 2: a schema change is DETECTED, never misread
# --------------------------------------------------------------------------


def test_v1_line_number_state_is_refused_not_reinterpreted(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 5)
    state = _state(
        tmp_path,
        {
            "version": 1,
            "watermarks": {SOURCE: {"path": str(ledger), "lines": 12}},
        },
    )
    result = _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    assert result.returncode == EXIT_SCHEMA
    assert "--migrate" in result.stderr


def test_future_schema_version_is_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 5)
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    body = json.loads(state.read_text(encoding="utf-8"))
    body["watermark_schema_version"] = 99
    state.write_text(json.dumps(body), encoding="utf-8")
    result = _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    assert result.returncode == EXIT_SCHEMA


def test_migrate_converts_a_line_number_to_the_row_that_line_falls_in(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, 6)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    target = next(i for i, ln in enumerate(lines, 1) if "entry 3" in ln) + 2
    state = _state(
        tmp_path,
        {
            "version": 1,
            "watermarks": {
                SOURCE: {
                    "path": str(ledger),
                    "lines": target,
                    "section_heading": SECTION,
                    "archive_dir": str(tmp_path / "archive"),
                }
            },
        },
    )
    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--migrate"])
    )
    assert out["anchor_heading"] == "## 2026-08-03 — entry 3"
    written = json.loads(state.read_text(encoding="utf-8"))
    assert written["watermark_schema_version"] == 2
    assert written["watermarks"][SOURCE]["anchor_heading"] == "## 2026-08-03 — entry 3"
    assert written["watermarks"][SOURCE]["anchor_digest"]
    assert "lines" not in written["watermarks"][SOURCE]


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


def test_resolve_counts_exactly_the_rows_after_the_anchor(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 7)
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    _out(_run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"]))
    for n in (8, 9, 10):
        ledger.write_text(
            ledger.read_text(encoding="utf-8") + _entry(n), encoding="utf-8"
        )
    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    )
    assert out["anchor_found_in"] == "live"
    assert out["unread_entries"] == 3
    resume = out["resume_line"]
    remainder = "\n".join(ledger.read_text(encoding="utf-8").splitlines()[resume - 1 :])
    assert "entry 7" not in remainder
    assert "entry 8" in remainder and "entry 10" in remainder


def test_bootstrap_state_reads_every_row(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 4)
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    )
    assert out["anchor_found_in"] == "bootstrap"
    assert out["unread_entries"] == 4


def test_a_rewritten_anchor_row_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 4)
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    _out(_run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"]))
    text = ledger.read_text(encoding="utf-8").replace("row 4 body line b", "REWRITTEN")
    ledger.write_text(text, encoding="utf-8")
    result = _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    assert result.returncode == EXIT_UNRESOLVED
    assert "digest" in result.stderr.lower()


def test_a_vanished_anchor_row_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 4)
    state = _v2_state(
        tmp_path,
        ledger,
        {"anchor_heading": "## 2026-07-01 — never existed", "anchor_digest": "0" * 12},
    )
    result = _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    assert result.returncode == EXIT_UNRESOLVED


def test_advance_does_not_run_when_resolution_failed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 4)
    state = _v2_state(
        tmp_path,
        ledger,
        {"anchor_heading": "## 2026-07-01 — never existed", "anchor_digest": "0" * 12},
    )
    before = state.read_bytes()
    result = _run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"])
    assert result.returncode == EXIT_UNRESOLVED
    assert state.read_bytes() == before


# --------------------------------------------------------------------------
# DoD 2 / done-proof 2 -- the rewrite the line watermark could not survive
# --------------------------------------------------------------------------


def test_the_watermark_survives_a_roll_and_names_the_rows_a_line_mark_would_skip(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, 12)
    archive_dir = tmp_path / "archive"
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})

    # read up to entry 12, both ways: heading anchor and the old line number
    _out(_run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"]))
    old_line_watermark = len(ledger.read_text(encoding="utf-8").splitlines())

    # three genuinely new rows arrive
    for n in (13, 14, 15):
        ledger.write_text(
            ledger.read_text(encoding="utf-8") + _entry(n), encoding="utf-8"
        )

    # ...then the file is split: the 12 oldest rows -- the anchor row among them --
    # move to the archive, leaving only the 3 unread ones live
    roll = subprocess.run(
        [
            sys.executable,
            str(_LOCK),
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
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert roll.returncode == 0, roll.stderr

    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    )

    # the anchor row (entry 12) went to the archive; the mark still resolves
    assert out["anchor_found_in"].startswith("archive:")
    # exactly the three rows appended after the mark are unread -- no more, no less
    assert out["unread_entries"] == 3
    remainder = "\n".join(
        ledger.read_text(encoding="utf-8").splitlines()[out["resume_line"] - 1 :]
    )
    for n in (13, 14, 15):
        assert f"entry {n}" in remainder
    assert "entry 12" not in remainder

    # ...and the old line-number watermark would have skipped every one of them,
    # because the split left the file shorter than the mark it stored.
    assert len(ledger.read_text(encoding="utf-8").splitlines()) < old_line_watermark
    assert out["skipped_by_line_watermark"] == 3


def test_advance_after_a_roll_records_the_new_tail(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 6)
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"])
    )
    assert out["advanced"] is True
    written = json.loads(state.read_text(encoding="utf-8"))["watermarks"][SOURCE]
    assert written["anchor_heading"] == "## 2026-08-06 — entry 6"
    assert len(written["anchor_digest"]) == 12


def test_duplicate_headings_are_disambiguated_by_digest(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    duplicate = "## 2026-08-02 — entry 2\n\nA SECOND ROW WITH THE SAME HEADING\n\n"
    ledger.write_text(ledger.read_text(encoding="utf-8") + duplicate, encoding="utf-8")
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})
    _out(_run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"]))
    ledger.write_text(ledger.read_text(encoding="utf-8") + _entry(9), encoding="utf-8")
    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    )
    assert out["unread_entries"] == 1


def test_rows_archived_before_they_were_read_are_still_reported_unread(
    tmp_path: Path,
) -> None:
    """A roll archives the OLDEST rows; it does not ask whether they were read.

    So a reader whose anchor lands in an archive can have unread rows on BOTH
    sides of the split. Dropping the archived ones would be the same silent
    skip the line-number watermark produced, reached by a different route.
    """
    ledger = _ledger(tmp_path, 6)
    archive_dir = tmp_path / "archive"
    state = _v2_state(tmp_path, ledger, {"anchor_heading": None, "anchor_digest": None})

    # read only as far as entry 2 ...
    _out(_run([str(ledger), "--state", str(state), "--source", SOURCE, "--advance"]))
    body = json.loads(state.read_text(encoding="utf-8"))
    body["watermarks"][SOURCE]["anchor_heading"] = "## 2026-08-02 — entry 2"
    body["watermarks"][SOURCE]["anchor_digest"] = None
    state.write_text(json.dumps(body), encoding="utf-8")
    assert (
        _out(
            _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
        )["unread_entries"]
        == 4
    )

    # ... then a roll archives entries 1-4, two of which were never read
    roll = subprocess.run(
        [
            sys.executable,
            str(_LOCK),
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
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert roll.returncode == 0, roll.stderr

    out = _out(
        _run([str(ledger), "--state", str(state), "--source", SOURCE, "--resolve"])
    )
    assert out["anchor_found_in"].startswith("archive:")
    assert out["unread_archived_entries"] == 2
    assert out["unread_archived_headings"] == [
        "## 2026-08-03 — entry 3",
        "## 2026-08-04 — entry 4",
    ]
    # still four unread rows in total -- the roll moved two of them, it did not
    # make them read
    assert out["unread_entries"] == 4
