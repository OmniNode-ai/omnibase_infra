# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17403: the ledger row model must be the one the ledger writer writes.

The defect, measured on the live rolling work ledger on 2026-09-16. The
section parser in ``scripts/ledger_lock.py`` defined a row as a markdown
heading (``^#{2,6} \\S``). ``append_text`` writes the caller's payload
verbatim and adds no heading, and essentially every lane appends a
timestamp-led line. §5 therefore held 8,219 lines that parsed as 29 rows.

Two chronic failures fall out of that single mismatch, and both are asserted
here:

* ``ledger_watermark.py --advance`` anchors on the TAIL row, whose body is
  open by construction. The next un-headed append lands inside that row, its
  digest changes, and ``--resolve`` exits 3 UNRESOLVED. It did so on every
  morning friction sweep from 2026-09-05 to 2026-09-15 -- eleven consecutive
  runs where the fleet's primary friction source was never read.
* ``--roll-section`` keeps ``--roll-keep-entries`` ROWS. With 29 rows and a
  keep of 40 it rolls nothing, and it exited 0 with a success receipt while
  reporting the section still at twice its configured cap. A trigger firing
  daily against that behaviour is indistinguishable from a trigger that never
  fired at all.

The corrective contract: a row starts where an append starts, an append that
cannot start a row is refused, and a roll that leaves the section over its cap
fails loudly instead of reporting success.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_LOCK = _REPO / "scripts" / "ledger_lock.py"
_WATERMARK = _REPO / "scripts" / "ledger_watermark.py"

SECTION = "## §5 Action Log (append-only)"
SOURCE = "rolling_work_ledger"

EXIT_SECTION_CAP = 74
EXIT_ROW_SHAPE = 76
EXIT_UNRESOLVED = 3


# --- OMN-18554: fixture rows are stamped from the LIVE clock ----------------
#
# ledger_lock.py in this repo now carries the OMN-17427 wall-clock guard, ported
# from the omni_home copy where it had been living uncommitted. That guard reads
# a row's OWN leading timestamp and refuses one stamped more than 5 minutes ahead
# of the wall clock or more than 24 hours behind it. Every fixture row below was
# written before this repo's script had that guard, so each carried a frozen date
# literal that is now weeks in the past and is correctly refused.
#
# `_live_stamp` rewrites only the LEADING timestamp, at call time, and leaves the
# rest of each row byte-identical. The literal stays in the source as the shape
# documentation it always was; what changes is that the row is honest about when
# it was written, which is the only thing the guard asks.
_LEADING_TS = re.compile(r"(?<![\d])20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _live_stamp(row: str, *, shift_seconds: int = 0) -> str:
    """Rewrite the row's FIRST timestamp to now (+shift), leaving all else."""
    moment = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=shift_seconds)
    return _LEADING_TS.sub(moment.strftime("%Y-%m-%dT%H:%M:%SZ"), row, count=1)


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _row(n: int) -> str:
    """One row in the shape lanes actually append: a UTC timestamp, then prose."""
    return (
        f"2026-09-{n:02d}T08:00:00Z | TERMINAL | lane=lane-{n} | ticket=OMN-1{n:03d} | "
        f"outcome=DONE_{n}\n"
    )


def _ledger(tmp_path: Path, rows: int, *, headed_tail: bool = False) -> Path:
    text = (
        "# Rolling work ledger\n\n## §2 Work Claims (LIVE)\n\n| ts | lane |\n\n"
        + SECTION
        + "\n\nAppend-only. Newest at the bottom.\n\n"
        + "".join(_row(i) for i in range(1, rows + 1))
    )
    if headed_tail:
        text += "### 2026-09-30 — a headed row\n\nbody\n\n"
    path = tmp_path / "ROLLING_WORK_LEDGER.md"
    path.write_text(text, encoding="utf-8")
    return path


def _receipt(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("ledger_lock: ROLL "):
            return json.loads(line[len("ledger_lock: ROLL ") :])
    raise AssertionError(f"no ROLL receipt on stdout:\n{stdout}")


def _v2_state(tmp_path: Path, ledger: Path) -> Path:
    path = tmp_path / "friction-sweep-state.json"
    path.write_text(
        json.dumps(
            {
                "watermark_schema_version": 2,
                "watermarks": {
                    SOURCE: {
                        "path": str(ledger),
                        "section_heading": SECTION,
                        "archive_dir": str(tmp_path / "archive"),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _watermark(
    ledger: Path, state: Path, mode: str
) -> subprocess.CompletedProcess[str]:
    return _run(
        _WATERMARK,
        [str(ledger), "--state", str(state), "--source", SOURCE, mode],
    )


# --------------------------------------------------------------------------
# The row model itself.
# --------------------------------------------------------------------------


def test_a_timestamp_led_append_starts_its_own_row(tmp_path: Path) -> None:
    """The unit the parser counts must be the unit the writer writes.

    Before OMN-17403 these twelve appends parsed as ZERO rows -- none of them
    carries a markdown heading -- and the whole section landed in the
    section's preamble.
    """
    ledger = _ledger(tmp_path, 12)
    result = _run(
        _LOCK,
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
            "4",
        ],
    )
    assert result.returncode == 0, result.stderr
    receipt = _receipt(result.stdout)
    assert receipt["entries_rolled"] == 8
    assert receipt["entries_kept"] == 4


def test_every_live_row_shape_starts_a_row(tmp_path: Path) -> None:
    """The four shapes counted on the live §5 on 2026-09-16, each its own row.

    Bare timestamp (6,918 lines), pipe-table timestamp (436), bullet-led
    timestamp (715) and markdown heading (29). A shape that does not start a
    row is a shape whose appends silently mutate the row above it.
    """
    ledger = _ledger(tmp_path, 0)
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "2026-09-16T08:00:00Z CLAIM lane=a\n"
        + "| 2026-09-16T08:01:00Z | TERMINAL | lane=b |\n"
        + "- 2026-09-16T08:02:00Z NOTE lane=c\n"
        + "### 2026-09-16 — a headed row\n\nbody\n",
        encoding="utf-8",
    )
    result = _run(
        _LOCK,
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
            "1",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(result.stdout)["entries_rolled"] == 3


def test_a_continuation_line_stays_inside_its_row(tmp_path: Path) -> None:
    """A wrapped body line is not a row. Splitting one would double-count it."""
    ledger = _ledger(tmp_path, 0)
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "2026-09-16T08:00:00Z CLAIM lane=a\n"
        + "  continued prose that wraps onto its own line\n"
        + "and more prose with no timestamp at all\n"
        + "2026-09-16T08:05:00Z TERMINAL lane=a\n",
        encoding="utf-8",
    )
    result = _run(
        _LOCK,
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
            "1",
        ],
    )
    assert result.returncode == 0, result.stderr
    receipt = _receipt(result.stdout)
    assert receipt["entries_rolled"] == 1
    archive = Path(receipt["archive"]).read_text(encoding="utf-8")
    assert "continued prose that wraps onto its own line" in archive
    assert "and more prose with no timestamp at all" in archive


# --------------------------------------------------------------------------
# (a) the watermark anchor survives an ordinary peer append.
# --------------------------------------------------------------------------


def test_anchor_survives_an_unheaded_peer_append(tmp_path: Path) -> None:
    """The eleven-day failure, reduced to three commands.

    ``--advance`` anchors on the tail. A peer lane appends one ordinary row.
    ``--resolve`` must find the anchor and report exactly one unread row --
    not exit 3 because the anchor's body grew underneath it.
    """
    ledger = _ledger(tmp_path, 3)
    state = _v2_state(tmp_path, ledger)

    advanced = _watermark(ledger, state, "--advance")
    assert advanced.returncode == 0, advanced.stderr

    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(_row(4))

    resolved = _watermark(ledger, state, "--resolve")
    assert resolved.returncode == 0, (
        "the anchor did not survive one ordinary peer append:\n" + resolved.stderr
    )
    payload = json.loads(resolved.stdout)
    assert payload["anchor_found_in"] == "live"
    assert payload["unread_entries"] == 1
    assert payload["unread_headings"] == [_row(4).rstrip("\n")]


def test_anchor_survives_a_roll_that_moves_it_into_the_archive(tmp_path: Path) -> None:
    """Row identity, not line number: the reader finds itself in the archive
    and still counts the rows the roll moved out from under it."""
    ledger = _ledger(tmp_path, 6)
    state = _v2_state(tmp_path, ledger)
    assert _watermark(ledger, state, "--advance").returncode == 0

    for n in range(7, 11):
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(_row(n))

    rolled = _run(
        _LOCK,
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
        ],
    )
    assert rolled.returncode == 0, rolled.stderr

    resolved = _watermark(ledger, state, "--resolve")
    assert resolved.returncode == 0, resolved.stderr
    payload = json.loads(resolved.stdout)
    assert payload["anchor_found_in"].startswith("archive:")
    assert payload["unread_entries"] == 4


def test_a_genuinely_rewritten_row_still_fails_closed(tmp_path: Path) -> None:
    """Widening the row model must not weaken the fail-closed guarantee."""
    ledger = _ledger(tmp_path, 3)
    state = _v2_state(tmp_path, ledger)
    assert _watermark(ledger, state, "--advance").returncode == 0

    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        text.replace("outcome=DONE_3", "outcome=EDITED"), encoding="utf-8"
    )

    resolved = _watermark(ledger, state, "--resolve")
    assert resolved.returncode == EXIT_UNRESOLVED
    assert "is in neither" in resolved.stderr or "digest changed" in resolved.stderr


# --------------------------------------------------------------------------
# The row-shape guard: an append that cannot start a row is refused.
# --------------------------------------------------------------------------


def test_append_that_does_not_start_a_row_is_refused(tmp_path: Path) -> None:
    """Without this, the row model is true only by the goodwill of callers.

    A payload that does not open a row appends INTO the row above it, which is
    exactly how the 2026-09-04 anchor was invalidated.
    """
    ledger = _ledger(tmp_path, 3)
    before = ledger.read_bytes()
    result = _run(
        _LOCK,
        [
            str(ledger),
            "--append",
            "just some prose with no timestamp",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "block",
        ],
    )
    assert result.returncode == EXIT_ROW_SHAPE, result.stdout + result.stderr
    assert ledger.read_bytes() == before
    assert "row" in result.stderr.lower()


def test_append_in_a_live_row_shape_is_accepted(tmp_path: Path) -> None:
    """The guard must accept every shape the fleet already writes."""
    # OMN-18554: stamped live, because this repo's ledger_lock.py now carries the
    # OMN-17427 clock guard. The two rows that are also rule-4 CLAIM shapes carry a
    # cost sentence for the same reason the claim-token fixtures do -- this test is
    # about the ROW-SHAPE guard, and a row refused earlier in the chain never
    # reaches it.
    priced = "est ~2 lane-hours; displaces nothing; (OMN-17403)"
    for index, payload in enumerate(
        (
            _live_stamp(
                f"2026-09-16T09:00:00Z | CLAIM | lane=x | OMN-17403 {priced} |"
            ),
            _live_stamp(
                f"| 2026-09-16T09:00:00Z | CLAIM | lane=x | OMN-17403 {priced} |"
            ),
            _live_stamp(f"- 2026-09-16T09:00:00Z CLAIM OMN-17403 lane=x {priced}"),
            "### 2026-09-16 — a headed row",
        )
    ):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        ledger = _ledger(case, 3)
        result = _run(
            _LOCK,
            [
                str(ledger),
                "--append",
                payload,
                "--section-heading",
                SECTION,
                "--max-section-rows",
                "100000",
                "--on-cap",
                "block",
            ],
        )
        assert result.returncode == 0, f"{payload!r}: {result.stderr}"
        assert payload in ledger.read_text(encoding="utf-8")


def test_the_guard_only_judges_the_first_line(tmp_path: Path) -> None:
    """A multi-line row is one row. Only its first line opens it."""
    ledger = _ledger(tmp_path, 3)
    # `friction=none` is here because this repo's ledger_lock.py now carries the
    # OMN-18274 friction guard, which requires every TERMINAL row to state its
    # friction one way or the other. Same reasoning as the rule-4 fixtures above:
    # a row refused earlier in the chain never reaches the row-shape guard.
    payload = _live_stamp(
        "2026-09-16T09:00:00Z | TERMINAL | lane=x | friction=none |"
        "\n  wrapped continuation\nmore prose"
    )
    result = _run(
        _LOCK,
        [
            str(ledger),
            "--append",
            payload,
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "100000",
            "--on-cap",
            "block",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "wrapped continuation" in ledger.read_text(encoding="utf-8")


def test_the_guard_is_scoped_to_a_capped_section(tmp_path: Path) -> None:
    """Callers that never opted into a section keep the pre-OMN-17023
    behaviour -- the guard governs the append-only section it was written
    for, not every file this tool can lock."""
    ledger = _ledger(tmp_path, 3)
    result = _run(_LOCK, [str(ledger), "--append", "free-form prose"])
    assert result.returncode == 0, result.stderr
    assert "free-form prose" in ledger.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# (b) a roll that does not get under the cap is a failure, not a success.
# --------------------------------------------------------------------------


def test_roll_that_leaves_the_section_over_cap_fails_loudly(tmp_path: Path) -> None:
    """The live 2026-09-16 reproduction: 8,220 lines, cap 4,000, keep 40,
    29 parsed rows -> entries_rolled 0, section_lines_after 8,220, exit 0.

    A trigger cannot tell that apart from a healthy no-op, which is why the
    section sat at twice its cap for twelve days while the receipt surface
    would have said the roll completed."""
    ledger = _ledger(tmp_path, 30)
    before = ledger.read_bytes()
    result = _run(
        _LOCK,
        [
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "10",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "40",
        ],
    )
    assert result.returncode == EXIT_SECTION_CAP, result.stdout + result.stderr
    assert ledger.read_bytes() == before
    assert "--roll-keep-entries" in result.stderr


def test_a_roll_that_gets_under_the_cap_succeeds(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 30)
    result = _run(
        _LOCK,
        [
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "20",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "4",
        ],
    )
    assert result.returncode == 0, result.stderr
    receipt = _receipt(result.stdout)
    assert receipt["entries_rolled"] == 26
    assert receipt["section_lines_after"] <= 20


def test_a_roll_under_the_cap_is_still_a_clean_no_op(tmp_path: Path) -> None:
    """Under its cap, nothing to do, exit 0 -- the normal daily case."""
    ledger = _ledger(tmp_path, 5)
    before = ledger.read_bytes()
    result = _run(
        _LOCK,
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
            "40",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(result.stdout)["entries_rolled"] == 0
    assert ledger.read_bytes() == before


def test_no_row_is_lost_by_a_roll(tmp_path: Path) -> None:
    """Positive control for the roll's append-only guarantee: every row
    present before is present in the live file or the archive afterwards."""
    ledger = _ledger(tmp_path, 30)
    before = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    result = _run(
        _LOCK,
        [
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION,
            "--max-section-rows",
            "20",
            "--on-cap",
            "roll",
            "--archive-dir",
            str(tmp_path / "archive"),
            "--roll-keep-entries",
            "4",
        ],
    )
    assert result.returncode == 0, result.stderr
    archive = Path(_receipt(result.stdout)["archive"]).read_text(encoding="utf-8")
    live = ledger.read_text(encoding="utf-8")
    missing = [line for line in before if line not in live and line not in archive]
    assert missing == [], missing


# --------------------------------------------------------------------------
# The repair path the exit-3 message has always named and never had.
# --------------------------------------------------------------------------


def _broken_anchor_state(tmp_path: Path, ledger: Path) -> Path:
    state = _v2_state(tmp_path, ledger)
    body = json.loads(state.read_text(encoding="utf-8"))
    body["watermarks"][SOURCE]["anchor_heading"] = (
        "2026-01-01T00:00:00Z | gone | lane=old"
    )
    body["watermarks"][SOURCE]["anchor_digest"] = "deadbeefcafe"
    state.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return state


def test_a_broken_anchor_has_a_repair_path(tmp_path: Path) -> None:
    """``--resolve`` says "Re-anchor deliberately" and, before OMN-17403, the
    tool offered no action that could do it: --resolve, --advance and
    --migrate all resolve first and all exit 3. Eleven consecutive friction
    sweeps correctly refused to hand-edit the state file, so the only
    remaining remedy was the one the rules forbid."""
    ledger = _ledger(tmp_path, 3)
    state = _broken_anchor_state(tmp_path, ledger)

    assert _watermark(ledger, state, "--advance").returncode == EXIT_UNRESOLVED

    repaired = _run(
        _WATERMARK,
        [
            str(ledger),
            "--state",
            str(state),
            "--source",
            SOURCE,
            "--reanchor",
            "--reason",
            "OMN-17403: the anchor predates the row-model fix",
        ],
    )
    assert repaired.returncode == 0, repaired.stderr

    resolved = _watermark(ledger, state, "--resolve")
    assert resolved.returncode == 0, resolved.stderr
    assert json.loads(resolved.stdout)["unread_entries"] == 0


def test_reanchor_records_what_it_abandoned(tmp_path: Path) -> None:
    """A repair that leaves no trace is indistinguishable from a hand edit."""
    ledger = _ledger(tmp_path, 3)
    state = _broken_anchor_state(tmp_path, ledger)
    result = _run(
        _WATERMARK,
        [
            str(ledger),
            "--state",
            str(state),
            "--source",
            SOURCE,
            "--reanchor",
            "--reason",
            "OMN-17403 repair",
        ],
    )
    assert result.returncode == 0, result.stderr
    history = json.loads(state.read_text(encoding="utf-8"))["watermarks"][SOURCE][
        "reanchor_history"
    ]
    assert len(history) == 1
    record = history[0]
    assert record["reason"] == "OMN-17403 repair"
    assert (
        record["abandoned_anchor_heading"] == "2026-01-01T00:00:00Z | gone | lane=old"
    )
    assert record["abandoned_anchor_digest"] == "deadbeefcafe"
    assert record["new_anchor_heading"] == _row(3).rstrip("\n")
    assert record["rows_in_section_at_reanchor"] == 3
    assert record["unread_rows_abandoned"] == "unknown"


def test_reanchor_refuses_when_the_anchor_already_resolves(tmp_path: Path) -> None:
    """The repair is for a broken anchor only. On a healthy one it would skip
    every unread row and call it maintenance -- the exact silent skip the
    watermark exists to prevent."""
    ledger = _ledger(tmp_path, 3)
    state = _v2_state(tmp_path, ledger)
    assert _watermark(ledger, state, "--advance").returncode == 0
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(_row(4))
    before = state.read_bytes()

    result = _run(
        _WATERMARK,
        [
            str(ledger),
            "--state",
            str(state),
            "--source",
            SOURCE,
            "--reanchor",
            "--reason",
            "not actually broken",
        ],
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "resolves" in result.stderr
    assert state.read_bytes() == before


def test_reanchor_requires_a_reason(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    state = _broken_anchor_state(tmp_path, ledger)
    before = state.read_bytes()
    result = _run(
        _WATERMARK,
        [str(ledger), "--state", str(state), "--source", SOURCE, "--reanchor"],
    )
    assert result.returncode == 2
    assert state.read_bytes() == before


def test_reason_without_reanchor_is_a_usage_error(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, 3)
    state = _v2_state(tmp_path, ledger)
    result = _run(
        _WATERMARK,
        [
            str(ledger),
            "--state",
            str(state),
            "--source",
            SOURCE,
            "--resolve",
            "--reason",
            "x",
        ],
    )
    assert result.returncode == 2
