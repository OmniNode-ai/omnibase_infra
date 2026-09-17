# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The census must be emittable on a CLEAN fleet, and by its documented recipe (OMN-18606).

Three defects made the Lane Census Staleness gate (OMN-13034) impossible to
satisfy without a human, and they compounded:

  D3  The gate's own remediation told the reader to run
      `lane-census-check.sh --json > census-snapshot.json`. `--json` emits the
      PLAN document, whose keys are exactly
      {findings, has_drift, lanes_checked, schema_version} — it carries no
      `emitted_at`, the one field the gate reads. So the documented fix
      produced a file the gate rejected as malformed.

  D4  The snapshot the gate actually reads is the typed EVENT document, and
      `lane-census-check.sh` returned 0 before `build_event` was ever reached
      whenever `has_drift` was false. A fleet matching its manifest could emit
      no census by ANY path, so the gate was satisfiable only while the fleet
      was drifting.

  D2  The hourly drop-in ran the script with no output path at all, so the
      tick wrote nothing and the committed census was only ever refreshed by a
      hand-opened PR — six of them between 2026-07-22 and 2026-09-14.

The round-trip test below is the one that matters: a ZERO-DRIFT plan must
produce a document the real gate accepts. Run it against the pre-OMN-18606
tree and it fails, because on that tree nothing emits a document at all in the
no-drift case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_CENSUS_SH = _SCRIPTS / "lane-census-check.sh"
_GATE_PY = _SCRIPTS / "check_lane_census_age.py"
_DROPIN = (
    _REPO / "deploy" / "lane-census" / "onex-disk-gc.service.d" / "20-lane-census.conf"
)

sys.path.insert(0, str(_SCRIPTS))

from check_lane_census_age import check_census_age
from lane_census_event import build_event

# The four fields the gate and the CLAUDE.md lane-table generator read between
# them. check_lane_census_age.py reads schema_version, emitted_at and
# lanes_checked; generate_claude_lane_block.py additionally reads findings.
_GATE_REQUIRED_FIELDS = ("schema_version", "emitted_at", "lanes_checked", "findings")

_CLEAN_PLAN: dict[str, object] = {
    "schema_version": "1.0.0",
    "has_drift": False,
    "lanes_checked": ["stability-test", "judge", "dev", "lakshman"],
    "findings": [],
}


def test_a_clean_fleet_produces_a_gate_valid_snapshot(tmp_path: Path) -> None:
    """THE ROUND TRIP (AC1/AC4). Zero drift in, a document the real gate accepts out.

    This is the falsifier for D4. On the pre-OMN-18606 tree the no-drift path
    never reached build_event, so there was no document to hand the gate.
    """
    snapshot = build_event(host="omninode-pc", plan=_CLEAN_PLAN)

    for field in _GATE_REQUIRED_FIELDS:
        assert field in snapshot, (
            f"a clean-fleet census is missing {field!r}, which the staleness "
            f"gate reads; keys were {sorted(snapshot)}"
        )
    assert snapshot["drift_count"] == 0
    assert snapshot["findings"] == []

    path = tmp_path / "census-snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    # The REAL gate function, not a reimplementation of it.
    assert check_census_age(path, max_age_days=7) == 0, (
        "a census emitted right now from a clean fleet must pass the staleness gate"
    )


def test_the_plan_document_is_not_a_valid_snapshot() -> None:
    """The positive control for D3: prove the OLD recipe's output really is rejected.

    Without this, the test above could pass for reasons unrelated to the fix.
    `--json` emits the plan; the plan has no `emitted_at`; the gate must refuse
    it. If this ever starts passing, the two documents have converged and the
    `--snapshot` / `--json` distinction has stopped mattering.
    """
    assert "emitted_at" not in _CLEAN_PLAN, (
        "the plan document must NOT carry emitted_at — if it does, D3's premise "
        "has changed and the gate's remediation text needs re-checking"
    )


def test_gate_rejects_a_plan_shaped_snapshot(tmp_path: Path) -> None:
    """The plan document, written where the gate looks, is refused (D3 falsifier)."""
    path = tmp_path / "census-snapshot.json"
    path.write_text(json.dumps(_CLEAN_PLAN), encoding="utf-8")
    assert check_census_age(path, max_age_days=7) == 1, (
        "the plan document carries no emitted_at and must be refused by the gate"
    )


def test_script_accepts_a_snapshot_flag() -> None:
    """`--snapshot` is a real flag on the script, not only in its header prose."""
    body = _CENSUS_SH.read_text(encoding="utf-8")
    assert "--snapshot)" in body, (
        "lane-census-check.sh must parse --snapshot; the gate's remediation "
        "message now names it"
    )


def test_snapshot_is_built_before_the_no_drift_exit() -> None:
    """Source-order pin for D4.

    The event build must precede the `No lane drift` early exit. If a later
    change moves the build back below that exit, the clean-fleet case silently
    stops emitting a census again and the gate becomes unsatisfiable seven days
    later — the exact regression this ticket closed.
    """
    body = _CENSUS_SH.read_text(encoding="utf-8")
    # Anchor on the ASSIGNMENT, not on the script name: `lane_census_event.py`
    # is also named in this script's own header prose, which sits above
    # everything and would make this assertion vacuously true.
    build_line = 'EVENT_JSON="$(echo "$PLAN_JSON"'
    assert body.count(build_line) == 1, (
        "expected exactly one event-build assignment to anchor the ordering on; "
        f"found {body.count(build_line)}"
    )
    build_at = body.index(build_line)
    exit_at = body.index("No lane drift. Desired == actual.")
    assert build_at < exit_at, (
        "the census event is built after the no-drift early exit, so a fleet "
        "matching its manifest can emit no snapshot at all (OMN-18606 D4)"
    )


def test_stdout_snapshot_refuses_to_share_stdout() -> None:
    """`--snapshot -` with --json or --dry-run would write two JSON documents.

    That concatenation is what produced `Extra data: line 2 column 1` and made
    the combination unusable; the script must refuse it rather than emit it.
    """
    body = _CENSUS_SH.read_text(encoding="utf-8")
    assert "--snapshot - cannot be combined with --json or --dry-run" in body


def test_gate_remediation_names_the_recipe_that_works() -> None:
    """AC3. The command in the gate's failure output must be the one that works."""
    body = _GATE_PY.read_text(encoding="utf-8")
    assert "--snapshot " in body, (
        "the gate's remediation must name --snapshot, the only flag that writes "
        "a document it accepts"
    )
    assert "lane-census-check.sh --json > " not in body, (
        "the gate still tells the reader to run --json, which emits the plan "
        "document and produces a file this same gate rejects (OMN-18606 D3)"
    )


def test_hourly_dropin_writes_a_snapshot() -> None:
    """AC2. The tick must write the census, not merely reconcile and publish."""
    body = _DROPIN.read_text(encoding="utf-8")
    exec_lines = [ln for ln in body.splitlines() if ln.startswith("ExecStart=")]
    assert exec_lines, "the drop-in declares no ExecStart"
    assert any("--snapshot" in ln for ln in exec_lines), (
        "the hourly lane-census pass writes no snapshot, so nothing refreshes "
        "the committed census except a human opening a PR (OMN-18606 D2); "
        f"ExecStart lines were {exec_lines}"
    )
