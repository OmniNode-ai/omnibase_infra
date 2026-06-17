# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""check_lane_census_age.py — CI gate: reject a census snapshot older than MAX_AGE_DAYS (OMN-13034).

THE CLASS FIX (retro B-6): the CLAUDE.md lane table must be a generated block
derived from a machine-readable census snapshot. The snapshot is produced by
lane-census-check.sh on .201 (the census timer). This gate asserts that:

  1. A census snapshot file exists at deploy/lane-census/census-snapshot.json.
  2. Its `emitted_at` timestamp is within MAX_AGE_DAYS (default 7) of now.
  3. The snapshot's `schema_version` is supported.

A census older than MAX_AGE_DAYS means the .201 timer has not run (or the
snapshot was never committed) — stale documentation is the failure we are
ratcheting against.

Exit codes:
  0 — snapshot present and fresh
  1 — snapshot missing, stale, or malformed (hard failure — blocks CI)

Usage:
  python scripts/check_lane_census_age.py
  python scripts/check_lane_census_age.py --max-age-days 7 --snapshot deploy/lane-census/census-snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_SNAPSHOT = _REPO / "deploy" / "lane-census" / "census-snapshot.json"
_DEFAULT_MAX_AGE_DAYS = 7
_SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}


def check_census_age(
    snapshot_path: Path,
    max_age_days: int,
    *,
    now: datetime | None = None,
) -> int:
    """Check that the census snapshot exists and is fresh.

    Returns 0 on success, 1 on failure. Prints diagnostic messages to stderr.
    """
    now = now or datetime.now(UTC)

    if not snapshot_path.exists():
        print(
            f"::error title=LANE-CENSUS-STALE::Census snapshot missing: "
            f"{snapshot_path.relative_to(_REPO) if snapshot_path.is_relative_to(_REPO) else snapshot_path}. "
            f"Run 'bash scripts/lane-census-check.sh --json > "
            f"deploy/lane-census/census-snapshot.json' on .201 and commit the "
            f"result. A missing snapshot means the lane table is undocumented — "
            f"the failure class OMN-13034 was written to prevent.",
            file=sys.stderr,
        )
        return 1

    try:
        with open(snapshot_path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"::error title=LANE-CENSUS-MALFORMED::Census snapshot is not valid "
            f"JSON: {exc}. Path: {snapshot_path}",
            file=sys.stderr,
        )
        return 1

    schema = snapshot.get("schema_version", "")
    if schema not in _SUPPORTED_SCHEMA_VERSIONS:
        print(
            f"::error title=LANE-CENSUS-MALFORMED::Unsupported census snapshot "
            f"schema_version {schema!r}. Expected one of "
            f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}.",
            file=sys.stderr,
        )
        return 1

    emitted_raw = snapshot.get("emitted_at", "")
    if not emitted_raw:
        print(
            "::error title=LANE-CENSUS-MALFORMED::Census snapshot missing "
            "'emitted_at' field.",
            file=sys.stderr,
        )
        return 1

    try:
        emitted = datetime.fromisoformat(emitted_raw)
    except ValueError as exc:
        print(
            f"::error title=LANE-CENSUS-MALFORMED::Census snapshot 'emitted_at' "
            f"is not a valid ISO-8601 timestamp: {emitted_raw!r} — {exc}",
            file=sys.stderr,
        )
        return 1

    # Ensure both datetimes are timezone-aware for comparison.
    if emitted.tzinfo is None:
        emitted = emitted.replace(tzinfo=UTC)

    age = now - emitted
    max_age = timedelta(days=max_age_days)

    if age > max_age:
        age_days = age.days
        print(
            f"::error title=LANE-CENSUS-STALE::Census snapshot is {age_days} days "
            f"old (emitted_at={emitted_raw!r}). Maximum allowed age is "
            f"{max_age_days} days. Re-run the lane census on .201 and commit an "
            f"updated deploy/lane-census/census-snapshot.json. Stale census = "
            f"undocumented runtime topology (retro B-6 / OMN-13034).",
            file=sys.stderr,
        )
        return 1

    lanes_checked = snapshot.get("lanes_checked", [])
    age_hours = int(age.total_seconds() / 3600)
    print(
        f"OK: census snapshot is {age_hours}h old "
        f"(emitted_at={emitted_raw!r}, lanes={lanes_checked})",
        file=sys.stdout,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CI gate: reject a lane census snapshot older than MAX_AGE_DAYS"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_DEFAULT_SNAPSHOT,
        help="Path to census-snapshot.json (default: deploy/lane-census/census-snapshot.json)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=_DEFAULT_MAX_AGE_DAYS,
        help=f"Maximum allowed census age in days (default: {_DEFAULT_MAX_AGE_DAYS})",
    )
    args = parser.parse_args(argv)
    return check_census_age(args.snapshot, args.max_age_days)


if __name__ == "__main__":
    raise SystemExit(main())
