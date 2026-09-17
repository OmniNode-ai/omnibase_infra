# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_census_refresh_decision.py — should the census refresh open a PR? (OMN-18606).

The refresh leg (``.github/workflows/lane-census-refresh.yml``) collects a fresh
census on the lab host several times a day. It must NOT open a pull request every
time: that would be four no-op PRs a day forever, which is how an automation
earns itself an exemption. This module is the pure decision that sits between
collection and PR-opening, so the rule is testable without a runner, a docker
socket, or a GitHub token.

WHY A SEPARATE PURE MODULE. The workflow step around it does I/O and cannot be
unit tested; the decision is where every interesting edge lives (a missing
committed file, a clock that went backwards, a topology change, an aging but
otherwise identical census). Keeping it pure is the same split the census itself
already uses — ``lane_census_inventory.py`` does all the docker I/O and
``lane_census_plan.py`` is a pure planner over its output.

THE RULES, in evaluation order. Order matters and the first two are guards:

1. The candidate must itself be a valid snapshot. A malformed candidate is an
   ERROR, never a refresh — writing it would replace a good committed census
   with a bad one and fail the staleness gate on the next PR.
2. The committed file missing or unreadable is a REFRESH. That is the
   bootstrap case and the corruption case, and both want a good file written.
3. A candidate NOT NEWER than the committed snapshot is a NO-OP, always. This is
   the clock-skew guard: the leg must never move ``emitted_at`` backwards,
   because the staleness gate reads that field and a backwards write would
   manufacture staleness out of a healthy host.
4. A changed ``alert_key`` is a REFRESH. That key is a content hash of the
   census findings (``lane_census_event._alert_key``), so a difference means the
   lane topology actually moved and the committed census now describes a fleet
   that no longer exists. This is the case worth a PR on its own merits, aging
   or not.
5. An aging committed census is a REFRESH. The threshold is deliberately well
   under the gate's own 7-day limit so the PR has days to land rather than
   racing the deadline.
6. Otherwise NO-OP: the committed census is recent and still describes the
   fleet accurately. Nothing to say.

Exit codes (CLI): ``0`` decision computed (read ``refresh`` from the JSON on
stdout), ``1`` the candidate is malformed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Matches check_lane_census_age.py. A snapshot whose schema this does not know
# is not one this module will reason about.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0"})

# Deliberately well inside the gate's 7-day limit (check_lane_census_age.py
# _DEFAULT_MAX_AGE_DAYS). The margin is the point: a refresh PR opened at day 3
# has four days to get reviewed, land, and propagate before anything goes red.
# A threshold at or near 7 would make every refresh a race against the gate.
DEFAULT_REFRESH_AFTER_DAYS = 3

REASON_COMMITTED_UNREADABLE = "committed_unreadable"
REASON_CANDIDATE_NOT_NEWER = "candidate_not_newer"
REASON_TOPOLOGY_CHANGED = "topology_changed"
REASON_AGING = "aging"
REASON_FRESH_AND_UNCHANGED = "fresh_and_unchanged"


class MalformedCandidateError(ValueError):
    """The freshly collected census is not a snapshot this leg may commit."""


@dataclass(frozen=True)
class Decision:
    """The refresh verdict and the one-line reason it will be reported under."""

    refresh: bool
    reason: str
    detail: str


def _parse_emitted_at(snapshot: dict[str, Any]) -> datetime | None:
    """Return a timezone-aware ``emitted_at``, or ``None`` if unusable."""
    raw = snapshot.get("emitted_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def validate_candidate(candidate: dict[str, Any]) -> datetime:
    """Return the candidate's ``emitted_at``, or raise if it is not committable.

    Guard 1. A candidate that would fail the staleness gate must never reach the
    repository, because committing it converts a healthy host into a red gate.
    """
    schema = candidate.get("schema_version", "")
    if schema not in _SUPPORTED_SCHEMA_VERSIONS:
        raise MalformedCandidateError(
            f"candidate census schema_version {schema!r} is not supported "
            f"(expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)})"
        )
    emitted = _parse_emitted_at(candidate)
    if emitted is None:
        raise MalformedCandidateError(
            f"candidate census has no usable 'emitted_at' "
            f"(got {candidate.get('emitted_at')!r}); the staleness gate reads "
            f"that field, so this snapshot is not committable"
        )
    return emitted


def decide_refresh(
    committed: dict[str, Any] | None,
    candidate: dict[str, Any],
    *,
    refresh_after_days: int = DEFAULT_REFRESH_AFTER_DAYS,
    now: datetime | None = None,
) -> Decision:
    """Decide whether ``candidate`` should replace ``committed`` in a PR.

    ``committed is None`` means the committed file is absent or unreadable.
    Raises :class:`MalformedCandidateError` if the candidate is not committable.
    """
    now = now or datetime.now(UTC)
    candidate_emitted = validate_candidate(candidate)

    if committed is None:
        return Decision(
            refresh=True,
            reason=REASON_COMMITTED_UNREADABLE,
            detail="no readable committed census; writing the collected one",
        )

    committed_emitted = _parse_emitted_at(committed)
    if committed_emitted is None:
        return Decision(
            refresh=True,
            reason=REASON_COMMITTED_UNREADABLE,
            detail="committed census carries no usable 'emitted_at'",
        )

    # Guard 2 — never move emitted_at backwards. A host whose clock disagrees
    # with the committed file must not be able to manufacture staleness.
    if candidate_emitted <= committed_emitted:
        return Decision(
            refresh=False,
            reason=REASON_CANDIDATE_NOT_NEWER,
            detail=(
                f"collected census ({candidate_emitted.isoformat()}) is not newer "
                f"than the committed one ({committed_emitted.isoformat()})"
            ),
        )

    committed_key = committed.get("alert_key")
    candidate_key = candidate.get("alert_key")
    if committed_key != candidate_key:
        return Decision(
            refresh=True,
            reason=REASON_TOPOLOGY_CHANGED,
            detail=(
                f"lane topology changed: alert_key {committed_key!r} -> "
                f"{candidate_key!r}"
            ),
        )

    age = now - committed_emitted
    if age >= timedelta(days=refresh_after_days):
        return Decision(
            refresh=True,
            reason=REASON_AGING,
            detail=(
                f"committed census is {age.days}d old, at or past the "
                f"{refresh_after_days}d refresh threshold"
            ),
        )

    return Decision(
        refresh=False,
        reason=REASON_FRESH_AND_UNCHANGED,
        detail=(
            f"committed census is {int(age.total_seconds() // 3600)}h old and "
            f"describes the same lane topology; nothing to say"
        ),
    )


def _load(path: Path) -> dict[str, Any] | None:
    """Read a snapshot, returning ``None`` for absent/unreadable/malformed."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--refresh-after-days", type=int, default=DEFAULT_REFRESH_AFTER_DAYS
    )
    args = parser.parse_args(argv)

    candidate = _load(args.candidate)
    if candidate is None:
        print(
            f"ERROR: candidate census at {args.candidate} is absent or not valid "
            f"JSON. A census that cannot be read is never committed.",
            file=sys.stderr,
        )
        return 1

    try:
        decision = decide_refresh(
            _load(args.committed),
            candidate,
            refresh_after_days=args.refresh_after_days,
        )
    except MalformedCandidateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    json.dump(asdict(decision), sys.stdout)
    sys.stdout.write("\n")
    print(
        f"lane-census-refresh: refresh={str(decision.refresh).lower()} "
        f"reason={decision.reason} — {decision.detail}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
