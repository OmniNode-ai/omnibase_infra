#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omninode-ci-required-context-probe.py -- detect required checks that never reported.

WHAT THIS IS (OMN-15550)
    A read-only probe that answers one question nothing else in the fleet asks:
    *which required status contexts did NOT show up?*

    It is executed on the ``.201`` host by
    ``deploy/maintenance/omninode-system-slack-report.sh`` inside ``collect()``,
    which means it rides the existing ``*/15`` alert cron, the existing Slack
    poster, the existing state-change de-duplication and the existing
    ``[OmniNode alert resolved]`` path. No new cron unit, no second Slack
    integration, no new dashboard.

THE DEFECT CLASS
    A required status check that never reports is ABSENT, not RED. Branch
    protection blocks the PR identically either way, but only one of the two is
    visible: a RED check has a row in every list, an ABSENT check has no row at
    all. So ``gh pr checks`` shows all-green, the CI Summary poller shows pass,
    every dashboard shows "no failures" -- and the PR is permanently
    unmergeable.

    Every existing surface iterates the checks that EXIST and grades them. This
    probe iterates the REQUIRED set and subtracts what is present. That
    set-difference is the whole point; without reading
    ``branches/<base>/protection/required_status_checks`` the missing case is
    not merely un-alarmed, it is unrepresentable.

WHY IT DOES NOT RUN IN GITHUB ACTIONS
    On 2026-07-30 (OMN-15536) ``omnibase_infra``'s ``ci.yml`` failed to
    assemble. ``CI Summary`` is the SOLE required context on that repo's dev
    branch, so it could not be produced for any PR and all 7 open PRs became
    unmergeable at once, for ~2.5h, discovered by a human noticing. A detector
    implemented inside the CI system it monitors would have failed to assemble
    alongside it. This runs on ``.201``, which is independent of GitHub Actions.

TRI-STATE, AND WHY ONLY ONE OF THE THREE ALARMS
    ABSENT  -- never reported. Invisible today. THIS is what alarms.
    PENDING -- reported, not concluded. Visible today; ageing it out is
               OMN-12560's job. Alarming here would page twice for one stall,
               so PENDING never alarms at any age.
    FAILED  -- reported, concluded not-success. Already visible everywhere.

    Counted and printed in the heartbeat either way, so "we looked and found
    none" is distinguishable from "we did not look" (the structural blindness
    called out in ``reference_detection_shelf_structurally_blind``).

TWO TRAPS THIS PROBE MUST NOT FALL INTO (both cost real time to find)
    1. PAGINATION. Heads in ``omnibase_infra`` carry 100-247 check-runs against
       a GitHub default page size of 30. A single-page probe reports contexts
       as ABSENT that are present on page 2 -- a false-positive generator that
       would train the alert to be ignored. Every list endpoint here follows
       ``Link: rel="next"`` to exhaustion.
    2. TWO SURFACES. A required context can be satisfied by a check-run OR by a
       legacy commit status, so a check-runs-only probe reports every
       commit-status context permanently absent. The present-set is the UNION
       of both endpoints. The worked example was ``CodeRabbit``, which posted a
       legacy commit status while everything else posted check-runs; it was
       removed in OMN-16933, but any GitHub App can report on that surface, so
       the union is not optional.

FAIL-CLOSED, BUT NOT PAGE-ON-EVERY-BLIP
    A GitHub API error is reported as a visible WARNING row -- never silently
    skipped, because "we could not look" must not render as "nothing is wrong".
    It is deliberately NOT CRITICAL: an unreachable API is not evidence that
    PRs are stranded, and paging on every network blip is how an alert channel
    gets muted. A genuinely stranded PR alarms on the next successful tick.

OUTPUT CONTRACT
    One ``ci|STATUS|key|detail`` row per finding on stdout, where STATUS is
    OK / WARNING / CRITICAL. The reporter's ``row_status()`` reads column 2 for
    this row shape and ``row_key()`` de-duplicates on ``ci|<key>``, so ``key``
    must be stable across ticks -- no timestamps, no ages, no flapping counts
    in it. Volatile numbers belong in ``detail``, which is not part of the
    de-duplication identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"  # url-authority-ok: GitHub's public API is an external SaaS control plane, not an ONEX service; it has no routing-authority/integration-catalog entry and cannot acquire one. Routing this probe through our own resolver would couple the outage detector to the infrastructure whose outages it exists to report.
OWNER = os.environ.get("OMNINODE_CI_PROBE_OWNER", "OmniNode-ai")

# Repos whose merge path is gated by required contexts. A repo absent from this
# list is not probed -- same posture as the maintenance-sync MANIFEST: being in
# the list is what makes it governed.
DEFAULT_REPOS = (
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnibase_compat",
    "omniclaude",
    "omnimarket",
    "onex_change_control",
    "omnidash",
)

# Minimum age before an absence can alarm, measured from when CI demonstrably
# began on the head. Below this, "absent" is indistinguishable from "has not
# started yet" on a freshly-pushed head.
GRACE_MINUTES = int(os.environ.get("OMNINODE_CI_ABSENT_GRACE_MINUTES", "20"))

# Above this age an absence alarms even while runs are still in flight. Without
# it, a permanently-hung run would hold a repo in a silent blind spot forever --
# the in-flight suppression below would never lift.
CEILING_MINUTES = int(os.environ.get("OMNINODE_CI_ABSENT_CEILING_MINUTES", "180"))

# How far back to scan workflow runs for the zero-job startup-failure signature.
ZEROJOB_WINDOW_MINUTES = int(os.environ.get("OMNINODE_CI_ZEROJOB_WINDOW_MINUTES", "90"))

# Runs whose wall-clock duration exceeds this cannot be startup failures; used
# as a cheap pre-filter so the expensive per-run jobs call is rare.
ZEROJOB_MAX_DURATION_SECONDS = 5

HTTP_TIMEOUT_SECONDS = int(os.environ.get("OMNINODE_CI_PROBE_TIMEOUT_SECONDS", "20"))

IN_FLIGHT_RUN_STATUSES = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending"}
)

# Conclusions that mean "this check reported and it is not a success". Present,
# therefore already visible, therefore not this probe's alarm.
FAILED_CONCLUSIONS = frozenset({"failure", "timed_out", "action_required", "stale"})


class ProbeError(RuntimeError):
    """A repo-scoped probe failure. Reported as WARNING, never silently dropped."""


def _now() -> datetime:
    """Current UTC time, overridable so grace arithmetic is deterministic in tests."""
    pinned = os.environ.get("OMNINODE_CI_PROBE_NOW")
    if pinned:
        return _parse_ts(pinned) or datetime.now(UTC)
    return datetime.now(UTC)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def emit(row: str) -> None:
    """Write one snapshot row to stdout.

    Raw stdout rather than ``print`` because stdout IS this script's interface:
    the reporter captures these lines verbatim into its snapshot, so no
    formatting layer may sit between the row and the caller.
    """
    sys.stdout.write(row + "\n")


def _sanitize(text: str) -> str:
    """Strip the field delimiter and newlines out of anything bound for a row."""
    return text.replace("|", "/").replace("\n", " ").replace("\r", " ").strip()


class GitHub:
    """Minimal paginating GitHub reader.

    Fixture mode (``OMNINODE_CI_PROBE_FIXTURE_DIR``) resolves each request to a
    recorded JSON file instead of the network. That is what lets the hermetic
    tests drive THIS file -- the artifact that actually runs on the host --
    against real recorded API payloads, rather than proving something about a
    re-implementation (``feedback_test_the_artifact_that_runs``).
    """

    def __init__(self, token: str | None, fixture_dir: str | None = None) -> None:
        self._token = token
        self._fixture_dir = fixture_dir

    @staticmethod
    def slug(path_and_query: str) -> str:
        """Stable filename for a request. Mirrored by the fixture generator."""
        out = []
        for ch in path_and_query:
            out.append(ch if ch.isalnum() else "_")
        return "".join(out).strip("_")[:180]

    def get(self, path: str) -> Any:
        if self._fixture_dir is not None:
            return self._get_fixture(path)
        return self._get_http(path)

    def _get_fixture(self, path: str) -> Any:
        assert self._fixture_dir is not None
        candidate = Path(self._fixture_dir) / (self.slug(path) + ".json")
        if not candidate.exists():
            raise ProbeError(f"no fixture for {path} (expected {candidate})")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("__error__"):
            raise ProbeError(str(payload["__error__"]))
        # `__next__` is the fixture stand-in for a `Link: rel="next"` header, so
        # multi-page reads exercise the same loop offline. Without it the
        # pagination trap in the module docstring would be untestable, and an
        # untested pagination path is how phantom absences ship.
        if isinstance(payload, dict):
            return _WithLink(payload, payload.get("__next__"))
        return payload

    def _get_http(self, path: str) -> Any:
        # S310: the scheme is fixed by API_ROOT above (https, literal), and
        # `path` is built from repo/branch/sha values, never from user input.
        request = urllib.request.Request(f"{API_ROOT}{path}")  # noqa: S310
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "omninode-ci-required-context-probe")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https API_ROOT
                request, timeout=HTTP_TIMEOUT_SECONDS
            ) as response:
                body = response.read().decode("utf-8")
                link = response.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            raise ProbeError(f"HTTP {exc.code} on {path}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProbeError(f"{type(exc).__name__} on {path}") from exc
        parsed = json.loads(body) if body else None
        return _WithLink(parsed, _next_link(link))

    def paginate(self, path: str, key: str | None = None) -> Iterator[Any]:
        """Yield every item across all pages.

        Following ``Link: rel="next"`` to exhaustion is load-bearing, not
        defensive: at a default page size of 30 against heads carrying 200+
        check-runs, a single-page read invents absences.
        """
        current: str | None = path
        seen_pages = 0
        while current:
            payload = self.get(current)
            nxt: str | None = None
            if isinstance(payload, _WithLink):
                nxt = payload.next_path
                payload = payload.value
            items = (
                payload.get(key, []) if (key and isinstance(payload, dict)) else payload
            )
            if isinstance(items, list):
                yield from items
            seen_pages += 1
            if seen_pages > 50:  # hard stop; no legitimate head has 5000 checks
                break
            current = nxt


class _WithLink:
    """Carries a page payload plus the parsed ``rel="next"`` path."""

    __slots__ = ("next_path", "value")

    def __init__(self, value: Any, next_path: str | None) -> None:
        self.value = value
        self.next_path = next_path

    def get(self, key: str, default: Any = None) -> Any:
        return self.value.get(key, default) if isinstance(self.value, dict) else default


def _next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2 or 'rel="next"' not in part:
            continue
        url = section[0].strip().lstrip("<").rstrip(">")
        split = urllib.parse.urlsplit(url)
        return split.path + ("?" + split.query if split.query else "")
    return None


def _unwrap(payload: Any) -> Any:
    return payload.value if isinstance(payload, _WithLink) else payload


def required_contexts(gh: GitHub, repo: str, branch: str) -> list[str]:
    path = (
        f"/repos/{OWNER}/{repo}/branches/"
        f"{urllib.parse.quote(branch, safe='')}/protection/required_status_checks"
    )
    try:
        payload = _unwrap(gh.get(path))
    except ProbeError as exc:
        if "HTTP 404" in str(exc):
            return []  # unprotected branch: nothing is required, nothing to miss
        raise
    contexts = payload.get("contexts", []) if isinstance(payload, dict) else []
    return [str(c) for c in contexts]


def observed_contexts(
    gh: GitHub, repo: str, sha: str
) -> tuple[dict[str, str], datetime | None]:
    """Return ``{context -> state}`` and the earliest moment CI reported on this head.

    ``state`` is one of ``pending`` / ``ok`` / ``failed``. The present-set is the
    union of check-runs and commit statuses because the two surfaces carry
    different producers -- see the module docstring.
    """
    states: dict[str, str] = {}
    earliest: datetime | None = None
    # A context can carry several check-runs when a job is re-run. Branch
    # protection honours the most recent one, so resolve by timestamp rather
    # than by arrival order -- the API does not promise chronological ordering,
    # and "whichever came last in the list wins" would make the verdict depend
    # on pagination boundaries. Observed live: omnibase_infra#2582 head
    # 144b2be7 carries both an in_progress and a completed `CI Summary`.
    newest_seen: dict[str, datetime] = {}

    for run in gh.paginate(
        f"/repos/{OWNER}/{repo}/commits/{sha}/check-runs?per_page=100", "check_runs"
    ):
        if not isinstance(run, dict):
            continue
        name = str(run.get("name", ""))
        if not name:
            continue
        started = _parse_ts(run.get("started_at"))
        if started and (earliest is None or started < earliest):
            earliest = started

        prior = newest_seen.get(name)
        if prior is not None and started is not None and started < prior:
            continue
        if started is not None:
            newest_seen[name] = started

        if run.get("status") != "completed":
            states[name] = "pending"
        else:
            conclusion = str(run.get("conclusion") or "")
            states[name] = "failed" if conclusion in FAILED_CONCLUSIONS else "ok"

    combined = _unwrap(
        gh.get(f"/repos/{OWNER}/{repo}/commits/{sha}/status?per_page=100")
    )
    statuses = combined.get("statuses", []) if isinstance(combined, dict) else []
    for status in statuses:
        if not isinstance(status, dict):
            continue
        context = str(status.get("context", ""))
        if not context:
            continue
        state = str(status.get("state", ""))
        if state == "pending":
            states.setdefault(context, "pending")
        elif state == "success":
            states[context] = "ok"
        else:
            states[context] = "failed"
        created = _parse_ts(status.get("created_at"))
        if created and (earliest is None or created < earliest):
            earliest = created

    return states, earliest


def has_in_flight_runs(gh: GitHub, repo: str, sha: str) -> bool:
    """True when any workflow run on this head is still queued or executing.

    Suppressing an absence while runs are in flight is what keeps a long
    fan-out pipeline (jobs gated behind ``needs:`` legitimately have no
    check-run for tens of minutes) from generating false positives. The ceiling
    above stops that suppression from becoming permanent.
    """
    runs = _unwrap(
        gh.get(f"/repos/{OWNER}/{repo}/actions/runs?head_sha={sha}&per_page=100")
    )
    entries = runs.get("workflow_runs", []) if isinstance(runs, dict) else []
    return any(
        isinstance(r, dict) and str(r.get("status", "")) in IN_FLIGHT_RUN_STATUSES
        for r in entries
    )


def probe_repo(
    gh: GitHub, repo: str, now: datetime
) -> tuple[list[str], dict[str, int]]:
    rows: list[str] = []
    tally = {"prs": 0, "required": 0, "absent": 0, "pending": 0, "failed": 0}

    pulls = list(gh.paginate(f"/repos/{OWNER}/{repo}/pulls?state=open&per_page=100"))
    protection_cache: dict[str, list[str]] = {}

    for pull in pulls:
        if not isinstance(pull, dict) or pull.get("draft"):
            continue
        number = pull.get("number")
        base = str((pull.get("base") or {}).get("ref", ""))
        sha = str((pull.get("head") or {}).get("sha", ""))
        if not (number and base and sha):
            continue

        if base not in protection_cache:
            protection_cache[base] = required_contexts(gh, repo, base)
        required = protection_cache[base]
        if not required:
            continue

        tally["prs"] += 1
        tally["required"] += len(required)

        states, earliest = observed_contexts(gh, repo, sha)
        missing = [ctx for ctx in required if ctx not in states]
        tally["pending"] += sum(1 for c in required if states.get(c) == "pending")
        tally["failed"] += sum(1 for c in required if states.get(c) == "failed")

        if not missing:
            continue

        # Age from when CI demonstrably began on this head. When nothing has
        # reported at all there is no such moment, so fall back to the head
        # commit's own timestamp rather than treating the head as brand new.
        anchor = earliest
        if anchor is None:
            anchor = _parse_ts(
                ((pull.get("head") or {}).get("repo") or {}).get("pushed_at")
            ) or _parse_ts(pull.get("created_at"))
        if anchor is None:
            continue
        age_minutes = int((now - anchor).total_seconds() // 60)

        if age_minutes < GRACE_MINUTES:
            continue
        if age_minutes < CEILING_MINUTES and has_in_flight_runs(gh, repo, sha):
            continue

        tally["absent"] += len(missing)
        for context in missing:
            key = _sanitize(f"absent/{repo}#{number}/{context}")
            detail = _sanitize(
                f"required context never reported: {age_minutes}m since CI started "
                f"on {sha[:8]}, {len(states)} other contexts present, "
                f"base={base} grace={GRACE_MINUTES}m"
            )
            rows.append(f"ci|CRITICAL|{key}|{detail}")

    return rows, tally


def probe_zero_job_runs(gh: GitHub, repo: str, now: datetime) -> list[str]:
    """Flag workflow runs that concluded having created zero jobs.

    This is the startup-failure signature and it has no benign reading: GitHub
    could not assemble the workflow file, so it never parsed the file's
    ``name:`` (the run renders as the raw path) and never created a job. It
    fires upstream of the absent context and is unambiguous on its own.
    """
    rows: list[str] = []
    runs = _unwrap(gh.get(f"/repos/{OWNER}/{repo}/actions/runs?per_page=50"))
    entries = runs.get("workflow_runs", []) if isinstance(runs, dict) else []

    for run in entries:
        if not isinstance(run, dict) or run.get("status") != "completed":
            continue
        conclusion = str(run.get("conclusion") or "")
        if conclusion in {"success", "skipped", "cancelled", "neutral"}:
            continue
        created = _parse_ts(run.get("created_at"))
        updated = _parse_ts(run.get("updated_at"))
        if (
            created is None
            or (now - created).total_seconds() > ZEROJOB_WINDOW_MINUTES * 60
        ):
            continue

        if conclusion != "startup_failure":
            # Cheap pre-filter: a run that did real work cannot be a startup
            # failure, so only near-instant runs are worth a jobs lookup.
            if updated is None:
                continue
            if (updated - created).total_seconds() > ZEROJOB_MAX_DURATION_SECONDS:
                continue
            jobs = _unwrap(
                gh.get(
                    f"/repos/{OWNER}/{repo}/actions/runs/{run.get('id')}/jobs?per_page=1"
                )
            )
            total = jobs.get("total_count", -1) if isinstance(jobs, dict) else -1
            if total != 0:
                continue

        key = _sanitize(f"zerojob/{repo}/{run.get('id')}")
        detail = _sanitize(
            f"workflow run concluded '{conclusion}' with 0 jobs (startup failure): "
            f"{run.get('name', '?')} head={str(run.get('head_sha', ''))[:8]} "
            f"branch={run.get('head_branch', '?')}"
        )
        rows.append(f"ci|CRITICAL|{key}|{detail}")

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repos",
        default=os.environ.get("OMNINODE_CI_PROBE_REPOS", ",".join(DEFAULT_REPOS)),
        help="comma-separated repo names to probe",
    )
    parser.add_argument(
        "--skip-zero-job-scan",
        action="store_true",
        help="skip the workflow-run startup-failure scan",
    )
    args = parser.parse_args(argv)

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    fixture_dir = os.environ.get("OMNINODE_CI_PROBE_FIXTURE_DIR") or None
    token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    gh = GitHub(token, fixture_dir)
    now = _now()

    rows: list[str] = []
    totals = {"prs": 0, "required": 0, "absent": 0, "pending": 0, "failed": 0}
    scanned = 0
    degraded: list[str] = []

    if not fixture_dir and not token:
        # Unauthenticated reads cannot see branch protection at all, so the
        # probe would report a clean board while being structurally blind.
        emit(
            "ci|WARNING|probe/credentials|no GH_PAT or GITHUB_TOKEN in environment; "
            "required-context probe cannot read branch protection and did NOT run"
        )
        return 0

    for repo in repos:
        try:
            repo_rows, tally = probe_repo(gh, repo, now)
            rows.extend(repo_rows)
            for name, value in tally.items():
                totals[name] += value
            if not args.skip_zero_job_scan:
                rows.extend(probe_zero_job_runs(gh, repo, now))
            scanned += 1
        except ProbeError as exc:
            degraded.append(repo)
            rows.append(
                f"ci|WARNING|probe/{_sanitize(repo)}|"
                f"probe did not complete, state unknown: {_sanitize(str(exc))}"
            )
        except Exception as exc:  # noqa: BLE001 - one repo must not kill the tick
            degraded.append(repo)
            rows.append(
                f"ci|WARNING|probe/{_sanitize(repo)}|"
                f"unexpected probe error, state unknown: {_sanitize(type(exc).__name__)}"
            )

    if scanned == 0:
        # "Scanned nothing" must never render as "found nothing wrong" -- that
        # is the detection-shelf blindness this heartbeat exists to make loud.
        rows.append(
            f"ci|WARNING|required-contexts|probe scanned 0 of {len(repos)} repos; "
            f"no required-context coverage this tick"
        )
    else:
        rows.append(
            f"ci|OK|required-contexts|scanned {scanned}/{len(repos)} repos, "
            f"{totals['prs']} open non-draft PRs, {totals['required']} required contexts; "
            f"absent={totals['absent']} pending={totals['pending']} failed={totals['failed']}"
            + (f" degraded={','.join(degraded)}" if degraded else "")
        )

    for row in rows:
        emit(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
