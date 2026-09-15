#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC5 -- read the deploy agent's onex-lab overlay record for one sha.

WHY THIS EXISTS
---------------
The k3s ``onex-lab`` overlay can only be applied from the lab host: the k3s admin
kubeconfig is ``0600 root``, the containerd socket is root-owned, and the runtime
image for the merged sha lives on that host's Docker daemon because the deploy
agent just built it. The runner fleet holds a READ-ONLY ``lab-ci-reader``
ServiceAccount kubeconfig (OMN-18188), which can read the lane and not apply to
it. So the apply belongs to the agent.

A GitHub Actions artifact, on the other hand, can only be created by a job inside
a run, and the lab-pass receipt surface is an artifact by deliberate design (see
``lab_pass_receipt.py``'s header: an artifact needs no new credential, no bucket
and no cluster access). So the receipt belongs to CI.

This script is the join: it polls the agent's own HTTP surface for the record
keyed by the merged sha and prints the record's checks as the JSON array
``lab_pass_receipt.py emit --checks-json`` consumes.

IT ALWAYS PRINTS A CHECK LIST, and that is the point. A receipt that exists only
when the apply worked cannot distinguish "the lab apply failed" from "nobody ran
it", which is the distinction rule 24(a)'s ``if: always()`` emitters exist for. A
timeout, an unreachable agent, a 404, a 500 and a malformed body are each turned
into a NAMED FAILING CHECK carrying what was actually read, so the receipt is
emitted either way and the reason is legible in it.

DESCENDANT TOLERANCE (OMN-18399)
---------------------------------
The exact-sha lookup above assumes the agent eventually writes a record keyed by
the sha THIS run requested. Under a busy ``dev`` branch it does not: the agent
keys its record by whatever HEAD ``git_pull()`` resolved to when it reached this
job (``deploy_agent/agent.py`` -- ``self._current_git_sha``), not by the sha of
the command that triggered the run. On a busy branch that is a LATER descendant,
so the exact record this script asks for is never written at all -- there is no
404-then-appears window, the key simply never exists. Compounding it, the agent's
synchronous rebuild blocks its own event loop (OMN-13760), so every poll during
that build times out rather than 404ing, which is exactly what was measured on
run 34976826412: 54 consecutive transport timeouts across the full 1800s window.

This mirrors OMN-18388, which fixed the same failure mode for the compose-dev
lane's ``check_dev_lane_staleness.py`` by accepting a lane running a DESCENDANT
of the merge sha as converged, resolved via GitHub's compare API rather than
local git (the job's checkout is depth-1). The mapping here is the same:
``compare/{requested}...{observed}`` status ``ahead`` means the observed record
is AHEAD of the requested sha -- it CONTAINS it, i.e. a descendant -- and
``behind`` means the observed record is an ANCESTOR of the requested sha, i.e.
older and still not applied. The relation logic is intentionally duplicated
rather than imported from ``check_dev_lane_staleness.py``: the two readers cover
different surfaces (an HTTP record vs. a running container) and must stay
independently correct even if one changes.

When the exact-sha poll exhausts its window, this script makes ONE follow-up
call to the agent's ``/lab-overlay-latest`` endpoint (the most recently written
record, any sha) and accepts it only when that record's sha is IDENTICAL to or a
DESCENDANT of the requested sha. An ANCESTOR or UNRELATED latest record, or no
latest record at all, keeps the original timeout failure -- descendant tolerance
never turns a genuinely stale or absent apply into a pass. The receipt's ``sha``
field is untouched by any of this (set by ``lab_pass_receipt.py emit --sha``
from the workflow's own merge-sha output); only the evidence text and the check
list content change, and the accepting check names the observed descendant sha
so the tolerance is visible to a reader.

STDLIB ONLY, plus the ``gh`` CLI binary already present on the runner (never a
pip package) for the same compare-API calls ``check_dev_lane_staleness.py``
already makes. The emitting job runs on a bare self-hosted runner and the
receipt module it feeds is stdlib-plus-hand-written-validation for a measured
reason: the first live run of that emitter died at import on a missing
third-party package after all its checks had already passed, and took the whole
delivery with it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

#: Only an exact commit. An abbreviated ref could resolve to more than one
#: commit, and a record fetched by prefix is a record about the wrong change.
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The check name that carries a transport or absence failure. One name, so a
#: reader grepping a run for the lab-overlay verdict finds it whatever went
#: wrong, instead of having to know the failure taxonomy first.
RECORD_CHECK = "lab_overlay_record"

# OMN-18399. What the LATEST recorded apply is, relative to the REQUESTED sha.
# ``compare/{base}...{head}`` describes HEAD relative to BASE, so with
# base=requested, head=observed: "ahead" means the observed record is ahead of
# the requested sha -- it contains it, i.e. a descendant.
RELATION_IDENTICAL = "identical"
RELATION_DESCENDANT = "descendant"
RELATION_ANCESTOR = "ancestor"
RELATION_UNRELATED = "unrelated"

_RELATION_BY_COMPARE_STATUS = {
    "identical": RELATION_IDENTICAL,
    "ahead": RELATION_DESCENDANT,
    "behind": RELATION_ANCESTOR,
    "diverged": RELATION_UNRELATED,
}

#: Relations that mean the latest applied record covers the requested change.
_CONTAINING_RELATIONS = frozenset({RELATION_IDENTICAL, RELATION_DESCENDANT})


def _truncate(text: str, limit: int = 400) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def fetch_once(url: str, timeout_seconds: float) -> tuple[int, str]:
    """GET the record. Never raises -- the failure IS the evidence."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - any transport failure is evidence
        return 0, f"{type(exc).__name__}: {exc}"


def _validate_checks_shape(checks: Any, *, url: str) -> list[dict[str, Any]] | None:
    """Structural validation shared by the exact-sha and latest-record readers.

    Returns the checks list unchanged when it is a well-formed, non-empty list
    of ``{name, ok, evidence}`` objects, or ``None`` when it is not. ``None``
    rather than a failing check here: the two callers disagree on what a
    malformed list MEANS (a hard failure for the exact record, "no usable
    fallback" for the latest one), so the shared helper reports only the fact.
    """
    if not isinstance(checks, list) or not checks:
        return None
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"name", "ok", "evidence"}:
            return None
    return checks


def checks_from_record(payload: Any, *, sha: str, url: str) -> list[dict[str, Any]]:
    """Validate one record and return its checks, or one failing check saying why.

    A record whose ``sha`` disagrees with the one requested is REFUSED rather than
    accepted: the name and the payload disagreeing is itself a finding, and it is
    the same check ``lab_pass_receipt.evaluate_gate`` makes on an artifact for the
    same reason.
    """
    if not isinstance(payload, dict):
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": f"GET {url} returned a {type(payload).__name__}, not an object",
            }
        ]
    if payload.get("sha") != sha:
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": (
                    f"GET {url} returned a record for sha {payload.get('sha')!r}; "
                    f"the request named {sha}. The reader refuses a record whose "
                    "payload and key disagree."
                ),
            }
        ]
    checks = _validate_checks_shape(payload.get("checks"), url=url)
    if checks is None:
        raw = payload.get("checks")
        if not isinstance(raw, list) or not raw:
            evidence = (
                f"GET {url} returned a record carrying no checks. A record with "
                "no checks asserts that nothing was verified."
            )
        else:
            bad = next(
                c
                for c in raw
                if not isinstance(c, dict) or set(c) != {"name", "ok", "evidence"}
            )
            evidence = (
                f"GET {url} returned a check that is not "
                f"{{name, ok, evidence}}: {_truncate(json.dumps(bad))}"
            )
        return [{"name": RECORD_CHECK, "ok": False, "evidence": evidence}]
    return checks


def _gh_compare_status(repo: str, base: str, head: str) -> str:
    """The ``status`` field of ``GET repos/{repo}/compare/{base}...{head}``.

    Via the ``gh`` CLI, not a raw HTTPS call: it is already the runner's
    credential path (``GH_TOKEN`` env, the job's own token, no new secret) and
    the same call ``check_dev_lane_staleness.py`` makes for the compose-dev
    lane. Raises on any subprocess or parse failure; the caller treats an
    unresolvable ancestry as a FAIL, never a silent pass.
    """
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/compare/{base}...{head}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        msg = f"gh api compare failed (exit {result.returncode}): {_truncate(result.stderr)}"
        raise RuntimeError(msg)
    payload = json.loads(result.stdout)
    status = payload.get("status")
    if not isinstance(status, str):
        msg = f"gh api compare returned no status field: {_truncate(result.stdout)}"
        raise RuntimeError(msg)
    return status


def resolve_relation(repo: str, requested_sha: str, observed_sha: str) -> str:
    """How ``observed_sha`` (the agent's latest applied record) relates to
    ``requested_sha`` (what this run asked for).

    Duplicated from, not imported from, ``check_dev_lane_staleness.py``'s
    identical mapping: the two guards read different surfaces (an HTTP record
    vs. a running container) and must stay independently correct even if one
    changes.
    """
    if requested_sha == observed_sha:
        return RELATION_IDENTICAL
    status = _gh_compare_status(repo, requested_sha, observed_sha)
    return _RELATION_BY_COMPARE_STATUS.get(status, RELATION_UNRELATED)


def resolve_via_latest(
    *,
    base_url: str,
    repo: str,
    sha: str,
    request_timeout_seconds: float,
    fetch: Any = None,
    compare: Any = resolve_relation,
) -> list[dict[str, Any]] | None:
    """One attempt to satisfy ``sha`` from the agent's most recently applied record.

    Returns:

    * a checks list, ``ok: True`` on its lead entry, when the latest record's
      sha is IDENTICAL to or a DESCENDANT of ``sha`` -- the change was applied,
      just not under the exact key this run requested;
    * a checks list, ``ok: False``, when the latest record's sha is an
      ANCESTOR of ``sha`` (older -- genuinely not applied yet) or UNRELATED, or
      when the ancestry itself cannot be resolved;
    * ``None`` when there is no latest record to compare against at all (the
      endpoint 404s, is unreachable, or returns something unparseable) -- the
      caller keeps its own, already-informative timeout message rather than
      replacing it with a less specific one.
    """
    if fetch is None:
        fetch = fetch_once
    url = f"{base_url.rstrip('/')}/lab-overlay-latest"
    status, body = fetch(url, request_timeout_seconds)
    if status != 200:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    observed_sha = payload.get("sha")
    if not isinstance(observed_sha, str) or not SHA_RE.match(observed_sha):
        return None
    checks = _validate_checks_shape(payload.get("checks"), url=url)
    if checks is None:
        return None

    try:
        relation = compare(repo, sha, observed_sha)
    except Exception as exc:  # noqa: BLE001 - unresolvable ancestry is a FAIL, not a crash
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": (
                    f"no exact record for {sha}; the agent's latest applied record "
                    f"is for {observed_sha}, but its relation to {sha} on {repo} "
                    f"could not be resolved: {type(exc).__name__}: {exc}"
                ),
            }
        ]

    if relation in _CONTAINING_RELATIONS:
        descendant_note = {
            "name": RECORD_CHECK,
            "ok": True,
            "evidence": (
                f"no exact record for {sha}; the agent's latest applied record is "
                f"for {observed_sha}, a {relation} of {sha} on {repo} -- accepted "
                "via descendant tolerance (OMN-18399). Checks below are from that "
                "record."
            ),
        }
        return [descendant_note, *checks]

    reason = (
        f"{observed_sha} is an ANCESTOR of {sha} -- older, not applied yet"
        if relation == RELATION_ANCESTOR
        else f"{observed_sha} is UNRELATED to {sha} on {repo}"
    )
    return [
        {
            "name": RECORD_CHECK,
            "ok": False,
            "evidence": (
                f"no exact record for {sha}; the agent's latest applied record is "
                f"for {observed_sha} ({reason})."
            ),
        }
    ]


def poll(
    *,
    base_url: str,
    sha: str,
    repo: str,
    wait_seconds: int,
    poll_interval_seconds: int,
    request_timeout_seconds: float,
    out: Any,
    sleep: Any = time.sleep,
    now: Any = time.monotonic,
    resolve_latest: Callable[..., list[dict[str, Any]] | None] = resolve_via_latest,
) -> list[dict[str, Any]]:
    """Poll until a record exists, the window closes, or a hard failure occurs.

    A ``404`` is the only retryable answer: it means the apply has not finished
    writing yet, which is expected -- the agent writes the record after a
    multi-minute apply that begins only once the compose lane has converged. A
    ``400``, a ``500`` or a malformed body is terminal, because retrying cannot
    change any of them and spending the window on a certainty is time the run
    could have spent reporting it.
    """
    url = f"{base_url.rstrip('/')}/lab-overlay/{sha}"
    deadline = now() + wait_seconds
    attempts = 0
    last = ""
    while True:
        attempts += 1
        status, body = fetch_once(url, request_timeout_seconds)
        if status == 200:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                return [
                    {
                        "name": RECORD_CHECK,
                        "ok": False,
                        "evidence": f"GET {url} -> 200 but not JSON: {exc}",
                    }
                ]
            print(f"record found after {attempts} attempt(s)", file=out)
            return checks_from_record(payload, sha=sha, url=url)
        if status in (400, 500):
            return [
                {
                    "name": RECORD_CHECK,
                    "ok": False,
                    "evidence": f"GET {url} -> {status} {_truncate(body)}",
                }
            ]
        last = f"{status} {_truncate(body, 160)}"
        remaining = deadline - now()
        if remaining <= 0:
            timeout_checks = [
                {
                    "name": RECORD_CHECK,
                    "ok": False,
                    "evidence": (
                        f"GET {url} never returned a record within {wait_seconds}s "
                        f"({attempts} attempt(s), last: {last}). The lab overlay "
                        "was not re-applied for this sha, or the deploy agent did "
                        "not reach the step that records it."
                    ),
                }
            ]
            # OMN-18399. Under busy dev the agent can have applied a LATER
            # descendant of `sha` and keyed its record by that head instead --
            # one follow-up call against /lab-overlay-latest, never widening
            # the window itself. `resolve_latest` returns None when there is
            # nothing to compare against, in which case the timeout evidence
            # above stands unchanged.
            fallback = resolve_latest(
                base_url=base_url,
                repo=repo,
                sha=sha,
                request_timeout_seconds=request_timeout_seconds,
            )
            if fallback is not None:
                print(
                    "exact record absent; resolved via /lab-overlay-latest "
                    "descendant tolerance",
                    file=out,
                )
                return fallback
            return timeout_checks
        print(f"attempt {attempts}: {last}; {int(remaining)}s left", file=out)
        sleep(min(poll_interval_seconds, max(1, int(remaining))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True, help="the merged 40-character sha")
    parser.add_argument(
        "--agent-url",
        required=True,
        help="base URL of the deploy agent's HTTP surface on the lab host",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "owner/repo, used only for the OMN-18399 descendant-tolerance "
            "fallback's gh api compare calls"
        ),
    )
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--out",
        required=True,
        help="path the checks JSON array is written to",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not SHA_RE.match(args.sha):
        print(
            f"::error::{args.sha!r} is not a 40-character lowercase commit sha. "
            "Refusing to resolve an abbreviated ref.",
            file=sys.stderr,
        )
        return 1
    checks = poll(
        base_url=args.agent_url,
        sha=args.sha,
        repo=args.repo,
        wait_seconds=args.wait_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        out=sys.stderr,
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
        handle.write("\n")
    failing = [check for check in checks if not check["ok"]]
    for check in checks:
        print(
            f"  [{'ok  ' if check['ok'] else 'FAIL'}] {check['name']}: "
            f"{check['evidence']}",
            file=sys.stderr,
        )
    # Exit 0 even when checks fail. The verdict is DERIVED from the checks by
    # `lab_pass_receipt.py emit`, which is the one place a verdict is computed, and
    # it exits non-zero on a FAIL receipt. A non-zero exit here would stop the
    # emit step from running at all and lose the very record this script fetched.
    print(
        f"wrote {len(checks)} check(s) to {args.out} ({len(failing)} failing)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
