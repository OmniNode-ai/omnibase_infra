# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Remote full-suite verification for the governed pre-push hook (OMN-16688).

WHAT THIS IS
------------
A third execution target for the heavy (full-suite) escalation in
``scripts/hooks/prepush_smart_tests.sh``, alongside ``.200`` (OMN-15059) and
the ``.201`` gate-runner container (OMN-16295): a **GitHub-hosted CI run of the
full suite, pinned to the exact HEAD sha about to be pushed**.

Every OmniNode repo is public, so GitHub-hosted minutes are free and unmetered,
and ``.github/workflows/ci.yml`` already runs the full sharded suite on
``ubuntu-latest``. When both local gate hosts are over the load threshold, the
hook can consume that remote run as evidence instead of refusing the push or
falling back to a degraded-evidence override grant.

WHAT THIS IS NOT
----------------
This is **not** a bypass, and it is deliberately not shaped like one:

* It accepts no ``PREPUSH_*`` env override and mints no local artifact. There is
  no file on disk to forge -- every answer is re-derived live from the GitHub
  API at the moment the hook asks.
* It cannot make the gate accept *less* work. It only accepts a run that
  executed the FULL suite; a selector-narrowed run is rejected exactly as a
  narrowed local run would be.
* A missing, pending, failed, or narrowed remote run returns non-zero, and the
  hook then behaves exactly as it does today.

THE THREE BINDINGS
------------------
``check`` returns 0 only when a CI run exists that satisfies all three:

1. **SHA-pinned.** ``run.head_sha`` equals the exact HEAD sha the hook is about
   to push -- byte equality on the full 40-char sha, never a prefix and never a
   branch name. A git sha is content-addressed, so a run on that sha is a run
   on that exact tree.
2. **Green.** ``run.status == "completed"`` and ``run.conclusion == "success"``.
3. **Full-suite shape.** The run's shard jobs are the FULL matrix: every job
   ``Tests (Split i/N)`` for ``i`` in ``1..N`` is present and succeeded, where
   ``N`` is :data:`FULL_SUITE_SPLIT_COUNT` -- imported from the selector itself
   (``scripts.ci.detect_test_paths``), never re-typed here, so the binding
   cannot drift from the selector's own definition of "full suite".

Binding 3 is what makes this safe. The selector emits ``split_count ==
_FULL_SUITE_SPLIT_COUNT`` **only** on ``is_full_suite=True``; narrowed
selections are capped far below it by ``_split_count_for``. So the shard
denominator observable in the job names is a faithful, forge-resistant witness
of ``is_full_suite`` -- it is produced by CI from the pushed tree, not supplied
by the caller. ``test_prepush_remote_verify.py`` pins that gap so a future
change to the narrowed ceiling cannot silently collapse the distinction.

CREDENTIALS
-----------
All GitHub reads go through ``gh api`` in a subprocess, reusing the operator's
existing ``gh auth`` session. No token is read, written, minted, or persisted by
this module, and no new secret is introduced anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.detect_test_paths import (  # sys.path set above
    _FULL_SUITE_SPLIT_COUNT,
)

#: Full-suite shard count, re-exported from the selector so this module and the
#: selector can never disagree about what "the full suite" means.
FULL_SUITE_SPLIT_COUNT: Final[int] = _FULL_SUITE_SPLIT_COUNT

#: Workflow file whose run carries the sharded full suite.
CI_WORKFLOW_FILE: Final[str] = "ci.yml"

#: ``Tests (Split 3/15)`` -> group 1 = "3", group 2 = "15".
_SHARD_JOB_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^Tests\s*\(Split\s*(\d+)\s*/\s*(\d+)\s*\)\s*$"
)

_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")

#: Bounded so a hung `gh` call can never wedge a `git push`.
_GH_TIMEOUT_SECONDS: Final[int] = 60


class RemoteVerifyError(RuntimeError):
    """Raised when the remote verification cannot be resolved at all.

    Distinct from "resolved, and the answer is no": callers must not conflate a
    failed *read* with a proven-absent run, so that an API outage can never be
    reported to a developer as "your suite failed".
    """


@dataclass(frozen=True)
class VerifyOutcome:
    """Result of one verification attempt."""

    ok: bool
    reason: str
    run_id: int | None = None
    run_url: str | None = None


def _gh_json(args: list[str]) -> Any:
    """Run ``gh`` and parse stdout as JSON, or raise :class:`RemoteVerifyError`."""
    try:
        completed = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=_GH_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RemoteVerifyError(
            "the 'gh' CLI is not installed or not on PATH; remote full-suite "
            "verification needs it to read GitHub Actions runs"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteVerifyError(
            f"'gh {' '.join(args)}' timed out after {_GH_TIMEOUT_SECONDS}s"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "<no stderr>"
        raise RemoteVerifyError(f"'gh {' '.join(args)}' failed: {stderr}")

    try:
        return json.loads(completed.stdout or "null")
    except json.JSONDecodeError as exc:
        raise RemoteVerifyError(
            f"'gh {' '.join(args)}' returned output that is not JSON"
        ) from exc


def resolve_repo_slug() -> str:
    """Return ``owner/name`` for the current repository."""
    payload = _gh_json(["repo", "view", "--json", "nameWithOwner"])
    slug = (payload or {}).get("nameWithOwner")
    if not isinstance(slug, str) or "/" not in slug:
        raise RemoteVerifyError(
            "could not resolve the GitHub repository for this checkout "
            "('gh repo view --json nameWithOwner' returned no usable slug)"
        )
    return slug


def _fetch_runs_for_sha(slug: str, head_sha: str) -> list[dict[str, Any]]:
    payload = _gh_json(
        [
            "api",
            f"repos/{slug}/actions/workflows/{CI_WORKFLOW_FILE}/runs"
            f"?head_sha={head_sha}&per_page=100",
        ]
    )
    runs = (payload or {}).get("workflow_runs")
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


def _fetch_jobs(slug: str, run_id: int) -> list[dict[str, Any]]:
    payload = _gh_json(
        ["api", "--paginate", f"repos/{slug}/actions/runs/{run_id}/jobs?per_page=100"]
    )
    # `gh api --paginate` concatenates pages; normalise both shapes.
    pages = payload if isinstance(payload, list) else [payload]
    jobs: list[dict[str, Any]] = []
    for page in pages:
        if isinstance(page, dict):
            entries = page.get("jobs")
            if isinstance(entries, list):
                jobs.extend(j for j in entries if isinstance(j, dict))
    return jobs


def run_is_full_suite_shaped(jobs: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return whether ``jobs`` prove a FULL-suite run, and why not if they do not.

    Binding 3. Requires every shard ``1..FULL_SUITE_SPLIT_COUNT`` to be present
    with the full denominator and to have concluded ``success``.
    """
    seen: dict[int, str | None] = {}
    denominators: set[int] = set()

    for job in jobs:
        name = job.get("name")
        if not isinstance(name, str):
            continue
        match = _SHARD_JOB_NAME_RE.match(name.strip())
        if match is None:
            continue
        index, denominator = int(match.group(1)), int(match.group(2))
        denominators.add(denominator)
        conclusion = job.get("conclusion")
        seen[index] = conclusion if isinstance(conclusion, str) else None

    if not seen:
        return False, "the run has no 'Tests (Split i/N)' shard jobs at all"

    if denominators != {FULL_SUITE_SPLIT_COUNT}:
        observed = ", ".join(str(d) for d in sorted(denominators))
        return False, (
            f"shard denominator is {{{observed}}}, not the full-suite "
            f"{FULL_SUITE_SPLIT_COUNT} -- this was a selector-NARROWED run, "
            "which is not full-suite evidence"
        )

    expected = set(range(1, FULL_SUITE_SPLIT_COUNT + 1))
    missing = sorted(expected - set(seen))
    if missing:
        return False, (
            f"shards {missing} are missing from the run "
            f"(expected all of 1..{FULL_SUITE_SPLIT_COUNT})"
        )

    not_green = sorted(i for i, c in seen.items() if c != "success")
    if not_green:
        detail = ", ".join(f"{i}={seen[i] or 'incomplete'}" for i in not_green)
        return False, f"shards did not all succeed ({detail})"

    return True, f"all {FULL_SUITE_SPLIT_COUNT} full-suite shards succeeded"


def check(head_sha: str, slug: str | None = None) -> VerifyOutcome:
    """Evaluate the three bindings for ``head_sha``.

    Returns an outcome whose ``ok`` is True only if a SHA-pinned, green,
    full-suite-shaped CI run exists. Raises :class:`RemoteVerifyError` if the
    GitHub read itself could not be completed.
    """
    if not _SHA_RE.match(head_sha):
        raise RemoteVerifyError(
            f"'{head_sha}' is not a full 40-character lowercase git sha; "
            "remote verification is pinned to an exact sha, never a prefix or "
            "a branch name"
        )

    resolved_slug = slug or resolve_repo_slug()
    runs = _fetch_runs_for_sha(resolved_slug, head_sha)
    if not runs:
        return VerifyOutcome(
            ok=False,
            reason=(
                f"no {CI_WORKFLOW_FILE} run exists for {head_sha[:12]} on "
                f"{resolved_slug} yet"
            ),
        )

    # Binding 1: re-assert sha equality locally. The query filters server-side,
    # but the gate must not delegate its own precondition to the API.
    pinned = [r for r in runs if r.get("head_sha") == head_sha]
    if not pinned:
        return VerifyOutcome(
            ok=False,
            reason=f"no run is pinned to the exact sha {head_sha[:12]}",
        )

    last_reason = "no candidate run satisfied the bindings"
    for run in sorted(pinned, key=lambda r: int(r.get("id") or 0), reverse=True):
        run_id = int(run.get("id") or 0)
        run_url = run.get("html_url") if isinstance(run.get("html_url"), str) else None

        # Binding 2.
        if run.get("status") != "completed":
            last_reason = (
                f"run {run_id} is still {run.get('status')} -- not completed yet"
            )
            continue
        if run.get("conclusion") != "success":
            last_reason = (
                f"run {run_id} concluded '{run.get('conclusion')}', not success"
            )
            continue

        # Binding 3.
        full_shaped, why = run_is_full_suite_shaped(_fetch_jobs(resolved_slug, run_id))
        if not full_shaped:
            last_reason = f"run {run_id} is green but not full-suite shaped: {why}"
            continue

        return VerifyOutcome(ok=True, reason=why, run_id=run_id, run_url=run_url)

    return VerifyOutcome(ok=False, reason=last_reason)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepush_remote_verify.py",
        description=(
            "Check whether a GitHub-hosted FULL-suite CI run has already passed "
            "for an exact HEAD sha, so the governed pre-push hook can consume it "
            "instead of running the heavy suite on an over-loaded local host."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser(
        "check",
        help="exit 0 iff a sha-pinned, green, full-suite CI run exists",
    )
    check_parser.add_argument(
        "--head-sha",
        required=True,
        help="the exact 40-character sha about to be pushed",
    )
    check_parser.add_argument(
        "--repo",
        default=None,
        help="owner/name (default: resolved from this checkout via gh)",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the outcome as JSON on stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        outcome = check(args.head_sha, slug=args.repo)
    except RemoteVerifyError as exc:
        # Exit 2: "could not determine", NOT "verified failed". The hook must
        # treat this as no-evidence and fall through to its existing refusal
        # path, never as a proven failure.
        if getattr(args, "as_json", False):
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"remote-verify: UNRESOLVED -- {exc}", file=sys.stderr)
        return 2

    if getattr(args, "as_json", False):
        print(
            json.dumps(
                {
                    "ok": outcome.ok,
                    "reason": outcome.reason,
                    "run_id": outcome.run_id,
                    "run_url": outcome.run_url,
                }
            )
        )
    elif outcome.ok:
        print(f"remote-verify: PASS -- {outcome.reason} ({outcome.run_url})")
    else:
        print(f"remote-verify: NO EVIDENCE -- {outcome.reason}", file=sys.stderr)

    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
