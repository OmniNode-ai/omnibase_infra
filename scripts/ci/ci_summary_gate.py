# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed verdict for the ``CI Summary`` required-context poller (OMN-14127).

Why this exists
---------------
``CI Summary`` is the single required branch-protection context for
``omnibase_infra`` (OMN-4497). It used to be a ``needs``-gated aggregator job
depending on ~20 upstream jobs. A ``needs``-gated job gets **no** GitHub
check-run until its ``needs`` reach a terminal state, so under self-hosted
runner-fleet saturation the gate jobs never terminalized and ``CI Summary`` was
**absent** — the PR wedged ``BLOCKED`` forever with 0 failing / 0 pending
checks and no auto-recovery.

The ``ci-summary`` workflow job is now a NO-``needs``, GitHub-hosted poller: its
check-run instantiates immediately (so the required context can never be
absent), and it calls this module in a loop against the current run's job list
until a terminal verdict is reached (or a bounded deadline fires → fail-closed).

Verdict policy — DEFAULT-DENY, FAIL-CLOSED
------------------------------------------
This module reproduces the *exact* strictness of the old needs-based
``ci-summary`` pass/fail condition and then adds a strictly-stronger safety net.
Three independent checks; all must be satisfied for success:

1. **Strict aggregate gates.** :data:`STRICT_GATE_JOBS` must each be *present*,
   *completed*, and conclude ``success`` — a ``skipped``/``failure``/
   ``cancelled`` conclusion fails the gate. These jobs are unconditional in
   ``ci.yml`` (no ``if:``), so they never legitimately skip on
   ``pull_request``/``merge_group``/``push``; treating a skip as a failure is
   the same fail-closed behavior the old ``== "success"`` condition had.

2. **Skippable aggregate gates.** :data:`SKIPPABLE_GATE_JOBS` must each be
   *present*, *completed*, and conclude ``success`` **or** ``skipped`` — these
   jobs carry a legitimate skip path (e.g. ``migration-integration`` skips on a
   docs-only diff; ``contract-sync-gate`` skips on ``push``), matching the old
   ``success || skipped`` condition.

3. **Default-deny failure sweep.** Any *other* job in the run that is *present*,
   *completed*, and whose conclusion is not ``success``/``skipped`` fails the
   gate — UNLESS it is the poller itself or one of a small, explicit
   :data:`SOFT_ALLOWLIST` of jobs that already exist in ``ci.yml`` as non-gating
   (advisory / warn-only / not in ci-summary's ``needs`` / not a required
   context). This sweep is what makes the poller *stricter* than the old gate:
   the old ``tests-gate`` greens when ``test-parallel`` is ``skipped``, so a
   failure in ``detect-changes`` / ``plugin-env-service-completeness`` /
   ``compose-required-env-coverage`` / ``contract-path-preflight`` (which skip
   ``test-parallel``) used to slip through silently. The sweep catches them.

The strict + skippable gates together are the **completeness anchor**: requiring
them present+good proves the whole substantive matrix actually ran and passed,
which prevents a *false green* before late-created jobs (``detect-changes`` →
``test-parallel`` → ``tests-gate``) have even been instantiated. If a gate is
missing or still running, the verdict is PENDING (poll again). At the caller's
deadline, PENDING is converted to FAILURE (fail-closed): the required context
always reaches a terminal state.

Exit codes: ``0`` success, ``1`` failure, ``2`` pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

# The poller's own job — excluded to avoid self-deadlock.
SELF_JOB_NAME = "CI Summary"

# Aggregate/leaf gates that the old needs-based ci-summary required STRICTLY
# (``== "success"``). Each is unconditional in ci.yml (no ``if:``) so it never
# legitimately skips on the gating events; a skip here fails closed. Names are
# the ``name:`` display strings the Actions jobs API returns (verified against
# ci.yml on 2026-07-07).
STRICT_GATE_JOBS: tuple[str, ...] = (
    "occ-preflight / eligibility",  # occ-preflight reusable gate
    "CI Tests Gate",  # tests-gate — aggregator over the split matrix
    "Lint",  # lint
    "ONEX Validators",  # onex-validation
    "Infra Node Handler Ownership",  # infra-node-handler-ownership
    "Migration Freeze Check",  # migration-freeze
    "Fingerprint Check",  # fingerprint-check
    "Demo Loop Gate",  # demo-loop-gate
    "Topic Enum Drift Check",  # topic-enum-drift
    "Topic Naming Lint",  # topic-naming-lint
    "Topic Drift Check",  # topic-drift-check
    "Arch Invariants (OMN-3343)",  # arch-invariants
    "Kafka Schema Handshake (OMN-3411)",  # schema-handshake
    "Writer-Migration Coupling Check",  # migration-required-check
)

# Gates the old ci-summary accepted as ``success`` OR ``skipped``. Each carries
# a legitimate skip path in ci.yml (docs-only diff, or event-scoped ``if:``).
SKIPPABLE_GATE_JOBS: tuple[str, ...] = (
    "Migration Integration Test",  # migration-integration (skips on docs-only)
    "Contract Compliance",  # compliance
    "Contract Compliance Check",  # contract-compliance
    "Contract Sync Gate (Wave C) [OMN-8915]",  # contract-sync-gate (skips on push)
)

# Every job the completeness anchor must observe present+good for SUCCESS.
GATE_JOBS: tuple[str, ...] = STRICT_GATE_JOBS + SKIPPABLE_GATE_JOBS

# Jobs that do NOT gate merge today (verified against ci.yml ci-summary ``needs``
# + the pass/fail condition, and against dev branch-protection required contexts
# on 2026-07-07). The default-deny sweep ignores these so it never newly-wedges
# a PR on a job that is already non-blocking. Keep this list SMALL and only add
# jobs that genuinely already exist in ci.yml as non-gating.
#
# Matching is prefix-aware (see :func:`_is_allowlisted`) so reusable-workflow
# callers — whose inner jobs surface as ``"<caller name> / <inner job>"`` — are
# covered by their caller entry (``zone-filter``, ``Runtime Boot Smoke
# (compose)``).
SOFT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # In ci-summary ``needs`` but deliberately EXCLUDED from the pass/fail
        # condition (advisory / informational):
        "Test-Failure Ratchet Gate",  # advisory (OMN-13867)
        "Version Pin Compliance",  # in needs, never checked in the condition
        # Not in ci-summary ``needs`` and not a required context:
        "Runtime Boot Smoke (compose)",  # advisory (OMN-9120); reusable caller
        "Cross-Repo Migration Conflicts",  # migration-conflict-check; not required
        "Kafka Boundary Compat (OMN-3256)",  # advisory; carries xfail known-drift
        "AI-Slop Pattern Check (strict, PR diff)",  # aislop-sweep gates the tree
        # Structural path filter — reusable caller, excluded from the condition:
        "zone-filter",  # zone-filter (reusable) inner jobs surface prefixed
    }
)

# Conclusions that count as "provably passed".
GOOD_CONCLUSIONS: frozenset[str] = frozenset({"success", "skipped"})

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PENDING = 2


@dataclass(frozen=True)
class JobState:
    """The latest-attempt state of a single workflow job."""

    name: str
    status: str  # queued | in_progress | completed | waiting | ...
    conclusion: str | None  # success | failure | cancelled | skipped | timed_out | None
    run_attempt: int


def _state_severity(job: JobState) -> int:
    """Rank same-attempt duplicate jobs by the most blocking state."""

    if job.status != "completed":
        return 2
    if job.conclusion not in GOOD_CONCLUSIONS:
        return 3
    return 1


def dedup_latest(jobs: list[dict[str, object]]) -> dict[str, JobState]:
    """Collapse the raw ``/runs/{id}/jobs`` array to one entry per job name.

    Uses the highest ``run_attempt`` so partial re-runs (``gh run rerun
    --failed``, which re-runs a subset in a new attempt) evaluate the freshest
    conclusion for each job while still seeing jobs that passed in an earlier
    attempt (fetch the endpoint with ``?filter=all``). Within the same attempt,
    duplicate display names keep the most blocking state so a failed matrix leg
    cannot be hidden by a later same-name success.
    """

    latest: dict[str, JobState] = {}
    for raw in jobs:
        name = str(raw.get("name") or "")
        if not name:
            continue
        try:
            attempt = int(str(raw.get("run_attempt") or 1))
        except (TypeError, ValueError):
            attempt = 1
        prev = latest.get(name)
        if prev is not None and attempt < prev.run_attempt:
            continue
        conclusion = raw.get("conclusion")
        current = JobState(
            name=name,
            status=str(raw.get("status") or ""),
            conclusion=None if conclusion is None else str(conclusion),
            run_attempt=attempt,
        )
        if (
            prev is not None
            and attempt == prev.run_attempt
            and _state_severity(current) < _state_severity(prev)
        ):
            continue
        latest[name] = current
    return latest


def _is_allowlisted(name: str, allowlist: frozenset[str]) -> bool:
    """Prefix-aware allowlist check.

    A reusable-workflow caller's inner jobs surface in the jobs API as
    ``"<caller display name> / <inner job name>"``; matching the caller segment
    lets a single allowlist entry cover all of its inner jobs.
    """

    if name in allowlist:
        return True
    caller = name.split(" / ", 1)[0]
    return caller in allowlist


def evaluate(
    jobs: list[dict[str, object]],
    *,
    self_name: str = SELF_JOB_NAME,
    strict_gates: tuple[str, ...] = STRICT_GATE_JOBS,
    skippable_gates: tuple[str, ...] = SKIPPABLE_GATE_JOBS,
    allowlist: frozenset[str] = SOFT_ALLOWLIST,
) -> tuple[int, str]:
    """Return ``(exit_code, human_report)`` for the current job snapshot."""

    latest = dedup_latest(jobs)
    gate_names = frozenset(strict_gates) | frozenset(skippable_gates)

    # (1) Strict aggregate gates: present + completed + conclusion == success.
    strict_failures = sorted(
        g
        for g in strict_gates
        if (
            (st := latest.get(g)) is not None
            and st.status == "completed"
            and st.conclusion != "success"
        )
    )

    # (2) Skippable aggregate gates: present + completed + success/skipped.
    skippable_failures = sorted(
        g
        for g in skippable_gates
        if (
            (st := latest.get(g)) is not None
            and st.status == "completed"
            and st.conclusion not in GOOD_CONCLUSIONS
        )
    )

    # (3) Default-deny sweep over every OTHER present+completed job.
    sweep_failures = sorted(
        j.name
        for name, j in latest.items()
        if name != self_name
        and name not in gate_names
        and not _is_allowlisted(name, allowlist)
        and j.status == "completed"
        and j.conclusion not in GOOD_CONCLUSIONS
    )

    # Completeness anchor: every gate must be present AND completed.
    gate_missing_or_pending = [
        g
        for g in (*strict_gates, *skippable_gates)
        if (latest.get(g) is None or latest[g].status != "completed")
    ]

    all_failures = strict_failures + skippable_failures + sweep_failures

    if all_failures:
        return EXIT_FAILURE, _report(
            "FAILURE",
            latest,
            strict_gates,
            skippable_gates,
            strict_failures,
            skippable_failures,
            sweep_failures,
            gate_missing_or_pending,
        )
    if gate_missing_or_pending:
        return EXIT_PENDING, _report(
            "PENDING",
            latest,
            strict_gates,
            skippable_gates,
            strict_failures,
            skippable_failures,
            sweep_failures,
            gate_missing_or_pending,
        )
    return EXIT_SUCCESS, _report(
        "SUCCESS",
        latest,
        strict_gates,
        skippable_gates,
        strict_failures,
        skippable_failures,
        sweep_failures,
        gate_missing_or_pending,
    )


def _report(
    verdict: str,
    latest: dict[str, JobState],
    strict_gates: tuple[str, ...],
    skippable_gates: tuple[str, ...],
    strict_failures: list[str],
    skippable_failures: list[str],
    sweep_failures: list[str],
    gate_missing_or_pending: list[str],
) -> str:
    lines = [f"CI Summary verdict: {verdict}", f"  jobs observed: {len(latest)}"]
    lines.append("  strict gates:")
    for g in strict_gates:
        st = latest.get(g)
        lines.append(
            f"    - {g}: <absent>"
            if st is None
            else f"    - {g}: {st.status}/{st.conclusion}"
        )
    lines.append("  skippable gates:")
    for g in skippable_gates:
        st = latest.get(g)
        lines.append(
            f"    - {g}: <absent>"
            if st is None
            else f"    - {g}: {st.status}/{st.conclusion}"
        )
    if strict_failures:
        lines.append(f"  strict-gate failures: {', '.join(strict_failures)}")
    if skippable_failures:
        lines.append(f"  skippable-gate failures: {', '.join(skippable_failures)}")
    if sweep_failures:
        lines.append(f"  default-deny sweep failures: {', '.join(sweep_failures)}")
    if gate_missing_or_pending:
        lines.append(f"  gates missing/pending: {', '.join(gate_missing_or_pending)}")
    return "\n".join(lines)


def _load_jobs(path: str | None) -> list[dict[str, object]]:
    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    data = json.loads(raw)
    # Accept either the raw endpoint object ({"jobs": [...]}) or a bare array.
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
    else:
        jobs = data
    if not isinstance(jobs, list):
        raise ValueError("jobs payload must be a list or an object with a 'jobs' array")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-file",
        default="-",
        help="Path to the GitHub Actions jobs JSON (default: stdin). Accepts the "
        "raw endpoint object or a bare array of job objects.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the verdict report and exit 0 regardless (diagnostics only).",
    )
    args = parser.parse_args(argv)

    jobs = _load_jobs(args.jobs_file)
    code, report = evaluate(jobs)
    print(report)
    if args.report_only:
        return EXIT_SUCCESS
    return code


if __name__ == "__main__":
    raise SystemExit(main())
