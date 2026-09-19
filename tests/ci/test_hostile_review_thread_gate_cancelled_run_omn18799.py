# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18799: the Hostile Review Thread Gate emits no verdict on a cancelled run.

Sibling of OMN-18793, resolving the other way and therefore the worse one.
`hostile-reviewer.yml` declares ``concurrency.cancel-in-progress: true``, so
every re-push cancels the run already in flight. On a cancelled run GitHub
still STARTS a job guarded by bare ``always()`` -- that is the whole
difference between ``always()`` and ``!cancelled()``.

Where the sibling gate then read a ``cancelled`` predecessor and posted a
hard red, this job has no ``needs.*.result`` triage at all: its single
evaluating step shells ``scripts/ci/check_hostile_review_threads.py``, which
consults GitHub thread state and nothing else. The reviewer that would have
posted this head's threads was cancelled before it could, so the script finds
zero blocking threads, exits 0, and a GREEN `Hostile Review Thread Gate`
lands on the pull-request head for a run that evaluated nothing.

Measured over four sampled cancelled runs, each read with
``gh api repos/OmniNode-ai/omnibase_infra/actions/runs/<id>/jobs``:

    35415401505  cancelled run  ->  Thread Gate: success  (step ran, success)
    35414662126  cancelled run  ->  Thread Gate: success  (step ran, success)
    35413698388  cancelled run  ->  Thread Gate: success  (step ran, success)
    35412846604  cancelled run  ->  Thread Gate: success  (step ran, success)

On 35414662126 the sibling `Hostile Review Gate` concluded ``cancelled`` with
zero steps in the SAME run, so the two jobs disagreed about whether anything
had happened. A false red is loud and is eventually fixed; a false green is
indistinguishable from a real verdict and is never noticed.

This is not a defect in the gate script, which is already fail-closed on an
unreachable or malformed GraphQL response and raises on an absent
``PR_NUMBER``. The false green comes entirely from the job being STARTED on a
run that was cancelled, so the fix is the condition and only the condition.

The properties below are asserted over the PARSED workflow, never over its
text, so a reshuffle that preserves the YAML while losing the property still
fails here. The first is the fix; the rest are the positive control, because
the cheapest wrong ways to make this test pass -- narrowing the condition to
``success()``, dropping the event guard, or deleting the evaluating step --
each trade a false green for a vacuous one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"
THREAD_GATE_JOB = "hostile-review-thread-gate"
GATE_SCRIPT = "scripts/ci/check_hostile_review_threads.py"

pytestmark = pytest.mark.unit


def _workflow() -> dict[str, Any]:
    with WORKFLOW.open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict), f"{WORKFLOW} did not parse to a mapping"
    return parsed


def _thread_gate_job() -> dict[str, Any]:
    jobs = _workflow().get("jobs")
    assert isinstance(jobs, dict), "the workflow declares no jobs mapping"
    job = jobs.get(THREAD_GATE_JOB)
    assert isinstance(job, dict), (
        f"{THREAD_GATE_JOB} is absent from {WORKFLOW.name}. This test pins that "
        "job's cancellation behaviour; if the job was renamed, repoint "
        "THREAD_GATE_JOB rather than deleting the assertion."
    )
    return job


def _thread_gate_condition() -> str:
    condition = _thread_gate_job().get("if")
    assert isinstance(condition, (str, bool)), (
        f"{THREAD_GATE_JOB} declares no `if:` condition at all, so it runs "
        "unconditionally -- including on a cancelled run, and including on an "
        "event that carries no pull request for it to read threads from."
    )
    return str(condition).replace(" ", "")


def _thread_gate_step_bodies() -> str:
    """Every `run:` body the thread gate declares, joined."""
    steps = _thread_gate_job().get("steps")
    assert isinstance(steps, list) and steps, f"{THREAD_GATE_JOB} declares no steps"
    bodies = [
        step["run"]
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    ]
    assert bodies, f"{THREAD_GATE_JOB} declares no run step"
    return "\n".join(bodies)


def test_the_thread_gate_does_not_run_on_a_cancelled_run() -> None:
    """A cancelled run must not mint a green thread-gate verdict.

    ``always()`` is true on a cancelled run; ``!cancelled()`` is not. The
    condition must be the second shape, so a superseded push skips the gate
    instead of publishing a success it never established.
    """
    condition = _thread_gate_condition()

    assert "always()" not in condition, (
        f"{THREAD_GATE_JOB} is guarded by always(), which is TRUE on a "
        "cancelled run. On every superseded push this job executes, finds no "
        "threads from a reviewer that was cancelled before it could post any, "
        "exits 0, and publishes a GREEN check for a run that evaluated "
        "nothing (OMN-18799). Use !cancelled() so a cancelled run skips the "
        "gate instead of passing it."
    )
    assert "!cancelled()" in condition, (
        f"{THREAD_GATE_JOB} must be guarded by !cancelled() so it still "
        "evaluates when the reviewer was skipped or failed -- unresolved "
        "threads outlive any single run -- while a cancelled run produces no "
        "verdict at all."
    )


def test_a_skipped_or_failed_reviewer_still_evaluates_the_thread_gate() -> None:
    """Positive control: the reason bare always() was chosen is preserved.

    The thread gate exists because unresolved threads outlive the run that
    posted them, so a draft-skipped or crashed reviewer must not needs-cascade
    this job into a vacuous skip. Narrowing the condition to ``success()`` --
    the cheapest way to silence a cancelled run -- would do exactly that, and
    a skipped check does not block a merge the way a failed one does.
    """
    condition = _thread_gate_condition()
    assert "success()" not in condition, (
        f"{THREAD_GATE_JOB} is now gated on success(), so a skipped or failed "
        "hostile-review job would cascade it into a skip. Unresolved threads "
        "from earlier runs still deserve the deterministic check; OMN-18799 "
        "changes WHEN this job runs, never WHAT it checks."
    )
    assert "failure()" not in condition, (
        f"{THREAD_GATE_JOB} is now gated on failure(), which would run it only "
        "when something upstream broke."
    )


def test_the_thread_gate_still_runs_only_on_a_pull_request() -> None:
    """Positive control: the event guard survives.

    The evaluating step reads ``github.event.pull_request.number`` into
    ``PR_NUMBER``. On any other event that is empty and the script raises on
    ``int("")``. Fail-closed, but a refusal on an event that has no threads to
    check is noise, not a verdict -- the guard is what keeps the job off those
    events entirely.
    """
    condition = _thread_gate_condition()
    assert "github.event_name=='pull_request'" in condition, (
        f"{THREAD_GATE_JOB} no longer restricts itself to pull_request events. "
        "On a push or schedule event PR_NUMBER is empty and the gate script "
        "raises, turning every such run red for a reason unrelated to threads."
    )


def test_the_thread_gate_still_runs_the_deterministic_thread_check() -> None:
    """Positive control: the evaluating step is still there and still shells
    the deterministic checker.

    A job whose condition is perfect and whose body does nothing is the other
    way to publish a green that established nothing.
    """
    body = _thread_gate_step_bodies()
    assert GATE_SCRIPT in body, (
        f"{THREAD_GATE_JOB} no longer runs {GATE_SCRIPT}, so its success "
        "conclusion is no longer evidence that unresolved hostile-reviewer "
        "threads were checked at all (OMN-17492 is the gate this job IS)."
    )


def test_the_workflow_still_cancels_superseded_runs() -> None:
    """The premise of the fix, pinned so it cannot silently disappear.

    If `cancel-in-progress` were ever turned off, `!cancelled()` would stop
    being load-bearing and a future reader would have no way to tell why the
    condition is written that way. Pinning the premise keeps the fix legible.
    """
    concurrency = _workflow().get("concurrency")
    assert isinstance(concurrency, dict), (
        "the workflow declares no concurrency block; OMN-18799's fix assumes "
        "superseded runs are cancelled."
    )
    assert concurrency.get("cancel-in-progress") is True, (
        "cancel-in-progress is no longer true. Re-read OMN-18799 before "
        "changing the thread gate's condition: the cancelled-run case it "
        "guards against arises from this setting."
    )
