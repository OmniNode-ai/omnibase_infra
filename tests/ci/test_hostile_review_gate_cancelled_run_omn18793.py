# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18793: the Hostile Review Gate emits no verdict on a cancelled run.

`hostile-reviewer.yml` declares ``concurrency.cancel-in-progress: true``, so
every re-push to a branch cancels the run already in flight. On a cancelled
run GitHub still starts a job guarded by bare ``always()`` -- that is what
``always()`` means, and it is the difference between it and ``!cancelled()``.
The gate job then read ``needs.occ-preflight.result``, found ``cancelled``,
and ran ``exit 1``, posting a hard ``failure`` check conclusion against the
pull-request head for a run that was never allowed to produce a verdict.

Measured by the OMN-18254 failure-rate alert on its first firing run,
35412814968 (2026-09-19T01:29:37Z): `Hostile Review Gate` failed on 9 of the
10 most recent heads in this repository, the worst check-class rate in the
report. Four consecutive samples were read and all four sat on runs whose own
conclusion was ``cancelled``:

    35412846604  cancelled  jonah/omn-18789-broker-readiness-consumer-sync-probe
    35412240460  cancelled  jonah/omn-18640-deploy-agent-force-recreate
    35411367811  cancelled  jonah/omn-18288-bump-occ-audit-pin
    35411353863  cancelled  bot/omnimarket-contract-pin-20260919010206

A check that is red nine times in ten for a reason unrelated to the code
teaches every reader to ignore it, which is worse than an absent gate.

The properties below are asserted over the PARSED workflow, never over its
text, so a reshuffle that preserves the YAML while losing the property still
fails here. The first is the fix; the second and third are the positive
control, because the cheapest wrong way to make this test pass is to delete
the fail-closed branches the gate exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"
GATE_JOB = "hostile-review-gate"

pytestmark = pytest.mark.unit


def _workflow() -> dict[str, Any]:
    with WORKFLOW.open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict), f"{WORKFLOW} did not parse to a mapping"
    return parsed


def _gate_job() -> dict[str, Any]:
    jobs = _workflow().get("jobs")
    assert isinstance(jobs, dict), "the workflow declares no jobs mapping"
    job = jobs.get(GATE_JOB)
    assert isinstance(job, dict), (
        f"{GATE_JOB} is absent from {WORKFLOW.name}. This test pins that job's "
        "cancellation behaviour; if the job was renamed, repoint GATE_JOB "
        "rather than deleting the assertion."
    )
    return job


def _gate_step_run() -> str:
    """The gate's single evaluating step, as its shell body."""
    steps = _gate_job().get("steps")
    assert isinstance(steps, list) and steps, f"{GATE_JOB} declares no steps"
    bodies = [
        step["run"]
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    ]
    assert bodies, f"{GATE_JOB} declares no run step"
    return "\n".join(bodies)


def test_the_gate_does_not_run_on_a_cancelled_run() -> None:
    """A cancelled run must not produce a gate verdict.

    ``always()`` is true on a cancelled run; ``!cancelled()`` is not. The
    condition must be the second shape, so a superseded push skips the gate
    instead of minting a red check for it.
    """
    condition = _gate_job().get("if")
    assert isinstance(condition, (str, bool)), (
        f"{GATE_JOB} declares no `if:` condition at all, so it runs "
        "unconditionally -- including on a cancelled run."
    )
    normalised = str(condition).replace(" ", "")

    assert "always()" not in normalised, (
        f"{GATE_JOB} is guarded by always(), which is TRUE on a cancelled run. "
        "On every superseded push this job executes, reads "
        "needs.occ-preflight.result == 'cancelled', and posts a hard failure "
        "for a run that never produced a verdict (OMN-18793). Use "
        "!cancelled() so a cancelled run skips the gate instead."
    )
    assert "!cancelled()" in normalised, (
        f"{GATE_JOB} must be guarded by !cancelled() so it still evaluates "
        "when an upstream job was skipped or failed -- a skipped reviewer must "
        "not needs-cascade the gate into a vacuous skip -- while a cancelled "
        "run produces no verdict at all."
    )


def test_a_real_preflight_failure_still_fails_the_gate_closed() -> None:
    """Positive control: the OCC-preflight refusal branch survives the fix.

    The cheapest wrong way to clear the red measured above is to stop failing
    on a bad preflight. That would convert a noisy gate into a silent one.
    """
    body = _gate_step_run()
    assert "needs.occ-preflight.result" in body, (
        "the gate no longer reads the OCC preflight result, so a genuinely "
        "failed preflight would pass unremarked."
    )
    assert '!= "success"' in body, (
        "the gate no longer refuses a non-success OCC preflight. Narrowing "
        "this to an equality against 'failure' would let a preflight that "
        "died for any other reason through."
    )


def test_a_reviewer_pipeline_failure_still_fails_the_gate_closed() -> None:
    """Positive control: the hostile-review pipeline-error branch survives.

    Since OMN-17492 the reviewer never exits non-zero on findings, so a
    failure there is a pipeline error and must still block.
    """
    body = _gate_step_run()
    assert "needs.hostile-review.result" in body, (
        "the gate no longer reads the reviewer job result, so a pipeline "
        "error in the adversarial review would pass unremarked."
    )
    assert body.count("exit 1") >= 2, (
        "the gate must retain both fail-closed branches -- the OCC-preflight "
        "one and the reviewer-pipeline one. OMN-18793 changes WHEN the job "
        "runs, never WHAT it refuses."
    )


def test_the_workflow_still_cancels_superseded_runs() -> None:
    """The premise of the fix, pinned so it cannot silently disappear.

    If `cancel-in-progress` were ever turned off, `!cancelled()` would stop
    being load-bearing and a future reader would have no way to tell why the
    condition is written that way. Pinning the premise keeps the fix legible.
    """
    concurrency = _workflow().get("concurrency")
    assert isinstance(concurrency, dict), (
        "the workflow declares no concurrency block; OMN-18793's fix assumes "
        "superseded runs are cancelled."
    )
    assert concurrency.get("cancel-in-progress") is True, (
        "cancel-in-progress is no longer true. Re-read OMN-18793 before "
        "changing the gate's condition: the cancelled-run case it guards "
        "against arises from this setting."
    )
