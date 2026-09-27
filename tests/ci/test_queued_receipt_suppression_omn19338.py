# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The queued path emits no receipt and asserts none either (OMN-19338).

OMN-18976 gave ``verify-lane-converged`` a third verdict: a merge whose turn in
the deploy agent's queue has not come has not FAILED a lab pass, it has not run
one, so the guard publishes ``emit_receipt: false`` and the emit step is
skipped. That output was read by exactly one step. The three steps after it --
assert-present, assert-own-identity and upload -- each carried a bare
``if: always()``, so on the designed queued path they asserted the presence of a
file the design had deliberately not written.

The upload is the one that did lasting damage. Its ``path:`` is the whole of
``LAB_PASS_DIR``, which on the queued path still holds the earlier probes'
leftovers, so ``if-no-files-found: error`` was satisfied by files that are not a
receipt and the step published a receipt-NAMED artifact carrying no
``receipt.json``. The consuming gate reads that as ``UNREADABLE``, which is not
in ``RERUN_ELIGIBLE_TOKENS``; the OMN-19233 re-run then refuses permanently and
a verdict the design made recoverable becomes terminal.

Measured 2026-09-25: subject ``644d82d6`` read ``token=PENDING`` at 05:15Z, its
493-byte artifact landed at 05:36Z, and the same subject read ``token=UNREADABLE
... artifact 10848737185 carries 0 receipt.json entries`` at 05:41Z.

These are shape invariants on the parsed workflow, asserted against the step
graph and never by matching prose. What they cannot do is replay a GitHub
Actions run, so they pin the CONDITION rather than the observed skip; the
behaviour of the re-emission path itself is covered by its own job's tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"

JOB = "verify-lane-converged"

#: The output the convergence guard publishes, and the only value that means
#: "this merge was queued, nothing ran, write nothing".
SUPPRESSION_CONDITION = "steps.converge.outputs.emit_receipt != 'false'"

#: Every step that reads or writes the compose-dev receipt. The emit step was
#: already guarded by OMN-18976; the other three are what OMN-19338 adds.
RECEIPT_STEPS = (
    "Emit the compose-dev lab-pass receipt",
    "Assert the compose-dev lab-pass receipt is present and non-empty",
    "Assert the compose-dev receipt on disk is this job's own",
    "Upload the compose-dev lab-pass receipt",
)


def _steps() -> dict[str, dict[str, Any]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][JOB]["steps"]
    return {step["name"]: step for step in steps if isinstance(step.get("name"), str)}


def test_every_receipt_step_is_suppressed_on_the_queued_path() -> None:
    """AC1. The queued verdict skips the whole receipt surface, not just emit."""
    steps = _steps()
    for name in RECEIPT_STEPS:
        assert name in steps, f"{name!r} is not a step of {JOB}; this test is stale"
        condition = steps[name].get("if", "")
        assert SUPPRESSION_CONDITION in condition, (
            f"{name!r} does not read the suppression output. On the OMN-18976 "
            f"queued path it will act on a receipt the guard deliberately did "
            f"not write. Its condition is {condition!r}."
        )


def test_the_upload_is_suppressed_so_no_receiptless_artifact_is_published() -> None:
    """AC1, the specific regression: a named artifact with no receipt in it.

    Asserted separately from the loop above because this is the step whose
    unguarded ``always()`` turned a recoverable PENDING into a terminal
    UNREADABLE, and because the two properties that combine to cause it --
    uploading the whole directory, and erroring only when the directory is
    empty -- are still true and still correct for the non-queued path.
    """
    upload = _steps()["Upload the compose-dev lab-pass receipt"]
    assert SUPPRESSION_CONDITION in upload.get("if", "")
    assert upload["with"]["path"] == "${{ env.LAB_PASS_DIR }}", (
        "the upload still publishes the whole directory, so a queued run that "
        "reached this step would still publish the leftovers as a receipt"
    )
    assert upload["with"]["if-no-files-found"] == "error"


def test_a_run_that_should_emit_still_asserts_its_receipt() -> None:
    """AC2. The guard narrows to one verdict; it does not disarm the checks.

    ``!= 'false'`` is true for ``true`` and true for the empty string a guard
    that died before writing any output leaves behind. Both of those paths owe
    a receipt and are still held to it. Only the literal suppression verdict
    is exempt, which is what keeps this a fix rather than a hole.
    """
    for name in RECEIPT_STEPS:
        condition = _steps()[name]["if"]
        assert "!= 'false'" in condition, (
            f"{name!r} must exempt only the literal 'false' verdict; an "
            f"equality test against 'true' would also exempt the empty "
            f"output of a crashed guard. Condition: {condition!r}"
        )
        assert "always()" in condition, (
            f"{name!r} lost always(); a failed lab pass would then stop "
            f"producing the record that tells 'it failed' from 'nobody ran it'"
        )


def test_the_bus_publish_is_left_alone() -> None:
    """AC3. Nothing outside the receipt surface changes shape.

    The verdict publish is advisory and ``continue-on-error``, so it cannot
    redden the job on any path and is deliberately not guarded here. Pinning
    that keeps a later "guard everything with always()" sweep from quietly
    making an observability publish conditional on a receipt.
    """
    publish = _steps()["Publish the compose-dev lab-pass verdict to the bus"]
    assert publish["if"] == "always()"
    assert publish["continue-on-error"] is True


def test_the_queued_path_records_its_outcome() -> None:
    """AC1: suppression alone would leave a green job that explains nothing.

    A skipped assertion and a skipped upload are invisible in a run summary, so
    a queued merge would render identically to a lane nobody probed -- and the
    reader who goes looking for the receipt finds no statement about why it is
    absent. The recorder is the complement of the suppression condition, so the
    two cannot drift apart: every run takes exactly one of the two branches.
    """
    step = _steps()["Record the queued outcome for a merge whose turn has not come"]
    condition = str(step.get("if", ""))
    assert "steps.converge.outputs.emit_receipt == 'false'" in condition, (
        "the recorder must be the exact complement of the suppression guard, "
        f"not {condition!r}"
    )
    body = str(step.get("run", ""))
    assert "GITHUB_STEP_SUMMARY" in body, (
        "the queued outcome must reach the run summary"
    )
    assert "QUEUED" in body, "the summary must name the queued token (AC1)"
    assert "PENDING" in body and "UNREADABLE" in body, (
        "the summary must say why NOT uploading is the point: absent reads as "
        "PENDING and stays recoverable, a receiptless artifact reads as "
        "UNREADABLE and does not"
    )


def test_reemit_job_runs_for_every_completed_verify_with_candidates() -> None:
    """OMN-19563: a source FAIL receipt must be copied as FAIL, never skipped.

    The verify job is failed by a source receipt whose health probes failed,
    but that artifact is still authoritative for every bounded re-emission
    candidate. Skipped and cancelled jobs have no completed observation; a
    non-empty candidate list from any other result must reach the re-emitter.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    condition = " ".join(str(workflow["jobs"]["reemit-queued-receipts"]["if"]).split())
    assert "always()" in condition
    assert "needs.verify-lane-converged.result != 'skipped'" in condition
    assert "needs.verify-lane-converged.result != 'cancelled'" in condition
    assert "needs.verify-lane-converged.result == 'success'" not in condition
    assert "reemit_candidates != ''" in condition
    assert "reemit_candidates != '[]'" in condition
