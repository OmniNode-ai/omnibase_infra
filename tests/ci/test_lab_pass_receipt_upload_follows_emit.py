# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19563: a receipt-named artifact is published only when a receipt was emitted.

WHAT BROKE, measured live 2026-09-25.

``verify-lane-converged`` skips its emit step for a merge whose turn in the
deploy agent's queue has not come (``emit_receipt == 'false'``, OMN-18976): a
queued merge writes NO receipt, so the delivery gate reads ABSENT or PENDING and
the PS-5 re-run selector re-runs the delivery once a PASS arrives.

The upload step did not share that condition. It ran under a bare ``always()``
and published the receipt directory, which also holds ``declared-groups.txt``
and ``lag-sample.json``, so ``if-no-files-found: error`` saw files and the upload
succeeded:

    run 36138224804 (queue_position_at_start=2), artifact 10866285353
      name    : lab-pass-receipt-compose-dev-0bbb82f121f2...
      content : declared-groups.txt, lag-sample.json   (no receipt.json)

The delivery gate reads the newest artifact of that name, finds zero
``receipt.json`` entries and refuses UNREADABLE -- not ABSENT. The re-run
selector skips UNREADABLE by design ("a compose-dev PASS does not change it"),
so when the PASS for sha 59361591 landed at 12:25:36Z, delivery run
36127915167 was never re-run. Three consecutive delivery runs refused this way.

The gate's reading is correct and is not relaxed here: an artifact under a
receipt's name that holds no receipt IS unreadable. The producer is fixed.

WHAT IS PINNED, as a property of the parsed workflow rather than as text: in
every job that emits a lab-pass receipt, the steps that assert that receipt and
the step that uploads it under a ``lab-pass-receipt-*`` name run under exactly
the emit step's own ``if:``. A condition that differs in either direction is the
defect: a weaker one publishes a receipt-named artifact with no receipt in it,
and a stronger one drops the FAIL receipt ``always()`` exists to keep.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: Every workflow that emits a lab-pass receipt and uploads it under a
#: ``lab-pass-receipt-*`` name.
WORKFLOWS = (
    WORKFLOW_DIR / "runtime-rebuild-trigger.yml",
    WORKFLOW_DIR / "runtime-rebuild-trigger-reusable.yml",
    WORKFLOW_DIR / "deliver-dev-candidate-to-staging.yml",
)

_EMIT = re.compile(r"lab_pass_receipt\.py\s+emit\b")
_VERIFY = re.compile(r"lab_pass_receipt\.py\s+verify\b")
_ASSERT = re.compile(
    r"assert_evidence_artifact\.py[\s\\]+--artifact\s+lab-pass-receipt"
)
_RECEIPT_ARTIFACT = "lab-pass-receipt-"


def _norm(condition: Any) -> str:
    """One spelling per condition: whitespace collapsed, absent means success()."""
    if condition is None:
        return "success()"
    return " ".join(str(condition).split())


def _receipt_jobs(
    workflow: Path,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """(job id, emit step, dependent steps) for every job with a receipt emit."""
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    found: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for job_id, job in (doc.get("jobs") or {}).items():
        steps = job.get("steps") or []
        emits = [s for s in steps if _EMIT.search(str(s.get("run", "")))]
        if not emits:
            continue
        assert len(emits) == 1, f"{workflow.name}:{job_id} has {len(emits)} emit steps"
        emit = emits[0]
        dependents = [
            s
            for s in steps
            if s is not emit
            and (
                _VERIFY.search(str(s.get("run", "")))
                or _ASSERT.search(str(s.get("run", "")))
                or (
                    "upload-artifact" in str(s.get("uses", ""))
                    and str((s.get("with") or {}).get("name", "")).startswith(
                        _RECEIPT_ARTIFACT
                    )
                )
            )
        ]
        found.append((job_id, emit, dependents))
    return found


_CASES = [
    pytest.param(wf, job_id, emit, dep, id=f"{wf.name}:{job_id}:{dep.get('name')}")
    for wf in WORKFLOWS
    for job_id, emit, deps in _receipt_jobs(wf)
    for dep in deps
]


@pytest.mark.unit
def test_every_receipt_job_has_an_upload_under_the_receipt_name() -> None:
    """Positive control: the selector finds the uploads it is meant to check."""
    uploads = [
        (wf.name, job_id)
        for wf in WORKFLOWS
        for job_id, _emit, deps in _receipt_jobs(wf)
        if any("upload-artifact" in str(d.get("uses", "")) for d in deps)
    ]
    assert ("runtime-rebuild-trigger.yml", "verify-lane-converged") in uploads
    assert (
        "runtime-rebuild-trigger-reusable.yml",
        "verify-sibling-converged",
    ) in uploads
    assert len(_CASES) >= 6, f"only {len(_CASES)} receipt-dependent steps found"


@pytest.mark.unit
@pytest.mark.parametrize(("workflow", "job_id", "emit", "step"), _CASES)
def test_receipt_steps_run_under_the_emit_condition(
    workflow: Path, job_id: str, emit: dict[str, Any], step: dict[str, Any]
) -> None:
    assert _norm(step.get("if")) == _norm(emit.get("if")), (
        f"{workflow.name} job {job_id!r}: step {step.get('name')!r} runs under "
        f"if: {_norm(step.get('if'))!r}, but the receipt it asserts or uploads is "
        f"emitted under if: {_norm(emit.get('if'))!r}. When the emit is skipped "
        "(a queued merge, OMN-18976) this step then asserts or publishes a "
        "receipt that was never written; an upload of the receipt directory "
        "without receipt.json reads UNREADABLE at the delivery gate and is never "
        "re-run (OMN-19563)."
    )
