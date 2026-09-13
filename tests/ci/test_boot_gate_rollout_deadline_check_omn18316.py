# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18316: the onex-lab lab-pass receipt covers the rollout-wedge invariant.

WHY A CHECK ON THE RECEIPT AND NOT ONLY A TEST IN THE OTHER REPOSITORY. The
invariant is that a wedged lab rollout REPORTS rather than hangs, which holds
only while every Deployment's ``progressDeadlineSeconds`` stays under
``apply_lab_lane.sh``'s own rollout wait. A gate that lives only in
omninode_infra's test suite is enforced on omninode_infra PRs and nowhere else,
so a candidate announced to staging carries no evidence either way. Operating
rule 24's whole point is that the receipt says what was and was not proven, and
an unnamed condition is an unproven one.

WHAT THIS MODULE HOLDS. Three things, each of which has its own way of going
quietly wrong:

1. The gate RUNS the check. A step that was deleted takes its receipt check
   with it and the receipt simply gets shorter, which no consumer notices.
2. The receipt NAMES it. A step that runs and reports nowhere is the same as a
   step that never ran, for anybody reading the artifact a week later.
3. The check's outcome is wired to the STEP, not to a literal. ``ok`` spelled
   as a constant is the failure mode the receipt model's own evidence rule
   exists to catch: a check that cannot fail is indistinguishable from a check
   that never ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVER = REPO_ROOT / ".github/workflows/deliver-dev-candidate-to-staging.yml"
GATE_JOB = "candidate-boot-gate"

#: The step that asserts the invariant, and the check name it feeds.
DEADLINE_STEP_ID = "rollout-deadline"
DEADLINE_CHECK_NAME = "rollout_deadline_bounded"

#: The gate omninode_infra owns. Named here, in one place, so a rename there
#: fails this module with the path rather than silently running nothing.
OMNINODE_INFRA_GATE = "tests/k8s/test_onex_lab_rollout_wedge_omn18316.py"


@pytest.fixture(scope="module")
def deliver() -> dict[str, Any]:
    return yaml.safe_load(DELIVER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate(deliver: dict[str, Any]) -> dict[str, Any]:
    assert GATE_JOB in deliver["jobs"], f"{DELIVER.name} has no {GATE_JOB!r} job"
    return deliver["jobs"][GATE_JOB]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps") or [])


def _step(job: dict[str, Any], step_id: str) -> dict[str, Any]:
    for step in _steps(job):
        if step.get("id") == step_id:
            return step
    msg = (
        f"the {GATE_JOB} job has no step with id {step_id!r}. Without it the "
        "onex-lab receipt carries no verdict on whether a wedged rollout can "
        "report itself, and the candidate is announced to staging with that "
        "condition unexamined."
    )
    raise AssertionError(msg)


def _emit_step(job: dict[str, Any]) -> dict[str, Any]:
    for step in _steps(job):
        if "lab_pass_receipt.py emit" in step.get("run", ""):
            return step
    msg = f"the {GATE_JOB} job emits no lab-pass receipt"
    raise AssertionError(msg)


def test_the_gate_runs_omninode_infras_own_rollout_wedge_gate(
    gate: dict[str, Any],
) -> None:
    """One authority for the invariant, not a second copy of the arithmetic.

    Two implementations of one rule in two repositories drift, and the drift is
    silent in the direction that matters: this repository's copy goes on
    passing while the manifests it describes have moved.
    """
    step = _step(gate, DEADLINE_STEP_ID)
    run = step.get("run", "")
    assert OMNINODE_INFRA_GATE in run, (
        f"the {DEADLINE_STEP_ID!r} step does not run {OMNINODE_INFRA_GATE}. The "
        "invariant is defined in the repository that owns the manifests; "
        "re-deriving it here is how the two stop agreeing."
    )
    assert "cd omninode_infra" in run, (
        f"the {DEADLINE_STEP_ID!r} step must run inside the omninode_infra "
        "checkout, against the revision this gate is proving -- not against "
        "anything in this repository's tree."
    )


def test_the_check_runs_even_when_the_boot_failed(gate: dict[str, Any]) -> None:
    """A failed boot is exactly when the deadline question is worth asking.

    The invariant is static: it holds or does not hold regardless of whether
    the candidate booted. Skipping it on failure would drop it from the receipt
    in the only run where a reader is going to look.
    """
    step = _step(gate, DEADLINE_STEP_ID)
    assert str(step.get("if", "")).strip() == "always()", (
        f"the {DEADLINE_STEP_ID!r} step must carry `if: always()`; on a failed "
        "boot it would otherwise be skipped, and a skipped step reports "
        "`fail` into a receipt for a reason that has nothing to do with the "
        "condition being checked."
    )


def test_the_receipt_carries_the_check(gate: dict[str, Any]) -> None:
    emit = _emit_step(gate).get("run", "")
    assert f'--check "{DEADLINE_CHECK_NAME}:' in emit, (
        f"the onex-lab receipt does not carry a {DEADLINE_CHECK_NAME!r} check. "
        "A step that runs and reports nowhere is, to every later reader of the "
        "artifact, a step that never ran."
    )


def test_the_checks_verdict_is_read_off_the_step_and_not_asserted(
    gate: dict[str, Any],
) -> None:
    """A hardcoded ``ok`` is a check that cannot fail.

    ``ModelLabPassCheck`` requires evidence on passing checks precisely because
    an ``ok: true`` with nothing behind it is indistinguishable from a check
    nobody ran. The same reasoning applies one level up: a verdict spelled as a
    literal rather than read from the step's own outcome is a claim, not a
    measurement.
    """
    emit = _emit_step(gate).get("run", "")
    line = next(
        (
            candidate
            for candidate in emit.splitlines()
            if f'--check "{DEADLINE_CHECK_NAME}:' in candidate
        ),
        None,
    )
    assert line is not None, f"no {DEADLINE_CHECK_NAME} check line in the emit step"
    assert f"steps.{DEADLINE_STEP_ID}.outcome" in line, (
        f"the {DEADLINE_CHECK_NAME!r} verdict must be read from "
        f"steps.{DEADLINE_STEP_ID}.outcome. As written it does not depend on "
        "whether the check passed."
    )


def test_the_check_carries_evidence_naming_what_it_asserted(
    gate: dict[str, Any],
) -> None:
    """Evidence is the third field, and it has to say something falsifiable."""
    emit = _emit_step(gate).get("run", "")
    line = next(
        candidate
        for candidate in emit.splitlines()
        if f'--check "{DEADLINE_CHECK_NAME}:' in candidate
    )
    # --check "<name>:<verdict expression>:<evidence>"
    evidence = line.split(f'--check "{DEADLINE_CHECK_NAME}:', 1)[1]
    for token in ("progressDeadlineSeconds", "apply_lab_lane.sh", "Recreate"):
        assert token in evidence, (
            f"the {DEADLINE_CHECK_NAME!r} evidence does not mention {token!r}. "
            "Evidence that does not name the thing measured cannot be checked "
            "by anyone reading the receipt."
        )
