# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18492: the onex-lab lab-pass receipt carries the contract-boot check.

THE CLAIM. A change that makes the runtime unable to boot is refused where the
candidate is proven, and the receipt says so. The detection itself is the
runtime's own ``discover_runtime_local_ingress_routes`` replayed over the
2026-09-16 incident bytes (``tests/integration/runtime/
test_contract_boot_incident_omn18492.py``); this module holds the three ways
the wiring around it goes quietly wrong, the same three OMN-18316 pinned for
the rollout-deadline check:

1. The gate RUNS the check. A deleted step takes its receipt check with it and
   the receipt gets shorter, which no consumer notices.
2. The receipt NAMES it, with evidence. A step that runs and reports nowhere is
   a step that never ran.
3. The verdict is read off the STEP, not spelled as a literal. A check that
   cannot fail is indistinguishable from a check that never ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import ModelLabPassCheck, parse_check_argument

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVER = REPO_ROOT / ".github/workflows/deliver-dev-candidate-to-staging.yml"
GATE_JOB = "candidate-boot-gate"

BOOT_STEP_ID = "contract-boot"
BOOT_CHECK_NAME = "contract_boot_invariants"

#: The two files the step runs. Named here so a rename fails with the path.
BOOT_TESTS = (
    "tests/integration/runtime/test_local_ingress_self_collision_omn18550.py",
    "tests/integration/runtime/test_contract_boot_incident_omn18492.py",
)


@pytest.fixture(scope="module")
def gate() -> dict[str, Any]:
    deliver = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
    assert GATE_JOB in deliver["jobs"], f"{DELIVER.name} has no {GATE_JOB!r} job"
    return deliver["jobs"][GATE_JOB]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps") or [])


def _boot_step(job: dict[str, Any]) -> dict[str, Any]:
    for step in _steps(job):
        if step.get("id") == BOOT_STEP_ID:
            return step
    msg = (
        f"the {GATE_JOB} job has no step with id {BOOT_STEP_ID!r}. Without it the "
        "onex-lab receipt carries no verdict on whether the contract set can "
        "boot, and a duplicate ingress alias reaches the lane as a crash loop."
    )
    raise AssertionError(msg)


def _emit_run(job: dict[str, Any]) -> str:
    for step in _steps(job):
        if "lab_pass_receipt.py emit" in step.get("run", ""):
            return str(step["run"])
    msg = f"the {GATE_JOB} job emits no lab-pass receipt"
    raise AssertionError(msg)


def _check_line(job: dict[str, Any]) -> str:
    prefix = f'--check "{BOOT_CHECK_NAME}:'
    for line in _emit_run(job).splitlines():
        if prefix in line:
            return line.strip()
    msg = (
        f"the onex-lab receipt does not carry a {BOOT_CHECK_NAME!r} check. A step "
        "that runs and reports nowhere is, to every later reader of the "
        "artifact, a step that never ran."
    )
    raise AssertionError(msg)


def test_the_step_runs_both_boot_test_files_in_this_checkout(
    gate: dict[str, Any],
) -> None:
    run = _boot_step(gate).get("run", "")
    assert "cd omnibase_infra" in run, (
        "the step must run in the checkout of the candidate's own commit, not "
        "in a tree this gate did not check out"
    )
    for path in BOOT_TESTS:
        assert path in run, f"the {BOOT_STEP_ID!r} step does not run {path}"
        assert (REPO_ROOT / path).is_file(), f"{path} does not exist"


def test_the_check_runs_even_when_the_boot_failed(gate: dict[str, Any]) -> None:
    step = _boot_step(gate)
    assert str(step.get("if", "")).strip() == "always()", (
        f"the {BOOT_STEP_ID!r} step must carry `if: always()`; on a failed boot "
        "it would be skipped, and a skipped step reports `fail` into the "
        "receipt for a reason unrelated to the condition checked"
    )


def test_the_step_comes_after_the_interpreter_it_needs(gate: dict[str, Any]) -> None:
    ids_and_names = [
        (step.get("id"), step.get("name", ""), step.get("uses", ""))
        for step in _steps(gate)
    ]
    uv_at = next(
        index
        for index, (_, _, uses) in enumerate(ids_and_names)
        if "astral-sh/setup-uv" in uses
    )
    boot_at = next(
        index
        for index, (step_id, _, _) in enumerate(ids_and_names)
        if step_id == BOOT_STEP_ID
    )
    assert uv_at < boot_at, (
        "the step invokes `uv run` and sits above the setup-uv step; it would "
        "exit 127 and the receipt would record a missing interpreter as "
        f"`{BOOT_CHECK_NAME}: fail` (the OMN-18364 failure)"
    )


def test_the_verdict_is_read_off_the_step_and_not_asserted(
    gate: dict[str, Any],
) -> None:
    line = _check_line(gate)
    assert f"steps.{BOOT_STEP_ID}.outcome" in line, (
        f"the {BOOT_CHECK_NAME!r} verdict must be read from "
        f"steps.{BOOT_STEP_ID}.outcome; as written it does not depend on "
        "whether the check passed"
    )


def test_the_evidence_names_what_was_asserted(gate: dict[str, Any]) -> None:
    evidence = _check_line(gate).split(f'--check "{BOOT_CHECK_NAME}:', 1)[1]
    for token in (
        "discover_runtime_local_ingress_routes",
        "local ingress alias",
        "pre-fix",
        "post-fix",
        "2026-09-16",
    ):
        assert token in evidence, (
            f"the {BOOT_CHECK_NAME!r} evidence does not mention {token!r}; "
            "evidence that does not name the thing measured cannot be checked "
            "by anyone reading the receipt"
        )


def _resolved_argument(gate: dict[str, Any], verdict: str) -> str:
    """The ``--check`` argument as the shell would pass it for one step outcome."""
    line = _check_line(gate)
    quoted = line.split("--check ", 1)[1].rstrip(" \\")
    assert quoted.startswith('"') and quoted.endswith('"')
    body = quoted[1:-1]
    expression_start = body.index("${{")
    expression_end = body.index("}}", expression_start) + 2
    return body[:expression_start] + verdict + body[expression_end:]


@pytest.mark.parametrize(("verdict", "ok"), [("ok", True), ("fail", False)])
def test_the_workflow_check_parses_to_a_receipt_check_with_evidence(
    gate: dict[str, Any], verdict: str, ok: bool
) -> None:
    """AC2: a passing run carries the check name with non-empty evidence."""
    check = parse_check_argument(_resolved_argument(gate, verdict))

    assert check.name == BOOT_CHECK_NAME
    assert check.ok is ok
    assert check.evidence.strip()


def test_the_receipt_refuses_this_check_without_evidence() -> None:
    """AC2: the model already refuses an evidence-free check; pin it on this name."""
    with pytest.raises(ValueError):
        parse_check_argument(f"{BOOT_CHECK_NAME}:ok:")
    with pytest.raises(ValueError):
        ModelLabPassCheck(name=BOOT_CHECK_NAME, ok=True, evidence="")
