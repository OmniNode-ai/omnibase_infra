# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict rules: PASS, FAIL, INCONCLUSIVE, DEV_INHERITED, and restored beside them.

Ticket: OMN-19572
"""

from __future__ import annotations

import json

import pytest

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_outcome import EnumLabProofOutcome
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_result import ModelLabProofResult
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)
from omnibase_infra.lab_proof.model_lab_proof_verdict_request import (
    ModelLabProofVerdictRequest,
)
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_verdict import (
    HandlerLabProofVerdict,
    LabProofVerdictError,
)
from tests.unit.nodes.node_lab_proof_plan_compute.lab_proof_fixtures import (
    HEAD,
    REGISTRY,
    core_plan,
    report_for,
)

pytestmark = pytest.mark.unit

_ID = EnumLabProofStepId
_C = EnumLabProofCheck
_O = EnumLabProofOutcome


def _mandatory() -> tuple[EnumLabProofCheck, ...]:
    registry = load_lab_proof_profile_registry(REGISTRY)
    return (
        registry.profile_for("OmniNode-ai/omnibase_core").variants[0].mandatory_checks
    )


def _judge(
    plan: ModelLabProofPlan,
    report: ModelLabProofRunReport,
    base: ModelLabProofResult | None = None,
) -> ModelLabProofResult:
    return HandlerLabProofVerdict().handle(
        ModelLabProofVerdictRequest(
            plan=plan, report=report, mandatory_checks=_mandatory(), base_result=base
        )
    )


def test_every_check_holding_is_a_pass_keyed_by_repo_pr_head_and_profile() -> None:
    plan = core_plan()
    result = _judge(plan, report_for(plan))
    assert result.outcome is _O.PASS
    assert result.failed_checks == ()
    assert result.receipt_key == (
        f"OmniNode-ai/omnibase_core#1766@{HEAD}:omnibase_core.foundation_override@1"
    )
    assert result.restored
    assert {check.check for check in result.checks} == set(_mandatory())


def test_a_harness_step_failing_is_inconclusive_never_fail() -> None:
    plan = core_plan()
    result = _judge(plan, report_for(plan, failing=frozenset({_ID.BUILD_BASE})))
    assert result.outcome is _O.INCONCLUSIVE
    assert any("harness step build_base" in reason for reason in result.reasons)


def test_a_moved_head_is_inconclusive_and_says_so() -> None:
    plan = core_plan()
    result = _judge(plan, report_for(plan, failing=frozenset({_ID.SUBJECT_REV})))
    assert result.outcome is _O.INCONCLUSIVE
    assert any("head moved" in reason for reason in result.reasons)


def test_an_unhealthy_runtime_is_a_fail() -> None:
    plan = core_plan()
    result = _judge(
        plan, report_for(plan, failing=frozenset({_ID.HEALTH_RUNTIME_MAIN}))
    )
    assert result.outcome is _O.FAIL
    assert result.failed_checks == (_C.RUNTIME_MAIN_HEALTHY,)


def test_an_override_that_does_not_build_is_a_fail() -> None:
    plan = core_plan()
    result = _judge(plan, report_for(plan, failing=frozenset({_ID.BUILD_OVERRIDE})))
    assert result.outcome is _O.FAIL
    assert _C.OVERRIDE_INSTALLED_IDENTITY in result.failed_checks


def test_installed_tree_differing_from_the_tree_under_test_is_a_fail() -> None:
    plan = core_plan()
    other = json.dumps({"files": 1200, "sha256": "cd" * 32})
    result = _judge(
        plan, report_for(plan, stdout={_ID.IDENTITY_RUNTIME_EFFECTS: other})
    )
    assert result.outcome is _O.FAIL
    assert result.failed_checks == (_C.OVERRIDE_INSTALLED_IDENTITY,)


def test_an_empty_tree_under_test_is_not_an_identity() -> None:
    plan = core_plan()
    empty = json.dumps({"files": 0, "sha256": "e3" * 32})
    result = _judge(
        plan,
        report_for(
            plan,
            stdout={
                _ID.SUBJECT_HASH: empty,
                _ID.IDENTITY_RUNTIME_MAIN: empty,
                _ID.IDENTITY_RUNTIME_EFFECTS: empty,
            },
        ),
    )
    assert _C.OVERRIDE_INSTALLED_IDENTITY in result.failed_checks


def test_a_wiring_failure_line_in_either_log_is_a_fail() -> None:
    plan = core_plan()
    hits = {"Auto-wiring failed for": 2, "Cannot register duplicate dispatcher ID": 0}
    result = _judge(
        plan, report_for(plan, patterns={_ID.WIRING_LOGS_RUNTIME_EFFECTS: hits})
    )
    assert result.outcome is _O.FAIL
    assert result.failed_checks == (_C.NO_WIRING_FAILURES,)


def test_focused_tests_are_n_a_when_the_pr_changes_none() -> None:
    plan = core_plan(changed=("src/omnibase_core/topics.py",))
    result = _judge(plan, report_for(plan))
    focused = next(c for c in result.checks if c.check is _C.FOCUSED_TESTS)
    assert focused.passed
    assert focused.detail.startswith("n/a")


def test_a_fail_reproduced_at_the_merge_base_is_dev_inherited() -> None:
    base_plan = core_plan(base_control=True)
    base = _judge(
        base_plan,
        report_for(
            base_plan,
            failing=frozenset({_ID.GOLDEN_CHAIN_DELEGATION, _ID.FOCUSED_TESTS}),
        ),
    )
    assert base.base_control and base.outcome is _O.FAIL
    plan = core_plan()
    head = _judge(
        plan, report_for(plan, failing=frozenset({_ID.GOLDEN_CHAIN_DELEGATION})), base
    )
    assert head.outcome is _O.DEV_INHERITED
    assert not head.base_control


def test_a_fail_the_merge_base_does_not_have_stays_a_fail() -> None:
    base_plan = core_plan(base_control=True)
    base = _judge(base_plan, report_for(base_plan))
    assert base.outcome is _O.PASS
    plan = core_plan()
    head = _judge(
        plan, report_for(plan, failing=frozenset({_ID.HEALTH_RUNTIME_MAIN})), base
    )
    assert head.outcome is _O.FAIL
    assert any("not at the merge base" in reason for reason in head.reasons)


def test_a_head_run_cannot_stand_in_for_a_base_control() -> None:
    plan = core_plan()
    not_base = _judge(
        plan, report_for(plan, failing=frozenset({_ID.HEALTH_RUNTIME_MAIN}))
    )
    with pytest.raises(LabProofVerdictError, match="not a base control"):
        _judge(
            plan,
            report_for(plan, failing=frozenset({_ID.HEALTH_RUNTIME_MAIN})),
            not_base,
        )


def test_negative_control_that_fails_is_reported_as_the_expected_fail() -> None:
    plan = core_plan(negative_control=True)
    result = _judge(
        plan,
        report_for(
            plan,
            failing=frozenset(
                {
                    _ID.HEALTH_RUNTIME_MAIN,
                    _ID.HEALTH_RUNTIME_EFFECTS,
                    _ID.IMPORT_SMOKE_RUNTIME_MAIN,
                    _ID.IMPORT_SMOKE_RUNTIME_EFFECTS,
                    _ID.GOLDEN_CHAIN_DELEGATION,
                }
            ),
        ),
    )
    assert result.negative_control
    assert result.outcome is _O.FAIL
    assert any("expected outcome" in reason for reason in result.reasons)


def test_residue_left_behind_clears_restored_but_not_the_outcome() -> None:
    plan = core_plan()
    result = _judge(plan, report_for(plan, failing=frozenset({_ID.RESIDUE_VOLUMES})))
    assert result.outcome is _O.PASS
    assert not result.restored
    assert any("residue_volumes: NOT OK" in line for line in result.residue_detail)


def test_a_zero_residue_without_its_positive_control_is_not_restored() -> None:
    plan = core_plan()
    result = _judge(
        plan, report_for(plan, failing=frozenset({_ID.RESIDUE_POSITIVE_CONTROL}))
    )
    assert not result.restored


def test_a_check_only_a_hand_run_recipe_evaluates_is_refused() -> None:
    plan = core_plan()
    with pytest.raises(LabProofVerdictError, match="changed_path_live"):
        HandlerLabProofVerdict().handle(
            ModelLabProofVerdictRequest(
                plan=plan,
                report=report_for(plan),
                mandatory_checks=(_C.CHANGED_PATH_LIVE,),
            )
        )


def test_a_report_of_another_run_is_refused() -> None:
    plan = core_plan()
    other = report_for(core_plan(base_control=True))
    with pytest.raises(LabProofVerdictError, match="is not a run of plan"):
        _judge(plan, other)
