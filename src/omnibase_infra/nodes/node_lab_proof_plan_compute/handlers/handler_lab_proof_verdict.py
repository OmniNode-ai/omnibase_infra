# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerLabProofVerdict — decides one proof run from its observations.

Canonical definition-B handler: ``handle(ModelLabProofVerdictRequest) ->
ModelLabProofResult``. Pure: the run effect recorded what happened; this
decides what it means, so a recorded run can be re-judged without a lab.

THE OUTCOME RULES (interim recipes common frame 8, plan section 6):

  INCONCLUSIVE   a harness step that had to succeed did not (host over its load
                 ceiling, a clone, a build of the pinned base image, a moved
                 head). Nothing is said about the PR. A pilot profile never turns
                 a harness failure into a FAIL.
  PASS           every mandatory check of the row held.
  FAIL           a mandatory check did not hold and no harness step failed.
  DEV_INHERITED  a FAIL whose failed checks all failed the same way in a base
                 control run of the same steps at the merge base: the defect is
                 dev's, not the PR's.

``restored`` (the zero-residue readback with its positive control) is reported
beside the outcome and never changes it.

Ticket: OMN-19572
"""

from __future__ import annotations

import json

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_outcome import EnumLabProofOutcome
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.model_lab_proof_check_result import (
    ModelLabProofCheckResult,
)
from omnibase_infra.lab_proof.model_lab_proof_observation import (
    ModelLabProofObservation,
)
from omnibase_infra.lab_proof.model_lab_proof_result import ModelLabProofResult
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)
from omnibase_infra.lab_proof.model_lab_proof_verdict_request import (
    ModelLabProofVerdictRequest,
)

_ID = EnumLabProofStepId
_C = EnumLabProofCheck

# Checks this handler can evaluate from a node run. A row naming any other
# check runs by hand and cannot be judged here.
NODE_EVALUATED_CHECKS: frozenset[EnumLabProofCheck] = frozenset(
    {
        _C.SUBJECT_HEAD_IDENTITY,
        _C.FOCUSED_TESTS,
        _C.OVERRIDE_INSTALLED_IDENTITY,
        _C.MIGRATION_GATE_HEALTHY,
        _C.RUNTIME_MAIN_HEALTHY,
        _C.RUNTIME_EFFECTS_HEALTHY,
        _C.CONSUMER_IMPORT_SMOKE,
        _C.NO_WIRING_FAILURES,
        _C.GOLDEN_CHAIN_DELEGATION,
    }
)


class LabProofVerdictError(ValueError):
    """The verdict cannot be computed from this request."""


def _describe(observation: ModelLabProofObservation | None) -> str:
    if observation is None:
        return "step not in the plan"
    if not observation.ran:
        return f"not run: {observation.skip_reason or 'skipped'}"
    if observation.timed_out:
        return f"timed out after {observation.attempts} attempt(s)"
    tail = observation.stdout_tail.strip().splitlines()
    last = tail[-1][:160] if tail else ""
    return f"exit {observation.exit_code}, {observation.attempts} attempt(s)" + (
        f", stdout: {last}" if last else ""
    )


def _ok(report: ModelLabProofRunReport, step_id: EnumLabProofStepId) -> bool:
    observation = report.get(step_id)
    return observation is not None and observation.ok


def _tree_hash(observation: ModelLabProofObservation | None) -> tuple[str, int] | None:
    if observation is None or not observation.ok:
        return None
    lines = observation.stdout_tail.strip().splitlines()
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    digest, files = parsed.get("sha256"), parsed.get("files")
    if not isinstance(digest, str) or not isinstance(files, int):
        return None
    return digest, files


class HandlerLabProofVerdict:
    """Pure verdict for one proof run."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: compute handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute, no external I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelLabProofVerdictRequest) -> ModelLabProofResult:
        """Evaluate the row's mandatory checks and decide the outcome."""
        plan, report = request.plan, request.report
        if report.run_key != plan.run_key:
            raise LabProofVerdictError(
                f"report {report.run_key} is not a run of plan {plan.run_key}"
            )
        unknown = sorted(set(request.mandatory_checks) - NODE_EVALUATED_CHECKS)
        if unknown:
            raise LabProofVerdictError(
                "checks not evaluated from a node run: " + ", ".join(unknown)
            )
        checks = tuple(self._check(check, report) for check in request.mandatory_checks)
        failed = tuple(result.check for result in checks if not result.passed)
        reasons: list[str] = []
        harness_abort = False
        if report.aborted_at is not None:
            aborted = report.get(report.aborted_at)
            if (
                aborted is not None
                and aborted.attribution is EnumLabProofAttribution.HARNESS
            ):
                harness_abort = True
                reasons.append(
                    f"harness step {report.aborted_at} failed ({_describe(aborted)}); "
                    "nothing is concluded about the PR"
                )
                if report.aborted_at is _ID.SUBJECT_REV:
                    reasons.append(
                        "the fetched commit is not the one under test: the head moved"
                    )
            else:
                reasons.append(
                    f"subject step {report.aborted_at} failed ({_describe(aborted)})"
                )
        if harness_abort:
            outcome = EnumLabProofOutcome.INCONCLUSIVE
        elif not failed:
            outcome = EnumLabProofOutcome.PASS
        else:
            outcome = EnumLabProofOutcome.FAIL
            reasons.append("failed checks: " + ", ".join(failed))
            base = request.base_result
            if base is not None and not plan.negative_control:
                if not base.base_control:
                    raise LabProofVerdictError("base_result is not a base control run")
                if base.outcome is EnumLabProofOutcome.INCONCLUSIVE:
                    reasons.append("base control was inconclusive; FAIL stands")
                elif set(failed) <= set(base.failed_checks):
                    outcome = EnumLabProofOutcome.DEV_INHERITED
                    reasons.append(
                        f"every failed check also failed at the merge base "
                        f"{base.proved_sha[:12]} (run {base.run_key}): dev-inherited"
                    )
                else:
                    only_head = sorted(set(failed) - set(base.failed_checks))
                    reasons.append(
                        "failed at the head but not at the merge base: "
                        + ", ".join(only_head)
                    )
        if plan.negative_control:
            reasons.append(
                "negative control run: FAIL is the expected outcome; PASS would mean "
                "these checks cannot fail"
            )
        residue = [
            observation
            for observation in report.observations
            if observation.phase is EnumLabProofStepPhase.RESIDUE
        ]
        residue_detail = tuple(
            f"{observation.step_id}: {'ok' if observation.ok else 'NOT OK'} "
            f"({_describe(observation)})"
            for observation in residue
        )
        restored = bool(residue) and all(observation.ok for observation in residue)
        subject = plan.subject
        return ModelLabProofResult(
            receipt_key=(
                f"{subject.repo}#{subject.pr_number}@{subject.head_sha}:"
                f"{plan.profile_key}@{plan.profile_version}"
            ),
            repo=subject.repo,
            pr_number=subject.pr_number,
            head_sha=subject.head_sha,
            base_sha=subject.base_sha,
            proved_sha=subject.proved_sha,
            profile_key=plan.profile_key,
            profile_version=plan.profile_version,
            variant_key=plan.variant_key,
            host=plan.host,
            run_key=plan.run_key,
            negative_control=plan.negative_control,
            base_control=subject.is_base_control,
            outcome=outcome,
            checks=checks,
            reasons=tuple(reasons),
            restored=restored,
            residue_detail=residue_detail,
            started_at=report.started_at,
            finished_at=report.finished_at,
            failed_checks=failed,
        )

    def _check(
        self, check: EnumLabProofCheck, report: ModelLabProofRunReport
    ) -> ModelLabProofCheckResult:
        def result(passed: bool, detail: str) -> ModelLabProofCheckResult:
            return ModelLabProofCheckResult(check=check, passed=passed, detail=detail)

        def single(step_id: EnumLabProofStepId) -> ModelLabProofCheckResult:
            observation = report.get(step_id)
            return result(_ok(report, step_id), f"{step_id}: {_describe(observation)}")

        def both(
            main: EnumLabProofStepId, effects: EnumLabProofStepId
        ) -> ModelLabProofCheckResult:
            detail = "; ".join(
                f"{step_id}: {_describe(report.get(step_id))}"
                for step_id in (main, effects)
            )
            return result(_ok(report, main) and _ok(report, effects), detail)

        if check is _C.SUBJECT_HEAD_IDENTITY:
            return single(_ID.SUBJECT_REV)
        if check is _C.FOCUSED_TESTS:
            if report.get(_ID.FOCUSED_TESTS) is None:
                return result(True, "n/a: the PR changes no test files")
            return single(_ID.FOCUSED_TESTS)
        if check is _C.MIGRATION_GATE_HEALTHY:
            return single(_ID.HEALTH_MIGRATION_GATE)
        if check is _C.RUNTIME_MAIN_HEALTHY:
            return single(_ID.HEALTH_RUNTIME_MAIN)
        if check is _C.RUNTIME_EFFECTS_HEALTHY:
            return single(_ID.HEALTH_RUNTIME_EFFECTS)
        if check is _C.GOLDEN_CHAIN_DELEGATION:
            return single(_ID.GOLDEN_CHAIN_DELEGATION)
        if check is _C.CONSUMER_IMPORT_SMOKE:
            return both(_ID.IMPORT_SMOKE_RUNTIME_MAIN, _ID.IMPORT_SMOKE_RUNTIME_EFFECTS)
        if check is _C.NO_WIRING_FAILURES:
            parts: list[str] = []
            passed = True
            for step_id in (
                _ID.WIRING_LOGS_RUNTIME_MAIN,
                _ID.WIRING_LOGS_RUNTIME_EFFECTS,
            ):
                observation = report.get(step_id)
                if observation is None or not observation.ok:
                    passed = False
                    parts.append(f"{step_id}: {_describe(observation)}")
                    continue
                hits = {k: v for k, v in observation.pattern_counts.items() if v}
                passed = passed and not hits
                parts.append(f"{step_id}: {hits or 'no failure lines'}")
            return result(passed, "; ".join(parts))
        # OVERRIDE_INSTALLED_IDENTITY
        expected = _tree_hash(report.get(_ID.SUBJECT_HASH))
        if expected is None or expected[1] == 0:
            return result(False, "no hash of the package tree under test")
        parts = []
        passed = True
        for step_id in (_ID.IDENTITY_RUNTIME_MAIN, _ID.IDENTITY_RUNTIME_EFFECTS):
            installed = _tree_hash(report.get(step_id))
            same = installed == expected
            passed = passed and same
            parts.append(
                f"{step_id}: "
                + (
                    f"{installed[0][:16]} ({installed[1]} files)"
                    if installed
                    else _describe(report.get(step_id))
                )
                + (" = tree under test" if same else " != tree under test")
            )
        return result(
            passed,
            f"tree under test {expected[0][:16]} ({expected[1]} files); "
            + "; ".join(parts),
        )


__all__ = ["NODE_EVALUATED_CHECKS", "HandlerLabProofVerdict", "LabProofVerdictError"]
