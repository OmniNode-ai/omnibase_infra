# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the validation executor effect node.

Tests:
- HandlerRunChecks handler properties and async handle()
- ModelCheckResult.is_blocking_failure() logic
- ModelExecutorResult computed properties and __bool__()
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.enums import (
    EnumCheckSeverity,
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.models.validation import (
    ModelCheckResult,
    ModelExecutorResult,
    ModelPlannedCheck,
    ModelValidationPlan,
)
from omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks import (
    HandlerRunChecks,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_planned_check(
    check_code: str = "CHECK-TEST-001",
    label: str = "Test check",
    severity: EnumCheckSeverity = EnumCheckSeverity.REQUIRED,
    enabled: bool = True,
    command: str = "poetry run pytest",
) -> ModelPlannedCheck:
    """Create a planned check for testing."""
    return ModelPlannedCheck(
        check_code=check_code,
        label=label,
        severity=severity,
        enabled=enabled,
        command=command,
    )


def _make_plan(
    checks: tuple[ModelPlannedCheck, ...] = (),
) -> ModelValidationPlan:
    """Create a validation plan for testing."""
    return ModelValidationPlan(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        checks=checks,
    )


def _make_check_result(
    check_code: str = "CHECK-TEST-001",
    label: str = "Test check",
    severity: EnumCheckSeverity = EnumCheckSeverity.REQUIRED,
    passed: bool = True,
    skipped: bool = False,
) -> ModelCheckResult:
    """Create a check result for testing."""
    return ModelCheckResult(
        check_code=check_code,
        label=label,
        severity=severity,
        passed=passed,
        skipped=skipped,
        message="test message",
        executed_at=datetime.now(tz=UTC),
    )


# ============================================================================
# HandlerRunChecks -- Properties
# ============================================================================


class TestHandlerRunChecksProperties:
    """Tests for HandlerRunChecks handler classification properties."""

    def test_handler_id(self) -> None:
        """handler_id returns the expected identifier."""
        handler = HandlerRunChecks()
        assert handler.handler_id == "handler-run-checks"

    def test_handler_type(self) -> None:
        """handler_type is INFRA_HANDLER (subprocess I/O)."""
        handler = HandlerRunChecks()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        """handler_category is EFFECT (side-effecting)."""
        handler = HandlerRunChecks()
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# ============================================================================
# HandlerRunChecks -- handle()
# ============================================================================


class TestHandlerRunChecksHandle:
    """Tests for HandlerRunChecks.handle() async method."""

    @pytest.mark.asyncio
    async def test_handle_enabled_checks_pass(self) -> None:
        """Enabled checks produce passed results in MVP skeleton."""
        handler = HandlerRunChecks()
        checks = tuple(
            _make_planned_check(
                check_code=f"CHECK-{i}",
                label=f"Check {i}",
                enabled=True,
            )
            for i in range(3)
        )
        plan = _make_plan(checks=checks)

        result = await handler.handle(plan, correlation_id=uuid4())

        assert len(result.check_results) == 3
        assert all(r.passed for r in result.check_results)
        assert all(not r.skipped for r in result.check_results)

    @pytest.mark.asyncio
    async def test_handle_disabled_checks_are_skipped(self) -> None:
        """Disabled checks produce skipped results."""
        handler = HandlerRunChecks()
        checks = (
            _make_planned_check(check_code="C-1", enabled=True),
            _make_planned_check(check_code="C-2", enabled=False),
            _make_planned_check(check_code="C-3", enabled=False),
        )
        plan = _make_plan(checks=checks)

        result = await handler.handle(plan, correlation_id=uuid4())

        assert len(result.check_results) == 3
        # First check: enabled -> passed
        assert result.check_results[0].passed is True
        assert result.check_results[0].skipped is False
        # Second and third: disabled -> skipped
        assert result.check_results[1].passed is False
        assert result.check_results[1].skipped is True
        assert result.check_results[2].passed is False
        assert result.check_results[2].skipped is True

    @pytest.mark.asyncio
    async def test_handle_empty_plan(self) -> None:
        """Empty plan produces an empty result with no check_results."""
        handler = HandlerRunChecks()
        plan = _make_plan(checks=())

        result = await handler.handle(plan, correlation_id=uuid4())

        assert len(result.check_results) == 0
        assert result.pass_count == 0
        assert result.fail_count == 0

    @pytest.mark.asyncio
    async def test_handle_result_references_plan(self) -> None:
        """Result references the plan_id and candidate_id from input."""
        handler = HandlerRunChecks()
        plan = _make_plan(checks=())

        result = await handler.handle(plan, correlation_id=uuid4())

        assert result.plan_id == plan.plan_id
        assert result.candidate_id == plan.candidate_id

    @pytest.mark.asyncio
    async def test_handle_records_total_duration(self) -> None:
        """Result has a non-negative total_duration_ms."""
        handler = HandlerRunChecks()
        plan = _make_plan(
            checks=(_make_planned_check(),),
        )

        result = await handler.handle(plan, correlation_id=uuid4())

        assert result.total_duration_ms >= 0.0


# ============================================================================
# ModelCheckResult
# ============================================================================


class TestModelCheckResult:
    """Tests for ModelCheckResult model and is_blocking_failure() logic."""

    def test_blocking_failure_required_failed(self) -> None:
        """A required check that failed and was not skipped is a blocking failure."""
        result = _make_check_result(
            severity=EnumCheckSeverity.REQUIRED, passed=False, skipped=False
        )
        assert result.is_blocking_failure() is True

    def test_not_blocking_when_passed(self) -> None:
        """A required check that passed is not a blocking failure."""
        result = _make_check_result(
            severity=EnumCheckSeverity.REQUIRED, passed=True, skipped=False
        )
        assert result.is_blocking_failure() is False

    def test_not_blocking_when_skipped(self) -> None:
        """A required check that was skipped is not a blocking failure."""
        result = _make_check_result(
            severity=EnumCheckSeverity.REQUIRED, passed=False, skipped=True
        )
        assert result.is_blocking_failure() is False

    def test_not_blocking_when_recommended(self) -> None:
        """A recommended check that failed is not a blocking failure."""
        result = _make_check_result(
            severity=EnumCheckSeverity.RECOMMENDED, passed=False, skipped=False
        )
        assert result.is_blocking_failure() is False

    def test_not_blocking_when_informational(self) -> None:
        """An informational check that failed is not a blocking failure."""
        result = _make_check_result(
            severity=EnumCheckSeverity.INFORMATIONAL, passed=False, skipped=False
        )
        assert result.is_blocking_failure() is False


# ============================================================================
# ModelExecutorResult
# ============================================================================


class TestModelExecutorResult:
    """Tests for ModelExecutorResult computed properties and __bool__."""

    def test_all_required_passed_true(self) -> None:
        """all_required_passed is True when all REQUIRED checks pass."""
        results = (
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=True),
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=True),
            _make_check_result(severity=EnumCheckSeverity.RECOMMENDED, passed=False),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.all_required_passed is True

    def test_all_required_passed_false(self) -> None:
        """all_required_passed is False when a REQUIRED check fails."""
        results = (
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=True),
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=False),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.all_required_passed is False

    def test_all_required_passed_skipped_treated_as_passing(self) -> None:
        """Skipped REQUIRED checks count as passing for all_required_passed."""
        results = (
            _make_check_result(
                severity=EnumCheckSeverity.REQUIRED, passed=False, skipped=True
            ),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.all_required_passed is True

    def test_has_blocking_failures_true(self) -> None:
        """has_blocking_failures is True when a required check failed."""
        results = (
            _make_check_result(
                severity=EnumCheckSeverity.REQUIRED, passed=False, skipped=False
            ),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.has_blocking_failures is True

    def test_has_blocking_failures_false(self) -> None:
        """has_blocking_failures is False when all pass."""
        results = (
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=True),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.has_blocking_failures is False

    def test_pass_count(self) -> None:
        """pass_count counts only passed checks."""
        results = (
            _make_check_result(passed=True),
            _make_check_result(passed=True),
            _make_check_result(passed=False),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.pass_count == 2

    def test_fail_count(self) -> None:
        """fail_count counts failed non-skipped checks."""
        results = (
            _make_check_result(passed=False, skipped=False),
            _make_check_result(passed=False, skipped=True),  # skipped -> not counted
            _make_check_result(passed=True, skipped=False),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert executor.fail_count == 1

    def test_bool_true_when_all_required_pass(self) -> None:
        """__bool__ returns True when all required checks pass."""
        results = (
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=True),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert bool(executor) is True

    def test_bool_false_when_required_fails(self) -> None:
        """__bool__ returns False when a required check fails."""
        results = (
            _make_check_result(severity=EnumCheckSeverity.REQUIRED, passed=False),
        )
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=results,
        )
        assert bool(executor) is False

    def test_bool_true_when_empty(self) -> None:
        """__bool__ returns True for empty results (vacuously all_required_passed)."""
        executor = ModelExecutorResult(
            plan_id=uuid4(),
            candidate_id=uuid4(),
            check_results=(),
        )
        assert bool(executor) is True
