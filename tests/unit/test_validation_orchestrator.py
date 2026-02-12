# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the validation orchestrator node.

Tests:
- HandlerBuildPlan handler properties and async handle()
- ModelPatternCandidate and ModelValidationPlan model construction
- Check catalog composition (7 required, 3 recommended, 2 informational)
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumCheckSeverity,
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_validation_orchestrator.handlers.handler_build_plan import (
    HandlerBuildPlan,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
    ModelPlannedCheck,
    ModelValidationPlan,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Fixtures (local to this module)
# ============================================================================


@pytest.fixture
def handler() -> HandlerBuildPlan:
    """Create a HandlerBuildPlan instance."""
    return HandlerBuildPlan()


@pytest.fixture
def candidate() -> ModelPatternCandidate:
    """Create a sample pattern candidate for testing."""
    return ModelPatternCandidate(
        candidate_id=uuid4(),
        pattern_id=uuid4(),
        source_path="/src/test_module",
        diff_summary="Test diff summary",
        changed_files=("src/foo.py", "src/bar.py"),
        risk_tags=("security",),
    )


# ============================================================================
# HandlerBuildPlan -- Properties
# ============================================================================


class TestHandlerBuildPlanProperties:
    """Tests for HandlerBuildPlan handler classification properties."""

    def test_handler_id(self, handler: HandlerBuildPlan) -> None:
        """handler_id returns the expected identifier."""
        assert handler.handler_id == "handler-build-plan"

    def test_handler_type(self, handler: HandlerBuildPlan) -> None:
        """handler_type is NODE_HANDLER."""
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER

    def test_handler_category(self, handler: HandlerBuildPlan) -> None:
        """handler_category is COMPUTE (pure transformation)."""
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE


# ============================================================================
# HandlerBuildPlan -- handle()
# ============================================================================


class TestHandlerBuildPlanHandle:
    """Tests for HandlerBuildPlan.handle() async method."""

    @pytest.mark.asyncio
    async def test_handle_produces_plan_with_12_checks(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """handle() produces a plan with exactly 12 MVP checks."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        assert isinstance(plan, ModelValidationPlan)
        assert len(plan.checks) == 12

    @pytest.mark.asyncio
    async def test_handle_plan_references_candidate(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Plan's candidate_id matches the input candidate."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        assert plan.candidate_id == candidate.candidate_id

    @pytest.mark.asyncio
    async def test_handle_plan_has_score_threshold(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Plan has default score_threshold of 0.8."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        assert plan.score_threshold == 0.8

    @pytest.mark.asyncio
    async def test_handle_all_checks_enabled(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """All checks in the plan are enabled by default."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        assert all(check.enabled for check in plan.checks)

    @pytest.mark.asyncio
    async def test_handle_check_catalog_7_required(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Check catalog contains 7 REQUIRED checks."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        required_count = sum(
            1 for c in plan.checks if c.severity == EnumCheckSeverity.REQUIRED
        )
        assert required_count == 7

    @pytest.mark.asyncio
    async def test_handle_check_catalog_3_recommended(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Check catalog contains 3 RECOMMENDED checks."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        recommended_count = sum(
            1 for c in plan.checks if c.severity == EnumCheckSeverity.RECOMMENDED
        )
        assert recommended_count == 3

    @pytest.mark.asyncio
    async def test_handle_check_catalog_2_informational(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Check catalog contains 2 INFORMATIONAL checks."""
        plan = await handler.handle(candidate, correlation_id=uuid4())
        informational_count = sum(
            1 for c in plan.checks if c.severity == EnumCheckSeverity.INFORMATIONAL
        )
        assert informational_count == 2

    @pytest.mark.asyncio
    async def test_handle_check_codes_match_catalog(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """All expected check_codes from the MVP catalog are present."""
        expected_codes = {
            "CHECK-PY-001",
            "CHECK-PY-002",
            "CHECK-TEST-001",
            "CHECK-TEST-002",
            "CHECK-VAL-001",
            "CHECK-VAL-002",
            "CHECK-RISK-001",
            "CHECK-RISK-002",
            "CHECK-RISK-003",
            "CHECK-OUT-001",
            "CHECK-COST-001",
            "CHECK-TIME-001",
        }
        plan = await handler.handle(candidate, correlation_id=uuid4())
        actual_codes = {c.check_code for c in plan.checks}
        assert actual_codes == expected_codes

    @pytest.mark.asyncio
    async def test_handle_plan_has_unique_plan_id(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """Each invocation produces a plan with a distinct plan_id."""
        plan_a = await handler.handle(candidate, correlation_id=uuid4())
        plan_b = await handler.handle(candidate, correlation_id=uuid4())
        assert plan_a.plan_id != plan_b.plan_id

    @pytest.mark.asyncio
    async def test_handle_auto_generates_correlation_id_when_none(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """handle() succeeds with correlation_id=None (auto-generates UUID)."""
        plan = await handler.handle(candidate, correlation_id=None)
        assert isinstance(plan, ModelValidationPlan)
        assert isinstance(plan.plan_id, UUID)
        assert len(plan.checks) == 12

    @pytest.mark.asyncio
    async def test_handle_omitted_correlation_id_defaults_to_none(
        self,
        handler: HandlerBuildPlan,
        candidate: ModelPatternCandidate,
    ) -> None:
        """handle() succeeds when correlation_id is omitted entirely."""
        plan = await handler.handle(candidate)
        assert isinstance(plan, ModelValidationPlan)
        assert len(plan.checks) == 12


# ============================================================================
# ModelPatternCandidate
# ============================================================================


class TestModelPatternCandidate:
    """Tests for ModelPatternCandidate Pydantic model."""

    def test_construction(self) -> None:
        """Model can be constructed with required fields."""
        cid = uuid4()
        pid = uuid4()
        candidate = ModelPatternCandidate(
            candidate_id=cid,
            pattern_id=pid,
            source_path="/src",
        )
        assert candidate.candidate_id == cid
        assert candidate.pattern_id == pid
        assert candidate.source_path == "/src"
        assert candidate.diff_summary == ""
        assert candidate.changed_files == ()
        assert candidate.risk_tags == ()

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        candidate = ModelPatternCandidate(
            candidate_id=uuid4(),
            pattern_id=uuid4(),
            source_path="/src",
        )
        with pytest.raises(ValidationError):
            candidate.pattern_id = uuid4()  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelPatternCandidate(
                candidate_id=uuid4(),
                pattern_id=uuid4(),
                source_path="/src",
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================================
# ModelValidationPlan
# ============================================================================


class TestModelValidationPlan:
    """Tests for ModelValidationPlan Pydantic model."""

    def test_construction(self) -> None:
        """Model can be constructed with required fields."""
        cid = uuid4()
        plan = ModelValidationPlan(
            candidate_id=cid,
            checks=(),
        )
        assert plan.candidate_id == cid
        assert plan.checks == ()
        assert plan.score_threshold == 0.8

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        plan = ModelValidationPlan(candidate_id=uuid4(), checks=())
        with pytest.raises(ValidationError):
            plan.score_threshold = 0.5  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelValidationPlan(
                candidate_id=uuid4(),
                checks=(),
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_score_threshold_bounds_low(self) -> None:
        """score_threshold below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelValidationPlan(
                candidate_id=uuid4(),
                checks=(),
                score_threshold=-0.1,
            )

    def test_score_threshold_bounds_high(self) -> None:
        """score_threshold above 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelValidationPlan(
                candidate_id=uuid4(),
                checks=(),
                score_threshold=1.1,
            )


# ============================================================================
# ModelPlannedCheck
# ============================================================================


class TestModelPlannedCheck:
    """Tests for ModelPlannedCheck Pydantic model."""

    def test_construction(self) -> None:
        """Model can be constructed with required fields."""
        check = ModelPlannedCheck(
            check_code="CHECK-PY-001",
            label="Typecheck",
            severity=EnumCheckSeverity.REQUIRED,
        )
        assert check.check_code == "CHECK-PY-001"
        assert check.label == "Typecheck"
        assert check.severity == EnumCheckSeverity.REQUIRED
        assert check.enabled is True
        assert check.command == ""
        assert check.timeout_ms == 0.0

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        check = ModelPlannedCheck(
            check_code="CHECK-PY-001",
            label="Typecheck",
            severity=EnumCheckSeverity.REQUIRED,
        )
        with pytest.raises(ValidationError):
            check.enabled = False  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelPlannedCheck(
                check_code="CHECK-PY-001",
                label="Typecheck",
                severity=EnumCheckSeverity.REQUIRED,
                unknown_field="bad",  # type: ignore[call-arg]
            )
