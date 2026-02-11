# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the pattern lifecycle effect node.

Tests:
- ModelLifecycleState pure transitions (with_verdict)
- Promotion ladder (2 consecutive PASSes)
- Demotion rules (2 consecutive FAILs, 3 -> suppress)
- QUARANTINE recording without tier change
- HandlerLifecycleUpdate handler properties and async handle()
- ModelLifecycleResult __bool__ behaviour
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLifecycleTier,
    EnumValidationVerdict,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.handlers.handler_lifecycle_update import (
    HandlerLifecycleUpdate,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models.model_lifecycle_result import (
    ModelLifecycleResult,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models.model_lifecycle_state import (
    ModelLifecycleState,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_state(
    pattern_id: UUID | None = None,
    tier: EnumLifecycleTier = EnumLifecycleTier.OBSERVED,
    pass_count: int = 0,
    fail_count: int = 0,
    total: int = 0,
) -> ModelLifecycleState:
    """Create a lifecycle state for testing."""
    return ModelLifecycleState(
        pattern_id=pattern_id if pattern_id is not None else uuid4(),
        current_tier=tier,
        consecutive_pass_count=pass_count,
        consecutive_fail_count=fail_count,
        total_validations=total,
    )


# ============================================================================
# ModelLifecycleState -- Construction
# ============================================================================


class TestLifecycleStateConstruction:
    """Tests for ModelLifecycleState construction and defaults."""

    def test_default_tier_is_observed(self) -> None:
        """Default tier is OBSERVED."""
        state = ModelLifecycleState(pattern_id=uuid4())
        assert state.current_tier == EnumLifecycleTier.OBSERVED

    def test_default_counters_are_zero(self) -> None:
        """Default counters start at zero."""
        state = ModelLifecycleState(pattern_id=uuid4())
        assert state.consecutive_pass_count == 0
        assert state.consecutive_fail_count == 0
        assert state.total_validations == 0

    def test_default_last_verdict_is_none(self) -> None:
        """Default last_verdict is None."""
        state = ModelLifecycleState(pattern_id=uuid4())
        assert state.last_verdict is None
        assert state.last_updated is None

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        state = ModelLifecycleState(pattern_id=uuid4())
        with pytest.raises(ValidationError):
            state.pattern_id = uuid4()  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelLifecycleState(
                pattern_id=uuid4(),
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================================
# ModelLifecycleState -- with_verdict(PASS)
# ============================================================================


class TestLifecycleStatePassVerdict:
    """Tests for applying PASS verdicts."""

    def test_pass_increments_pass_count(self) -> None:
        """PASS increments consecutive_pass_count."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.consecutive_pass_count == 1

    def test_pass_resets_fail_count(self) -> None:
        """PASS resets consecutive_fail_count to 0."""
        state = _make_state(fail_count=2)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.consecutive_fail_count == 0

    def test_pass_increments_total(self) -> None:
        """PASS increments total_validations."""
        state = _make_state(total=5)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.total_validations == 6

    def test_pass_records_verdict(self) -> None:
        """PASS is recorded as last_verdict."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.last_verdict == EnumValidationVerdict.PASS

    def test_pass_sets_last_updated(self) -> None:
        """PASS sets last_updated timestamp."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.last_updated is not None

    def test_single_pass_no_promotion(self) -> None:
        """A single PASS does not promote the tier."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.current_tier == EnumLifecycleTier.OBSERVED

    def test_two_consecutive_passes_promote(self) -> None:
        """2 consecutive PASSes promote OBSERVED -> SUGGESTED."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED, pass_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.current_tier == EnumLifecycleTier.SUGGESTED

    def test_promotion_resets_pass_count(self) -> None:
        """After promotion, pass counter resets to 0."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED, pass_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        assert new_state.consecutive_pass_count == 0

    def test_promotion_full_ladder(self) -> None:
        """Walk the full promotion ladder with 2-pass streaks."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED)

        # OBSERVED -> SUGGESTED (2 passes)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        assert state.current_tier == EnumLifecycleTier.SUGGESTED

        # SUGGESTED -> SHADOW_APPLY (2 passes)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        assert state.current_tier == EnumLifecycleTier.SHADOW_APPLY

        # SHADOW_APPLY -> PROMOTED (2 passes)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        assert state.current_tier == EnumLifecycleTier.PROMOTED

        # PROMOTED -> DEFAULT (2 passes)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        state = state.with_verdict(EnumValidationVerdict.PASS)
        assert state.current_tier == EnumLifecycleTier.DEFAULT

    def test_no_promotion_past_default(self) -> None:
        """DEFAULT tier does not promote further (can_promote is False)."""
        state = _make_state(tier=EnumLifecycleTier.DEFAULT, pass_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.PASS)
        # Should stay at DEFAULT, pass_count increments but no promotion
        assert new_state.current_tier == EnumLifecycleTier.DEFAULT


# ============================================================================
# ModelLifecycleState -- with_verdict(FAIL)
# ============================================================================


class TestLifecycleStateFailVerdict:
    """Tests for applying FAIL verdicts."""

    def test_fail_increments_fail_count(self) -> None:
        """FAIL increments consecutive_fail_count."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.consecutive_fail_count == 1

    def test_fail_resets_pass_count(self) -> None:
        """FAIL resets consecutive_pass_count to 0."""
        state = _make_state(pass_count=3)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.consecutive_pass_count == 0

    def test_fail_records_verdict(self) -> None:
        """FAIL is recorded as last_verdict."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.last_verdict == EnumValidationVerdict.FAIL

    def test_single_fail_no_demotion(self) -> None:
        """A single FAIL does not demote the tier."""
        state = _make_state(tier=EnumLifecycleTier.SUGGESTED)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.current_tier == EnumLifecycleTier.SUGGESTED

    def test_two_consecutive_fails_demote(self) -> None:
        """2 consecutive FAILs demote SUGGESTED -> OBSERVED."""
        state = _make_state(tier=EnumLifecycleTier.SUGGESTED, fail_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.current_tier == EnumLifecycleTier.OBSERVED

    def test_three_consecutive_fails_suppress(self) -> None:
        """3 consecutive FAILs suppress the pattern."""
        state = _make_state(tier=EnumLifecycleTier.SUGGESTED, fail_count=2)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.current_tier == EnumLifecycleTier.SUPPRESSED

    def test_observed_demotion_stays_at_observed(self) -> None:
        """2 consecutive FAILs at OBSERVED floor stay at OBSERVED (demoted() returns OBSERVED)."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED, fail_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.current_tier == EnumLifecycleTier.OBSERVED

    def test_three_fails_at_observed_suppresses(self) -> None:
        """3 consecutive FAILs even from OBSERVED -> SUPPRESSED."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED, fail_count=2)
        new_state = state.with_verdict(EnumValidationVerdict.FAIL)
        assert new_state.current_tier == EnumLifecycleTier.SUPPRESSED


# ============================================================================
# ModelLifecycleState -- with_verdict(QUARANTINE)
# ============================================================================


class TestLifecycleStateQuarantineVerdict:
    """Tests for applying QUARANTINE verdicts."""

    def test_quarantine_does_not_change_tier(self) -> None:
        """QUARANTINE records the verdict without changing the tier."""
        state = _make_state(tier=EnumLifecycleTier.SUGGESTED)
        new_state = state.with_verdict(EnumValidationVerdict.QUARANTINE)
        assert new_state.current_tier == EnumLifecycleTier.SUGGESTED

    def test_quarantine_records_verdict(self) -> None:
        """QUARANTINE is recorded as last_verdict."""
        state = _make_state()
        new_state = state.with_verdict(EnumValidationVerdict.QUARANTINE)
        assert new_state.last_verdict == EnumValidationVerdict.QUARANTINE

    def test_quarantine_increments_total(self) -> None:
        """QUARANTINE increments total_validations."""
        state = _make_state(total=3)
        new_state = state.with_verdict(EnumValidationVerdict.QUARANTINE)
        assert new_state.total_validations == 4

    def test_quarantine_preserves_pass_count(self) -> None:
        """QUARANTINE does not reset pass or fail counters."""
        state = _make_state(pass_count=1, fail_count=1)
        new_state = state.with_verdict(EnumValidationVerdict.QUARANTINE)
        # model_copy preserves fields not in update dict
        assert new_state.consecutive_pass_count == 1
        assert new_state.consecutive_fail_count == 1


# ============================================================================
# ModelLifecycleState -- can_promote() and is_suppressed()
# ============================================================================


class TestLifecycleStateHelpers:
    """Tests for can_promote() and is_suppressed() convenience methods."""

    def test_can_promote_delegates_to_tier(self) -> None:
        """can_promote delegates to the tier enum."""
        state = _make_state(tier=EnumLifecycleTier.OBSERVED)
        assert state.can_promote() is True

        state = _make_state(tier=EnumLifecycleTier.DEFAULT)
        assert state.can_promote() is False

    def test_is_suppressed(self) -> None:
        """is_suppressed returns True only when tier is SUPPRESSED."""
        state = _make_state(tier=EnumLifecycleTier.SUPPRESSED)
        assert state.is_suppressed() is True

        state = _make_state(tier=EnumLifecycleTier.OBSERVED)
        assert state.is_suppressed() is False


# ============================================================================
# HandlerLifecycleUpdate -- Properties
# ============================================================================


class TestHandlerLifecycleUpdateProperties:
    """Tests for HandlerLifecycleUpdate handler classification properties."""

    def test_handler_id(self) -> None:
        """handler_id returns the expected identifier."""
        handler = HandlerLifecycleUpdate()
        assert handler.handler_id == "handler-lifecycle-update"

    def test_handler_type(self) -> None:
        """handler_type is INFRA_HANDLER."""
        handler = HandlerLifecycleUpdate()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        """handler_category is EFFECT."""
        handler = HandlerLifecycleUpdate()
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# ============================================================================
# HandlerLifecycleUpdate -- handle()
# ============================================================================


class TestHandlerLifecycleUpdateHandle:
    """Tests for HandlerLifecycleUpdate.handle() async method."""

    @pytest.mark.asyncio
    async def test_handle_pass_verdict(self) -> None:
        """Applying PASS verdict returns a result with correct tier."""
        handler = HandlerLifecycleUpdate()
        state = _make_state(tier=EnumLifecycleTier.OBSERVED)

        result = await handler.handle(
            pattern_id=uuid4(),
            verdict=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
            current_state=state,
        )

        assert isinstance(result, ModelLifecycleResult)
        assert result.previous_tier == EnumLifecycleTier.OBSERVED
        assert result.new_tier == EnumLifecycleTier.OBSERVED  # 1 pass, no promotion
        assert result.tier_changed is False
        assert result.verdict_applied == EnumValidationVerdict.PASS
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_handle_fail_verdict(self) -> None:
        """Applying FAIL verdict returns a result with correct tier."""
        handler = HandlerLifecycleUpdate()
        state = _make_state(tier=EnumLifecycleTier.OBSERVED)

        result = await handler.handle(
            pattern_id=uuid4(),
            verdict=EnumValidationVerdict.FAIL,
            correlation_id=uuid4(),
            current_state=state,
        )

        assert result.previous_tier == EnumLifecycleTier.OBSERVED
        assert result.new_tier == EnumLifecycleTier.OBSERVED  # 1 fail, no demotion
        assert result.tier_changed is False
        assert result.verdict_applied == EnumValidationVerdict.FAIL

    @pytest.mark.asyncio
    async def test_handle_none_current_state(self) -> None:
        """None current_state creates a default state at OBSERVED tier."""
        handler = HandlerLifecycleUpdate()
        pid = uuid4()

        result = await handler.handle(
            pattern_id=pid,
            verdict=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
            current_state=None,
        )

        assert result.previous_tier == EnumLifecycleTier.OBSERVED
        assert result.pattern_id == pid
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_handle_promotion_tier_changed(self) -> None:
        """Tier change is reflected correctly in the result."""
        handler = HandlerLifecycleUpdate()
        # State with 1 consecutive pass -- next pass triggers promotion
        state = _make_state(
            tier=EnumLifecycleTier.OBSERVED,
            pass_count=1,
        )

        result = await handler.handle(
            pattern_id=uuid4(),
            verdict=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
            current_state=state,
        )

        assert result.previous_tier == EnumLifecycleTier.OBSERVED
        assert result.new_tier == EnumLifecycleTier.SUGGESTED
        assert result.tier_changed is True

    @pytest.mark.asyncio
    async def test_handle_includes_correlation_id(self) -> None:
        """Result includes the correlation_id from the call."""
        handler = HandlerLifecycleUpdate()
        cid = uuid4()

        result = await handler.handle(
            pattern_id=uuid4(),
            verdict=EnumValidationVerdict.PASS,
            correlation_id=cid,
            current_state=None,
        )

        assert result.correlation_id == cid


# ============================================================================
# ModelLifecycleResult -- __bool__
# ============================================================================


class TestModelLifecycleResultBool:
    """Tests for ModelLifecycleResult __bool__ non-standard behaviour."""

    def test_bool_true_when_no_error(self) -> None:
        """__bool__ returns True when error field is empty."""
        result = ModelLifecycleResult(
            pattern_id=uuid4(),
            previous_tier=EnumLifecycleTier.OBSERVED,
            new_tier=EnumLifecycleTier.OBSERVED,
            tier_changed=False,
            verdict_applied=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
            error="",
        )
        assert bool(result) is True

    def test_bool_false_when_error_set(self) -> None:
        """__bool__ returns False when error field is non-empty."""
        result = ModelLifecycleResult(
            pattern_id=uuid4(),
            previous_tier=EnumLifecycleTier.OBSERVED,
            new_tier=EnumLifecycleTier.OBSERVED,
            tier_changed=False,
            verdict_applied=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
            error="Something went wrong",
            error_code="LIFECYCLE_UPDATE_ERROR",
        )
        assert bool(result) is False

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelLifecycleResult(
                pattern_id=uuid4(),
                previous_tier=EnumLifecycleTier.OBSERVED,
                new_tier=EnumLifecycleTier.OBSERVED,
                tier_changed=False,
                verdict_applied=EnumValidationVerdict.PASS,
                correlation_id=uuid4(),
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        result = ModelLifecycleResult(
            pattern_id=uuid4(),
            previous_tier=EnumLifecycleTier.OBSERVED,
            new_tier=EnumLifecycleTier.OBSERVED,
            tier_changed=False,
            verdict_applied=EnumValidationVerdict.PASS,
            correlation_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            result.error = "mutated"  # type: ignore[misc]
