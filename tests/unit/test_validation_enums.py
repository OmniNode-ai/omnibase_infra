# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for validation pipeline enums.

Tests the four new enums introduced in OMN-2147:
- EnumCheckSeverity
- EnumValidationVerdict
- EnumLifecycleTier
- EnumAdjudicatorState
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import (
    EnumAdjudicatorState,
    EnumCheckSeverity,
    EnumLifecycleTier,
    EnumValidationVerdict,
)

# ============================================================================
# EnumCheckSeverity
# ============================================================================


class TestEnumCheckSeverity:
    """Tests for EnumCheckSeverity enum."""

    def test_required_value(self) -> None:
        """REQUIRED member has string value 'required'."""
        assert EnumCheckSeverity.REQUIRED.value == "required"

    def test_recommended_value(self) -> None:
        """RECOMMENDED member has string value 'recommended'."""
        assert EnumCheckSeverity.RECOMMENDED.value == "recommended"

    def test_informational_value(self) -> None:
        """INFORMATIONAL member has string value 'informational'."""
        assert EnumCheckSeverity.INFORMATIONAL.value == "informational"

    def test_blocks_verdict_required(self) -> None:
        """REQUIRED severity blocks the verdict on failure."""
        assert EnumCheckSeverity.REQUIRED.blocks_verdict() is True

    def test_blocks_verdict_recommended(self) -> None:
        """RECOMMENDED severity does not block the verdict."""
        assert EnumCheckSeverity.RECOMMENDED.blocks_verdict() is False

    def test_blocks_verdict_informational(self) -> None:
        """INFORMATIONAL severity does not block the verdict."""
        assert EnumCheckSeverity.INFORMATIONAL.blocks_verdict() is False

    def test_str_serialization(self) -> None:
        """__str__ returns the raw value for serialization."""
        assert str(EnumCheckSeverity.REQUIRED) == "required"
        assert str(EnumCheckSeverity.RECOMMENDED) == "recommended"
        assert str(EnumCheckSeverity.INFORMATIONAL) == "informational"

    def test_member_count(self) -> None:
        """Enum has exactly 3 members."""
        assert len(EnumCheckSeverity) == 3

    def test_is_str_enum(self) -> None:
        """Enum members are strings."""
        assert isinstance(EnumCheckSeverity.REQUIRED, str)


# ============================================================================
# EnumValidationVerdict
# ============================================================================


class TestEnumValidationVerdict:
    """Tests for EnumValidationVerdict enum."""

    def test_pass_value(self) -> None:
        """PASS member has string value 'pass'."""
        assert EnumValidationVerdict.PASS.value == "pass"

    def test_fail_value(self) -> None:
        """FAIL member has string value 'fail'."""
        assert EnumValidationVerdict.FAIL.value == "fail"

    def test_quarantine_value(self) -> None:
        """QUARANTINE member has string value 'quarantine'."""
        assert EnumValidationVerdict.QUARANTINE.value == "quarantine"

    def test_str_serialization(self) -> None:
        """__str__ returns the raw value for serialization."""
        assert str(EnumValidationVerdict.PASS) == "pass"
        assert str(EnumValidationVerdict.FAIL) == "fail"
        assert str(EnumValidationVerdict.QUARANTINE) == "quarantine"

    def test_member_count(self) -> None:
        """Enum has exactly 3 members."""
        assert len(EnumValidationVerdict) == 3

    def test_is_str_enum(self) -> None:
        """Enum members are strings."""
        assert isinstance(EnumValidationVerdict.PASS, str)


# ============================================================================
# EnumLifecycleTier
# ============================================================================


class TestEnumLifecycleTier:
    """Tests for EnumLifecycleTier enum."""

    def test_all_values(self) -> None:
        """All tier members have expected string values."""
        assert EnumLifecycleTier.OBSERVED.value == "observed"
        assert EnumLifecycleTier.SUGGESTED.value == "suggested"
        assert EnumLifecycleTier.SHADOW_APPLY.value == "shadow_apply"
        assert EnumLifecycleTier.PROMOTED.value == "promoted"
        assert EnumLifecycleTier.DEFAULT.value == "default"
        assert EnumLifecycleTier.SUPPRESSED.value == "suppressed"

    def test_member_count(self) -> None:
        """Enum has exactly 6 members."""
        assert len(EnumLifecycleTier) == 6

    # -- can_promote() ---

    def test_can_promote_observed(self) -> None:
        """OBSERVED tier can be promoted."""
        assert EnumLifecycleTier.OBSERVED.can_promote() is True

    def test_can_promote_suggested(self) -> None:
        """SUGGESTED tier can be promoted."""
        assert EnumLifecycleTier.SUGGESTED.can_promote() is True

    def test_can_promote_shadow_apply(self) -> None:
        """SHADOW_APPLY tier can be promoted."""
        assert EnumLifecycleTier.SHADOW_APPLY.can_promote() is True

    def test_can_promote_promoted(self) -> None:
        """PROMOTED tier can be promoted to DEFAULT."""
        assert EnumLifecycleTier.PROMOTED.can_promote() is True

    def test_can_promote_default(self) -> None:
        """DEFAULT tier cannot be promoted further."""
        assert EnumLifecycleTier.DEFAULT.can_promote() is False

    def test_can_promote_suppressed(self) -> None:
        """SUPPRESSED tier cannot be promoted."""
        assert EnumLifecycleTier.SUPPRESSED.can_promote() is False

    # -- promoted() ---

    def test_promoted_observed(self) -> None:
        """OBSERVED promotes to SUGGESTED."""
        assert EnumLifecycleTier.OBSERVED.promoted() == EnumLifecycleTier.SUGGESTED

    def test_promoted_suggested(self) -> None:
        """SUGGESTED promotes to SHADOW_APPLY."""
        assert EnumLifecycleTier.SUGGESTED.promoted() == EnumLifecycleTier.SHADOW_APPLY

    def test_promoted_shadow_apply(self) -> None:
        """SHADOW_APPLY promotes to PROMOTED."""
        assert EnumLifecycleTier.SHADOW_APPLY.promoted() == EnumLifecycleTier.PROMOTED

    def test_promoted_promoted(self) -> None:
        """PROMOTED promotes to DEFAULT."""
        assert EnumLifecycleTier.PROMOTED.promoted() == EnumLifecycleTier.DEFAULT

    def test_promoted_default_raises(self) -> None:
        """DEFAULT cannot be promoted -- raises ValueError."""
        with pytest.raises(ValueError, match="Cannot promote from tier default"):
            EnumLifecycleTier.DEFAULT.promoted()

    def test_promoted_suppressed_raises(self) -> None:
        """SUPPRESSED cannot be promoted -- raises ValueError."""
        with pytest.raises(ValueError, match="Cannot promote from tier suppressed"):
            EnumLifecycleTier.SUPPRESSED.promoted()

    # -- demoted() ---

    def test_demoted_default(self) -> None:
        """DEFAULT demotes to PROMOTED."""
        assert EnumLifecycleTier.DEFAULT.demoted() == EnumLifecycleTier.PROMOTED

    def test_demoted_promoted(self) -> None:
        """PROMOTED demotes to SHADOW_APPLY."""
        assert EnumLifecycleTier.PROMOTED.demoted() == EnumLifecycleTier.SHADOW_APPLY

    def test_demoted_shadow_apply(self) -> None:
        """SHADOW_APPLY demotes to SUGGESTED."""
        assert EnumLifecycleTier.SHADOW_APPLY.demoted() == EnumLifecycleTier.SUGGESTED

    def test_demoted_suggested(self) -> None:
        """SUGGESTED demotes to OBSERVED."""
        assert EnumLifecycleTier.SUGGESTED.demoted() == EnumLifecycleTier.OBSERVED

    def test_demoted_observed_stays(self) -> None:
        """OBSERVED demotes to OBSERVED (floor)."""
        assert EnumLifecycleTier.OBSERVED.demoted() == EnumLifecycleTier.OBSERVED

    def test_demoted_suppressed_raises(self) -> None:
        """SUPPRESSED cannot be demoted -- raises ValueError."""
        with pytest.raises(ValueError, match="Cannot demote from tier suppressed"):
            EnumLifecycleTier.SUPPRESSED.demoted()

    def test_str_serialization(self) -> None:
        """__str__ returns the raw value for serialization."""
        assert str(EnumLifecycleTier.OBSERVED) == "observed"
        assert str(EnumLifecycleTier.DEFAULT) == "default"
        assert str(EnumLifecycleTier.SUPPRESSED) == "suppressed"


# ============================================================================
# EnumAdjudicatorState
# ============================================================================


class TestEnumAdjudicatorState:
    """Tests for EnumAdjudicatorState enum."""

    def test_collecting_value(self) -> None:
        """COLLECTING member has string value 'collecting'."""
        assert EnumAdjudicatorState.COLLECTING.value == "collecting"

    def test_adjudicating_value(self) -> None:
        """ADJUDICATING member has string value 'adjudicating'."""
        assert EnumAdjudicatorState.ADJUDICATING.value == "adjudicating"

    def test_verdict_emitted_value(self) -> None:
        """VERDICT_EMITTED member has string value 'verdict_emitted'."""
        assert EnumAdjudicatorState.VERDICT_EMITTED.value == "verdict_emitted"

    def test_str_serialization(self) -> None:
        """__str__ returns the raw value for serialization."""
        assert str(EnumAdjudicatorState.COLLECTING) == "collecting"
        assert str(EnumAdjudicatorState.ADJUDICATING) == "adjudicating"
        assert str(EnumAdjudicatorState.VERDICT_EMITTED) == "verdict_emitted"

    def test_member_count(self) -> None:
        """Enum has exactly 3 members."""
        assert len(EnumAdjudicatorState) == 3

    def test_is_str_enum(self) -> None:
        """Enum members are strings."""
        assert isinstance(EnumAdjudicatorState.COLLECTING, str)
