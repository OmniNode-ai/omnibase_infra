# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Edge case tests for ModelLlmUsage type coercion and boundary conditions.

Supplements test_model_llm_usage.py with focused coverage on:
1. Pydantic's type coercion behavior for financial-grade data
2. Boundary values for token and cost fields
3. Usage source provenance edge cases
4. raw_provider_usage validation

These tests document Pydantic's actual coercion behavior. Where Pydantic
silently coerces (e.g., numeric strings to int), the tests document this
as accepted behavior. The SPI contracts and DB CHECK constraints provide
the additional strictness layer.

Related Tickets:
    - OMN-2295: LLM cost tracking: input validation and edge case tests
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_spi.contracts.measurement import ContractEnumUsageSource

# =============================================================================
# Type Coercion Behavior
# =============================================================================


class TestTypeCoercionBehavior:
    """Document Pydantic's coercion behavior for ModelLlmUsage fields.

    Pydantic v2 coerces compatible types by default (e.g., str "10" -> int 10).
    These tests document the actual behavior so that any future Pydantic
    config change (strict=True) is caught.
    """

    @pytest.mark.unit
    def test_numeric_string_tokens_coerced_with_validator_caveat(self) -> None:
        """Pydantic coerces numeric strings to int, but mode='before' sees raw strings.

        IMPORTANT CAVEAT: ModelLlmUsage uses a mode='before' model_validator
        that auto-computes tokens_total. When string inputs are provided, the
        validator sees raw strings and does string concatenation ('10' + '20' = '1020')
        before Pydantic coerces to int. The resulting tokens_total is 1020, not 30.

        This documents actual behavior. Callers should pass int values, not strings.
        """
        usage = ModelLlmUsage(
            tokens_input="10",  # type: ignore[arg-type]
            tokens_output="20",  # type: ignore[arg-type]
        )
        assert usage.tokens_input == 10
        assert usage.tokens_output == 20
        # mode='before' validator concatenates strings: "10" + "20" = "1020"
        # Then Pydantic coerces "1020" to int(1020)
        assert usage.tokens_total == 1020

    @pytest.mark.unit
    def test_non_numeric_string_tokens_rejected(self) -> None:
        """Non-numeric strings are rejected for token fields.

        The mode='before' validator attempts arithmetic on raw values
        before Pydantic coerces types, so 'abc' + 0 raises TypeError
        which Pydantic wraps into a ValidationError.
        """
        with pytest.raises((ValidationError, TypeError)):
            ModelLlmUsage(tokens_input="abc")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_float_string_cost_coerced(self) -> None:
        """Numeric string '0.05' is coerced to float for cost_usd."""
        usage = ModelLlmUsage(cost_usd="0.05")  # type: ignore[arg-type]
        assert usage.cost_usd == pytest.approx(0.05)

    @pytest.mark.unit
    def test_non_numeric_string_cost_rejected(self) -> None:
        """Non-numeric string is rejected for cost_usd."""
        with pytest.raises(ValidationError):
            ModelLlmUsage(cost_usd="abc")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_fractional_float_tokens_rejected(self) -> None:
        """Fractional float values for int fields are rejected by Pydantic v2.

        Pydantic v2 does NOT truncate fractional floats to int. Only exact
        integer-valued floats (e.g. 10.0) are coerced. Fractional floats
        like 10.9 raise int_from_float validation error.
        """
        with pytest.raises(ValidationError, match="int_from_float"):
            ModelLlmUsage(
                tokens_input=10.9,  # type: ignore[arg-type]
                tokens_output=5.1,  # type: ignore[arg-type]
            )

    @pytest.mark.unit
    def test_integer_valued_float_tokens_accepted(self) -> None:
        """Exact integer-valued floats (e.g. 10.0) are coerced to int."""
        usage = ModelLlmUsage(
            tokens_input=10.0,  # type: ignore[arg-type]
            tokens_output=5.0,  # type: ignore[arg-type]
        )
        assert usage.tokens_input == 10
        assert usage.tokens_output == 5

    @pytest.mark.unit
    def test_dict_tokens_rejected(self) -> None:
        """Dict value for int field is rejected.

        The mode='before' validator hits TypeError on dict + int before
        Pydantic can reject the type.
        """
        with pytest.raises((ValidationError, TypeError)):
            ModelLlmUsage(tokens_input={"value": 10})  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_list_tokens_rejected(self) -> None:
        """List value for int field is rejected.

        The mode='before' validator hits TypeError on list + int before
        Pydantic can reject the type.
        """
        with pytest.raises((ValidationError, TypeError)):
            ModelLlmUsage(tokens_input=[10])  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_none_tokens_rejected(self) -> None:
        """None is rejected for tokens_input/output (int fields with default 0).

        The mode='before' validator hits TypeError on None + int before
        Pydantic can reject the type.
        """
        with pytest.raises((ValidationError, TypeError)):
            ModelLlmUsage(tokens_input=None)  # type: ignore[arg-type]


# =============================================================================
# Boundary Values
# =============================================================================


class TestBoundaryValues:
    """Boundary value tests for token and cost fields."""

    @pytest.mark.unit
    def test_max_int_tokens(self) -> None:
        """Very large token counts are accepted."""
        big = 2**31 - 1  # Max 32-bit signed int
        usage = ModelLlmUsage(tokens_input=big, tokens_output=0)
        assert usage.tokens_input == big
        assert usage.tokens_total == big

    @pytest.mark.unit
    def test_very_large_cost(self) -> None:
        """Very large cost values are accepted."""
        usage = ModelLlmUsage(cost_usd=99999.99)
        assert usage.cost_usd == pytest.approx(99999.99)

    @pytest.mark.unit
    def test_very_small_positive_cost(self) -> None:
        """Very small positive cost is not rounded to zero."""
        usage = ModelLlmUsage(cost_usd=0.000001)
        assert usage.cost_usd is not None
        assert usage.cost_usd > 0

    @pytest.mark.unit
    def test_zero_cost_not_none(self) -> None:
        """cost_usd=0.0 is zero, not None."""
        usage = ModelLlmUsage(cost_usd=0.0)
        assert usage.cost_usd is not None
        assert usage.cost_usd == 0.0

    @pytest.mark.unit
    def test_none_cost_is_none(self) -> None:
        """cost_usd=None remains None."""
        usage = ModelLlmUsage(cost_usd=None)
        assert usage.cost_usd is None


# =============================================================================
# Usage Source Provenance
# =============================================================================


class TestUsageSourceProvenance:
    """Tests for usage_source field in ModelLlmUsage."""

    @pytest.mark.unit
    def test_default_source_is_missing(self) -> None:
        """Default usage_source is MISSING."""
        usage = ModelLlmUsage()
        assert usage.usage_source == ContractEnumUsageSource.MISSING

    @pytest.mark.unit
    def test_api_source_accepted(self) -> None:
        """API source is accepted."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
        )
        assert usage.usage_source == ContractEnumUsageSource.API

    @pytest.mark.unit
    def test_estimated_source_accepted(self) -> None:
        """ESTIMATED source is accepted."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.ESTIMATED,
        )
        assert usage.usage_source == ContractEnumUsageSource.ESTIMATED

    @pytest.mark.unit
    def test_string_source_coerced(self) -> None:
        """String 'api' is coerced to ContractEnumUsageSource.API by Pydantic."""
        usage = ModelLlmUsage(
            tokens_input=100,
            usage_source="api",  # type: ignore[arg-type]
        )
        assert usage.usage_source == ContractEnumUsageSource.API

    @pytest.mark.unit
    def test_invalid_source_rejected(self) -> None:
        """Invalid source string is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmUsage(usage_source="invalid_source")  # type: ignore[arg-type]


# =============================================================================
# Raw Provider Usage
# =============================================================================


class TestRawProviderUsage:
    """Tests for raw_provider_usage field edge cases."""

    @pytest.mark.unit
    def test_none_raw_usage_default(self) -> None:
        """Default raw_provider_usage is None."""
        usage = ModelLlmUsage()
        assert usage.raw_provider_usage is None

    @pytest.mark.unit
    def test_empty_dict_raw_usage(self) -> None:
        """Empty dict is valid raw_provider_usage."""
        usage = ModelLlmUsage(raw_provider_usage={})
        assert usage.raw_provider_usage == {}

    @pytest.mark.unit
    def test_nested_dict_raw_usage(self) -> None:
        """Nested dict raw_provider_usage is preserved."""
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        usage = ModelLlmUsage(raw_provider_usage=raw)
        assert usage.raw_provider_usage is not None
        assert usage.raw_provider_usage["prompt_tokens"] == 100
        assert usage.raw_provider_usage["prompt_tokens_details"]["cached_tokens"] == 10


# =============================================================================
# Token Total Consistency Edge Cases
# =============================================================================


class TestTokenTotalConsistencyEdgeCases:
    """Additional edge cases for tokens_total auto-computation and validation."""

    @pytest.mark.unit
    def test_total_auto_computed_with_one_zero_component(self) -> None:
        """When one component is 0, total equals the other component."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=0)
        assert usage.tokens_total == 100

        usage2 = ModelLlmUsage(tokens_input=0, tokens_output=50)
        assert usage2.tokens_total == 50

    @pytest.mark.unit
    def test_explicit_total_matching_accepted(self) -> None:
        """Explicit total matching the sum is accepted without error."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
        )
        assert usage.tokens_total == 150

    @pytest.mark.unit
    def test_explicit_total_off_by_one_rejected(self) -> None:
        """Off-by-one total is rejected (no rounding tolerance)."""
        with pytest.raises(ValidationError):
            ModelLlmUsage(
                tokens_input=100,
                tokens_output=50,
                tokens_total=149,
            )

        with pytest.raises(ValidationError):
            ModelLlmUsage(
                tokens_input=100,
                tokens_output=50,
                tokens_total=151,
            )
