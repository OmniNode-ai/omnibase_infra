# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive tests for ModelLlmUsage.

Tests cover:
- Default construction with all-zero tokens and cost_usd=None
- Auto-computation of tokens_total from tokens_input + tokens_output
- Consistency validation when tokens_total is explicitly provided
- Ambiguous sentinel fix: tokens_total=None means auto-compute, explicit 0 is
  validated against tokens_input + tokens_output
- Field-level ge=0 constraints on all token fields and cost_usd
- cost_usd optional float validation
- Immutability (frozen=True) and extra field rejection (extra="forbid")
- Serialization round-trips (model_dump, model_dump_json, model_validate)
- from_attributes config, hashability, and equality
- model_validator passthrough for non-dict inputs

OMN-2103: Phase 3 shared LLM models - ModelLlmUsage
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage

# ============================================================================
# Construction Tests
# ============================================================================


class TestConstruction:
    """Tests for ModelLlmUsage default and explicit construction."""

    def test_default_construction(self) -> None:
        """ModelLlmUsage() produces all-zero tokens and cost_usd=None."""
        usage = ModelLlmUsage()

        assert usage.tokens_input == 0
        assert usage.tokens_output == 0
        assert usage.tokens_total == 0
        assert usage.cost_usd is None

    def test_auto_compute_total(self) -> None:
        """tokens_total is auto-computed when not explicitly provided."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50)

        assert usage.tokens_total == 150

    def test_auto_compute_when_total_none(self) -> None:
        """Explicit tokens_total=None triggers auto-computation."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50, tokens_total=None)

        assert usage.tokens_total == 150

    def test_explicit_total_zero_with_nonzero_components_raises(self) -> None:
        """Explicit tokens_total=0 with non-zero input+output raises ValueError.

        This verifies the sentinel-ambiguity fix: 0 is no longer conflated
        with 'not provided'.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(tokens_input=100, tokens_output=50, tokens_total=0)

        error_text = str(exc_info.value)
        assert "0" in error_text
        assert "150" in error_text

    def test_explicit_total_zero_with_zero_components_valid(self) -> None:
        """Explicit tokens_total=0 with input=0 and output=0 is valid."""
        usage = ModelLlmUsage(tokens_input=0, tokens_output=0, tokens_total=0)

        assert usage.tokens_total == 0

    def test_explicit_total_matching_sum(self) -> None:
        """Explicit tokens_total matching the sum is accepted."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50, tokens_total=150)

        assert usage.tokens_total == 150

    def test_explicit_total_mismatch_raises(self) -> None:
        """Explicit tokens_total != sum raises ValueError."""
        with pytest.raises(ValidationError):
            ModelLlmUsage(tokens_input=100, tokens_output=50, tokens_total=200)

    def test_explicit_total_mismatch_error_message(self) -> None:
        """Error message contains the mismatched values."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(tokens_input=100, tokens_output=50, tokens_total=200)

        error_text = str(exc_info.value)
        assert "200" in error_text
        assert "150" in error_text

    def test_only_input_tokens(self) -> None:
        """Only tokens_input provided results in total = input."""
        usage = ModelLlmUsage(tokens_input=100)

        assert usage.tokens_output == 0
        assert usage.tokens_total == 100

    def test_only_output_tokens(self) -> None:
        """Only tokens_output provided results in total = output."""
        usage = ModelLlmUsage(tokens_output=50)

        assert usage.tokens_input == 0
        assert usage.tokens_total == 50

    def test_zero_input_zero_output(self) -> None:
        """Both zero means total remains zero."""
        usage = ModelLlmUsage(tokens_input=0, tokens_output=0)

        assert usage.tokens_total == 0


# ============================================================================
# Tokens Validation Tests
# ============================================================================


class TestTokensValidation:
    """Tests for ge=0 constraints on token fields."""

    def test_negative_input_rejected(self) -> None:
        """tokens_input=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(tokens_input=-1)

        assert "tokens_input" in str(exc_info.value).lower()

    def test_negative_output_rejected(self) -> None:
        """tokens_output=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(tokens_output=-1)

        assert "tokens_output" in str(exc_info.value).lower()

    def test_negative_total_rejected(self) -> None:
        """tokens_total=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(tokens_total=-1)

        assert "tokens_total" in str(exc_info.value).lower()


# ============================================================================
# Cost Validation Tests
# ============================================================================


class TestCostValidation:
    """Tests for cost_usd field validation."""

    def test_cost_usd_none_default(self) -> None:
        """cost_usd defaults to None."""
        usage = ModelLlmUsage()

        assert usage.cost_usd is None

    def test_cost_usd_accepts_float(self) -> None:
        """cost_usd=0.05 is accepted."""
        usage = ModelLlmUsage(cost_usd=0.05)

        assert usage.cost_usd == pytest.approx(0.05)

    def test_cost_usd_zero_accepted(self) -> None:
        """cost_usd=0.0 is accepted by ge=0.0 constraint."""
        usage = ModelLlmUsage(cost_usd=0.0)

        assert usage.cost_usd == 0.0

    def test_cost_usd_negative_rejected(self) -> None:
        """cost_usd=-0.01 is rejected by ge=0.0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(cost_usd=-0.01)

        assert "cost_usd" in str(exc_info.value).lower()

    def test_cost_usd_large_value(self) -> None:
        """cost_usd=999.99 is accepted."""
        usage = ModelLlmUsage(cost_usd=999.99)

        assert usage.cost_usd == pytest.approx(999.99)


# ============================================================================
# Immutability Tests
# ============================================================================


class TestImmutability:
    """Tests for frozen model immutability and extra field rejection."""

    def test_frozen_immutability(self) -> None:
        """Assigning to any field on a frozen model raises ValidationError."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50)

        with pytest.raises(ValidationError):
            usage.tokens_input = 999  # type: ignore[misc]

        with pytest.raises(ValidationError):
            usage.tokens_output = 999  # type: ignore[misc]

        with pytest.raises(ValidationError):
            usage.tokens_total = 999  # type: ignore[misc]

        with pytest.raises(ValidationError):
            usage.cost_usd = 1.0  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmUsage(
                tokens_input=10,
                extra_field="unexpected",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value).lower()


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for model serialization and deserialization round-trips."""

    def test_model_dump_roundtrip(self) -> None:
        """model_dump -> model_validate preserves auto-computed total."""
        original = ModelLlmUsage(tokens_input=100, tokens_output=50)

        data = original.model_dump()
        restored = ModelLlmUsage.model_validate(data)

        assert restored.tokens_input == 100
        assert restored.tokens_output == 50
        assert restored.tokens_total == 150
        assert restored == original

    def test_model_dump_json(self) -> None:
        """model_dump_json produces valid JSON with expected fields."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50, cost_usd=0.03)

        json_str = usage.model_dump_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["tokens_input"] == 100
        assert parsed["tokens_output"] == 50
        assert parsed["tokens_total"] == 150
        assert parsed["cost_usd"] == pytest.approx(0.03)

    def test_from_attributes_config(self) -> None:
        """from_attributes=True allows creation from objects with attributes."""

        class UsageData:
            """Simple class with matching attributes."""

            def __init__(self) -> None:
                self.tokens_input = 200
                self.tokens_output = 80
                self.tokens_total = 280
                self.cost_usd = None

        source = UsageData()
        usage = ModelLlmUsage.model_validate(source)

        assert usage.tokens_input == 200
        assert usage.tokens_output == 80
        assert usage.tokens_total == 280
        assert usage.cost_usd is None

    def test_hashable(self) -> None:
        """Frozen model instances are hashable and can be used in sets."""
        usage_a = ModelLlmUsage(tokens_input=100, tokens_output=50)
        usage_b = ModelLlmUsage(tokens_input=100, tokens_output=50)

        assert hash(usage_a) == hash(usage_b)

        usage_set = {usage_a, usage_b}
        assert len(usage_set) == 1

        usage_dict = {usage_a: "value"}
        assert usage_dict[usage_b] == "value"

    def test_equality(self) -> None:
        """Instances with the same values are equal."""
        usage_a = ModelLlmUsage(tokens_input=100, tokens_output=50)
        usage_b = ModelLlmUsage(tokens_input=100, tokens_output=50)

        assert usage_a == usage_b

        usage_c = ModelLlmUsage(tokens_input=200, tokens_output=50)
        assert usage_a != usage_c

    def test_validator_non_dict_passthrough(self) -> None:
        """model_validate from a non-dict object passes through the validator."""

        class UsageAttrs:
            """Object with matching attributes for from_attributes validation."""

            def __init__(self) -> None:
                self.tokens_input = 10
                self.tokens_output = 20
                self.tokens_total = 30
                self.cost_usd = None

        source = UsageAttrs()
        usage = ModelLlmUsage.model_validate(source)

        assert usage.tokens_input == 10
        assert usage.tokens_output == 20
        assert usage.tokens_total == 30
