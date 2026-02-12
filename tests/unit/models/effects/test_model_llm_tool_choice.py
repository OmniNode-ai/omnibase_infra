# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ModelLlmToolChoice.

This test suite validates:
- Basic model construction for all four modes (auto, none, required, function)
- model_validator(mode="after") enforcement of function_name consistency
- Pydantic field-level validation (Literal, min_length)
- Frozen immutability and extra="forbid"
- Serialization roundtrip and identity semantics

Test Organization:
    - TestModelLlmToolChoiceConstruction: Valid construction paths (4 tests)
    - TestModelLlmToolChoiceValidatorLogic: model_validator enforcement (8 tests)
    - TestModelLlmToolChoiceImmutability: Frozen / extra="forbid" (2 tests)
    - TestModelLlmToolChoiceSerialization: Dump/roundtrip, hash, equality (4 tests)

Coverage Goals:
    - 100% code coverage for model_llm_tool_choice.py
    - All four modes tested for valid construction
    - All validator error paths exercised
    - All ConfigDict constraints verified
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)

# ==============================================================================
# Construction
# ==============================================================================


class TestModelLlmToolChoiceConstruction:
    """Test valid construction for each mode."""

    def test_mode_auto_valid(self) -> None:
        """mode='auto' with function_name=None constructs successfully."""
        choice = ModelLlmToolChoice(mode="auto")

        assert choice.mode == "auto"
        assert choice.function_name is None

    def test_mode_none_valid(self) -> None:
        """mode='none' with function_name=None constructs successfully."""
        choice = ModelLlmToolChoice(mode="none")

        assert choice.mode == "none"
        assert choice.function_name is None

    def test_mode_required_valid(self) -> None:
        """mode='required' with function_name=None constructs successfully."""
        choice = ModelLlmToolChoice(mode="required")

        assert choice.mode == "required"
        assert choice.function_name is None

    def test_mode_function_with_name_valid(self) -> None:
        """mode='function' with a non-empty function_name constructs successfully."""
        choice = ModelLlmToolChoice(mode="function", function_name="get_weather")

        assert choice.mode == "function"
        assert choice.function_name == "get_weather"


# ==============================================================================
# ValidatorLogic
# ==============================================================================


class TestModelLlmToolChoiceValidatorLogic:
    """Test model_validator enforcement and field-level validation."""

    def test_mode_is_required(self) -> None:
        """Omitting mode raises ValidationError (required field)."""
        with pytest.raises(ValidationError):
            ModelLlmToolChoice()  # type: ignore[call-arg]

    def test_mode_invalid_value_rejected(self) -> None:
        """mode='invalid' is rejected by the Literal constraint."""
        with pytest.raises(ValidationError):
            ModelLlmToolChoice(mode="invalid")  # type: ignore[arg-type]

    def test_mode_function_without_name_raises(self) -> None:
        """mode='function' without function_name raises ValueError via model_validator."""
        with pytest.raises(ValidationError, match="function_name is required"):
            ModelLlmToolChoice(mode="function")

    def test_mode_auto_with_function_name_raises(self) -> None:
        """mode='auto' with function_name raises ValueError via model_validator."""
        with pytest.raises(ValidationError, match="function_name must be None"):
            ModelLlmToolChoice(mode="auto", function_name="fn")

    def test_mode_none_with_function_name_raises(self) -> None:
        """mode='none' with function_name raises ValueError via model_validator."""
        with pytest.raises(ValidationError, match="function_name must be None"):
            ModelLlmToolChoice(mode="none", function_name="fn")

    def test_mode_required_with_function_name_raises(self) -> None:
        """mode='required' with function_name raises ValueError via model_validator."""
        with pytest.raises(ValidationError, match="function_name must be None"):
            ModelLlmToolChoice(mode="required", function_name="fn")

    def test_function_name_min_length_rejects_empty(self) -> None:
        """mode='function' with empty string function_name is rejected by min_length=1."""
        with pytest.raises(ValidationError):
            ModelLlmToolChoice(mode="function", function_name="")

    def test_function_name_single_char_valid(self) -> None:
        """mode='function' with a single-character function_name passes min_length=1."""
        choice = ModelLlmToolChoice(mode="function", function_name="f")

        assert choice.function_name == "f"


# ==============================================================================
# Immutability
# ==============================================================================


class TestModelLlmToolChoiceImmutability:
    """Test frozen model and extra field constraints."""

    def test_frozen_immutability(self) -> None:
        """Assigning to mode or function_name raises on a frozen model."""
        choice = ModelLlmToolChoice(mode="auto")

        with pytest.raises(ValidationError):
            choice.mode = "none"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            choice.function_name = "fn"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Passing an unexpected field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelLlmToolChoice(mode="auto", unexpected_field="value")  # type: ignore[call-arg]


# ==============================================================================
# Serialization
# ==============================================================================


class TestModelLlmToolChoiceSerialization:
    """Test dump/validate roundtrip, hashability, and equality."""

    def test_model_dump_roundtrip_auto(self) -> None:
        """model_dump -> model_validate roundtrip preserves auto mode."""
        original = ModelLlmToolChoice(mode="auto")
        data = original.model_dump()
        restored = ModelLlmToolChoice.model_validate(data)

        assert restored.mode == original.mode
        assert restored.function_name == original.function_name
        assert restored == original

    def test_model_dump_roundtrip_function(self) -> None:
        """model_dump -> model_validate roundtrip preserves function mode."""
        original = ModelLlmToolChoice(mode="function", function_name="get_weather")
        data = original.model_dump()
        restored = ModelLlmToolChoice.model_validate(data)

        assert restored.mode == original.mode
        assert restored.function_name == original.function_name
        assert restored == original

    def test_hashable(self) -> None:
        """Frozen model instances are hashable and can be used in sets."""
        choice_a = ModelLlmToolChoice(mode="auto")
        choice_b = ModelLlmToolChoice(mode="function", function_name="fn")

        # Should not raise
        result = {choice_a, choice_b}
        assert len(result) == 2

    def test_equality(self) -> None:
        """Two instances with the same mode and function_name are equal."""
        choice_a = ModelLlmToolChoice(mode="function", function_name="get_weather")
        choice_b = ModelLlmToolChoice(mode="function", function_name="get_weather")

        assert choice_a == choice_b
        assert hash(choice_a) == hash(choice_b)
