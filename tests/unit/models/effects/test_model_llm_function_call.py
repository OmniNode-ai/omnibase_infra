# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive tests for ModelLlmFunctionCall.

Tests cover:
- Valid construction with required fields
- Field validation (name min_length, required fields)
- Immutability (frozen=True)
- Serialization (model_dump, model_dump_json, model_validate roundtrip)
- from_attributes config
- Hashability and equality

OMN-2103: Phase 3 shared LLM models - ModelLlmFunctionCall
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)

# ============================================================================
# Construction Tests
# ============================================================================


class TestConstruction:
    """Tests for ModelLlmFunctionCall construction and required fields."""

    def test_valid_construction(self) -> None:
        """Test valid construction with name and arguments."""
        call = ModelLlmFunctionCall(name="get_weather", arguments='{"city":"London"}')

        assert call.name == "get_weather"
        assert call.arguments == '{"city":"London"}'

    def test_name_field_is_required(self) -> None:
        """Test that omitting name raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmFunctionCall(arguments='{"city":"London"}')  # type: ignore[call-arg]

    def test_arguments_field_is_required(self) -> None:
        """Test that omitting arguments raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmFunctionCall(name="get_weather")  # type: ignore[call-arg]


# ============================================================================
# Field Validation Tests
# ============================================================================


class TestFieldValidation:
    """Tests for field-level validation constraints."""

    def test_name_min_length_rejects_empty(self) -> None:
        """Test that name='' is rejected by min_length=1 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmFunctionCall(name="", arguments="{}")

        assert "name" in str(exc_info.value).lower()

    def test_name_single_char_accepted(self) -> None:
        """Test that a single-character name is accepted."""
        call = ModelLlmFunctionCall(name="f", arguments="{}")

        assert call.name == "f"

    def test_arguments_empty_string_accepted(self) -> None:
        """Test that arguments='' is accepted (no min_length on arguments)."""
        call = ModelLlmFunctionCall(name="func", arguments="")

        assert call.arguments == ""

    def test_arguments_with_valid_json(self) -> None:
        """Test that arguments containing valid JSON round-trips correctly."""
        json_str = '{"a":1}'
        call = ModelLlmFunctionCall(name="func", arguments=json_str)

        assert call.arguments == json_str
        parsed = json.loads(call.arguments)
        assert parsed == {"a": 1}

    def test_arguments_with_non_json_string(self) -> None:
        """Test that non-JSON string is accepted (arguments is a plain str field)."""
        call = ModelLlmFunctionCall(name="func", arguments="not json")

        assert call.arguments == "not json"


# ============================================================================
# Immutability Tests
# ============================================================================


class TestImmutability:
    """Tests for frozen model immutability."""

    def test_frozen_immutability_name(self) -> None:
        """Test that assigning to name raises ValidationError on frozen model."""
        call = ModelLlmFunctionCall(name="func", arguments="{}")

        with pytest.raises(ValidationError):
            call.name = "other"  # type: ignore[misc]

    def test_frozen_immutability_arguments(self) -> None:
        """Test that assigning to arguments raises ValidationError on frozen model."""
        call = ModelLlmFunctionCall(name="func", arguments="{}")

        with pytest.raises(ValidationError):
            call.arguments = '{"new":true}'  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmFunctionCall(
                name="func",
                arguments="{}",
                extra_field="x",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value).lower()


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for model serialization and deserialization."""

    def test_model_dump_roundtrip(self) -> None:
        """Test model_dump -> model_validate preserves data."""
        original = ModelLlmFunctionCall(
            name="get_weather", arguments='{"city":"London"}'
        )

        data = original.model_dump()
        restored = ModelLlmFunctionCall.model_validate(data)

        assert restored.name == original.name
        assert restored.arguments == original.arguments
        assert restored == original

    def test_model_dump_json(self) -> None:
        """Test that model_dump_json produces a valid JSON string."""
        call = ModelLlmFunctionCall(name="get_weather", arguments='{"city":"London"}')

        json_str = call.model_dump_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"] == '{"city":"London"}'

    def test_from_attributes_config(self) -> None:
        """Test from_attributes=True allows creation from objects with attributes."""

        class FunctionCallData:
            """Simple class with matching attributes."""

            def __init__(self) -> None:
                self.name = "get_weather"
                self.arguments = '{"city":"London"}'

        source = FunctionCallData()
        call = ModelLlmFunctionCall.model_validate(source)

        assert call.name == source.name
        assert call.arguments == source.arguments


# ============================================================================
# Hashability and Equality Tests
# ============================================================================


class TestHashabilityAndEquality:
    """Tests for hashability and equality comparison."""

    def test_hashable(self) -> None:
        """Test that hash() works and equal instances produce the same hash."""
        call_a = ModelLlmFunctionCall(name="func", arguments='{"a":1}')
        call_b = ModelLlmFunctionCall(name="func", arguments='{"a":1}')

        assert hash(call_a) == hash(call_b)

        # Can be used in a set
        call_set = {call_a, call_b}
        assert len(call_set) == 1

        # Can be used as dict key
        call_dict = {call_a: "value"}
        assert call_dict[call_b] == "value"

    def test_equality_same_values(self) -> None:
        """Test that instances with the same values are equal."""
        call_a = ModelLlmFunctionCall(name="func", arguments='{"a":1}')
        call_b = ModelLlmFunctionCall(name="func", arguments='{"a":1}')

        assert call_a == call_b

    def test_equality_different_values(self) -> None:
        """Test that instances with different values are not equal."""
        call_a = ModelLlmFunctionCall(name="func_a", arguments='{"a":1}')
        call_b = ModelLlmFunctionCall(name="func_b", arguments='{"b":2}')

        assert call_a != call_b
