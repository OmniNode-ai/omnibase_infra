# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive tests for ModelLlmToolCall.

Tests cover:
- Valid construction with required and default fields
- Field validation (id, type literal, function type enforcement)
- Dict-to-nested-model coercion by Pydantic
- Frozen immutability (frozen=True)
- Extra field rejection (extra='forbid')
- Nested ModelLlmFunctionCall accessibility
- Serialization roundtrip (model_dump, model_dump_json)
- Equality and hashability of frozen instances

OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall

# ============================================================================
# Construction Tests
# ============================================================================


class TestModelLlmToolCallConstruction:
    """Tests for valid construction of ModelLlmToolCall."""

    def test_valid_construction(self) -> None:
        """Test construction with id, function, and default type='function'."""
        fn = ModelLlmFunctionCall(name="search", arguments='{"q": "hello"}')
        tc = ModelLlmToolCall(id="call_abc123", function=fn)

        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function is fn

    def test_type_defaults_to_function(self) -> None:
        """Test that omitting type defaults to 'function'."""
        fn = ModelLlmFunctionCall(name="get_weather", arguments='{"city": "NYC"}')
        tc = ModelLlmToolCall(id="call_001", function=fn)

        assert tc.type == "function"

    def test_type_accepts_explicit_function(self) -> None:
        """Test that explicitly passing type='function' is accepted."""
        fn = ModelLlmFunctionCall(name="lookup", arguments="{}")
        tc = ModelLlmToolCall(id="call_002", type="function", function=fn)

        assert tc.type == "function"


# ============================================================================
# Field Validation Tests
# ============================================================================


class TestModelLlmToolCallFieldValidation:
    """Tests for field validation rules on ModelLlmToolCall."""

    def test_type_literal_rejects_invalid(self) -> None:
        """Test that type='tool' (or any non-'function' value) raises ValidationError."""
        fn = ModelLlmFunctionCall(name="f", arguments="{}")
        with pytest.raises(ValidationError):
            ModelLlmToolCall(id="call_x", type="tool", function=fn)  # type: ignore[arg-type]

    def test_id_is_required(self) -> None:
        """Test that omitting id raises ValidationError."""
        fn = ModelLlmFunctionCall(name="f", arguments="{}")
        with pytest.raises(ValidationError):
            ModelLlmToolCall(function=fn)  # type: ignore[call-arg]

    def test_id_empty_string_accepted(self) -> None:
        """Test that id='' is accepted (no min_length constraint on id)."""
        fn = ModelLlmFunctionCall(name="f", arguments="{}")
        tc = ModelLlmToolCall(id="", function=fn)

        assert tc.id == ""

    def test_function_is_required(self) -> None:
        """Test that omitting function raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmToolCall(id="call_missing")  # type: ignore[call-arg]

    def test_function_must_be_correct_type(self) -> None:
        """Test that function='string' raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmToolCall(id="call_bad", function="string")  # type: ignore[arg-type]

    def test_function_from_dict_coercion(self) -> None:
        """Test that passing a dict for function auto-coerces to ModelLlmFunctionCall."""
        tc = ModelLlmToolCall(
            id="call_dict",
            function={"name": "f", "arguments": "{}"},  # type: ignore[arg-type]
        )

        assert isinstance(tc.function, ModelLlmFunctionCall)
        assert tc.function.name == "f"
        assert tc.function.arguments == "{}"


# ============================================================================
# Immutability Tests
# ============================================================================


class TestModelLlmToolCallImmutability:
    """Tests for frozen immutability of ModelLlmToolCall."""

    def test_frozen_immutability(self) -> None:
        """Test that assigning to id, type, or function raises ValidationError."""
        fn = ModelLlmFunctionCall(name="f", arguments="{}")
        tc = ModelLlmToolCall(id="call_frozen", function=fn)

        with pytest.raises(ValidationError):
            tc.id = "new_id"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            tc.type = "function"  # type: ignore[misc]

        fn2 = ModelLlmFunctionCall(name="g", arguments="{}")
        with pytest.raises(ValidationError):
            tc.function = fn2  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise ValidationError (extra='forbid')."""
        fn = ModelLlmFunctionCall(name="f", arguments="{}")
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmToolCall(
                id="call_extra",
                function=fn,
                unexpected="nope",  # type: ignore[call-arg]
            )

        assert "unexpected" in str(exc_info.value)


# ============================================================================
# Serialization Tests
# ============================================================================


class TestModelLlmToolCallSerialization:
    """Tests for serialization and deserialization of ModelLlmToolCall."""

    def test_nested_function_accessible(self) -> None:
        """Test that nested function fields are accessible via dot notation."""
        fn = ModelLlmFunctionCall(name="search", arguments='{"q": "test"}')
        tc = ModelLlmToolCall(id="call_nested", function=fn)

        assert tc.function.name == "search"
        assert tc.function.arguments == '{"q": "test"}'

    def test_model_dump_roundtrip(self) -> None:
        """Test that model_dump -> model_validate roundtrip preserves nested structure."""
        fn = ModelLlmFunctionCall(name="get_data", arguments='{"id": 42}')
        original = ModelLlmToolCall(id="call_rt", function=fn)

        dumped = original.model_dump()
        restored = ModelLlmToolCall.model_validate(dumped)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.function.name == original.function.name
        assert restored.function.arguments == original.function.arguments
        assert restored == original

    def test_model_dump_json(self) -> None:
        """Test that model_dump_json includes nested function correctly."""
        fn = ModelLlmFunctionCall(name="calculate", arguments='{"x": 1}')
        tc = ModelLlmToolCall(id="call_json", function=fn)

        json_str = tc.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == "call_json"
        assert parsed["type"] == "function"
        assert parsed["function"]["name"] == "calculate"
        assert parsed["function"]["arguments"] == '{"x": 1}'

    def test_equality_and_hashability(self) -> None:
        """Test that instances with the same nested values are equal and hashable."""
        fn1 = ModelLlmFunctionCall(name="f", arguments='{"a": 1}')
        fn2 = ModelLlmFunctionCall(name="f", arguments='{"a": 1}')
        tc1 = ModelLlmToolCall(id="call_eq", function=fn1)
        tc2 = ModelLlmToolCall(id="call_eq", function=fn2)

        # Equality
        assert tc1 == tc2

        # Hashability
        assert hash(tc1) == hash(tc2)

        # Usable in set
        tc_set = {tc1, tc2}
        assert len(tc_set) == 1

        # Usable as dict key
        tc_dict = {tc1: "value"}
        assert tc_dict[tc2] == "value"

        # Different instances are not equal
        fn3 = ModelLlmFunctionCall(name="g", arguments="{}")
        tc3 = ModelLlmToolCall(id="call_diff", function=fn3)
        assert tc1 != tc3
        assert hash(tc1) != hash(tc3)
