# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelLlmToolDefinition.

Tests validate:
- Valid construction with defaults and explicit values
- Literal type field validation
- Required function field enforcement
- Dict coercion for nested ModelLlmFunctionDef
- Frozen immutability and extra field rejection
- Serialization round-trip and JSON output
- Equality and hashability

Test Organization:
    - TestModelLlmToolDefinitionConstruction: Basic instantiation (3 tests)
    - TestModelLlmToolDefinitionFieldValidation: Validation constraints (4 tests)
    - TestModelLlmToolDefinitionImmutability: Frozen / extra forbid (2 tests)
    - TestModelLlmToolDefinitionSerialization: Dump / JSON / equality (3 tests)
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_function_def import (
    ModelLlmFunctionDef,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)

# ==============================================================================
# Construction
# ==============================================================================


class TestModelLlmToolDefinitionConstruction:
    """Tests for basic model instantiation."""

    def test_valid_construction(self) -> None:
        """Test construction with explicit function and default type."""
        fn = ModelLlmFunctionDef(name="fn")
        defn = ModelLlmToolDefinition(function=fn)

        assert defn.type == "function"
        assert defn.function is fn

    def test_type_defaults_to_function(self) -> None:
        """Test that omitting type defaults to 'function'."""
        defn = ModelLlmToolDefinition(function=ModelLlmFunctionDef(name="search"))

        assert defn.type == "function"

    def test_type_literal_rejects_invalid(self) -> None:
        """Test that a non-'function' type value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmToolDefinition(
                type="custom",  # type: ignore[arg-type]
                function=ModelLlmFunctionDef(name="fn"),
            )

        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any(e["loc"] == ("type",) for e in errors)


# ==============================================================================
# Field Validation
# ==============================================================================


class TestModelLlmToolDefinitionFieldValidation:
    """Tests for field-level validation constraints."""

    def test_function_is_required(self) -> None:
        """Test that omitting function raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmToolDefinition()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("function",) for e in errors)

    def test_function_from_dict_coercion(self) -> None:
        """Test that a raw dict is auto-coerced into ModelLlmFunctionDef."""
        defn = ModelLlmToolDefinition(function={"name": "fn"})  # type: ignore[arg-type]

        assert isinstance(defn.function, ModelLlmFunctionDef)
        assert defn.function.name == "fn"
        assert defn.function.description == ""
        assert defn.function.parameters == {}

    def test_function_invalid_type_rejected(self) -> None:
        """Test that a non-dict/non-model value for function raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmToolDefinition(function="string")  # type: ignore[arg-type]

    def test_nested_function_fields_accessible(self) -> None:
        """Test that nested ModelLlmFunctionDef fields are accessible."""
        fn = ModelLlmFunctionDef(
            name="get_weather",
            description="Return current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )
        defn = ModelLlmToolDefinition(function=fn)

        assert defn.function.name == "get_weather"
        assert defn.function.description == "Return current weather for a city."
        assert defn.function.parameters["type"] == "object"
        assert "city" in defn.function.parameters["properties"]


# ==============================================================================
# Immutability
# ==============================================================================


class TestModelLlmToolDefinitionImmutability:
    """Tests for frozen model and extra field rejection."""

    def test_frozen_immutability(self) -> None:
        """Test that assigning to type or function raises ValidationError."""
        defn = ModelLlmToolDefinition(function=ModelLlmFunctionDef(name="fn"))

        with pytest.raises(ValidationError):
            defn.type = "function"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            defn.function = ModelLlmFunctionDef(name="other")  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that supplying an extra field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmToolDefinition(
                function=ModelLlmFunctionDef(name="fn"),
                unknown_field="value",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


# ==============================================================================
# Serialization
# ==============================================================================


class TestModelLlmToolDefinitionSerialization:
    """Tests for model_dump, JSON output, and equality."""

    def test_model_dump_roundtrip(self) -> None:
        """Test that model_dump preserves nested ModelLlmFunctionDef."""
        fn = ModelLlmFunctionDef(
            name="search",
            description="Search the web.",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        defn = ModelLlmToolDefinition(function=fn)
        dumped = defn.model_dump()

        assert dumped == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }

        # Round-trip: reconstruct from dump
        reconstructed = ModelLlmToolDefinition(**dumped)
        assert reconstructed == defn

    def test_model_dump_json(self) -> None:
        """Test that model_dump_json produces valid JSON with nested structure."""
        defn = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(
                name="calc",
                parameters={"type": "object"},
            ),
        )
        raw_json = defn.model_dump_json()
        parsed = json.loads(raw_json)

        assert parsed["type"] == "function"
        assert parsed["function"]["name"] == "calc"
        assert parsed["function"]["parameters"] == {"type": "object"}
        assert parsed["function"]["description"] == ""

    def test_equality_and_hashability(self) -> None:
        """Test that instances with same nested values are equal.

        Note: ModelLlmFunctionDef contains a ``parameters: dict[str, Any]``
        field, which makes Pydantic's frozen-model ``__hash__`` raise
        ``TypeError`` because ``dict`` is unhashable.  We verify equality
        works correctly and document the hashing limitation.
        """
        # Equality with dict parameters
        fn_a = ModelLlmFunctionDef(name="fn", description="desc", parameters={"a": 1})
        fn_b = ModelLlmFunctionDef(name="fn", description="desc", parameters={"a": 1})
        defn_a = ModelLlmToolDefinition(function=fn_a)
        defn_b = ModelLlmToolDefinition(function=fn_b)

        assert defn_a == defn_b

        # Different values are not equal
        defn_c = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="other"),
        )
        assert defn_a != defn_c

        # Hashing raises TypeError due to nested dict field
        with pytest.raises(TypeError, match="unhashable type"):
            hash(defn_a)
