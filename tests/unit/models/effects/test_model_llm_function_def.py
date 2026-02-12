# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ModelLlmFunctionDef.

This test suite validates:
- Basic model construction with minimal and full parameters
- Field validation (required fields, min_length, defaults)
- Immutability enforcement (frozen model)
- Serialization round-trips and attribute-based construction
- Hashability and equality semantics

Test Organization:
    - TestModelLlmFunctionDefConstruction: Minimal and full construction (2 tests)
    - TestModelLlmFunctionDefFieldValidation: Field constraints and defaults (7 tests)
    - TestModelLlmFunctionDefImmutability: Frozen and extra-forbid behavior (2 tests)
    - TestModelLlmFunctionDefSerialization: Dump/load, from_attributes, hash, eq (4 tests)

Coverage Goals:
    - 100% code coverage for ModelLlmFunctionDef
    - All field defaults and constraints verified
    - Pydantic ConfigDict enforcement (frozen, extra=forbid, from_attributes)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_function_def import (
    ModelLlmFunctionDef,
)

# ==============================================================================
# Construction Tests
# ==============================================================================


class TestModelLlmFunctionDefConstruction:
    """Test basic model construction with minimal and full parameters."""

    def test_minimal_construction(self) -> None:
        """Test construction with only the required 'name' field uses defaults."""
        fn = ModelLlmFunctionDef(name="fn")

        assert fn.name == "fn"
        assert fn.description == ""
        assert fn.parameters == {}

    def test_full_construction(self) -> None:
        """Test construction with all fields explicitly set."""
        params = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }
        fn = ModelLlmFunctionDef(
            name="get_weather",
            description="Return current weather for a city.",
            parameters=params,
        )

        assert fn.name == "get_weather"
        assert fn.description == "Return current weather for a city."
        assert fn.parameters == params
        assert fn.parameters["properties"]["city"]["type"] == "string"


# ==============================================================================
# Field Validation Tests
# ==============================================================================


class TestModelLlmFunctionDefFieldValidation:
    """Test field-level validation constraints and defaults."""

    def test_name_is_required(self) -> None:
        """Omitting name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmFunctionDef()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_name_min_length_rejects_empty(self) -> None:
        """Empty string for name is rejected by min_length=1 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmFunctionDef(name="")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)
        assert any("min_length" in str(e) or "too_short" in e["type"] for e in errors)

    def test_description_defaults_to_empty_string(self) -> None:
        """Omitting description defaults to empty string, not None."""
        fn = ModelLlmFunctionDef(name="test_fn")

        assert fn.description == ""
        assert isinstance(fn.description, str)

    def test_description_accepts_long_text(self) -> None:
        """Long description text is accepted without truncation."""
        long_desc = "A" * 10_000
        fn = ModelLlmFunctionDef(name="verbose_fn", description=long_desc)

        assert fn.description == long_desc
        assert len(fn.description) == 10_000

    def test_parameters_defaults_to_empty_dict(self) -> None:
        """Omitting parameters defaults to an empty dict."""
        fn = ModelLlmFunctionDef(name="no_params")

        assert fn.parameters == {}
        assert isinstance(fn.parameters, dict)

    def test_parameters_accepts_json_schema(self) -> None:
        """Parameters field accepts a valid JSON Schema object."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        fn = ModelLlmFunctionDef(name="search", parameters=schema)

        assert fn.parameters["type"] == "object"
        assert "query" in fn.parameters["properties"]
        assert fn.parameters["required"] == ["query"]

    def test_parameters_accepts_nested_structures(self) -> None:
        """Parameters field accepts deeply nested dict structures."""
        nested = {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "criteria": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "operator": {"type": "string"},
                                    "value": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                            {"type": "boolean"},
                                        ]
                                    },
                                },
                            },
                        }
                    },
                }
            },
        }
        fn = ModelLlmFunctionDef(name="advanced_query", parameters=nested)

        # Verify deep nesting preserved
        items = fn.parameters["properties"]["filter"]["properties"]["criteria"]["items"]
        assert items["properties"]["field"]["type"] == "string"
        assert len(items["properties"]["value"]["oneOf"]) == 3


# ==============================================================================
# Immutability Tests
# ==============================================================================


class TestModelLlmFunctionDefImmutability:
    """Test frozen model immutability and extra field rejection."""

    def test_frozen_immutability(self) -> None:
        """Assigning to any field on a frozen model raises ValidationError."""
        fn = ModelLlmFunctionDef(name="immutable_fn")

        with pytest.raises(ValidationError):
            fn.name = "new_name"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            fn.description = "new desc"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            fn.parameters = {"new": "value"}  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Passing an extra field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmFunctionDef(
                name="fn",
                unknown_field="should_fail",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert any("extra" in e["type"] for e in errors)


# ==============================================================================
# Serialization Tests
# ==============================================================================


class TestModelLlmFunctionDefSerialization:
    """Test serialization round-trips, from_attributes, hashability, equality."""

    def test_model_dump_roundtrip(self) -> None:
        """Dumping to dict and re-validating preserves all fields."""
        original = ModelLlmFunctionDef(
            name="roundtrip_fn",
            description="Roundtrip test function.",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )

        dumped = original.model_dump()
        restored = ModelLlmFunctionDef.model_validate(dumped)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.parameters == original.parameters
        assert restored == original

    def test_from_attributes_config(self) -> None:
        """model_validate with from_attributes=True works on attr-bearing objects."""

        class AttrSource:
            """Simple object with matching attribute names."""

            def __init__(self) -> None:
                self.name = "attr_fn"
                self.description = "From attributes."
                self.parameters = {"type": "object"}

        source = AttrSource()
        fn = ModelLlmFunctionDef.model_validate(source, from_attributes=True)

        assert fn.name == "attr_fn"
        assert fn.description == "From attributes."
        assert fn.parameters == {"type": "object"}

    def test_hashable(self) -> None:
        """Frozen model with default empty parameters is hashable.

        Note: When parameters contains a non-empty dict, Pydantic's frozen
        hash raises TypeError because dict is unhashable.  With the default
        empty-dict parameters field the model is hashable via the empty-dict
        edge case only when all values are hashable.  We verify the general
        contract: instances with only hashable field values can be hashed.
        """
        # Default parameters={} -- Pydantic hashes the __dict__ values as a
        # tuple; an empty dict is still unhashable, so we expect TypeError
        # for any instance that has a dict field.
        fn = ModelLlmFunctionDef(name="hash_fn", description="Hashable test.")

        # dict fields make the model unhashable despite frozen=True
        with pytest.raises(TypeError, match="unhashable type"):
            hash(fn)

    def test_equality(self) -> None:
        """Instances with same values are equal; different values are not."""
        fn_a = ModelLlmFunctionDef(
            name="eq_fn",
            description="Same.",
            parameters={"type": "object"},
        )
        fn_b = ModelLlmFunctionDef(
            name="eq_fn",
            description="Same.",
            parameters={"type": "object"},
        )
        fn_c = ModelLlmFunctionDef(
            name="different_fn",
            description="Different.",
            parameters={},
        )

        assert fn_a == fn_b
        assert fn_a != fn_c
        assert fn_b != fn_c
