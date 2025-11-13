#!/usr/bin/env python3
"""
JSONB Validation Examples.

This module demonstrates the three approaches to JSONB field validation:
1. JsonbField() helper function (Recommended)
2. Manual Field() with validator decorator
3. JsonbValidatedModel base class (Automatic)

These examples can be used as reference when implementing entity models.
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from omninode_bridge.infrastructure.validation import (
    JsonbField,
    JsonbValidatedModel,
    validate_jsonb_fields,
)

# ============================================================================
# Approach 1: JsonbField() Helper (Recommended)
# ============================================================================


class ModelExample1(BaseModel):
    """
    Example using JsonbField() helper function.

    This is the RECOMMENDED approach as it's:
    - Concise and readable
    - Automatically includes JSONB annotation
    - Type-safe with full IDE support
    """

    id: UUID = Field(default_factory=uuid4)
    namespace: str = Field(..., min_length=1, max_length=255)

    # JSONB fields using JsonbField() helper
    metadata: dict[str, Any] = JsonbField(
        default_factory=dict, description="Workflow execution metadata"
    )

    config_data: dict[str, Any] = JsonbField(
        default_factory=dict, description="Configuration settings"
    )

    # Regular fields work as expected
    status: str = Field(..., min_length=1, max_length=50)


# ============================================================================
# Approach 2: Manual Field() with Validator Decorator
# ============================================================================


class ModelExample2(BaseModel):
    """
    Example using manual Field() with validator decorator.

    This approach is useful when:
    - Migrating existing models
    - Need explicit control over json_schema_extra
    - Want to see the full Field() configuration
    """

    id: UUID = Field(default_factory=uuid4)
    namespace: str = Field(..., min_length=1, max_length=255)

    # JSONB fields with manual annotation
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow execution metadata",
        json_schema_extra={"db_type": "jsonb"},
    )

    config_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration settings",
        json_schema_extra={"db_type": "jsonb"},
    )

    status: str = Field(..., min_length=1, max_length=50)

    # Add validator to enforce JSONB annotations
    @model_validator(mode="after")
    def _validate_jsonb(self) -> "ModelExample2":
        return validate_jsonb_fields(self)


# ============================================================================
# Approach 3: JsonbValidatedModel Base Class (Automatic)
# ============================================================================


class ModelExample3(JsonbValidatedModel):
    """
    Example using JsonbValidatedModel base class.

    This approach is useful when:
    - Building new models from scratch
    - Want automatic validation without decorators
    - Prefer inheritance-based configuration
    """

    id: UUID = Field(default_factory=uuid4)
    namespace: str = Field(..., min_length=1, max_length=255)

    # JSONB fields - validation is automatic via base class
    metadata: dict[str, Any] = JsonbField(
        default_factory=dict, description="Workflow execution metadata"
    )

    config_data: dict[str, Any] = JsonbField(
        default_factory=dict, description="Configuration settings"
    )

    status: str = Field(..., min_length=1, max_length=50)


# ============================================================================
# Error Examples (What NOT to do)
# ============================================================================


class ModelBadExample1(BaseModel):
    """
    ❌ INCORRECT: Missing json_schema_extra for dict field.

    This will raise a ValueError when validated.
    """

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Missing JSONB annotation",
    )

    @model_validator(mode="after")
    def _validate_jsonb(self) -> "ModelBadExample1":
        # This will raise ValueError due to missing annotation
        return validate_jsonb_fields(self)


class ModelBadExample2(BaseModel):
    """
    ❌ INCORRECT: Wrong db_type in json_schema_extra.

    This will raise a ValueError when validated.
    """

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Wrong db_type annotation",
        json_schema_extra={"db_type": "json"},  # Should be "jsonb"
    )

    @model_validator(mode="after")
    def _validate_jsonb(self) -> "ModelBadExample2":
        # This will raise ValueError due to wrong db_type
        return validate_jsonb_fields(self)


# ============================================================================
# Usage Examples
# ============================================================================


def example_usage() -> None:
    """Demonstrate proper usage of JSONB validated models."""

    # Example 1: JsonbField() helper
    model1 = ModelExample1(
        namespace="production",
        metadata={"version": "1.0", "author": "system"},
        config_data={"timeout_ms": 5000, "retry_count": 3},
        status="active",
    )
    print(f"Model 1: {model1.model_dump_json(indent=2)}")

    # Example 2: Manual Field() with validator
    model2 = ModelExample2(
        namespace="staging",
        metadata={"version": "1.0", "author": "user"},
        config_data={"timeout_ms": 3000, "retry_count": 2},
        status="pending",
    )
    print(f"Model 2: {model2.model_dump_json(indent=2)}")

    # Example 3: JsonbValidatedModel base class
    model3 = ModelExample3(
        namespace="development",
        metadata={"version": "1.0", "author": "developer"},
        config_data={"timeout_ms": 1000, "retry_count": 1},
        status="testing",
    )
    print(f"Model 3: {model3.model_dump_json(indent=2)}")

    # Verify JSONB annotations are present
    for model_class in [ModelExample1, ModelExample2, ModelExample3]:
        fields = model_class.model_fields
        for field_name, field_info in fields.items():
            if field_info.annotation.__origin__ is dict:  # type: ignore
                assert field_info.json_schema_extra == {
                    "db_type": "jsonb"
                }, f"Missing JSONB annotation on {model_class.__name__}.{field_name}"

    print("✅ All models have correct JSONB annotations")


def example_errors() -> None:
    """Demonstrate validation errors for incorrect models."""

    try:
        # This will raise ValueError: missing json_schema_extra
        ModelBadExample1(metadata={"test": "data"})
    except ValueError as e:
        print(f"❌ Expected error: {e}")

    try:
        # This will raise ValueError: wrong db_type
        ModelBadExample2(metadata={"test": "data"})
    except ValueError as e:
        print(f"❌ Expected error: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("JSONB Validation Examples")
    print("=" * 80)
    print()

    print("Valid Usage Examples:")
    print("-" * 80)
    example_usage()
    print()

    print("Error Examples:")
    print("-" * 80)
    example_errors()
