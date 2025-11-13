#!/usr/bin/env python3
"""
JSONB Field Validation Utilities for Pydantic Models.

This module provides utilities to enforce PostgreSQL JSONB type annotations
in Pydantic entity models. It ensures all JSONB fields are properly marked
with `json_schema_extra={"db_type": "jsonb"}` for correct database mapping.

ONEX v2.0 Compliance:
- Type-safe JSONB field definitions
- Automatic validation of database type annotations
- Clear error messages for missing annotations
- Pydantic v2 compatibility

Usage:
    # Option 1: Use JsonbField() helper function (Recommended)
    from omninode_bridge.infrastructure.validation.jsonb_validators import JsonbField

    class MyModel(BaseModel):
        metadata: dict[str, Any] = JsonbField(
            default_factory=dict,
            description="JSONB metadata field"
        )

    # Option 2: Use model validator for existing models
    from omninode_bridge.infrastructure.validation.jsonb_validators import (
        validate_jsonb_fields
    )

    class MyModel(BaseModel):
        metadata: dict[str, Any] = Field(
            default_factory=dict,
            json_schema_extra={"db_type": "jsonb"}
        )

        @model_validator(mode="after")
        def _validate_jsonb(self) -> "MyModel":
            return validate_jsonb_fields(self)
"""

from collections.abc import Callable
from typing import Any, Optional, TypeVar, Union

from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo

# Type variable for generic default factory
T = TypeVar("T")

# Type aliases for better clarity
JsonbDefault = Union[dict[str, Any], list[Any], None]
JsonbFactory = Optional[Callable[[], JsonbDefault]]


def JsonbField(
    default: Union[JsonbDefault, type(...)] = ...,  # type: ignore[valid-type]
    *,
    default_factory: JsonbFactory = None,
    alias: Optional[str] = None,
    alias_priority: Optional[int] = None,
    validation_alias: Optional[str] = None,
    serialization_alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    examples: Optional[list[Any]] = None,
    exclude: Optional[bool] = None,
    discriminator: Optional[str] = None,
    deprecated: Optional[str] = None,
    frozen: Optional[bool] = None,
    validate_default: Optional[bool] = None,
    repr: bool = True,
    init: Optional[bool] = None,
    init_var: Optional[bool] = None,
    kw_only: Optional[bool] = None,
    **extra: Any,
) -> FieldInfo:
    """
    Create a Pydantic Field with automatic JSONB database type annotation.

    This is a convenience wrapper around Pydantic's Field() that automatically
    adds `json_schema_extra={"db_type": "jsonb"}` to ensure proper PostgreSQL
    JSONB type mapping.

    Type Safety Notes:
        - default accepts JsonbDefault types (dict[str, Any], list[Any], None) or ... for required
        - default_factory must return JsonbDefault if provided
        - Return type is FieldInfo for proper Pydantic integration

    Args:
        default: Default value for the field (dict, list, None, or ... for required)
        default_factory: Factory function returning JsonbDefault (e.g., dict, list)
        alias: Alternative name for the field
        alias_priority: Priority of alias resolution
        validation_alias: Alias used during validation
        serialization_alias: Alias used during serialization
        title: Field title for schema generation
        description: Human-readable description of the field
        examples: Example values for documentation
        exclude: Whether to exclude field from serialization
        discriminator: Discriminator field for union types
        deprecated: Deprecation message
        frozen: Whether field is immutable after initialization
        validate_default: Whether to validate default value
        repr: Whether to include in __repr__
        init: Whether to include in __init__
        init_var: Whether field is init-only variable
        kw_only: Whether field is keyword-only
        **extra: Additional field configuration

    Returns:
        FieldInfo: Pydantic field with JSONB annotation

    Example:
        >>> from typing import Any
        >>> from pydantic import BaseModel
        >>> from omninode_bridge.infrastructure.validation.jsonb_validators import JsonbField
        >>>
        >>> class ModelWorkflow(BaseModel):
        ...     metadata: dict[str, Any] = JsonbField(
        ...         default_factory=dict,
        ...         description="Workflow execution metadata"
        ...     )
        ...
        >>> workflow = ModelWorkflow()
        >>> assert workflow.model_fields['metadata'].json_schema_extra == {"db_type": "jsonb"}

    Note:
        The field type annotation MUST be `dict[str, Any]` for JSONB fields.
        Using other types (like `dict`, `Dict[str, str]`, etc.) will work
        but may not represent the actual JSONB flexibility correctly.
    """
    # Ensure json_schema_extra includes db_type marker
    json_schema_extra = extra.pop("json_schema_extra", {})
    if not isinstance(json_schema_extra, dict):
        json_schema_extra = {}
    json_schema_extra["db_type"] = "jsonb"

    return Field(
        default=default,
        default_factory=default_factory,
        alias=alias,
        alias_priority=alias_priority,
        validation_alias=validation_alias,
        serialization_alias=serialization_alias,
        title=title,
        description=description,
        examples=examples,
        exclude=exclude,
        discriminator=discriminator,
        deprecated=deprecated,
        frozen=frozen,
        validate_default=validate_default,
        repr=repr,
        init=init,
        init_var=init_var,
        kw_only=kw_only,
        json_schema_extra=json_schema_extra,
        **extra,
    )


def validate_jsonb_fields(model: BaseModel) -> BaseModel:
    """
    Validate that all dict[str, Any] fields have proper JSONB annotations.

    This validator checks that fields typed as `dict[str, Any]` (or similar
    dictionary types) include `json_schema_extra={"db_type": "jsonb"}` to
    ensure correct PostgreSQL JSONB type mapping.

    Args:
        model: Pydantic model instance to validate

    Returns:
        BaseModel: The validated model instance (unchanged)

    Raises:
        ValueError: If a dict field is missing JSONB annotation

    Example:
        >>> from typing import Any
        >>> from pydantic import BaseModel, model_validator
        >>> from omninode_bridge.infrastructure.validation.jsonb_validators import (
        ...     validate_jsonb_fields
        ... )
        >>>
        >>> class MyModel(BaseModel):
        ...     data: dict[str, Any] = Field(
        ...         default_factory=dict,
        ...         json_schema_extra={"db_type": "jsonb"}
        ...     )
        ...
        ...     @model_validator(mode="after")
        ...     def _validate_jsonb(self) -> "MyModel":
        ...         return validate_jsonb_fields(self)

    Note:
        This validator runs in "after" mode, meaning it validates the model
        after all field validators have run. It does not modify the model,
        only checks for compliance.
    """
    model_class = type(model)
    field_info: dict[str, FieldInfo] = model_class.model_fields

    for field_name, field in field_info.items():
        # Get field annotation
        annotation = field.annotation

        # Check if field is a dict type (various representations)
        is_dict_field = False
        if hasattr(annotation, "__origin__"):
            # Handle generic types like dict[str, Any]
            is_dict_field = annotation.__origin__ is dict
        elif annotation is dict:
            # Handle plain dict annotation
            is_dict_field = True

        # If it's a dict field, verify JSONB annotation
        if is_dict_field:
            json_schema_extra = field.json_schema_extra or {}

            if not isinstance(json_schema_extra, dict):
                raise ValueError(
                    f"Field '{field_name}' in {model_class.__name__} has invalid "
                    f"json_schema_extra (expected dict, got {type(json_schema_extra).__name__})"
                )

            if json_schema_extra.get("db_type") != "jsonb":
                raise ValueError(
                    f"Field '{field_name}' in {model_class.__name__} is typed as dict "
                    f"but missing required json_schema_extra={{'db_type': 'jsonb'}}. "
                    f"\n\nUse JsonbField() helper or add json_schema_extra manually:\n"
                    f"  # Option 1 (Recommended):\n"
                    f"  {field_name}: dict[str, Any] = JsonbField(default_factory=dict)\n\n"
                    f"  # Option 2:\n"
                    f"  {field_name}: dict[str, Any] = Field(\n"
                    f"      default_factory=dict,\n"
                    f"      json_schema_extra={{'db_type': 'jsonb'}}\n"
                    f"  )"
                )

    return model


class JsonbValidatedModel(BaseModel):
    """
    Base model with automatic JSONB field validation.

    This base class automatically validates that all dict[str, Any] fields
    have proper JSONB annotations. Inherit from this class to enable automatic
    validation without manually adding the validator decorator.

    Example:
        >>> from typing import Any
        >>> from omninode_bridge.infrastructure.validation.jsonb_validators import (
        ...     JsonbValidatedModel,
        ...     JsonbField
        ... )
        >>>
        >>> class MyModel(JsonbValidatedModel):
        ...     metadata: dict[str, Any] = JsonbField(
        ...         default_factory=dict,
        ...         description="Workflow metadata"
        ...     )
        ...
        >>> model = MyModel()  # Automatically validated

    Note:
        This is the simplest way to enforce JSONB validation across all models.
        Simply inherit from JsonbValidatedModel instead of BaseModel.
    """

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "JsonbValidatedModel":
        """Automatically validate JSONB fields on all inheriting models."""
        return validate_jsonb_fields(self)


# Convenience exports
__all__ = [
    "JsonbField",
    "validate_jsonb_fields",
    "JsonbValidatedModel",
]
