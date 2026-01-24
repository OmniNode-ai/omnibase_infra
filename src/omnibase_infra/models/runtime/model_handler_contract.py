# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Contract Model for Contract-Driven Discovery.

This module provides ModelHandlerContract, a Pydantic model that represents the
schema for handler contract YAML files. It replaces manual field extraction in
HandlerPluginLoader with automatic validation and type coercion.

The model supports dual field names for flexibility:
- handler_name / name
- capability_tags / tags

See Also:
    - HandlerPluginLoader: Uses this model to validate contracts
    - ModelLoadedHandler: Output model after successful loading
    - EnumHandlerTypeCategory: Handler behavioral classification

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Handler Plugin Loader implementation.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums.enum_handler_type_category import EnumHandlerTypeCategory
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.runtime.model_contract_security_config import (
    ModelContractSecurityConfig,
)

# Handler type category mapping from contract strings to enum
_HANDLER_TYPE_CATEGORY_MAP: dict[str, EnumHandlerTypeCategory] = {
    "compute": EnumHandlerTypeCategory.COMPUTE,
    "effect": EnumHandlerTypeCategory.EFFECT,
    "nondeterministic_compute": EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
}


class ModelHandlerContract(BaseModel):
    """Pydantic model for handler contract YAML schema.

    Validates and parses handler contract YAML files with automatic type
    coercion and field aliasing. Supports dual field names for flexibility
    in contract authoring.

    Field Aliases:
        - handler_name: Also accepts 'name'
        - capability_tags: Also accepts 'tags'

    Handler Type Conversion:
        The handler_type field accepts string values (e.g., 'compute', 'effect',
        'nondeterministic_compute') and converts them to EnumHandlerTypeCategory
        enum values. Case-insensitive matching is supported.

    Attributes:
        handler_name: Unique identifier for the handler. Required field
            that accepts both 'handler_name' and 'name' keys in YAML.
        handler_class: Fully qualified class path for the handler
            (e.g., 'myapp.handlers.AuthHandler'). Required field.
        handler_type: Behavioral classification of the handler. Required field
            that accepts string values and converts to EnumHandlerTypeCategory.
        capability_tags: List of tags for handler discovery and filtering.
            Optional field that accepts both 'capability_tags' and 'tags' keys.

    Example:
        >>> from omnibase_infra.models.runtime import ModelHandlerContract
        >>> # Parse from YAML dict with 'name' alias
        >>> contract = ModelHandlerContract.model_validate({
        ...     "name": "auth.handler",
        ...     "handler_class": "myapp.handlers.AuthHandler",
        ...     "handler_type": "compute",
        ...     "tags": ["auth", "validation"],
        ... })
        >>> contract.handler_name
        'auth.handler'
        >>> contract.handler_type
        <EnumHandlerTypeCategory.COMPUTE: 'compute'>

        >>> # Parse with canonical field names
        >>> contract = ModelHandlerContract.model_validate({
        ...     "handler_name": "db.handler",
        ...     "handler_class": "myapp.handlers.DbHandler",
        ...     "handler_type": "effect",
        ...     "capability_tags": ["database", "postgres"],
        ... })

    See Also:
        - :class:`HandlerPluginLoader`: Uses this model for contract validation.
        - :class:`ModelLoadedHandler`: Output model after successful loading.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    handler_name: str = Field(
        ...,
        min_length=1,
        validation_alias="name",
        description="Unique identifier for the handler (also accepts 'name')",
    )
    handler_class: str = Field(
        ...,
        min_length=1,
        description="Fully qualified handler class path (e.g., 'myapp.handlers.AuthHandler')",
    )
    handler_type: EnumHandlerTypeCategory = Field(
        ...,
        description="Behavioral classification (compute, effect, nondeterministic_compute)",
    )
    capability_tags: list[str] = Field(
        default_factory=list,
        validation_alias="tags",
        description="Tags for handler discovery and filtering (also accepts 'tags')",
    )
    protocol_type: str | None = Field(
        default=None,
        description=(
            "Protocol type identifier for registry (e.g., 'db', 'http', 'consul'). "
            "If not provided, defaults to handler_name with 'handler-' prefix stripped."
        ),
    )
    security: ModelContractSecurityConfig | None = Field(
        default=None,
        description="Optional security configuration for the handler",
    )
    handler_version: ModelSemVer | None = Field(
        default=None,
        description="Handler version in semantic versioning format. If not provided, defaults to 1.0.0",
    )

    @field_validator("handler_type", mode="before")
    @classmethod
    def convert_handler_type(cls, value: object) -> EnumHandlerTypeCategory:
        """Convert string handler_type to EnumHandlerTypeCategory.

        Accepts case-insensitive string values and converts them to the
        appropriate enum value. Also accepts EnumHandlerTypeCategory directly.

        Args:
            value: String handler type name or EnumHandlerTypeCategory.

        Returns:
            EnumHandlerTypeCategory enum value.

        Raises:
            ValueError: If the value is not a valid handler type.
        """
        # Already an enum
        if isinstance(value, EnumHandlerTypeCategory):
            return value

        # Convert string to enum
        if isinstance(value, str):
            normalized = value.lower().strip()
            if normalized in _HANDLER_TYPE_CATEGORY_MAP:
                return _HANDLER_TYPE_CATEGORY_MAP[normalized]

            valid_types = ", ".join(_HANDLER_TYPE_CATEGORY_MAP.keys())
            raise ValueError(
                f"Invalid handler_type '{value}'. Valid types: {valid_types}"
            )

        raise ValueError(
            f"handler_type must be a string or EnumHandlerTypeCategory, got {type(value).__name__}"
        )

    @field_validator("capability_tags", mode="before")
    @classmethod
    def normalize_capability_tags(cls, value: object) -> list[str]:
        """Normalize capability_tags to a list of strings.

        Handles various input formats:
        - None -> empty list
        - Single string -> list with one element
        - List with mixed types -> filter to strings only

        Args:
            value: Raw value from YAML (None, str, or list).

        Returns:
            List of string capability tags.
        """
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, list):
            # Filter to only string tags, warn about invalid types
            valid_tags: list[str] = []
            for tag in value:
                if isinstance(tag, str):
                    valid_tags.append(tag)
                else:
                    logger.warning(
                        "Non-string capability tag filtered out: value=%r, type=%s",
                        tag,
                        type(tag).__name__,
                    )
            return valid_tags

        return []

    @model_validator(mode="after")
    def set_defaults(self) -> ModelHandlerContract:
        """Set default values for protocol_type and handler_version.

        Protocol Type:
            If protocol_type is None, derives it from handler_name by stripping
            the 'handler-' prefix. If handler_name doesn't have that prefix,
            uses the full handler_name as protocol_type.

            Guards against empty derived protocol_type which would produce
            invalid registry keys.

        Handler Version:
            If handler_version is None, sets it to the default version 1.0.0.

        Returns:
            Self with protocol_type and handler_version populated.

        Raises:
            ValueError: If derived protocol_type would be empty (e.g., handler_name
                is exactly "handler-" with nothing after the prefix).

        Example:
            >>> contract = ModelHandlerContract(
            ...     handler_name="handler-db",
            ...     handler_class="myapp.handlers.DbHandler",
            ...     handler_type="effect",
            ... )
            >>> contract.protocol_type
            'db'
            >>> contract.handler_version
            ModelSemVer(major=1, minor=0, patch=0)

            >>> contract = ModelHandlerContract(
            ...     handler_name="custom-handler",
            ...     handler_class="myapp.handlers.Custom",
            ...     handler_type="compute",
            ... )
            >>> contract.protocol_type
            'custom-handler'
        """
        # Set default handler_version if not provided
        if self.handler_version is None:
            self.handler_version = ModelSemVer(major=1, minor=0, patch=0)

        if self.protocol_type is None:
            prefix = "handler-"
            if self.handler_name.startswith(prefix):
                derived = self.handler_name[len(prefix) :]
                if not derived:
                    raise ProtocolConfigurationError(
                        f"Cannot derive protocol_type from handler_name '{self.handler_name}': "
                        f"result would be empty. Either provide an explicit protocol_type "
                        f"or use a handler_name with content after the 'handler-' prefix."
                    )
                self.protocol_type = derived
            else:
                self.protocol_type = self.handler_name

        # Final guard: ensure protocol_type is never empty
        if not self.protocol_type:
            raise ProtocolConfigurationError(
                f"protocol_type cannot be empty. handler_name='{self.handler_name}', "
                f"derived protocol_type='{self.protocol_type}'. "
                f"Provide a non-empty protocol_type or handler_name."
            )

        # Log successful contract validation for debugging
        logger.debug(
            "Handler contract validated successfully",
            extra={
                "handler_name": self.handler_name,
                "handler_class": self.handler_class,
                "protocol_type": self.protocol_type,
                "handler_type": self.handler_type.value,
                "handler_version": str(self.handler_version),
            },
        )

        return self


__all__ = ["ModelHandlerContract"]
