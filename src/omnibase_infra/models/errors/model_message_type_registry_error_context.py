# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Message Type Registry Error Context Model.

This module defines the configuration model for message type registry error context,
encapsulating registry-specific fields to reduce __init__ parameter count
while maintaining strong typing per ONEX standards.

Related:
    - OMN-937: Central Message Type Registry implementation
    - MessageTypeRegistryError: Error class that uses this model

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumMessageCategory


class ModelMessageTypeRegistryErrorContext(BaseModel):
    """Configuration model for message type registry error context.

    Encapsulates registry-specific fields for message type registry errors
    to reduce __init__ parameter count while maintaining strong typing.
    This follows the ONEX pattern of using configuration models to
    bundle related parameters.

    Attributes:
        message_type: The message type that caused the error (if applicable)
        domain: The domain involved in the error (if applicable)
        category: The category involved in the error (if applicable)

    Example:
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>> registry_context = ModelMessageTypeRegistryErrorContext(
        ...     message_type="UnknownType",
        ...     domain="myapp.events",
        ...     category=EnumMessageCategory.EVENT,
        ... )
        >>> raise MessageTypeRegistryError(
        ...     "Message type not found",
        ...     registry_context=registry_context,
        ... )

    .. versionadded:: 0.5.0
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    message_type: str | None = Field(
        default=None,
        description="The message type that caused the error (if applicable)",
    )
    domain: str | None = Field(
        default=None,
        description="The domain involved in the error (if applicable)",
    )
    category: EnumMessageCategory | None = Field(
        default=None,
        description="The category involved in the error (if applicable)",
    )


__all__ = ["ModelMessageTypeRegistryErrorContext"]
