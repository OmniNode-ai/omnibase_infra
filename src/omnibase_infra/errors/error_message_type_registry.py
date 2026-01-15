# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Type Registry Error.

Provides the MessageTypeRegistryError class for message type registry operations.

Related:
    - OMN-937: Central Message Type Registry implementation
    - RegistryMessageType: Registry that raises this error

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = [
    "MessageTypeRegistryError",
]

from omnibase_core.enums import EnumCoreErrorCode

from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class MessageTypeRegistryError(RuntimeHostError):
    """Error raised when message type registry operations fail.

    Used for:
    - Missing message type mappings
    - Category constraint violations
    - Domain constraint violations
    - Registration validation failures

    Extends RuntimeHostError for consistency with infrastructure error patterns.

    Example:
        >>> from omnibase_infra.errors import ModelInfraErrorContext
        >>> from uuid import uuid4
        >>> try:
        ...     handlers = registry.get_handlers("UnknownType", category, domain)
        ... except MessageTypeRegistryError as e:
        ...     print(f"Handler not found: {e}")
        ...
        >>> # With correlation context
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     correlation_id=uuid4(),
        ...     operation="get_handlers",
        ... )
        >>> raise MessageTypeRegistryError(
        ...     "Message type not found",
        ...     message_type="UnknownType",
        ...     context=context,
        ... )

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        message: str,
        message_type: str | None = None,
        domain: str | None = None,
        category: EnumMessageCategory | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize MessageTypeRegistryError.

        Args:
            message: Human-readable error message
            message_type: The message type that caused the error (if applicable)
            domain: The domain involved in the error (if applicable)
            category: The category involved in the error (if applicable)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Build extra context dict
        extra: dict[str, object] = dict(extra_context)
        if message_type is not None:
            extra["message_type"] = message_type
        if domain is not None:
            extra["domain"] = domain
        if category is not None:
            extra["category"] = category.value

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            context=context,
            **extra,
        )
