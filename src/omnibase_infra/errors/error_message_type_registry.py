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
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.errors.model_message_type_registry_error_context import (
    ModelMessageTypeRegistryErrorContext,
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
        >>> from omnibase_infra.models.errors import ModelMessageTypeRegistryErrorContext
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>> from uuid import uuid4
        >>> try:
        ...     handlers = registry.get_handlers("UnknownType", category, domain)
        ... except MessageTypeRegistryError as e:
        ...     print(f"Handler not found: {e}")
        ...
        >>> # With correlation context and registry context
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     correlation_id=uuid4(),
        ...     operation="get_handlers",
        ... )
        >>> registry_context = ModelMessageTypeRegistryErrorContext(
        ...     message_type="UnknownType",
        ...     domain="myapp.events",
        ...     category=EnumMessageCategory.EVENT,
        ... )
        >>> raise MessageTypeRegistryError(
        ...     "Message type not found",
        ...     registry_context=registry_context,
        ...     context=context,
        ... )

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        message: str,
        registry_context: ModelMessageTypeRegistryErrorContext | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize MessageTypeRegistryError.

        Args:
            message: Human-readable error message
            registry_context: Bundled registry-specific context (message_type, domain, category)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Build extra context dict from registry context
        extra: dict[str, object] = dict(extra_context)
        if registry_context is not None:
            if registry_context.message_type is not None:
                extra["message_type"] = registry_context.message_type
            if registry_context.domain is not None:
                extra["domain"] = registry_context.domain
            if registry_context.category is not None:
                extra["category"] = registry_context.category.value

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            context=context,
            **extra,
        )
