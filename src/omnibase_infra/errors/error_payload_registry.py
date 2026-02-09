# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Payload Registry Error.

Provides the PayloadRegistryError class for payload registry operations.

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
    - RegistryPayload: Registry that raises this error

.. versionadded:: 0.7.0
"""

from __future__ import annotations

__all__ = [
    "PayloadRegistryError",
]

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class PayloadRegistryError(RuntimeHostError):
    """Error raised when payload registry operations fail.

    Used for:
    - Unregistered (payload_type, version) lookups
    - Registration after freeze
    - Invalid model class registration
    - Duplicate registration attempts

    Extends RuntimeHostError for consistency with infrastructure error patterns.

    Example:
        >>> from omnibase_infra.errors import PayloadRegistryError
        >>> try:
        ...     model_cls = registry.resolve("UnknownType", "1.0.0")
        ... except PayloadRegistryError as e:
        ...     print(f"Payload type not found: {e}")
        ...

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        message: str,
        *,
        payload_type: str | None = None,
        version: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize PayloadRegistryError.

        Args:
            message: Human-readable error message
            payload_type: The payload type that caused the error
            version: The version that caused the error
            context: Bundled infrastructure context for correlation_id
            **extra_context: Additional context information
        """
        extra: dict[str, object] = dict(extra_context)
        if payload_type is not None:
            extra["payload_type"] = payload_type
        if version is not None:
            extra["version"] = version

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            context=context,
            **extra,
        )
