# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Registry Error Class.

This module defines the EventBusRegistryError for event bus registry operations.
"""

from typing import Any, cast

from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class EventBusRegistryError(RuntimeHostError):
    """Error raised when event bus registry operations fail.

    Used for:
    - Attempting to get an unregistered event bus kind
    - Registration failures (duplicate registration, protocol validation)
    - Event bus class validation failures during registration
    - Invalid bus_kind identifiers

    Extends RuntimeHostError as this is an infrastructure-layer runtime concern.

    Example:
        >>> from omnibase_infra.errors import EventBusRegistryError
        >>> from omnibase_infra.errors import ModelInfraErrorContext
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> from uuid import uuid4

        >>> # Bus kind not found
        >>> try:
        ...     bus_cls = registry.get("unknown_bus")
        ... except EventBusRegistryError as e:
        ...     print(f"Event bus not found: {e}")

        >>> # With context
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     operation="get_bus",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise EventBusRegistryError(
        ...     "Event bus kind not registered",
        ...     bus_kind="kafka",
        ...     context=context,
        ... )

        >>> # Protocol validation failure
        >>> raise EventBusRegistryError(
        ...     "Event bus class does not implement ProtocolEventBus",
        ...     bus_kind="custom",
        ...     bus_class="InvalidBus",
        ... )

        >>> # Duplicate registration
        >>> raise EventBusRegistryError(
        ...     "Event bus kind is already registered",
        ...     bus_kind="inmemory",
        ...     existing_class="EventBusInmemory",
        ... )
    """

    def __init__(
        self,
        message: str,
        bus_kind: str | None = None,
        bus_class: str | None = None,
        available_kinds: list[str] | None = None,
        existing_class: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize EventBusRegistryError.

        Args:
            message: Human-readable error message
            bus_kind: The bus kind that caused the error (if applicable)
            bus_class: The bus class name that caused the error (if applicable)
            available_kinds: List of currently registered bus kinds (for error context)
            existing_class: Name of already registered class (for duplicate registration)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Add domain-specific fields to extra_context if provided
        if bus_kind is not None:
            extra_context["bus_kind"] = bus_kind
        if bus_class is not None:
            extra_context["bus_class"] = bus_class
        if available_kinds is not None:
            extra_context["available_kinds"] = available_kinds
        if existing_class is not None:
            extra_context["existing_class"] = existing_class

        # NOTE: Cast required for mypy - **dict[str, object] doesn't satisfy **context: Any
        super().__init__(
            message=message,
            context=context,
            **cast("dict[str, Any]", extra_context),
        )


__all__ = ["EventBusRegistryError"]
