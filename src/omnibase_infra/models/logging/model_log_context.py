# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Structured Log Context Model.

This module provides the ModelLogContext Pydantic model for structured logging
in ONEX infrastructure components. It replaces the common union pattern
``dict[str, str | int | float]`` with a strongly-typed model that maintains
flexibility while providing type safety.

Design Pattern:
    ModelLogContext is a structured data model that carries logging context
    through infrastructure operations. It includes:
    - Operation identification (operation, service_name)
    - Correlation tracking (correlation_id)
    - Performance metrics (duration_ms, retry_count)
    - Extensible context (extra dict with string values)

    The model provides builder methods for common use cases and a ``to_dict()``
    method for integration with standard logging formatters.

Thread Safety:
    ModelLogContext is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Sentinel Values:
    This model uses sentinel values instead of nullable unions to minimize
    union count in the codebase (OMN-1002). The sentinel convention is:
    - Empty string ("") for string fields means "not set"
    - -1.0 for float fields means "not set"
    - -1 for int fields means "not set"

    Use the ``has_*`` properties to check if a field has been set.

Union Reduction:
    This model replaces the pattern::

        def build_log_context() -> dict[str, str | int | float]:
            return {
                "operation": "dispatch",
                "correlation_id": str(uuid),
                "duration_ms": 42.5,
                "retry_count": 3,
            }

    With a structured approach that:
    - Eliminates unions in favor of typed fields with sentinel defaults
    - Provides IDE autocompletion and type checking
    - Integrates with standard logging via ``to_dict()``
    - Ensures consistent log context structure across components

Example:
    >>> from uuid import uuid4
    >>> from omnibase_infra.models.logging import ModelLogContext
    >>>
    >>> # Create context using builder methods
    >>> ctx = (
    ...     ModelLogContext.for_operation("dispatch")
    ...     .with_correlation_id(str(uuid4()))
    ...     .with_duration_ms(42.5)
    ...     .with_retry_count(3)
    ...     .with_service_name("kafka-event-bus")
    ... )
    >>>
    >>> # Use in logging
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Event dispatched", extra=ctx.to_dict())
    >>>
    >>> # Direct construction
    >>> ctx = ModelLogContext(
    ...     operation="subscribe",
    ...     service_name="kafka",
    ...     correlation_id=str(uuid4()),
    ... )

See Also:
    omnibase_infra.models.dispatch.ModelDispatchContext: Dispatch-specific context
    omnibase_infra.models.dispatch.ModelDispatchResult: Dispatch result with metrics

.. versionadded:: 0.6.0
    Added as part of OMN-1002 Union Reduction Phase 2.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import PrimitiveValue
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# Sentinel constants for "not set" values
_SENTINEL_STR: str = ""
_SENTINEL_FLOAT: float = -1.0
_SENTINEL_INT: int = -1


class ModelLogContext(BaseModel):
    """
    Structured logging context for ONEX infrastructure operations.

    Provides type-safe logging context that replaces the common pattern of
    ``dict[str, str | int | float]`` with explicit fields and builder methods.

    Sentinel Values:
        This model uses sentinel values to indicate "not set" instead of None:
        - Empty string ("") for string fields
        - -1.0 for duration_ms
        - -1 for retry_count

        Use ``has_*`` properties to check if a field is set.

    Attributes:
        operation: The operation being logged (e.g., "dispatch", "connect", "subscribe").
        correlation_id: Correlation ID for request tracing. Empty string if not set.
        duration_ms: Duration of the operation in milliseconds. -1.0 if not set.
        retry_count: Number of retries attempted. -1 if not set.
        service_name: Name of the service performing the operation. Empty string if not set.
        topic: Topic name for event-related operations. Empty string if not set.
        group_id: Consumer group ID for subscription operations. Empty string if not set.
        error_type: Error type name when logging errors. Empty string if not set.
        extra: Additional string-typed context fields.

    Example:
        >>> ctx = ModelLogContext(
        ...     operation="dispatch",
        ...     service_name="kafka-event-bus",
        ...     duration_ms=42.5,
        ... )
        >>> ctx.to_dict()
        {'operation': 'dispatch', 'service_name': 'kafka-event-bus', 'duration_ms': 42.5}

    .. versionadded:: 0.6.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Required Fields ----
    operation: str = Field(
        ...,
        description="The operation being logged (e.g., 'dispatch', 'connect', 'subscribe').",
        min_length=1,
    )

    # ---- Correlation Tracking ----
    correlation_id: str = Field(
        default=_SENTINEL_STR,
        description="Correlation ID for request tracing. Empty string if not set.",
    )

    # ---- Performance Metrics ----
    duration_ms: float = Field(
        default=_SENTINEL_FLOAT,
        description="Duration in milliseconds. -1.0 if not set. Must be >= 0 when set.",
    )

    retry_count: int = Field(
        default=_SENTINEL_INT,
        description="Number of retries attempted. -1 if not set. Must be >= 0 when set.",
    )

    # ---- Service Identification ----
    service_name: str = Field(
        default=_SENTINEL_STR,
        description="Name of the service performing the operation. Empty string if not set.",
    )

    # ---- Event/Topic Context ----
    topic: str = Field(
        default=_SENTINEL_STR,
        description="Topic name for event-related operations. Empty string if not set.",
    )

    group_id: str = Field(
        default=_SENTINEL_STR,
        description="Consumer group ID for subscription operations. Empty string if not set.",
    )

    # ---- Error Context ----
    error_type: str = Field(
        default=_SENTINEL_STR,
        description="Error type name when logging errors. Empty string if not set.",
    )

    # ---- Extensible Context ----
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Additional string-typed context fields for extensibility.",
    )

    # ---- Sentinel Check Properties ----

    @property
    def has_correlation_id(self) -> bool:
        """Check if correlation_id is set (not empty string)."""
        return self.correlation_id != _SENTINEL_STR

    @property
    def has_duration_ms(self) -> bool:
        """Check if duration_ms is set (not -1.0)."""
        return self.duration_ms >= 0

    @property
    def has_retry_count(self) -> bool:
        """Check if retry_count is set (not -1)."""
        return self.retry_count >= 0

    @property
    def has_service_name(self) -> bool:
        """Check if service_name is set (not empty string)."""
        return self.service_name != _SENTINEL_STR

    @property
    def has_topic(self) -> bool:
        """Check if topic is set (not empty string)."""
        return self.topic != _SENTINEL_STR

    @property
    def has_group_id(self) -> bool:
        """Check if group_id is set (not empty string)."""
        return self.group_id != _SENTINEL_STR

    @property
    def has_error_type(self) -> bool:
        """Check if error_type is set (not empty string)."""
        return self.error_type != _SENTINEL_STR

    def to_dict(self) -> dict[str, PrimitiveValue]:
        """
        Convert to dictionary for use with logging formatters.

        Returns a dictionary containing only fields that are set (excludes
        sentinel values), suitable for passing to ``logging.Logger`` methods
        via the ``extra`` parameter.

        This method provides integration with standard logging interfaces that
        expect ``dict[str, str | int | float]`` for logging context.

        Returns:
            Dictionary with string keys and string/int/float values.
            Only includes fields that are actually set (not sentinel values).

        Example:
            >>> ctx = ModelLogContext(
            ...     operation="dispatch",
            ...     duration_ms=42.5,
            ...     retry_count=3,
            ... )
            >>> ctx.to_dict()
            {'operation': 'dispatch', 'duration_ms': 42.5, 'retry_count': 3}

        .. versionadded:: 0.6.0
        """
        result: dict[str, PrimitiveValue] = {"operation": self.operation}

        if self.has_correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.has_duration_ms:
            result["duration_ms"] = self.duration_ms
        if self.has_retry_count:
            result["retry_count"] = self.retry_count
        if self.has_service_name:
            result["service_name"] = self.service_name
        if self.has_topic:
            result["topic"] = self.topic
        if self.has_group_id:
            result["group_id"] = self.group_id
        if self.has_error_type:
            result["error_type"] = self.error_type

        # Add extra fields
        result.update(self.extra)

        return result

    # ---- Builder Methods ----

    @classmethod
    def for_operation(cls, operation: str) -> ModelLogContext:
        """
        Create a log context for a specific operation.

        This is the primary entry point for building log contexts using
        the fluent builder pattern.

        Args:
            operation: The operation being logged.

        Returns:
            A new ModelLogContext instance.

        Example:
            >>> ctx = ModelLogContext.for_operation("dispatch")
            >>> ctx.operation
            'dispatch'

        .. versionadded:: 0.6.0
        """
        return cls(operation=operation)

    def with_correlation_id(self, correlation_id: str) -> ModelLogContext:
        """
        Add correlation ID to the context.

        Args:
            correlation_id: Correlation ID for request tracing.

        Returns:
            A new ModelLogContext with correlation_id set.

        Example:
            >>> from uuid import uuid4
            >>> ctx = (
            ...     ModelLogContext.for_operation("dispatch")
            ...     .with_correlation_id(str(uuid4()))
            ... )

        .. versionadded:: 0.6.0
        """
        return self.model_copy(update={"correlation_id": correlation_id})

    def with_duration_ms(self, duration_ms: float) -> ModelLogContext:
        """
        Add duration metric to the context.

        Args:
            duration_ms: Duration in milliseconds (must be >= 0).

        Returns:
            A new ModelLogContext with duration_ms set.

        Raises:
            ProtocolConfigurationError: If duration_ms is negative.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("query")
            ...     .with_duration_ms(15.3)
            ... )

        .. versionadded:: 0.6.0
        """
        if duration_ms < 0:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="with_duration_ms",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "duration_ms must be >= 0", context=context
            )
        return self.model_copy(update={"duration_ms": duration_ms})

    def with_retry_count(self, retry_count: int) -> ModelLogContext:
        """
        Add retry count to the context.

        Args:
            retry_count: Number of retries (must be >= 0).

        Returns:
            A new ModelLogContext with retry_count set.

        Raises:
            ProtocolConfigurationError: If retry_count is negative.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("connect")
            ...     .with_retry_count(3)
            ... )

        .. versionadded:: 0.6.0
        """
        if retry_count < 0:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="with_retry_count",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "retry_count must be >= 0", context=context
            )
        return self.model_copy(update={"retry_count": retry_count})

    def with_service_name(self, service_name: str) -> ModelLogContext:
        """
        Add service name to the context.

        Args:
            service_name: Name of the service.

        Returns:
            A new ModelLogContext with service_name set.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("publish")
            ...     .with_service_name("kafka-event-bus")
            ... )

        .. versionadded:: 0.6.0
        """
        return self.model_copy(update={"service_name": service_name})

    def with_topic(self, topic: str) -> ModelLogContext:
        """
        Add topic name to the context.

        Args:
            topic: Topic name for event operations.

        Returns:
            A new ModelLogContext with topic set.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("subscribe")
            ...     .with_topic("dev.user.events.v1")
            ... )

        .. versionadded:: 0.6.0
        """
        return self.model_copy(update={"topic": topic})

    def with_group_id(self, group_id: str) -> ModelLogContext:
        """
        Add consumer group ID to the context.

        Args:
            group_id: Consumer group ID.

        Returns:
            A new ModelLogContext with group_id set.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("consume")
            ...     .with_group_id("user-service-consumers")
            ... )

        .. versionadded:: 0.6.0
        """
        return self.model_copy(update={"group_id": group_id})

    def with_error_type(self, error_type: str) -> ModelLogContext:
        """
        Add error type to the context.

        Args:
            error_type: Name of the error type (e.g., class name).

        Returns:
            A new ModelLogContext with error_type set.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("connect")
            ...     .with_error_type("InfraConnectionError")
            ... )

        .. versionadded:: 0.6.0
        """
        return self.model_copy(update={"error_type": error_type})

    def with_extra(self, key: str, value: str) -> ModelLogContext:
        """
        Add an extra context field.

        Use this method for context fields that don't fit the standard fields.
        The value must be a string to maintain type consistency.

        Args:
            key: The context field name.
            value: The context field value (must be string).

        Returns:
            A new ModelLogContext with the extra field added.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("dispatch")
            ...     .with_extra("dispatcher_id", "user-event-dispatcher")
            ...     .with_extra("route_id", "user-route")
            ... )

        .. versionadded:: 0.6.0
        """
        new_extra = dict(self.extra)
        new_extra[key] = value
        return self.model_copy(update={"extra": new_extra})

    def with_extras(self, extras: dict[str, str]) -> ModelLogContext:
        """
        Add multiple extra context fields at once.

        Args:
            extras: Dictionary of extra context fields (all values must be strings).

        Returns:
            A new ModelLogContext with the extra fields added.

        Example:
            >>> ctx = (
            ...     ModelLogContext.for_operation("batch_process")
            ...     .with_extras({
            ...         "batch_id": "batch-123",
            ...         "batch_size": "100",  # Note: must be string
            ...     })
            ... )

        .. versionadded:: 0.6.0
        """
        new_extra = dict(self.extra)
        new_extra.update(extras)
        return self.model_copy(update={"extra": new_extra})

    # ---- Factory Methods for Common Patterns ----

    @classmethod
    def for_event_bus(
        cls,
        operation: str,
        service_name: str,
        *,
        topic: str = _SENTINEL_STR,
        group_id: str = _SENTINEL_STR,
        correlation_id: str = _SENTINEL_STR,
    ) -> ModelLogContext:
        """
        Create log context for event bus operations.

        Factory method for common event bus logging patterns.

        Args:
            operation: The event bus operation (e.g., "publish", "subscribe", "start").
            service_name: Name of the event bus service.
            topic: Topic name. Empty string if not applicable.
            group_id: Consumer group ID. Empty string if not applicable.
            correlation_id: Correlation ID. Empty string if not applicable.

        Returns:
            A ModelLogContext configured for event bus operations.

        Example:
            >>> ctx = ModelLogContext.for_event_bus(
            ...     operation="publish",
            ...     service_name="kafka-event-bus",
            ...     topic="dev.user.events.v1",
            ... )

        .. versionadded:: 0.6.0
        """
        return cls(
            operation=operation,
            service_name=service_name,
            topic=topic,
            group_id=group_id,
            correlation_id=correlation_id,
        )

    @classmethod
    def for_dispatch(
        cls,
        *,
        dispatcher_id: str = _SENTINEL_STR,
        route_id: str = _SENTINEL_STR,
        topic: str = _SENTINEL_STR,
        correlation_id: str = _SENTINEL_STR,
        duration_ms: float = _SENTINEL_FLOAT,
    ) -> ModelLogContext:
        """
        Create log context for dispatch operations.

        Factory method for message dispatch logging patterns.

        Args:
            dispatcher_id: Dispatcher identifier. Empty string if not applicable.
            route_id: Route identifier. Empty string if not applicable.
            topic: Topic name. Empty string if not applicable.
            correlation_id: Correlation ID. Empty string if not applicable.
            duration_ms: Duration in milliseconds. -1.0 if not applicable.

        Returns:
            A ModelLogContext configured for dispatch operations.

        Raises:
            ProtocolConfigurationError: If duration_ms is negative (except sentinel -1.0).

        Example:
            >>> ctx = ModelLogContext.for_dispatch(
            ...     dispatcher_id="user-dispatcher",
            ...     route_id="user-route",
            ...     topic="dev.user.events.v1",
            ...     duration_ms=15.3,
            ... )

        .. versionadded:: 0.6.0
        """
        # Validate duration_ms: allow sentinel (-1.0) or non-negative values
        if duration_ms != _SENTINEL_FLOAT and duration_ms < 0:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="for_dispatch",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "duration_ms must be >= 0 when set", context=context
            )

        extra: dict[str, str] = {}
        if dispatcher_id != _SENTINEL_STR:
            extra["dispatcher_id"] = dispatcher_id
        if route_id != _SENTINEL_STR:
            extra["route_id"] = route_id

        return cls(
            operation="dispatch",
            topic=topic,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            extra=extra,
        )

    @classmethod
    def for_connection(
        cls,
        operation: str,
        service_name: str,
        *,
        host: str = _SENTINEL_STR,
        port: int = _SENTINEL_INT,
        retry_count: int = _SENTINEL_INT,
        correlation_id: str = _SENTINEL_STR,
    ) -> ModelLogContext:
        """
        Create log context for connection operations.

        Factory method for infrastructure connection logging patterns.

        Args:
            operation: The connection operation (e.g., "connect", "disconnect", "reconnect").
            service_name: Name of the service being connected to.
            host: Host name (sanitized, no credentials). Empty string if not applicable.
            port: Port number. -1 if not applicable.
            retry_count: Retry count. -1 if not applicable.
            correlation_id: Correlation ID. Empty string if not applicable.

        Returns:
            A ModelLogContext configured for connection operations.

        Raises:
            ProtocolConfigurationError: If retry_count is negative (except sentinel -1).

        Example:
            >>> ctx = ModelLogContext.for_connection(
            ...     operation="connect",
            ...     service_name="postgresql",
            ...     host="db.example.com",
            ...     port=5432,
            ...     retry_count=2,
            ... )

        .. versionadded:: 0.6.0
        """
        # Validate retry_count: allow sentinel (-1) or non-negative values
        if retry_count != _SENTINEL_INT and retry_count < 0:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="for_connection",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "retry_count must be >= 0 when set", context=context
            )

        extra: dict[str, str] = {}
        if host != _SENTINEL_STR:
            extra["host"] = host
        if port >= 0:
            extra["port"] = str(port)

        return cls(
            operation=operation,
            service_name=service_name,
            retry_count=retry_count,
            correlation_id=correlation_id,
            extra=extra,
        )

    @classmethod
    def for_error(
        cls,
        operation: str,
        error_type: str,
        *,
        service_name: str = _SENTINEL_STR,
        correlation_id: str = _SENTINEL_STR,
        retry_count: int = _SENTINEL_INT,
    ) -> ModelLogContext:
        """
        Create log context for error logging.

        Factory method for error logging patterns.

        Args:
            operation: The operation that failed.
            error_type: Name of the error type/class.
            service_name: Service name. Empty string if not applicable.
            correlation_id: Correlation ID. Empty string if not applicable.
            retry_count: Retry count when error occurred. -1 if not applicable.

        Returns:
            A ModelLogContext configured for error logging.

        Raises:
            ProtocolConfigurationError: If retry_count is negative (except sentinel -1).

        Example:
            >>> ctx = ModelLogContext.for_error(
            ...     operation="publish",
            ...     error_type="InfraConnectionError",
            ...     service_name="kafka",
            ...     correlation_id="abc-123",
            ... )

        .. versionadded:: 0.6.0
        """
        # Validate retry_count: allow sentinel (-1) or non-negative values
        if retry_count != _SENTINEL_INT and retry_count < 0:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="for_error",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "retry_count must be >= 0 when set", context=context
            )

        return cls(
            operation=operation,
            error_type=error_type,
            service_name=service_name,
            correlation_id=correlation_id,
            retry_count=retry_count,
        )


__all__ = ["ModelLogContext"]
