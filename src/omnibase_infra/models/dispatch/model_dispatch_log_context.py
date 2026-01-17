# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatch log context model for structured logging in the dispatch engine.

This model replaces the union-heavy dict pattern in MessageDispatchEngine's
``_build_log_context`` method. Instead of building a dict with 9+ nullable
parameters, this model provides typed fields with clear semantics.

The model supports two usage patterns:
1. Direct construction with all relevant fields
2. Builder pattern for incremental construction

**Sentinel Values**:
This model uses sentinel values internally instead of ``None``:
- String fields: empty string ``""`` means "not set"
- Numeric fields: ``-1`` means "not set"
- UUID fields: nil UUID (all zeros) means "not set"
- Enum fields: kept as ``None`` (unavoidable for type safety)

**Input Conversion**:
Constructors accept ``None`` and convert to sentinel values for convenience.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 2 (OMN-1002).

.. versionchanged:: 0.6.1
    Refactored to use sentinel values, reducing union count from 25 to 3.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.types import PrimitiveValue
from omnibase_infra.enums import EnumMessageCategory

# Sentinel values for "not set" state
_SENTINEL_STR: str = ""
_SENTINEL_INT: int = -1
_SENTINEL_FLOAT: float = -1.0
_SENTINEL_UUID: UUID = UUID(int=0)  # Nil UUID (00000000-0000-0000-0000-000000000000)

# Type alias for log context dict values (reduces union count)
type LogContextDict = dict[str, PrimitiveValue]


class ModelDispatchLogContext(BaseModel):
    """Structured log context for dispatch engine operations.

    This model provides a type-safe alternative to the dict-based log context.
    Uses sentinel values instead of ``None`` to minimize union types.

    **Sentinel Values**:
    - ``topic=""``: Not set
    - ``message_type=""``: Not set
    - ``dispatcher_id=""``: Not set
    - ``dispatcher_count=-1``: Not set
    - ``duration_ms=-1.0``: Not set
    - ``correlation_id=UUID(int=0)``: Not set (nil UUID)
    - ``trace_id=UUID(int=0)``: Not set (nil UUID)

    **None Handling**:
    Constructors accept ``None`` for any field and convert to the sentinel value.
    This provides a convenient API for optional fields.

    The model is designed for immutability (frozen=True) for thread-safety.

    Attributes:
        topic: The topic being dispatched to. Empty string if not set.
        category: The message category (EVENT, COMMAND, INTENT). None if not set.
        message_type: The message type being dispatched. Empty string if not set.
        dispatcher_id: Dispatcher ID or comma-separated list of IDs. Empty if not set.
        dispatcher_count: Number of dispatchers matched. -1 if not set.
        duration_ms: Dispatch duration in milliseconds. -1.0 if not set.
        correlation_id: Correlation ID from envelope. Nil UUID if not set.
        trace_id: Trace ID from envelope. Nil UUID if not set.
        error_code: Error code if dispatch failed. None if not set.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>> ctx = ModelDispatchLogContext(
        ...     topic="dev.user.events.v1",
        ...     category=EnumMessageCategory.EVENT,
        ...     message_type="UserCreatedEvent",
        ...     dispatcher_count=2,
        ...     correlation_id=uuid4(),
        ... )
        >>> log_dict = ctx.to_dict()

    .. versionadded:: 0.6.0
    .. versionchanged:: 0.6.1
        Refactored to use sentinel values instead of nullable unions.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # String fields use empty string sentinel
    topic: str = Field(
        default=_SENTINEL_STR,
        description="The topic being dispatched to. Empty string if not set.",
    )
    message_type: str = Field(
        default=_SENTINEL_STR,
        description="The message type being dispatched. Empty string if not set.",
    )
    dispatcher_id: str = Field(
        default=_SENTINEL_STR,
        description="Dispatcher ID or comma-separated list of IDs. Empty if not set.",
    )

    # Numeric fields use -1 sentinel
    dispatcher_count: int = Field(
        default=_SENTINEL_INT,
        description="Number of dispatchers matched. -1 if not set.",
    )
    duration_ms: float = Field(
        default=_SENTINEL_FLOAT,
        description="Dispatch duration in milliseconds. -1.0 if not set.",
    )

    # UUID fields use nil UUID sentinel
    correlation_id: UUID = Field(
        default=_SENTINEL_UUID,
        description="Correlation ID from envelope. Nil UUID if not set.",
    )
    trace_id: UUID = Field(
        default=_SENTINEL_UUID,
        description="Trace ID from envelope. Nil UUID if not set.",
    )

    # Enum fields keep None (unavoidable for type safety without modifying enums)
    category: EnumMessageCategory | None = Field(
        default=None,
        description="The message category (EVENT, COMMAND, INTENT).",
    )
    error_code: EnumCoreErrorCode | None = Field(
        default=None,
        description="Error code if dispatch failed.",
    )

    # Validators to convert None to sentinel values (backwards compatibility)
    # Note: Using `object` instead of unions to minimize union count per OMN-1002
    @field_validator("topic", "message_type", "dispatcher_id", mode="before")
    @classmethod
    def _convert_none_to_str_sentinel(cls, v: object) -> str:
        """Convert None to empty string sentinel."""
        if v is None:
            return _SENTINEL_STR
        if isinstance(v, str):
            return v
        return str(v)

    @field_validator("dispatcher_count", mode="before")
    @classmethod
    def _convert_none_to_int_sentinel(cls, v: object) -> int:
        """Convert None to -1 sentinel."""
        if v is None:
            return _SENTINEL_INT
        if isinstance(v, int):
            return v
        # Fallback for numeric strings - cast to int
        return int(str(v))

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _convert_none_to_float_sentinel(cls, v: object) -> float:
        """Convert None to -1.0 sentinel."""
        if v is None:
            return _SENTINEL_FLOAT
        if isinstance(v, int | float):
            return float(v)
        # Fallback for numeric strings - cast to float
        return float(str(v))

    @field_validator("correlation_id", "trace_id", mode="before")
    @classmethod
    def _convert_none_to_uuid_sentinel(cls, v: object) -> UUID:
        """Convert None to nil UUID sentinel."""
        if v is None:
            return _SENTINEL_UUID
        if isinstance(v, UUID):
            return v
        return UUID(str(v))

    def _is_set_str(self, value: str) -> bool:
        """Check if a string field is set (not sentinel)."""
        return value != _SENTINEL_STR

    def _is_set_int(self, value: int) -> bool:
        """Check if an int field is set (not sentinel)."""
        return value != _SENTINEL_INT

    def _is_set_float(self, value: float) -> bool:
        """Check if a float field is set (not sentinel)."""
        return value != _SENTINEL_FLOAT

    def _is_set_uuid(self, value: UUID) -> bool:
        """Check if a UUID field is set (not nil UUID)."""
        return value != _SENTINEL_UUID

    def to_dict(self) -> LogContextDict:
        """Convert to dictionary for use with logging formatters.

        Returns a dictionary containing only fields that are set (non-sentinel),
        suitable for passing to ``logging.Logger`` methods via the ``extra`` parameter.

        UUIDs are converted to strings for JSON serialization.
        Enums are converted to their string values.
        Sentinel values are excluded from the output.

        Returns:
            Dictionary with string keys and JSON-compatible values.

        Example:
            >>> from uuid import uuid4
            >>> ctx = ModelDispatchLogContext(
            ...     topic="dev.user.events.v1",
            ...     duration_ms=42.5,
            ...     correlation_id=uuid4(),
            ... )
            >>> d = ctx.to_dict()
            >>> "topic" in d
            True
            >>> "category" in d  # None values excluded
            False

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Updated to exclude sentinel values instead of None values.
        """
        result: LogContextDict = {}

        # String fields - exclude empty string sentinel
        if self._is_set_str(self.topic):
            result["topic"] = self.topic
        if self._is_set_str(self.message_type):
            result["message_type"] = self.message_type
        if self._is_set_str(self.dispatcher_id):
            result["dispatcher_id"] = self.dispatcher_id

        # Numeric fields - exclude -1 sentinel
        if self._is_set_int(self.dispatcher_count):
            result["dispatcher_count"] = self.dispatcher_count
        if self._is_set_float(self.duration_ms):
            # Round to 3 decimals: microsecond precision sufficient for log analysis
            result["duration_ms"] = round(self.duration_ms, 3)

        # UUID fields - exclude nil UUID sentinel
        if self._is_set_uuid(self.correlation_id):
            result["correlation_id"] = str(self.correlation_id)
        if self._is_set_uuid(self.trace_id):
            result["trace_id"] = str(self.trace_id)

        # Enum fields - exclude None (these still use unions)
        if self.category is not None:
            result["category"] = self.category.value
        if self.error_code is not None:
            result["error_code"] = self.error_code.name

        return result

    @classmethod
    def for_dispatch_start(
        cls,
        topic: str,
        category: EnumMessageCategory,
        correlation_id: UUID = _SENTINEL_UUID,
        trace_id: UUID = _SENTINEL_UUID,
    ) -> ModelDispatchLogContext:
        """Create context for dispatch start logging.

        Factory method for the common pattern of logging dispatch start.

        Args:
            topic: The topic being dispatched to.
            category: The message category.
            correlation_id: Correlation ID. Defaults to nil UUID (not set).
            trace_id: Trace ID. Defaults to nil UUID (not set).

        Returns:
            A ModelDispatchLogContext for dispatch start.

        Example:
            >>> from omnibase_infra.enums import EnumMessageCategory
            >>> ctx = ModelDispatchLogContext.for_dispatch_start(
            ...     topic="dev.user.events.v1",
            ...     category=EnumMessageCategory.EVENT,
            ... )

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Changed to use sentinel defaults instead of None.
        """
        return cls(
            topic=topic,
            category=category,
            correlation_id=correlation_id,
            trace_id=trace_id,
        )

    @classmethod
    def for_dispatch_complete(
        cls,
        topic: str,
        category: EnumMessageCategory,
        message_type: str,
        dispatcher_id: str,
        dispatcher_count: int,
        duration_ms: float,
        correlation_id: UUID = _SENTINEL_UUID,
        trace_id: UUID = _SENTINEL_UUID,
    ) -> ModelDispatchLogContext:
        """Create context for successful dispatch completion.

        Factory method for the common pattern of logging dispatch completion.

        Args:
            topic: The topic dispatched to.
            category: The message category.
            message_type: The message type.
            dispatcher_id: Comma-separated list of dispatcher IDs.
            dispatcher_count: Number of dispatchers executed.
            duration_ms: Total dispatch duration.
            correlation_id: Correlation ID. Defaults to nil UUID (not set).
            trace_id: Trace ID. Defaults to nil UUID (not set).

        Returns:
            A ModelDispatchLogContext for dispatch completion.

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Changed to use sentinel defaults instead of None.
        """
        return cls(
            topic=topic,
            category=category,
            message_type=message_type,
            dispatcher_id=dispatcher_id,
            dispatcher_count=dispatcher_count,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            trace_id=trace_id,
        )

    @classmethod
    def for_dispatch_error(
        cls,
        topic: str,
        error_code: EnumCoreErrorCode,
        duration_ms: float = _SENTINEL_FLOAT,
        category: EnumMessageCategory | None = None,
        message_type: str = _SENTINEL_STR,
        dispatcher_id: str = _SENTINEL_STR,
        dispatcher_count: int = _SENTINEL_INT,
        correlation_id: UUID = _SENTINEL_UUID,
        trace_id: UUID = _SENTINEL_UUID,
    ) -> ModelDispatchLogContext:
        """Create context for dispatch error logging.

        Factory method for the common pattern of logging dispatch errors.

        Args:
            topic: The topic that failed.
            error_code: The error code.
            duration_ms: Duration before failure. Defaults to -1.0 (not set).
            category: Message category. Defaults to None (not set).
            message_type: Message type. Defaults to empty string (not set).
            dispatcher_id: Dispatcher ID. Defaults to empty string (not set).
            dispatcher_count: Dispatcher count. Defaults to -1 (not set).
            correlation_id: Correlation ID. Defaults to nil UUID (not set).
            trace_id: Trace ID. Defaults to nil UUID (not set).

        Returns:
            A ModelDispatchLogContext for dispatch error.

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Changed to use sentinel defaults instead of None.
        """
        return cls(
            topic=topic,
            category=category,
            message_type=message_type,
            dispatcher_id=dispatcher_id,
            dispatcher_count=dispatcher_count,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            trace_id=trace_id,
            error_code=error_code,
        )


__all__ = ["ModelDispatchLogContext"]
