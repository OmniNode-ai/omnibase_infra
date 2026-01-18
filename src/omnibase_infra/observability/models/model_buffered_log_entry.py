# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for buffered log entries in structured logging sink.

This module defines ModelBufferedLogEntry, the structured representation of
log entries stored in the SinkLoggingStructured buffer. The model ensures
type-safe log entry handling while supporting JSON-compatible context values.

Design Decisions:
    - Uses frozen=True for thread-safety (entries are immutable once created)
    - Context values use JsonType for JSON serialization compatibility
    - Required context keys are validated at flush time, not emit time,
      to maintain hot-path performance (emit() must not block)

Thread Safety:
    ModelBufferedLogEntry is immutable (frozen=True), making instances
    thread-safe for concurrent read access during flush operations.

See Also:
    - SinkLoggingStructured: The sink that uses this model
    - ModelLoggingSinkConfig: Configuration model for the sink
    - EnumRequiredLogContextKey: Required context keys enum
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumLogLevel
from omnibase_core.types import JsonType


class ModelBufferedLogEntry(BaseModel):
    """Buffered log entry for structured logging sink.

    Represents a single log entry buffered by SinkLoggingStructured. The model
    is immutable (frozen) to ensure thread-safety when entries are accessed
    during flush operations.

    Attributes:
        level: Log level from EnumLogLevel (DEBUG, INFO, WARNING, ERROR, etc.)
        message: Log message content. Should be a complete, self-contained
            message suitable for structured logging.
        context: Structured context data for the log entry. Values may be
            any JSON-compatible type (str, int, float, bool, None, list, dict).
        timestamp: ISO-8601 timestamp captured at emit() time.

    Context Field Types:
        The context dictionary supports JSON-compatible value types to ensure
        proper serialization. While dict[str, str] was previously used for
        simplicity, JsonType allows richer context:

        ```python
        context = {
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "retry_count": 3,           # int is valid
            "duration_ms": 42.5,        # float is valid
            "is_retry": True,           # bool is valid
            "tags": ["hot-path", "db"], # list is valid
            "metadata": {"version": 1}, # dict is valid
        }
        ```

    Example:
        ```python
        from datetime import datetime, UTC
        from omnibase_core.enums import EnumLogLevel

        entry = ModelBufferedLogEntry(
            level=EnumLogLevel.INFO,
            message="Database query completed",
            context={
                "correlation_id": "abc-123",
                "duration_ms": 15.3,
                "rows_affected": 42,
            },
            timestamp=datetime.now(UTC),
        )
        ```

    .. versionadded:: 0.7.0
        Replaced NamedTuple with Pydantic model for enhanced type safety.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow EnumLogLevel
    )

    level: EnumLogLevel = Field(
        ...,
        description="Log level from EnumLogLevel enum.",
    )

    message: str = Field(
        ...,
        description="Log message content.",
        min_length=1,
    )

    context: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Structured context data with JSON-compatible values.",
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp captured at emit() time (UTC).",
    )


__all__: list[str] = [
    "ModelBufferedLogEntry",
]
