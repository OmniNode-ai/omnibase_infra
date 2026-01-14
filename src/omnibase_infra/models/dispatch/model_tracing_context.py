# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Tracing Context Model for Dispatch Operations.

This module provides the ModelTracingContext model that groups distributed tracing
fields (correlation_id, trace_id, span_id) into a single sub-model. This reduces
the number of optional fields in parent models like ModelDispatchResult.

Design Pattern:
    ModelTracingContext is a pure data model that captures tracing metadata for
    distributed request tracking:
    - correlation_id: Links related operations across services
    - trace_id: Distributed trace identifier (e.g., OpenTelemetry)
    - span_id: Span within a trace for this specific operation

    This model uses sentinel values instead of nullable unions to minimize
    union count in the codebase (OMN-1002).

Sentinel Values:
    - Nil UUID (00000000-0000-0000-0000-000000000000) means "not set"
    - Use the ``has_*`` properties to check if a field has been set

Thread Safety:
    ModelTracingContext is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Union Reduction:
    This model replaces three separate ``UUID | None`` fields with a single
    ``ModelTracingContext | None`` field in parent models, reducing union count
    by 2 per parent model that uses it.

Example:
    >>> from uuid import uuid4
    >>> from omnibase_infra.models.dispatch import ModelTracingContext
    >>>
    >>> # Create tracing context
    >>> ctx = ModelTracingContext(
    ...     correlation_id=uuid4(),
    ...     trace_id=uuid4(),
    ...     span_id=uuid4(),
    ... )
    >>>
    >>> # Check if fields are set
    >>> ctx.has_correlation_id
    True
    >>> ctx.has_trace_id
    True
    >>>
    >>> # Create from event envelope
    >>> ctx = ModelTracingContext.from_envelope(envelope)

See Also:
    omnibase_infra.models.dispatch.ModelDispatchResult: Uses this for tracing
    omnibase_infra.models.dispatch.ModelDispatchContext: Context with time injection

.. versionadded:: 0.7.0
    Added as part of OMN-1004 Optional Field Audit (Task 4.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

# Sentinel UUID for "not set" state
_SENTINEL_UUID: UUID = UUID(int=0)  # Nil UUID (00000000-0000-0000-0000-000000000000)


class ModelTracingContext(BaseModel):
    """
    Tracing context for distributed request tracking.

    Groups correlation_id, trace_id, and span_id into a single model to reduce
    the number of optional fields in parent models. Uses sentinel values (nil UUID)
    instead of None to minimize union count.

    Sentinel Values:
        - Nil UUID (00000000-0000-0000-0000-000000000000) means "not set"
        - Use ``has_correlation_id``, ``has_trace_id``, ``has_span_id`` to check

    Null Coercion:
        Constructors accept ``None`` for any field and convert to the sentinel value.

    Attributes:
        correlation_id: Correlation ID linking related operations. Nil UUID if not set.
        trace_id: Distributed trace ID (e.g., OpenTelemetry). Nil UUID if not set.
        span_id: Span ID within the trace. Nil UUID if not set.

    Example:
        >>> from uuid import uuid4
        >>> ctx = ModelTracingContext(
        ...     correlation_id=uuid4(),
        ...     trace_id=uuid4(),
        ... )
        >>> ctx.has_correlation_id
        True
        >>> ctx.has_span_id  # Not set
        False

    .. versionadded:: 0.7.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Tracing Fields ----
    correlation_id: UUID = Field(
        default=_SENTINEL_UUID,
        description="Correlation ID linking related operations. Nil UUID if not set.",
    )
    trace_id: UUID = Field(
        default=_SENTINEL_UUID,
        description="Distributed trace ID (e.g., OpenTelemetry). Nil UUID if not set.",
    )
    span_id: UUID = Field(
        default=_SENTINEL_UUID,
        description="Span ID within the trace. Nil UUID if not set.",
    )

    # ---- Validators for None-to-Sentinel Conversion ----
    @field_validator("correlation_id", "trace_id", "span_id", mode="before")
    @classmethod
    def _convert_none_to_uuid_sentinel(cls, v: object) -> UUID:
        """Convert None to nil UUID sentinel for null coercion."""
        if v is None:
            return _SENTINEL_UUID
        if isinstance(v, UUID):
            return v
        # Handle string UUID conversion
        return UUID(str(v))

    # ---- Sentinel Check Properties ----
    @property
    def has_correlation_id(self) -> bool:
        """Check if correlation_id is set (not nil UUID)."""
        return self.correlation_id != _SENTINEL_UUID

    @property
    def has_trace_id(self) -> bool:
        """Check if trace_id is set (not nil UUID)."""
        return self.trace_id != _SENTINEL_UUID

    @property
    def has_span_id(self) -> bool:
        """Check if span_id is set (not nil UUID)."""
        return self.span_id != _SENTINEL_UUID

    @property
    def is_empty(self) -> bool:
        """Check if all tracing fields are unset (all nil UUIDs)."""
        return (
            not self.has_correlation_id
            and not self.has_trace_id
            and not self.has_span_id
        )

    # ---- Factory Methods ----
    @classmethod
    def empty(cls) -> ModelTracingContext:
        """Create an empty tracing context with all fields unset.

        Returns:
            ModelTracingContext with all fields set to nil UUID.

        Example:
            >>> ctx = ModelTracingContext.empty()
            >>> ctx.is_empty
            True

        .. versionadded:: 0.7.0
        """
        return cls()

    @classmethod
    def from_envelope(
        cls,
        envelope: ModelEventEnvelope[object],
    ) -> ModelTracingContext:
        """Create tracing context from an event envelope.

        Extracts correlation_id, trace_id, and span_id from the envelope.
        Missing fields are set to nil UUID sentinel.

        Typing Note:
            Uses ``ModelEventEnvelope[object]`` instead of ``Any`` per CLAUDE.md
            guidance. This method extracts tracing metadata regardless of payload
            type; the ``object`` type parameter signals any payload is accepted.

        Args:
            envelope: The event envelope to extract tracing info from.

        Returns:
            ModelTracingContext populated from the envelope.

        Example:
            >>> ctx = ModelTracingContext.from_envelope(envelope)
            >>> ctx.has_correlation_id
            True

        .. versionadded:: 0.7.0
        """
        return cls(
            correlation_id=envelope.correlation_id,
            trace_id=envelope.trace_id,
            span_id=envelope.span_id,
        )

    @classmethod
    def from_uuids(
        cls,
        correlation_id: UUID | None = None,
        trace_id: UUID | None = None,
        span_id: UUID | None = None,
    ) -> ModelTracingContext:
        """Create tracing context from individual UUIDs.

        Convenience factory for creating context from optional UUID values.
        None values are converted to nil UUID sentinel.

        Args:
            correlation_id: Correlation ID. None means not set.
            trace_id: Trace ID. None means not set.
            span_id: Span ID. None means not set.

        Returns:
            ModelTracingContext with provided values.

        Example:
            >>> from uuid import uuid4
            >>> ctx = ModelTracingContext.from_uuids(
            ...     correlation_id=uuid4(),
            ... )
            >>> ctx.has_correlation_id
            True
            >>> ctx.has_trace_id
            False

        .. versionadded:: 0.7.0
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
        )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary with string values for logging.

        Returns a dictionary containing only fields that are set (non-nil UUID),
        with UUID values converted to strings.

        Returns:
            Dictionary with string keys and string values.

        Example:
            >>> from uuid import uuid4
            >>> ctx = ModelTracingContext(correlation_id=uuid4())
            >>> d = ctx.to_dict()
            >>> "correlation_id" in d
            True
            >>> "trace_id" in d  # Not set
            False

        .. versionadded:: 0.7.0
        """
        result: dict[str, str] = {}
        if self.has_correlation_id:
            result["correlation_id"] = str(self.correlation_id)
        if self.has_trace_id:
            result["trace_id"] = str(self.trace_id)
        if self.has_span_id:
            result["span_id"] = str(self.span_id)
        return result


__all__ = ["ModelTracingContext"]
