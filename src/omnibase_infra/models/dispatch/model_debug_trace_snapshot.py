# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Debug trace snapshot model.

This module defines the serializable trace metadata snapshot that replaces
live envelope references in the dispatch contract.

Design Rationale:
    The dispatch boundary is a **serialization boundary**. Anything crossing
    this boundary must be:
    - Loggable (observability)
    - Replayable (debugging)
    - Transportable (Kafka, distributed dispatch)
    - Inspectable offline (audit pipelines)

    Live Python object references violate all of these requirements.
    This snapshot model provides the trace metadata handlers need for
    debugging without creating dependencies on internal envelope types.

Warning:
    This model exists solely for debugging and observability.
    It must **never** be used for business logic.

    The data in this snapshot is:
    - Non-authoritative (may not reflect the complete envelope state)
    - Metadata-only (does not include payload content)
    - Immutable (cannot be used to modify the original envelope)

.. versionadded:: 0.2.8
    Added as part of OMN-1518 - Strict JSON-safe dispatch contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDebugTraceSnapshot(BaseModel):
    """Serializable, non-authoritative trace snapshot.

    This model captures trace metadata from the original envelope for
    debugging and observability purposes. It replaces live envelope
    references to maintain transport-safe dispatch contracts.

    All fields are optional because:
    1. Not all envelopes have all metadata fields
    2. Extraction may fail gracefully without blocking dispatch
    3. Debug data should never cause handler failures

    Attributes:
        event_type: The event type identifier (e.g., "UserCreated").
        correlation_id: Correlation ID for distributed tracing (serialized UUID).
        trace_id: Trace ID for span correlation (serialized UUID).
        causation_id: Causation ID linking to parent event (serialized UUID).
        topic: The Kafka/event topic this message was received on.
        timestamp: When the event was created (ISO 8601 format).
        partition_key: The partition key for Kafka routing.

    Example:
        >>> snapshot = ModelDebugTraceSnapshot(
        ...     correlation_id="550e8400-e29b-41d4-a716-446655440000",
        ...     trace_id="660e8400-e29b-41d4-a716-446655440001",
        ...     topic="dev.user.events.v1",
        ...     timestamp="2025-01-27T12:00:00Z",
        ... )
        >>> snapshot.model_dump()
        {'event_type': None, 'correlation_id': '550e8400-...', ...}

    Warning:
        This snapshot is **non-authoritative**. It exists solely for
        debugging and observability. Do not use it for business logic.

    .. versionadded:: 0.2.8
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    event_type: str | None = Field(
        default=None,
        description="Event type identifier (e.g., 'UserCreated').",
    )

    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for distributed tracing (serialized UUID).",
    )

    trace_id: str | None = Field(
        default=None,
        description="Trace ID for span correlation (serialized UUID).",
    )

    causation_id: str | None = Field(
        default=None,
        description="Causation ID linking to parent event (serialized UUID).",
    )

    topic: str | None = Field(
        default=None,
        description="Event topic this message was received on.",
    )

    timestamp: str | None = Field(
        default=None,
        description="Event creation timestamp (ISO 8601 format).",
    )

    partition_key: str | None = Field(
        default=None,
        description="Partition key for Kafka routing.",
    )
