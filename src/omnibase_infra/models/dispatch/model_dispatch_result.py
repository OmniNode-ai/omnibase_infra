# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Result Model.

Represents the result of a dispatch operation, including status, timing metrics,
and any outputs produced by the dispatcher. Used for observability, debugging,
and result propagation in the dispatch engine.

Design Pattern:
    ModelDispatchResult is a pure data model that captures the complete outcome
    of a dispatch operation:
    - Status (success, error, timeout, etc.)
    - Timing metrics (duration, timestamps)
    - Dispatcher outputs (for successful dispatches)
    - Error information (for failed dispatches)
    - Tracing context (correlation IDs, trace IDs)

    This model is produced by the dispatch engine after each dispatch operation
    and can be used for logging, metrics collection, and error handling.

Thread Safety:
    ModelDispatchResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

JsonType Recursion Fix (OMN-1274):
    The ``error_details`` field uses ``dict[str, object]`` instead of the
    recursive ``JsonType`` type alias. Here is why:

    **The Original Problem:**
    ``JsonType`` was a recursive type alias::

        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None

    Pydantic 2.x performs eager schema generation at class definition time,
    causing ``RecursionError`` when expanding recursive type aliases::

        JsonType -> dict[str, JsonType] | list[JsonType] | ...
                 -> dict[str, dict[str, JsonType] | ...] | ...
                 -> ... (infinite recursion)

    **Why dict[str, object] is Correct for error_details:**
    Error details are structured diagnostic information, always represented as
    key-value dictionaries (e.g., ``{"retry_count": 3, "service": "db"}``).
    They do NOT need to support arrays or primitives at the root level.

    Using ``dict[str, object]`` provides:
    - Correct semantics: Error details are always dictionaries
    - Type safety: Pydantic validates the outer structure
    - No recursion: ``object`` avoids recursive type expansion

    **Caveats:**
    - Values are typed as ``object`` (no static type checking)
    - For fields needing full JSON support, use ``JsonType`` from
      ``omnibase_core.types`` (now fixed via TypeAlias pattern)

Example:
    >>> from omnibase_infra.models.dispatch import (
    ...     ModelDispatchResult,
    ...     ModelDispatchOutputs,
    ...     EnumDispatchStatus,
    ... )
    >>> from uuid import uuid4
    >>> from datetime import datetime, UTC
    >>>
    >>> # Create a successful dispatch result
    >>> result = ModelDispatchResult(
    ...     dispatch_id=uuid4(),
    ...     status=EnumDispatchStatus.SUCCESS,
    ...     route_id="user-events-route",
    ...     dispatcher_id="user-event-dispatcher",
    ...     topic="dev.user.events.v1",
    ...     message_type="UserCreatedEvent",
    ...     duration_ms=45.2,
    ...     outputs=ModelDispatchOutputs(topics=["dev.notification.commands.v1"]),
    ... )
    >>>
    >>> result.is_successful()
    True

See Also:
    omnibase_infra.models.dispatch.ModelDispatchRoute: Routing rule model
    omnibase_infra.models.dispatch.EnumDispatchStatus: Dispatch status enum
    ADR: ``docs/decisions/adr-any-type-pydantic-workaround.md`` (historical)
    Pydantic issue: https://github.com/pydantic/pydantic/issues/3278
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_metadata import ModelDispatchMetadata
from omnibase_infra.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs


class ModelDispatchResult(BaseModel):
    """
    Result of a dispatch operation.

    Captures the complete outcome of routing a message to a dispatcher,
    including status, timing, outputs, and error information.

    Attributes:
        dispatch_id: Unique identifier for this dispatch operation.
        status: The outcome status of the dispatch operation.
        route_id: Identifier of the route that was matched (if any).
        dispatcher_id: Identifier of the dispatcher that was invoked (if any).
        topic: The topic the message was dispatched to.
        message_category: The category of the dispatched message.
        message_type: The specific type of the message (if known).
        duration_ms: Time taken for the dispatch operation in milliseconds.
        started_at: Timestamp when the dispatch started.
        completed_at: Timestamp when the dispatch completed.
        outputs: List of topics where dispatcher outputs were published.
        output_count: Number of outputs produced by the dispatcher.
        error_message: Error message if the dispatch failed.
        error_code: Error code if the dispatch failed.
        error_details: Additional JSON-serializable error details for debugging.
        retry_count: Number of times this dispatch was retried.
        correlation_id: Correlation ID from the original message.
        trace_id: Distributed trace ID for observability.
        span_id: Trace span ID for this dispatch operation.
        metadata: Optional additional metadata about the dispatch.

    Example:
        >>> result = ModelDispatchResult(
        ...     dispatch_id=uuid4(),
        ...     status=EnumDispatchStatus.HANDLER_ERROR,
        ...     route_id="order-route",
        ...     dispatcher_id="order-dispatcher",
        ...     topic="dev.order.commands.v1",
        ...     message_category=EnumMessageCategory.COMMAND,
        ...     error_message="Database connection failed",
        ...     error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
        ... )
        >>> result.is_error()
        True
        >>> result.requires_retry()
        False
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Dispatch Identity ----
    dispatch_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this dispatch operation.",
    )

    # ---- Status ----
    status: EnumDispatchStatus = Field(
        ...,
        description="The outcome status of the dispatch operation.",
    )

    # ---- Route and Dispatcher Info ----
    route_id: str | None = Field(
        default=None,
        description="Identifier of the route that was matched (if any).",
    )
    dispatcher_id: str | None = Field(
        default=None,
        description="Identifier of the dispatcher that was invoked (if any).",
    )

    # ---- Message Info ----
    topic: str = Field(
        ...,
        description="The topic the message was dispatched to.",
        min_length=1,
    )
    message_category: EnumMessageCategory | None = Field(
        default=None,
        description="The category of the dispatched message.",
    )
    message_type: str | None = Field(
        default=None,
        description="The specific type of the message (if known).",
    )

    # ---- Timing Metrics ----
    duration_ms: float | None = Field(
        default=None,
        description="Time taken for the dispatch operation in milliseconds.",
        ge=0,
    )
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    started_at: datetime = Field(
        ...,
        description="Timestamp when the dispatch started (UTC, must be explicitly provided).",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the dispatch completed (UTC).",
    )

    # ---- Dispatcher Outputs ----
    outputs: ModelDispatchOutputs | None = Field(
        default=None,
        description="Validated output topics where dispatcher outputs were published.",
    )
    output_count: int = Field(
        default=0,
        description="Number of outputs produced by the dispatcher.",
        ge=0,
    )
    output_events: list[BaseModel] = Field(
        default_factory=list,
        description=(
            "List of output events produced by the dispatcher that need to be "
            "published to output_topic. These are raw Pydantic models that will "
            "be wrapped in ModelEventEnvelope by the kernel before publishing."
        ),
    )

    # ---- Error Information ----
    error_message: str | None = Field(
        default=None,
        description="Error message if the dispatch failed.",
    )
    error_code: EnumCoreErrorCode | None = Field(
        default=None,
        description="Error code if the dispatch failed.",
    )
    error_details: dict[str, object] = Field(
        default_factory=dict,
        description="Additional JSON-serializable error details for debugging.",
    )

    # ---- Retry Information ----
    retry_count: int = Field(
        default=0,
        description="Number of times this dispatch was retried.",
        ge=0,
    )

    # ---- Tracing Context ----
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID from the original message (auto-generated if not provided).",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Distributed trace ID for observability.",
    )
    span_id: UUID | None = Field(
        default=None,
        description="Trace span ID for this dispatch operation.",
    )

    # ---- Optional Metadata ----
    metadata: ModelDispatchMetadata | None = Field(
        default=None,
        description="Optional additional metadata about the dispatch.",
    )

    def is_successful(self) -> bool:
        """
        Check if this dispatch was successful.

        Returns:
            True if status is SUCCESS, False otherwise

        Example:
            >>> from datetime import datetime, UTC
            >>> result = ModelDispatchResult(
            ...     dispatch_id=uuid4(),
            ...     status=EnumDispatchStatus.SUCCESS,
            ...     topic="test.events",
            ...     started_at=datetime.now(UTC),
            ... )
            >>> result.is_successful()
            True
        """
        return self.status.is_successful()

    def is_error(self) -> bool:
        """
        Check if this dispatch resulted in an error.

        Returns:
            True if the status represents an error condition, False otherwise

        Example:
            >>> from datetime import datetime, UTC
            >>> result = ModelDispatchResult(
            ...     dispatch_id=uuid4(),
            ...     status=EnumDispatchStatus.HANDLER_ERROR,
            ...     topic="test.events",
            ...     started_at=datetime.now(UTC),
            ...     error_message="Dispatcher failed",
            ... )
            >>> result.is_error()
            True
        """
        return self.status.is_error()

    def requires_retry(self) -> bool:
        """
        Check if this dispatch should be retried.

        Returns:
            True if the status indicates a retriable failure, False otherwise

        Example:
            >>> from datetime import datetime, UTC
            >>> result = ModelDispatchResult(
            ...     dispatch_id=uuid4(),
            ...     status=EnumDispatchStatus.TIMEOUT,
            ...     topic="test.events",
            ...     started_at=datetime.now(UTC),
            ... )
            >>> result.requires_retry()
            True
        """
        return self.status.requires_retry()

    def is_terminal(self) -> bool:
        """
        Check if this dispatch is in a terminal state.

        Returns:
            True if the dispatch has completed (success or failure), False otherwise
        """
        return self.status.is_terminal()

    def with_error(
        self,
        status: EnumDispatchStatus,
        message: str,
        code: EnumCoreErrorCode | None = None,
        details: dict[str, object] | None = None,
    ) -> "ModelDispatchResult":
        """
        Create a new result with error information.

        Args:
            status: The error status
            message: Error message
            code: Optional error code (EnumCoreErrorCode)
            details: Optional JSON-serializable error details

        Returns:
            New ModelDispatchResult with error information

        Example:
            >>> from datetime import datetime, UTC
            >>> result = ModelDispatchResult(
            ...     dispatch_id=uuid4(),
            ...     status=EnumDispatchStatus.ROUTED,
            ...     topic="test.events",
            ...     started_at=datetime.now(UTC),
            ... )
            >>> error_result = result.with_error(
            ...     EnumDispatchStatus.HANDLER_ERROR,
            ...     "Dispatcher failed",
            ...     code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
            ... )
        """
        return self.model_copy(
            update={
                "status": status,
                "error_message": message,
                "error_code": code,
                "error_details": details if details is not None else {},
                "completed_at": datetime.now(UTC),
            }
        )

    def with_success(
        self,
        outputs: ModelDispatchOutputs | None = None,
        output_count: int | None = None,
    ) -> "ModelDispatchResult":
        """
        Create a new result marked as successful.

        Args:
            outputs: Optional ModelDispatchOutputs with validated output topics
            output_count: Optional count of outputs (defaults to len(outputs))

        Returns:
            New ModelDispatchResult marked as SUCCESS

        Example:
            >>> from datetime import datetime, UTC
            >>> result = ModelDispatchResult(
            ...     dispatch_id=uuid4(),
            ...     status=EnumDispatchStatus.ROUTED,
            ...     topic="test.events",
            ...     started_at=datetime.now(UTC),
            ... )
            >>> success_result = result.with_success(
            ...     outputs=ModelDispatchOutputs(topics=["output.topic.v1"]),
            ...     output_count=1,
            ... )
        """
        resolved_outputs: ModelDispatchOutputs = (
            outputs if outputs is not None else ModelDispatchOutputs()
        )
        count = output_count if output_count is not None else len(resolved_outputs)
        return self.model_copy(
            update={
                "status": EnumDispatchStatus.SUCCESS,
                "outputs": resolved_outputs,
                "output_count": count,
                "completed_at": datetime.now(UTC),
            }
        )

    def with_duration(self, duration_ms: float) -> "ModelDispatchResult":
        """
        Create a new result with duration set.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            New ModelDispatchResult with duration set
        """
        return self.model_copy(
            update={
                "duration_ms": duration_ms,
                "completed_at": datetime.now(UTC),
            }
        )


__all__ = ["ModelDispatchResult"]
