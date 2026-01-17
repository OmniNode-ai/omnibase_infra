# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Context Model for Time Injection Control.

This module provides the dispatch context model that enforces ONEX architecture
rules for time injection. In ONEX:

- **Reducers** and **Compute** nodes (pure/deterministic) must NEVER receive `now`
- **Orchestrators**, **Effects**, and **Runtime Hosts** CAN receive `now`

The ModelDispatchContext provides factory methods that enforce these rules at
dispatch time, preventing accidental time injection into reducers.

Design Pattern:
    ModelDispatchContext is an immutable data model that carries dispatch metadata
    through the runtime. It includes:
    - Correlation tracking (correlation_id, trace_id)
    - Time injection (now) - only for non-reducer node types
    - Node kind classification (REDUCER, ORCHESTRATOR, EFFECT, COMPUTE)
    - Optional metadata for extensibility

    Factory methods enforce time injection rules at construction time:
    - for_reducer() - Creates context WITHOUT time injection
    - for_compute() - Creates context WITHOUT time injection
    - for_orchestrator() - Creates context WITH time injection
    - for_effect() - Creates context WITH time injection
    - for_runtime_host() - Creates context WITH time injection

Thread Safety:
    ModelDispatchContext is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> from omnibase_infra.models.dispatch import ModelDispatchContext
    >>>
    >>> # Create context for a reducer (no time injection allowed)
    >>> reducer_ctx = ModelDispatchContext.for_reducer(
    ...     correlation_id=uuid4(),
    ...     trace_id=uuid4(),
    ... )
    >>> assert reducer_ctx.now is None
    >>> assert not reducer_ctx.has_time_injection
    >>>
    >>> # Create context for an orchestrator (time injection allowed)
    >>> orchestrator_ctx = ModelDispatchContext.for_orchestrator(
    ...     correlation_id=uuid4(),
    ...     now=datetime.now(UTC),
    ... )
    >>> assert orchestrator_ctx.now is not None
    >>> assert orchestrator_ctx.has_time_injection

Time Capture Semantics:
    The ``now`` field represents the time when the dispatch context was created
    (dispatch time), NOT when the handler begins execution. This is important
    for understanding the timing model:

    1. MessageDispatchEngine creates ModelDispatchContext with ``now=datetime.now(UTC)``
    2. Context is passed to the dispatcher function
    3. Dispatcher function begins execution (microseconds to milliseconds later)

    For most applications, this timing difference is negligible. If your handler
    requires sub-millisecond timing precision, capture ``datetime.now(UTC)`` at
    the start of your handler function rather than relying on ``context.now``.

See Also:
    omnibase_core.enums.EnumNodeKind: Node type classification
    omnibase_infra.models.dispatch.ModelDispatcherRegistration: Dispatcher metadata
    omnibase_infra.runtime.service_message_dispatch_engine: Uses this context for dispatch
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.models.dispatch.model_dispatch_metadata import ModelDispatchMetadata


class ModelDispatchContext(BaseModel):
    """
    Dispatch context carrying correlation and time injection metadata.

    Enforces ONEX architecture rules for time injection:
    - Reducers must NEVER receive `now` (deterministic execution)
    - Orchestrators and Effects CAN receive `now` (time-dependent decisions)

    Use factory methods (for_reducer, for_compute, for_orchestrator, for_effect,
    for_runtime_host) to ensure proper time injection rules are enforced at
    construction time.

    Attributes:
        correlation_id: Unique identifier for request tracing across services.
        trace_id: Optional trace identifier for distributed tracing systems.
        now: Injected current time (None for reducers, required validation).
        node_kind: The ONEX node type this context is for.
        metadata: Optional additional metadata for extensibility.

    Example:
        >>> # Use factory methods for proper enforcement
        >>> ctx = ModelDispatchContext.for_reducer(correlation_id=uuid4())
        >>> ctx.validate_for_node_kind()  # Returns True
        >>>
        >>> # Direct construction is allowed but requires careful use
        >>> ctx = ModelDispatchContext(
        ...     correlation_id=uuid4(),
        ...     node_kind=EnumNodeKind.REDUCER,
        ...     now=None,  # Must be None for reducers
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Correlation Tracking ----
    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for request tracing across services.",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Optional trace identifier for distributed tracing systems.",
    )

    # ---- Time Injection ----
    now: datetime | None = Field(
        default=None,
        description=(
            "Injected current time for time-dependent operations. Represents dispatch "
            "time (when context was created), NOT handler execution time. "
            "Must be None for REDUCER and COMPUTE nodes to ensure deterministic execution."
        ),
    )

    # ---- Node Classification ----
    node_kind: EnumNodeKind = Field(
        ...,
        description="The ONEX node type this context is for.",
    )

    # ---- Optional Metadata ----
    metadata: ModelDispatchMetadata | None = Field(
        default=None,
        description="Optional additional metadata for extensibility.",
    )

    def _is_invalid_time_injection(self) -> bool:
        """Check if this context has invalid time injection for deterministic nodes.

        Returns:
            True if this is a deterministic node (REDUCER/COMPUTE) with time injection.
        """
        return (
            self.node_kind in (EnumNodeKind.REDUCER, EnumNodeKind.COMPUTE)
            and self.now is not None
        )

    @model_validator(mode="after")
    def _validate_deterministic_node_no_time(self) -> "ModelDispatchContext":
        """Validate that deterministic nodes do not receive time injection.

        This validator enforces the ONEX architecture rule that REDUCER and
        COMPUTE nodes must be deterministic and therefore cannot receive `now`.

        Raises:
            ValueError: If node_kind is REDUCER or COMPUTE and now is not None.

        Returns:
            Self if validation passes.
        """
        if self._is_invalid_time_injection():
            msg = (
                f"{self.node_kind.value.upper()} nodes cannot receive time injection. "
                f"{self.node_kind.value.capitalize()} nodes must be deterministic - "
                f"use for_{self.node_kind.value.lower()}() factory method."
            )
            raise ValueError(msg)
        return self

    @property
    def has_time_injection(self) -> bool:
        """Check if this context has time injection enabled.

        Returns:
            True if `now` is not None, False otherwise.

        Example:
            >>> ctx = ModelDispatchContext.for_reducer(correlation_id=uuid4())
            >>> ctx.has_time_injection
            False
            >>> ctx = ModelDispatchContext.for_effect(
            ...     correlation_id=uuid4(),
            ...     now=datetime.now(UTC),
            ... )
            >>> ctx.has_time_injection
            True
        """
        return self.now is not None

    def validate_for_node_kind(self) -> Literal[True]:
        """Validate that the context is appropriate for its node kind.

        This method provides explicit validation that can be called
        at dispatch time to ensure time injection rules are enforced.

        For REDUCER and COMPUTE nodes, this validates that `now` is None.
        For other node types, this always returns True.

        The return type `Literal[True]` signals that this method either
        returns True or raises an exception - it never returns False.

        Returns:
            Literal[True]: Always returns True if validation passes.

        Raises:
            ValueError: If node_kind is REDUCER or COMPUTE and now is not None.

        Example:
            >>> ctx = ModelDispatchContext.for_reducer(correlation_id=uuid4())
            >>> ctx.validate_for_node_kind()
            True
            >>> # Invalid context would raise ValueError
        """
        if self._is_invalid_time_injection():
            msg = (
                f"Dispatch context validation failed: "
                f"{self.node_kind.value.upper()} nodes cannot receive time injection "
                f"(now={self.now}). {self.node_kind.value.capitalize()} nodes must be deterministic."
            )
            raise ValueError(msg)
        return True

    # ---- Factory Methods ----

    @classmethod
    def for_reducer(
        cls,
        correlation_id: UUID,
        trace_id: UUID | None = None,
        metadata: ModelDispatchMetadata | None = None,
    ) -> "ModelDispatchContext":
        """Create dispatch context for a REDUCER node.

        Reducers are pure state aggregators that must be deterministic.
        This factory method enforces that NO time injection is provided.

        Args:
            correlation_id: Unique identifier for request tracing.
            trace_id: Optional trace identifier for distributed tracing.
            metadata: Optional additional metadata for extensibility.

        Returns:
            ModelDispatchContext configured for REDUCER execution.

        Example:
            >>> ctx = ModelDispatchContext.for_reducer(
            ...     correlation_id=uuid4(),
            ...     trace_id=uuid4(),
            ...     metadata=ModelDispatchMetadata(source_node="kafka"),
            ... )
            >>> assert ctx.now is None
            >>> assert ctx.node_kind == EnumNodeKind.REDUCER
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            now=None,
            node_kind=EnumNodeKind.REDUCER,
            metadata=metadata,
        )

    @classmethod
    def for_orchestrator(
        cls,
        correlation_id: UUID,
        now: datetime,
        trace_id: UUID | None = None,
        metadata: ModelDispatchMetadata | None = None,
    ) -> "ModelDispatchContext":
        """Create dispatch context for an ORCHESTRATOR node.

        Orchestrators coordinate workflows and can make time-dependent
        decisions. This factory method includes time injection.

        Args:
            correlation_id: Unique identifier for request tracing.
            now: Current time for time-dependent decisions.
            trace_id: Optional trace identifier for distributed tracing.
            metadata: Optional additional metadata for extensibility.

        Returns:
            ModelDispatchContext configured for ORCHESTRATOR execution.

        Example:
            >>> from datetime import datetime, UTC
            >>> ctx = ModelDispatchContext.for_orchestrator(
            ...     correlation_id=uuid4(),
            ...     now=datetime.now(UTC),
            ... )
            >>> assert ctx.now is not None
            >>> assert ctx.node_kind == EnumNodeKind.ORCHESTRATOR
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            now=now,
            node_kind=EnumNodeKind.ORCHESTRATOR,
            metadata=metadata,
        )

    @classmethod
    def for_effect(
        cls,
        correlation_id: UUID,
        now: datetime,
        trace_id: UUID | None = None,
        metadata: ModelDispatchMetadata | None = None,
    ) -> "ModelDispatchContext":
        """Create dispatch context for an EFFECT node.

        Effects handle external I/O operations and can make time-dependent
        decisions (e.g., TTL calculations, timeout handling).
        This factory method includes time injection.

        Args:
            correlation_id: Unique identifier for request tracing.
            now: Current time for time-dependent decisions.
            trace_id: Optional trace identifier for distributed tracing.
            metadata: Optional additional metadata for extensibility.

        Returns:
            ModelDispatchContext configured for EFFECT execution.

        Example:
            >>> from datetime import datetime, UTC
            >>> ctx = ModelDispatchContext.for_effect(
            ...     correlation_id=uuid4(),
            ...     now=datetime.now(UTC),
            ...     metadata=ModelDispatchMetadata(target_node="database"),
            ... )
            >>> assert ctx.now is not None
            >>> assert ctx.node_kind == EnumNodeKind.EFFECT
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            now=now,
            node_kind=EnumNodeKind.EFFECT,
            metadata=metadata,
        )

    @classmethod
    def for_compute(
        cls,
        correlation_id: UUID,
        trace_id: UUID | None = None,
        metadata: ModelDispatchMetadata | None = None,
    ) -> "ModelDispatchContext":
        """Create dispatch context for a COMPUTE node.

        Compute nodes are pure transformations that must be deterministic.
        Like reducers, this factory method enforces that NO time injection
        is provided.

        Args:
            correlation_id: Unique identifier for request tracing.
            trace_id: Optional trace identifier for distributed tracing.
            metadata: Optional additional metadata for extensibility.

        Returns:
            ModelDispatchContext configured for COMPUTE execution.

        Example:
            >>> ctx = ModelDispatchContext.for_compute(
            ...     correlation_id=uuid4(),
            ...     trace_id=uuid4(),
            ...     metadata=ModelDispatchMetadata(routing_decision="sha256"),
            ... )
            >>> assert ctx.now is None
            >>> assert ctx.node_kind == EnumNodeKind.COMPUTE
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            now=None,
            node_kind=EnumNodeKind.COMPUTE,
            metadata=metadata,
        )

    @classmethod
    def for_runtime_host(
        cls,
        correlation_id: UUID,
        now: datetime,
        trace_id: UUID | None = None,
        metadata: ModelDispatchMetadata | None = None,
    ) -> "ModelDispatchContext":
        """Create dispatch context for a RUNTIME_HOST node.

        Runtime hosts are infrastructure components that require time injection
        for operational decisions (e.g., health checks, monitoring, scheduling).
        This factory method includes time injection.

        Args:
            correlation_id: Unique identifier for request tracing.
            now: Current time for infrastructure operations.
            trace_id: Optional trace identifier for distributed tracing.
            metadata: Optional additional metadata for extensibility.

        Returns:
            ModelDispatchContext configured for RUNTIME_HOST execution.

        Example:
            >>> from datetime import datetime, UTC
            >>> ctx = ModelDispatchContext.for_runtime_host(
            ...     correlation_id=uuid4(),
            ...     now=datetime.now(UTC),
            ...     metadata=ModelDispatchMetadata(source_node="infra-hub-1"),
            ... )
            >>> assert ctx.now is not None
            >>> assert ctx.node_kind == EnumNodeKind.RUNTIME_HOST
        """
        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            now=now,
            node_kind=EnumNodeKind.RUNTIME_HOST,
            metadata=metadata,
        )


__all__ = ["ModelDispatchContext"]
