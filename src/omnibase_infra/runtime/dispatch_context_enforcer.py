# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Context Enforcer.

Enforces ONEX time injection rules at dispatch time based on dispatcher's node kind.
This is a critical component for maintaining ONEX architectural invariants:

- **Reducers** NEVER receive `now` (deterministic execution required)
- **Orchestrators** ALWAYS receive `now` (time-dependent coordination)
- **Effects** ALWAYS receive `now` (time-dependent I/O operations)
- **Compute** NEVER receives `now` (pure transformation, deterministic)

Design Pattern:
    The DispatchContextEnforcer acts as a factory that creates appropriately
    configured ModelDispatchContext instances based on the dispatcher's node_kind.
    It extracts correlation metadata from the envelope and enforces time injection
    rules at context creation time.

    This separation of concerns keeps the MessageDispatchEngine focused on routing
    while delegating context creation and validation to this specialized component.

Thread Safety:
    DispatchContextEnforcer is stateless and thread-safe. All methods can be
    called concurrently without synchronization.

Typing Note (ModelEventEnvelope[object]):
    Methods in this class use ``ModelEventEnvelope[object]`` instead of ``Any``
    per CLAUDE.md guidance: "Use ``object`` for generic payloads".

    This is intentional:
    - CLAUDE.md mandates "NEVER use ``Any``" for type annotations
    - Context creation extracts correlation metadata from envelopes regardless
      of payload type; the enforcer operates on envelope metadata, not content
    - ``object`` signals "any object payload" while maintaining type safety

Related:
    - OMN-973: Time injection enforcement at dispatch
    - ModelDispatchContext: The context model with factory methods
    - ProtocolMessageDispatcher: Protocol defining dispatcher node_kind

Example:
    >>> from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer
    >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
    >>>
    >>> enforcer = DispatchContextEnforcer()
    >>>
    >>> # For a reducer dispatcher - NO time injection
    >>> ctx = enforcer.create_context_for_dispatcher(reducer_dispatcher, envelope)
    >>> assert ctx.now is None
    >>>
    >>> # For an orchestrator dispatcher - WITH time injection
    >>> ctx = enforcer.create_context_for_dispatcher(orchestrator_dispatcher, envelope)
    >>> assert ctx.now is not None

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = ["DispatchContextEnforcer"]

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.enums import EnumCoreErrorCode, EnumNodeKind
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.runtime.registry_dispatcher import ProtocolMessageDispatcher

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope


class DispatchContextEnforcer:
    """
    Enforces time injection rules based on dispatcher's node kind.

    Creates appropriately configured dispatch contexts for each dispatcher,
    ensuring that:
    - Reducers and Compute nodes NEVER receive time injection
    - Orchestrators and Effects ALWAYS receive time injection

    This enforcer provides a single point of enforcement for ONEX's time
    injection rules, preventing accidental violations that could compromise
    reducer determinism.

    Key Invariants:
        - Reducer dispatchers receive context with `now=None`
        - Compute dispatchers receive context with `now=None`
        - Orchestrator dispatchers receive context with `now=<current_time>`
        - Effect dispatchers receive context with `now=<current_time>`

    Example:
        >>> enforcer = DispatchContextEnforcer()
        >>>
        >>> # Create context for reducer (no time)
        >>> reducer_ctx = enforcer.create_context_for_dispatcher(
        ...     reducer_dispatcher, envelope
        ... )
        >>> assert reducer_ctx.now is None
        >>>
        >>> # Create context for orchestrator (with time)
        >>> orch_ctx = enforcer.create_context_for_dispatcher(
        ...     orchestrator_dispatcher, envelope
        ... )
        >>> assert orch_ctx.now is not None

    Thread Safety:
        This class is stateless and thread-safe. All methods can be called
        concurrently without synchronization.

    .. versionadded:: 0.5.0
    """

    def create_context_for_node_kind(
        self,
        node_kind: EnumNodeKind,
        envelope: ModelEventEnvelope[object],
        dispatcher_id: str,
    ) -> ModelDispatchContext:
        """
        Create appropriate dispatch context based on node_kind.

        This is the core method that implements time injection rules. Both
        ``create_context_for_dispatcher()`` and external callers (e.g.,
        ``MessageDispatchEngine._create_context_for_entry()``) delegate to
        this method to ensure a single source of truth for context creation.

        Time Injection Rules:
            - REDUCER: No time injection (deterministic state aggregation)
            - COMPUTE: No time injection (pure transformation)
            - ORCHESTRATOR: With time injection (coordination)
            - EFFECT: With time injection (I/O operations)
            - RUNTIME_HOST: With time injection (infrastructure)

        Args:
            node_kind: The ONEX node kind determining time injection behavior.
            envelope: The event envelope containing correlation metadata.
            dispatcher_id: Identifier for the dispatcher (used in error messages).

        Returns:
            ModelDispatchContext configured appropriately for the node kind.

        Raises:
            ModelOnexError: If node_kind is unrecognized (INTERNAL_ERROR).

        Time Semantics:
            The ``now`` field is captured at context creation time (dispatch time),
            NOT at handler execution time. For ORCHESTRATOR, EFFECT, and RUNTIME_HOST
            nodes, this means:

            - ``now`` represents when the context was created
            - Handler execution may occur microseconds to milliseconds later
            - For most use cases, this drift is negligible
            - If sub-millisecond precision is required, handlers should capture
              their own time at the start of execution

        Example:
            >>> ctx = enforcer.create_context_for_node_kind(
            ...     EnumNodeKind.REDUCER, envelope, "my-dispatcher"
            ... )
            >>> assert ctx.now is None  # No time for reducers

        .. versionadded:: 0.5.0

        .. versionchanged:: 0.5.x
            Error code for unrecognized node_kind changed from VALIDATION_ERROR
            to INTERNAL_ERROR. This reflects that unhandled cases are internal
            implementation errors (missing switch cases), not validation failures.
        """
        # Extract correlation metadata from envelope
        correlation_id = envelope.correlation_id or uuid4()
        trace_id = envelope.trace_id

        # Route to appropriate factory based on node kind
        if node_kind == EnumNodeKind.REDUCER:
            return ModelDispatchContext.for_reducer(
                correlation_id=correlation_id,
                trace_id=trace_id,
            )

        if node_kind == EnumNodeKind.COMPUTE:
            return ModelDispatchContext.for_compute(
                correlation_id=correlation_id,
                trace_id=trace_id,
            )

        if node_kind == EnumNodeKind.ORCHESTRATOR:
            # Timestamp captured at context creation (dispatch time).
            # Drift from actual handler execution is microseconds in practice.
            return ModelDispatchContext.for_orchestrator(
                correlation_id=correlation_id,
                trace_id=trace_id,
                now=datetime.now(UTC),
            )

        if node_kind == EnumNodeKind.EFFECT:
            return ModelDispatchContext.for_effect(
                correlation_id=correlation_id,
                trace_id=trace_id,
                now=datetime.now(UTC),
            )

        if node_kind == EnumNodeKind.RUNTIME_HOST:
            return ModelDispatchContext.for_runtime_host(
                correlation_id=correlation_id,
                trace_id=trace_id,
                now=datetime.now(UTC),
            )

        # Unknown node kind - should never happen as all EnumNodeKind values are handled above.
        # If we reach here, a new enum value was added without updating this switch.
        raise ModelOnexError(
            message=f"Unhandled node_kind '{node_kind}' for dispatcher "
            f"'{dispatcher_id}'. This is an internal error - missing case handler.",
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
        )

    def create_context_for_dispatcher(
        self,
        dispatcher: ProtocolMessageDispatcher,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchContext:
        """
        Create appropriate dispatch context based on dispatcher's node_kind.

        Examines the dispatcher's node_kind and creates a context with or
        without time injection according to ONEX rules:

        - REDUCER: No time injection (deterministic)
        - COMPUTE: No time injection (pure transformation)
        - ORCHESTRATOR: With time injection (coordination)
        - EFFECT: With time injection (I/O operations)
        - RUNTIME_HOST: With time injection (infrastructure)

        Args:
            dispatcher: The dispatcher to create context for.
            envelope: The event envelope containing correlation metadata.

        Returns:
            ModelDispatchContext configured appropriately for the node kind.

        Raises:
            ModelOnexError: If node_kind is unrecognized (INTERNAL_ERROR).

        Example:
            >>> ctx = enforcer.create_context_for_dispatcher(dispatcher, envelope)
            >>> if dispatcher.node_kind == EnumNodeKind.REDUCER:
            ...     assert ctx.now is None
            ... else:
            ...     assert ctx.now is not None

        .. versionadded:: 0.5.0
        """
        return self.create_context_for_node_kind(
            node_kind=dispatcher.node_kind,
            envelope=envelope,
            dispatcher_id=dispatcher.dispatcher_id,
        )

    def validate_no_time_injection_for_reducer(
        self,
        context: ModelDispatchContext,
    ) -> None:
        """
        Validate that a reducer context does not have time injection.

        This method provides an explicit validation checkpoint that can be
        called before dispatching to a reducer to ensure no time injection
        has occurred (e.g., via manual context construction).

        Args:
            context: The dispatch context to validate.

        Raises:
            ModelOnexError: If context has time injection for a reducer
                (VALIDATION_FAILED).

        Example:
            >>> # This will raise if context has now != None
            >>> enforcer.validate_no_time_injection_for_reducer(reducer_ctx)

        .. versionadded:: 0.5.0
        """
        if context.node_kind == EnumNodeKind.REDUCER and context.now is not None:
            raise ModelOnexError(
                message="REDUCER nodes cannot receive time injection. "
                f"Dispatch context has now={context.now} but reducers must be "
                "deterministic. This is an ONEX architectural violation.",
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            )

    def validate_no_time_injection_for_compute(
        self,
        context: ModelDispatchContext,
    ) -> None:
        """
        Validate that a compute context does not have time injection.

        This method provides an explicit validation checkpoint that can be
        called before dispatching to a compute node to ensure no time injection
        has occurred (e.g., via manual context construction).

        Args:
            context: The dispatch context to validate.

        Raises:
            ModelOnexError: If context has time injection for a compute node
                (VALIDATION_FAILED).

        Example:
            >>> # This will raise if context has now != None
            >>> enforcer.validate_no_time_injection_for_compute(compute_ctx)

        .. versionadded:: 0.5.0
        """
        if context.node_kind == EnumNodeKind.COMPUTE and context.now is not None:
            raise ModelOnexError(
                message="COMPUTE nodes cannot receive time injection. "
                f"Dispatch context has now={context.now} but compute nodes must be "
                "deterministic. This is an ONEX architectural violation.",
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            )

    def validate_no_time_injection_for_deterministic_node(
        self,
        context: ModelDispatchContext,
    ) -> None:
        """
        Validate that a deterministic node context does not have time injection.

        This method provides an explicit validation checkpoint that can be
        called before dispatching to a REDUCER or COMPUTE node to ensure no
        time injection has occurred (e.g., via manual context construction).

        Args:
            context: The dispatch context to validate.

        Raises:
            ModelOnexError: If context has time injection for a REDUCER or COMPUTE
                node (VALIDATION_FAILED).

        Example:
            >>> # This will raise if context has now != None for deterministic nodes
            >>> enforcer.validate_no_time_injection_for_deterministic_node(ctx)

        .. versionadded:: 0.5.0
        """
        if (
            context.node_kind in (EnumNodeKind.REDUCER, EnumNodeKind.COMPUTE)
            and context.now is not None
        ):
            raise ModelOnexError(
                message=f"{context.node_kind.value.upper()} nodes cannot receive time injection. "
                f"Dispatch context has now={context.now} but {context.node_kind.value.lower()} "
                "nodes must be deterministic. This is an ONEX architectural violation.",
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            )

    def requires_time_injection(self, node_kind: EnumNodeKind) -> bool:
        """
        Check if a node kind requires time injection.

        This method can be used for logging, metrics, or conditional logic
        based on whether time injection is expected for a given node kind.

        Args:
            node_kind: The ONEX node kind to check.

        Returns:
            True if the node kind requires time injection, False otherwise.

        Time Injection Requirements:
            - REDUCER: False (deterministic, no time)
            - COMPUTE: False (pure transformation, no time)
            - ORCHESTRATOR: True (coordination needs time)
            - EFFECT: True (I/O operations need time)
            - RUNTIME_HOST: True (infrastructure needs time)

        Example:
            >>> enforcer.requires_time_injection(EnumNodeKind.REDUCER)
            False
            >>> enforcer.requires_time_injection(EnumNodeKind.ORCHESTRATOR)
            True

        .. versionadded:: 0.5.0
        """
        return node_kind in {
            EnumNodeKind.ORCHESTRATOR,
            EnumNodeKind.EFFECT,
            EnumNodeKind.RUNTIME_HOST,
        }

    def forbids_time_injection(self, node_kind: EnumNodeKind) -> bool:
        """
        Check if a node kind forbids time injection.

        This is the inverse of requires_time_injection and is useful for
        validation logic that needs to explicitly check for forbidden cases.

        Args:
            node_kind: The ONEX node kind to check.

        Returns:
            True if the node kind forbids time injection, False otherwise.

        Time Injection Forbidden:
            - REDUCER: True (must be deterministic)
            - COMPUTE: True (must be pure transformation)
            - ORCHESTRATOR: False (needs time)
            - EFFECT: False (needs time)
            - RUNTIME_HOST: False (needs time)

        Example:
            >>> enforcer.forbids_time_injection(EnumNodeKind.REDUCER)
            True
            >>> enforcer.forbids_time_injection(EnumNodeKind.EFFECT)
            False

        .. versionadded:: 0.5.0
        """
        return node_kind in {
            EnumNodeKind.REDUCER,
            EnumNodeKind.COMPUTE,
        }
