# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chain-Aware Dispatch for Correlation and Causation Chain Validation.

Provides dispatch wrappers and helper functions that enforce correlation and
causation chain validation before message dispatch. This ensures all messages
in a workflow maintain proper traceability for distributed debugging.

Design Principles:
    - **Wrapper Pattern**: ChainAwareDispatcher wraps MessageDispatchEngine without
      modifying it, adding chain validation as a cross-cutting concern.
    - **Helper Functions**: Standalone functions for envelope creation and validation
      can be used independently of the wrapper for flexible integration.
    - **Fail-Fast**: Chain violations are detected and raised before dispatch,
      preventing malformed messages from entering the event stream.

Chain Requirements:
    1. **Correlation Propagation**: Child messages must inherit parent's correlation_id
    2. **Causation Chain**: Child's causation_id must equal parent's envelope_id
    3. **New Message ID**: Child must have its own unique envelope_id

Thread Safety:
    ChainAwareDispatcher is stateless (only holds references to engine and validator).
    All validation and helper functions are pure functions that produce fresh objects.
    Thread safety depends on the wrapped MessageDispatchEngine's thread safety.

Typing Note (ModelEventEnvelope[object]):
    Functions in this module use ``ModelEventEnvelope[object]`` instead of ``Any``
    per CLAUDE.md guidance: "Use ``object`` for generic payloads".

    This is intentional:
    - CLAUDE.md mandates "NEVER use ``Any``" for type annotations
    - Chain validation functions must work with envelopes containing any payload
      type, as they validate correlation/causation chains regardless of content
    - The ``object`` type parameter signals "any object payload" while maintaining
      type safety (unlike ``Any`` which disables type checking)
    - When creating child envelopes with specific payload types, use the generic
      ``create_child_envelope[T]()`` method which preserves type information

Usage:
    >>> from omnibase_infra.runtime import ChainAwareDispatcher, MessageDispatchEngine
    >>> from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    >>>
    >>> # Create chain-aware dispatcher
    >>> engine = MessageDispatchEngine()
    >>> engine.freeze()  # After registration
    >>> dispatcher = ChainAwareDispatcher(engine)
    >>>
    >>> # Create child envelope from parent
    >>> child = dispatcher.create_child_envelope(parent, payload=MyEvent(...))
    >>>
    >>> # Dispatch with chain validation
    >>> result = await dispatcher.dispatch_with_parent(
    ...     topic="dev.events.v1",
    ...     envelope=child,
    ...     parent_envelope=parent,
    ... )

Related:
    - OMN-951: Enforce Correlation and Causation Chain Validation
    - ChainPropagationValidator: Core validation logic
    - MessageDispatchEngine: Underlying dispatch engine

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = [
    "ChainAwareDispatcher",
    "propagate_chain_context",
    "validate_dispatch_chain",
]

from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

from omnibase_infra.errors.error_chain_propagation import ChainPropagationError
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.validation.validator_chain_propagation import (
    ChainPropagationValidator,
    get_correlation_id,
    get_message_id,
)

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

# TypeVar for payload types when creating child envelopes
T = TypeVar("T")


def propagate_chain_context(
    parent: ModelEventEnvelope[object],
    child: ModelEventEnvelope[object],
) -> ModelEventEnvelope[object]:
    """Propagate chain context from parent to child envelope.

    Creates a new envelope with the same content as the child but with
    proper chain context copied from the parent:
    - correlation_id: Copied from parent (workflow trace)
    - causation_id: Set to parent's envelope_id (parent-child link)

    This function does NOT modify the original envelopes. It returns a new
    envelope with updated chain fields.

    Args:
        parent: The parent message envelope containing the source chain context.
        child: The child message envelope to receive chain context.

    Returns:
        A new ModelEventEnvelope with chain context properly set.
        The envelope_id remains the child's original envelope_id.

    Example:
        >>> parent = ModelEventEnvelope(payload=ParentEvent(...), correlation_id=uuid4())
        >>> child = ModelEventEnvelope(payload=ChildEvent(...))
        >>> linked_child = propagate_chain_context(parent, child)
        >>> linked_child.correlation_id == parent.correlation_id
        True
        >>> linked_child.causation_id == parent.envelope_id
        True

    Note:
        If the child already has chain context that matches the parent,
        the returned envelope will be identical to the input child.
    """
    parent_message_id = get_message_id(parent)
    parent_correlation_id = get_correlation_id(parent)

    # Build update dict for chain context
    updates: dict[str, UUID | None] = {}

    # Propagate correlation_id from parent
    if parent_correlation_id is not None:
        updates["correlation_id"] = parent_correlation_id

    # Set causation_id to parent's message_id (envelope_id)
    updates["causation_id"] = parent_message_id

    # Create new envelope with updated chain context
    # model_copy creates a shallow copy with specified field updates
    return child.model_copy(update=updates)


def validate_dispatch_chain(
    parent: ModelEventEnvelope[object],
    child: ModelEventEnvelope[object],
) -> None:
    """Validate chain propagation and raise error if invalid.

    Validates that the child envelope properly maintains chain context from
    the parent. This is a convenience function that combines validation and
    error raising for use in dispatch handlers.

    Validation checks:
        1. Child's correlation_id matches parent's correlation_id
        2. Child's causation_id equals parent's envelope_id

    Args:
        parent: The parent message envelope.
        child: The child message envelope to validate.

    Raises:
        ChainPropagationError: If any chain violations are detected.
            Contains the list of violations for inspection.

    Example:
        >>> try:
        ...     validate_dispatch_chain(parent, child)
        ...     # Chain is valid, proceed with dispatch
        ...     await engine.dispatch(topic, child)
        ... except ChainPropagationError as e:
        ...     logger.error("Chain validation failed: %s", e.format_violations_for_logging())
        ...     raise

    Note:
        This function uses the default singleton validator for efficiency.
        For custom validation behavior, use ChainPropagationValidator directly.
    """
    validator = ChainPropagationValidator()
    violations = validator.validate_chain(parent, child)

    if violations:
        # Get correlation_id for error context
        correlation_id = get_correlation_id(parent)

        # Build error context
        context = ModelInfraErrorContext(
            operation="validate_dispatch_chain",
            correlation_id=correlation_id,
        )

        raise ChainPropagationError(
            message="Dispatch chain validation failed",
            violations=violations,
            context=context,
        )


class ChainAwareDispatcher:
    """Dispatch wrapper that enforces correlation/causation chain validation.

    Wraps the MessageDispatchEngine to add chain validation before dispatch.
    This ensures all messages in a workflow maintain proper traceability for
    distributed debugging and observability.

    The wrapper does NOT modify the underlying engine. It adds a validation
    layer that:
    1. Validates chain context before dispatch (dispatch_with_parent)
    2. Provides helpers to create properly linked child envelopes

    Design Rationale:
        Chain validation is a cross-cutting concern that affects all message
        dispatch operations. By wrapping the engine instead of modifying it:
        - MessageDispatchEngine remains focused on routing concerns
        - Chain validation can be added/removed without engine changes
        - Testing is simplified (mock either layer independently)
        - Different validation policies can be applied via different wrappers

    Attributes:
        engine: The underlying MessageDispatchEngine for dispatch.
        validator: The ChainPropagationValidator for chain validation.

    Thread Safety:
        ChainAwareDispatcher is stateless and thread-safe for concurrent use
        after initialization. Thread safety during dispatch depends on the
        wrapped MessageDispatchEngine's thread safety guarantees.

    Example:
        >>> from omnibase_infra.runtime import ChainAwareDispatcher, MessageDispatchEngine
        >>>
        >>> # Setup
        >>> engine = MessageDispatchEngine()
        >>> engine.register_dispatcher(...)
        >>> engine.freeze()
        >>>
        >>> # Create chain-aware wrapper
        >>> chain_dispatcher = ChainAwareDispatcher(engine)
        >>>
        >>> # Create child envelope from parent
        >>> child = chain_dispatcher.create_child_envelope(
        ...     parent_envelope=parent,
        ...     payload=MyEvent(...),
        ... )
        >>>
        >>> # Dispatch with chain validation
        >>> result = await chain_dispatcher.dispatch_with_parent(
        ...     topic="dev.events.v1",
        ...     envelope=child,
        ...     parent_envelope=parent,
        ... )

    See Also:
        - MessageDispatchEngine: Underlying dispatch engine
        - ChainPropagationValidator: Validation logic
        - propagate_chain_context: Standalone context propagation

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        engine: MessageDispatchEngine,
        validator: ChainPropagationValidator | None = None,
    ) -> None:
        """Initialize ChainAwareDispatcher.

        Args:
            engine: The MessageDispatchEngine to wrap for dispatch operations.
                Must be frozen before dispatch methods are called.
            validator: Optional ChainPropagationValidator for chain validation.
                If None, creates a default validator instance.

        Example:
            >>> engine = MessageDispatchEngine()
            >>> engine.freeze()
            >>> dispatcher = ChainAwareDispatcher(engine)
            >>>
            >>> # With custom validator
            >>> custom_validator = ChainPropagationValidator()
            >>> dispatcher = ChainAwareDispatcher(engine, validator=custom_validator)
        """
        self._engine = engine
        self._validator = (
            validator if validator is not None else ChainPropagationValidator()
        )

    @property
    def engine(self) -> MessageDispatchEngine:
        """Get the underlying MessageDispatchEngine.

        Returns:
            The wrapped MessageDispatchEngine instance.
        """
        return self._engine

    @property
    def validator(self) -> ChainPropagationValidator:
        """Get the ChainPropagationValidator.

        Returns:
            The ChainPropagationValidator used for chain validation.
        """
        return self._validator

    async def dispatch_with_parent(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
        parent_envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult:
        """Dispatch with chain validation against parent envelope.

        Validates that the envelope properly maintains chain context from the
        parent envelope, then dispatches using the underlying engine. If chain
        validation fails, raises ChainPropagationError before dispatch.

        Validation checks:
            1. envelope.correlation_id == parent_envelope.correlation_id
            2. envelope.causation_id == parent_envelope.envelope_id

        Args:
            topic: The topic to dispatch to (e.g., "dev.user.events.v1").
            envelope: The child message envelope to dispatch.
            parent_envelope: The parent envelope for chain validation.

        Returns:
            ModelDispatchResult from the underlying engine dispatch.

        Raises:
            ChainPropagationError: If chain validation fails. Contains the
                list of violations for inspection and logging.
            ModelOnexError: If dispatch fails (from underlying engine).

        Example:
            >>> # Create child with proper chain context
            >>> child = chain_dispatcher.create_child_envelope(parent, payload=event)
            >>>
            >>> # Dispatch with validation
            >>> try:
            ...     result = await chain_dispatcher.dispatch_with_parent(
            ...         topic="dev.events.v1",
            ...         envelope=child,
            ...         parent_envelope=parent,
            ...     )
            ...     if result.is_successful():
            ...         logger.info("Dispatch successful")
            ... except ChainPropagationError as e:
            ...     logger.error("Chain broken: %s", e.format_violations_for_logging())

        Note:
            This method validates BEFORE dispatch. If you need to dispatch
            regardless of chain validity (with warnings), use the underlying
            engine directly and validate separately.
        """
        # Validate chain before dispatch
        violations = self._validator.validate_chain(parent_envelope, envelope)

        if violations:
            # Get correlation_id for error context
            correlation_id = get_correlation_id(parent_envelope)

            # Build error context
            context = ModelInfraErrorContext(
                operation="dispatch_with_parent",
                target_name=topic,
                correlation_id=correlation_id,
            )

            raise ChainPropagationError(
                message="Chain validation failed before dispatch",
                violations=violations,
                context=context,
            )

        # Chain is valid, proceed with dispatch
        return await self._engine.dispatch(topic, envelope)

    def create_child_envelope(
        self,
        parent_envelope: ModelEventEnvelope[object],
        payload: T,
        message_type: str | None = None,
    ) -> ModelEventEnvelope[T]:
        """Create a properly linked child envelope from a parent.

        Creates a new envelope with:
        - A new unique envelope_id (generated)
        - correlation_id copied from parent (workflow trace)
        - causation_id set to parent's envelope_id (parent-child link)
        - The provided payload
        - Optional message_type override

        This is the recommended way to create child envelopes that maintain
        proper chain context for dispatch.

        Args:
            parent_envelope: The parent envelope to inherit chain context from.
            payload: The payload for the new child envelope.
            message_type: Optional message type override. If None, uses the
                class name of the payload.

        Returns:
            A new ModelEventEnvelope with proper chain context and the
            provided payload. The envelope is ready for dispatch.

        Example:
            >>> # Create child from parent
            >>> child = chain_dispatcher.create_child_envelope(
            ...     parent_envelope=parent,
            ...     payload=UserCreatedEvent(user_id="123"),
            ... )
            >>>
            >>> # Child has proper chain context
            >>> child.correlation_id == parent.correlation_id
            True
            >>> child.causation_id == parent.envelope_id
            True
            >>> child.envelope_id != parent.envelope_id  # New ID
            True
            >>>
            >>> # With custom message type
            >>> child = chain_dispatcher.create_child_envelope(
            ...     parent_envelope=parent,
            ...     payload=event,
            ...     message_type="CustomEventType",
            ... )

        Note:
            The trace_id is NOT copied from parent. Each envelope may have
            its own trace context. For distributed tracing, correlation_id
            provides workflow-level correlation, while trace_id is for
            span-level tracing.
        """
        # Local import to avoid circular import issues with omnibase_core
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope as EnvelopeClass,
        )

        # Get parent chain context
        parent_message_id = get_message_id(parent_envelope)
        parent_correlation_id = get_correlation_id(parent_envelope)

        # Generate new envelope_id for child
        child_envelope_id = uuid4()

        # Determine message type
        effective_message_type = (
            message_type if message_type is not None else type(payload).__name__
        )

        # Create child envelope with proper chain context.
        # The type annotation ModelEventEnvelope[T] enables static type checking of the payload,
        # but Python's generics are erased at runtime - the envelope accepts any payload type.
        envelope: ModelEventEnvelope[T] = EnvelopeClass(
            envelope_id=child_envelope_id,
            payload=payload,
            message_type=effective_message_type,
            correlation_id=parent_correlation_id,
            causation_id=parent_message_id,
        )
        return envelope
