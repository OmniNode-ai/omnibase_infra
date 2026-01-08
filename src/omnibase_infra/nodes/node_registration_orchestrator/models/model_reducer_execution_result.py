# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer execution result model for the registration orchestrator.

This model replaces the tuple pattern `tuple[ModelReducerState, tuple[ModelRegistrationIntent, ...]]`
that was used for reducer return values. By using a single model type, we eliminate
the tuple pattern while providing richer context and self-documenting field names.

Design Pattern:
    ModelReducerExecutionResult replaces tuple returns from ProtocolReducer.reduce()
    with a strongly-typed, self-documenting model. This follows the ONEX principle
    of using Pydantic models instead of tuple returns for method results.

    The model is intentionally frozen (immutable) to ensure thread safety when
    the same result is passed between components. The intents field uses an
    immutable tuple rather than a mutable list to maintain full immutability.

Intent Typing:
    The intents field accepts any model inheriting from ModelRegistryIntent.
    Intent types self-register via the @RegistryIntent.register() decorator,
    enabling dynamic type resolution during deserialization without explicit unions.

    This registry-based approach:
    - Eliminates duplicate union definitions across modules
    - Allows future intent types to be added by simply implementing ModelRegistryIntent
      and registering with @RegistryIntent.register("kind")
    - Uses the `kind` field as a discriminator for type resolution
    - Follows ONEX duck typing principles while maintaining type safety

    Current implementations:
    - ModelConsulRegistrationIntent (kind="consul"): Consul service registration
    - ModelPostgresUpsertIntent (kind="postgres"): PostgreSQL upsert operations

    For type hints in external code, use ProtocolRegistrationIntent from
    omnibase_infra.nodes.node_registration_orchestrator.protocols for duck-typed
    signatures, or ModelRegistryIntent for concrete model requirements.

Thread Safety:
    ModelReducerExecutionResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access. All fields use immutable
    types (ModelReducerState is also frozen, intents is a tuple).

Example:
    >>> from omnibase_infra.nodes.node_registration_orchestrator.models import (
    ...     ModelReducerExecutionResult,
    ...     ModelReducerState,
    ... )
    >>>
    >>> # Create a result with no intents
    >>> result = ModelReducerExecutionResult.empty()
    >>> result.state.processed_node_ids
    frozenset()
    >>> result.intents
    ()
    >>>
    >>> # Create from existing state with intents
    >>> state = ModelReducerState(pending_registrations=2)
    >>> result = ModelReducerExecutionResult(state=state, intents=(intent1, intent2))

.. versionadded:: 0.7.0
    Created as part of tuple-to-model conversion work (OMN-1007).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registry_intent import (
    ModelRegistryIntent,
    RegistryIntent,
)


class ModelReducerExecutionResult(BaseModel):
    """Result of reducer execution containing state and generated intents.

    This model replaces the `tuple[ModelReducerState, tuple[ModelRegistrationIntent, ...]]`
    pattern with a strongly-typed container that provides:
    - Self-documenting field names (state, intents)
    - Factory methods for common patterns (empty, no_change, with_intents)
    - Full immutability for thread safety (frozen model with tuple intents)
    - Protocol-based intent typing for extensibility (ProtocolRegistrationIntent)

    Attributes:
        state: The updated reducer state after processing an event.
        intents: Tuple of registration intents to be executed by the effect node.
            May be empty if no infrastructure operations are needed.

    Warning:
        **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to return
        ``True`` only when intents are present (i.e., ``has_intents`` is True). This
        differs from typical Pydantic model behavior where ``bool(model)`` always
        returns ``True`` for any valid model instance.

        This design enables idiomatic conditional checks for work to be done::

            if result:
                # Process intents - there is work to do
                execute_intents(result.intents)
            else:
                # No intents - skip processing
                pass

        If you need to check model validity instead, use explicit attribute access::

            # Check for intents (uses __bool__)
            if result:
                ...

            # Check model is valid (always True for constructed instance)
            if result is not None:
                ...

            # Explicit intent check (preferred for clarity)
            if result.has_intents:
                ...

    Example:
        >>> # Create result with state and intents
        >>> state = ModelReducerState(pending_registrations=2)
        >>> result = ModelReducerExecutionResult(
        ...     state=state,
        ...     intents=(consul_intent, postgres_intent),
        ... )
        >>> result.has_intents
        True
        >>> result.intent_count
        2

        >>> # Create empty result (no state changes, no intents)
        >>> result = ModelReducerExecutionResult.empty()
        >>> result.has_intents
        False

    .. versionadded:: 0.7.0
        Created as part of OMN-1007 tuple-to-model conversion.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    state: ModelReducerState = Field(
        ...,
        description="The updated reducer state after processing the event.",
    )
    intents: tuple[ModelRegistryIntent, ...] = Field(
        default=(),
        description=(
            "Tuple of registration intents to be executed by the effect node. "
            "All intents must inherit from ModelRegistryIntent and be registered "
            "with RegistryIntent."
        ),
    )
    # Note: When serializing this model, use model_dump(serialize_as_any=True)
    # to ensure intent fields (like 'payload') that exist only on concrete types
    # are included. Without this, Pydantic serializes according to the declared
    # base type (ModelRegistryIntent) which omits subclass-specific fields.

    @field_validator("intents", mode="before")
    @classmethod
    def validate_intents(cls, v: object) -> tuple[ModelRegistryIntent, ...]:
        """Resolve intent types dynamically from RegistryIntent.

        When deserializing from JSON/dict, uses the 'kind' field to look up
        the correct concrete intent class from the registry.

        Args:
            v: Raw input value (tuple, list, or sequence of dicts/models)

        Returns:
            Tuple of validated ModelRegistryIntent instances

        Raises:
            ValueError: If intent kind is not registered or dict missing 'kind' field
        """
        if v is None:
            return ()

        # Validate input is a proper sequence (not str/bytes)
        if not isinstance(v, Sequence) or isinstance(v, str | bytes):
            raise ValueError(
                f"intents must be a tuple or Sequence (excluding str/bytes), "
                f"got {type(v).__name__}"
            )

        result: list[ModelRegistryIntent] = []
        for item in v:
            if isinstance(item, dict):
                kind = item.get("kind")
                if kind is None:
                    raise ValueError("Intent dict missing required 'kind' field")
                try:
                    intent_cls = RegistryIntent.get_type(kind)
                except KeyError as e:
                    raise ValueError(str(e)) from e
                result.append(intent_cls.model_validate(item))
            elif isinstance(item, ModelRegistryIntent):
                result.append(item)
            else:
                raise ValueError(
                    f"Intent must be dict or ModelRegistryIntent, got {type(item).__name__}"
                )
        return tuple(result)

    @property
    def has_intents(self) -> bool:
        """Check if the result contains any intents.

        Returns:
            True if intents tuple is non-empty, False otherwise.

        Example:
            >>> ModelReducerExecutionResult.empty().has_intents
            False
            >>> result = ModelReducerExecutionResult(
            ...     state=ModelReducerState.initial(),
            ...     intents=(some_intent,),
            ... )
            >>> result.has_intents
            True

        .. versionadded:: 0.7.0
        """
        return len(self.intents) > 0

    @property
    def intent_count(self) -> int:
        """Get the number of intents in the result.

        Returns:
            Number of intents in the intents tuple.

        Example:
            >>> ModelReducerExecutionResult.empty().intent_count
            0

        .. versionadded:: 0.7.0
        """
        return len(self.intents)

    @classmethod
    def empty(cls) -> ModelReducerExecutionResult:
        """Create an empty result with initial state and no intents.

        Use this factory when the reducer determines no action is needed
        (e.g., duplicate event, filtered event).

        Returns:
            ModelReducerExecutionResult with initial state and empty intents.

        Example:
            >>> result = ModelReducerExecutionResult.empty()
            >>> result.state.processed_node_ids
            frozenset()
            >>> result.intents
            ()

        .. versionadded:: 0.7.0
        """
        return cls(state=ModelReducerState.initial(), intents=())

    @classmethod
    def no_change(cls, state: ModelReducerState) -> ModelReducerExecutionResult:
        """Create a result with existing state and no intents.

        Use this factory when the reducer determines the event should be
        filtered but state should be preserved (e.g., already processed node).

        Args:
            state: The current reducer state to preserve.

        Returns:
            ModelReducerExecutionResult with preserved state and empty intents.

        Example:
            >>> state = ModelReducerState(pending_registrations=5)
            >>> result = ModelReducerExecutionResult.no_change(state)
            >>> result.state.pending_registrations
            5
            >>> result.intents
            ()

        .. versionadded:: 0.7.0
        """
        return cls(state=state, intents=())

    @classmethod
    def with_intents(
        cls,
        state: ModelReducerState,
        intents: Sequence[ModelRegistryIntent],
    ) -> ModelReducerExecutionResult:
        """Create a result with state and intents.

        Args:
            state: The updated reducer state.
            intents: Sequence of registration intents to execute. Each intent
                must inherit from ModelRegistryIntent and be registered with
                RegistryIntent (e.g., ModelConsulRegistrationIntent,
                ModelPostgresUpsertIntent). Will be converted to an immutable tuple.

        Returns:
            ModelReducerExecutionResult with the provided state and intents.

        Example:
            >>> state = ModelReducerState(pending_registrations=2)
            >>> result = ModelReducerExecutionResult.with_intents(
            ...     state=state,
            ...     intents=[consul_intent, postgres_intent],
            ... )
            >>> result.intent_count
            2

        .. versionadded:: 0.7.0
        """
        return cls(state=state, intents=tuple(intents))

    def __bool__(self) -> bool:
        """Allow using result in boolean context to check for pending work.

        Returns True if the result contains any intents, indicating that
        infrastructure operations need to be performed.

        Warning:
            This differs from typical Pydantic model behavior where ``bool(model)``
            always returns ``True`` for any valid model instance. Here, ``bool(result)``
            returns ``False`` for valid results with no intents.

            Use ``result.has_intents`` for explicit, self-documenting code.
            Use ``result is not None`` if you need to check model existence.

        Returns:
            True if intents tuple is non-empty, False otherwise.

        Example:
            >>> result_with_work = ModelReducerExecutionResult.with_intents(
            ...     state=ModelReducerState.initial(),
            ...     intents=(some_intent,),
            ... )
            >>> bool(result_with_work)
            True

            >>> result_no_work = ModelReducerExecutionResult.empty()
            >>> bool(result_no_work)  # False even though model is valid!
            False

            >>> # Idiomatic usage
            >>> if result_no_work:
            ...     print("Has work to do")
            ... else:
            ...     print("No work needed")
            No work needed

        .. versionadded:: 0.7.0
        """
        return self.has_intents

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String showing state summary and intent count.

        .. versionadded:: 0.7.0
        """
        processed = len(self.state.processed_node_ids)
        pending = self.state.pending_registrations
        return (
            f"ModelReducerExecutionResult("
            f"processed={processed}, "
            f"pending={pending}, "
            f"intents={self.intent_count})"
        )


__all__ = ["ModelReducerExecutionResult"]
