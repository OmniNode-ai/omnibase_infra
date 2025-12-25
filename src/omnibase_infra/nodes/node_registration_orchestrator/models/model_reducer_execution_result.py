# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer execution result model for the registration orchestrator.

This model replaces the tuple pattern `tuple[ModelReducerState, list[ModelRegistrationIntent]]`
that was used for reducer return values. By using a single model type, we eliminate
the tuple pattern while providing richer context and self-documenting field names.

Design Pattern:
    ModelReducerExecutionResult replaces tuple returns from ProtocolReducer.reduce()
    with a strongly-typed, self-documenting model. This follows the ONEX principle
    of using Pydantic models instead of tuple returns for method results.

    The model is intentionally frozen (immutable) to ensure thread safety when
    the same result is passed between components.

Thread Safety:
    ModelReducerExecutionResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Migration:
    Use the factory methods for gradual migration:
    - `from_legacy_tuple()`: Convert old tuple returns to new model
    - `to_legacy_tuple()`: Convert back to tuple where needed

    After all callers are migrated, the legacy methods can be removed.

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
    []
    >>>
    >>> # Create from existing state with intents
    >>> state = ModelReducerState(pending_registrations=2)
    >>> result = ModelReducerExecutionResult(state=state, intents=[intent1, intent2])

.. versionadded:: 0.7.0
    Created as part of tuple-to-model conversion work (OMN-1007).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_state import (
    ModelReducerState,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_registration_intent import (
        ModelConsulRegistrationIntent,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_upsert_intent import (
        ModelPostgresUpsertIntent,
    )

# Import the concrete types for runtime validation
from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_registration_intent import (
    ModelConsulRegistrationIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_upsert_intent import (
    ModelPostgresUpsertIntent,
)

RegistrationIntentUnion = ModelConsulRegistrationIntent | ModelPostgresUpsertIntent
"""Type alias for registration intent union without discriminator annotation.

This alias represents the same union as ``ModelRegistrationIntent`` in
``model_registration_intent.py``, but without the ``Annotated[..., Field(discriminator=...)]``
wrapper. We define it here rather than importing ``ModelRegistrationIntent`` to avoid
circular imports between these modules.

Use this alias for:
    - Type hints in method signatures and field definitions
    - Cases where Pydantic's discriminated union behavior is not needed

Use ``ModelRegistrationIntent`` from ``model_registration_intent.py`` when:
    - Pydantic needs to discriminate between intent types based on the ``kind`` field
    - Deserializing JSON where the discriminator determines the concrete type

.. versionadded:: 0.7.0
    Created as part of OMN-1007 tuple-to-model conversion work.
"""


class ModelReducerExecutionResult(BaseModel):
    """Result of reducer execution containing state and generated intents.

    This model replaces the `tuple[ModelReducerState, list[ModelRegistrationIntent]]`
    pattern with a strongly-typed container that provides:
    - Self-documenting field names (state, intents)
    - Factory methods for common patterns (empty, no_change)
    - Legacy compatibility methods for gradual migration
    - Immutability for thread safety

    Attributes:
        state: The updated reducer state after processing an event.
        intents: List of registration intents to be executed by the effect node.
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
        ...     intents=[consul_intent, postgres_intent],
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
    intents: list[RegistrationIntentUnion] = Field(
        default_factory=list,
        description="List of registration intents to be executed by the effect node.",
    )

    @property
    def has_intents(self) -> bool:
        """Check if the result contains any intents.

        Returns:
            True if intents list is non-empty, False otherwise.

        Example:
            >>> ModelReducerExecutionResult.empty().has_intents
            False
            >>> result = ModelReducerExecutionResult(
            ...     state=ModelReducerState.initial(),
            ...     intents=[some_intent],
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
            Number of intents in the intents list.

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
            []

        .. versionadded:: 0.7.0
        """
        return cls(state=ModelReducerState.initial(), intents=[])

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
            []

        .. versionadded:: 0.7.0
        """
        return cls(state=state, intents=[])

    @classmethod
    def with_intents(
        cls,
        state: ModelReducerState,
        intents: Sequence[RegistrationIntentUnion],
    ) -> ModelReducerExecutionResult:
        """Create a result with state and intents.

        Args:
            state: The updated reducer state.
            intents: Sequence of registration intents to execute.

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
        return cls(state=state, intents=list(intents))

    @classmethod
    def from_legacy_tuple(
        cls,
        result: tuple[ModelReducerState, Sequence[RegistrationIntentUnion]],
    ) -> ModelReducerExecutionResult:
        """Create from legacy tuple-based reducer result.

        This factory method handles the conversion from the old tuple pattern
        `tuple[ModelReducerState, list[ModelRegistrationIntent]]` to the new
        model structure.

        Args:
            result: Legacy reducer result as (state, intents).

        Returns:
            ModelReducerExecutionResult with equivalent values.

        Example:
            >>> state = ModelReducerState.initial()
            >>> legacy_result = (state, [intent1, intent2])
            >>> result = ModelReducerExecutionResult.from_legacy_tuple(legacy_result)
            >>> result.state == state
            True
            >>> result.intent_count
            2

        .. versionadded:: 0.7.0
        """
        state, intents = result
        return cls(state=state, intents=list(intents))

    def to_legacy_tuple(
        self,
    ) -> tuple[ModelReducerState, list[RegistrationIntentUnion]]:
        """Convert back to legacy tuple format.

        This method enables gradual migration by allowing conversion back
        to the original tuple format where needed.

        Returns:
            Tuple of (state, intents).

        Example:
            >>> result = ModelReducerExecutionResult.empty()
            >>> state, intents = result.to_legacy_tuple()
            >>> len(intents)
            0

        .. versionadded:: 0.7.0
        """
        return (self.state, list(self.intents))

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
            True if intents is non-empty, False otherwise.

        Example:
            >>> result_with_work = ModelReducerExecutionResult.with_intents(
            ...     state=ModelReducerState.initial(),
            ...     intents=[some_intent],
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


__all__ = ["ModelReducerExecutionResult", "RegistrationIntentUnion"]
