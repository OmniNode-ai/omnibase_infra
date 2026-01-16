# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for registration orchestrator.

Thread Safety:
    This model is fully immutable (frozen=True) with immutable field types.
    The ``intent_results`` field uses tuple instead of list to ensure
    complete immutability for thread-safe concurrent access.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_registration_orchestrator.models.model_intent_execution_result import (
    ModelIntentExecutionResult,
)


class ModelOrchestratorOutput(BaseModel):
    """Output from the registration orchestrator workflow.

    Provides comprehensive results of the orchestrated registration workflow,
    including per-target success status and detailed execution metrics.

    This model is fully immutable to support thread-safe concurrent access.
    All collection fields use immutable types (tuple instead of list).

    Attributes:
        correlation_id: Correlation ID for distributed tracing.
        status: Overall workflow status - success, partial, or failed.
        consul_applied: Whether Consul registration succeeded.
        postgres_applied: Whether PostgreSQL registration succeeded.
        consul_error: Error message from Consul registration if any.
        postgres_error: Error message from PostgreSQL registration if any.
        intent_results: Immutable tuple of results for each executed intent.
        total_execution_time_ms: Total workflow execution time in milliseconds.

    Example:
        >>> output = ModelOrchestratorOutput(
        ...     correlation_id=uuid4(),
        ...     status="success",
        ...     intent_results=[result1, result2],  # list auto-converted to tuple
        ...     total_execution_time_ms=150.5,
        ... )
        >>> isinstance(output.intent_results, tuple)
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    status: Literal["success", "partial", "failed"] = Field(
        ...,
        description="Overall workflow status",
    )
    consul_applied: bool = Field(
        default=False,
        description="Whether Consul registration succeeded",
    )
    postgres_applied: bool = Field(
        default=False,
        description="Whether PostgreSQL registration succeeded",
    )
    consul_error: str | None = Field(
        default=None,
        description="Consul error message if any",
    )
    postgres_error: str | None = Field(
        default=None,
        description="PostgreSQL error message if any",
    )
    intent_results: tuple[ModelIntentExecutionResult, ...] = Field(
        default=(),
        description="Immutable tuple of results for each intent execution",
    )
    total_execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total workflow execution time in milliseconds",
    )

    @field_validator("intent_results", mode="before")
    @classmethod
    def _coerce_intent_results_to_tuple(
        cls, v: object
    ) -> tuple[ModelIntentExecutionResult, ...]:
        """Convert list/sequence to tuple for immutability.

        This validator ensures explicit handling of all input types rather than
        silent fallback to empty tuple, which could mask invalid input. In strict
        mode, all elements must already be ModelIntentExecutionResult instances -
        no silent type coercion from dicts or other types.

        Args:
            v: The input value to coerce. Must be either a tuple or a Sequence
                (excluding str and bytes) containing only ModelIntentExecutionResult
                instances.

        Returns:
            A tuple of ModelIntentExecutionResult items.

        Raises:
            ValueError: If the input is None, str, bytes, or any other non-sequence
                type, OR if any element is not a ModelIntentExecutionResult instance.
                This ensures invalid input types are explicitly rejected rather than
                silently coerced.

        Edge Cases:
            - ``None``: Raises ValueError (explicit rejection)
            - ``str`` or ``bytes``: Raises ValueError (not valid sequences for this field)
            - Empty tuple ``()``: Passed through (same as default)
            - Empty list ``[]``: Converted to empty tuple
            - Other non-sequence types (int, dict, etc.): Raises ValueError
            - Sequence with dict elements: Raises ValueError (strict mode - no coercion)
            - Sequence with ModelIntentExecutionResult: Converts to tuple

        Example:
            >>> # Valid inputs
            >>> _coerce_intent_results_to_tuple([result1, result2])
            (result1, result2)
            >>> _coerce_intent_results_to_tuple(())
            ()
            >>> # Invalid inputs raise ValueError
            >>> _coerce_intent_results_to_tuple(None)  # Raises ValueError
            >>> _coerce_intent_results_to_tuple("not a sequence")  # Raises ValueError
            >>> _coerce_intent_results_to_tuple([{"target": "consul"}])  # Raises ValueError
        """
        if isinstance(v, tuple):
            # Validate tuple contents in strict mode
            for i, item in enumerate(v):
                if not isinstance(item, ModelIntentExecutionResult):
                    raise ValueError(
                        f"intent_results[{i}] must be a ModelIntentExecutionResult, "
                        f"got {type(item).__name__}"
                    )
            return v  # type: ignore[return-value]  # NOTE: runtime type validated above
        if isinstance(v, Sequence) and not isinstance(v, str | bytes):
            # Validate and convert to tuple - strict mode requires model instances
            result: list[ModelIntentExecutionResult] = []
            for i, item in enumerate(v):
                if not isinstance(item, ModelIntentExecutionResult):
                    raise ValueError(
                        f"intent_results[{i}] must be a ModelIntentExecutionResult, "
                        f"got {type(item).__name__}"
                    )
                result.append(item)
            return tuple(result)
        raise ValueError(
            f"intent_results must be a tuple or Sequence (excluding str/bytes), "
            f"got {type(v).__name__}"
        )


__all__ = ["ModelOrchestratorOutput"]
