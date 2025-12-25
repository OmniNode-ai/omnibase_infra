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
from typing import Any, Literal
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
        cls, v: Any
    ) -> tuple[ModelIntentExecutionResult, ...]:
        """Convert list/sequence to tuple for immutability."""
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # Runtime validated by Pydantic
        if isinstance(v, Sequence) and not isinstance(v, str | bytes):
            return tuple(v)  # type: ignore[return-value]  # Runtime validated by Pydantic
        # For unrecognized types, return empty tuple (Pydantic will validate)
        return ()


__all__ = ["ModelOrchestratorOutput"]
