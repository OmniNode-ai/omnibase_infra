# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for registration orchestrator."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIntentExecutionResult(BaseModel):
    """Result of executing a single intent.

    Captures the outcome of a single registration intent execution,
    including success/failure status, timing, and error details.

    Attributes:
        intent_kind: The type of intent that was executed (e.g., 'consul', 'postgres').
        success: Whether the execution completed successfully.
        error: Error message if execution failed, None otherwise.
        execution_time_ms: Time taken to execute the intent in milliseconds.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    intent_kind: str = Field(
        ...,
        min_length=1,
        description="The intent kind that was executed",
    )
    success: bool = Field(
        ...,
        description="Whether execution succeeded",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Execution time in milliseconds",
    )


class ModelOrchestratorOutput(BaseModel):
    """Output from the registration orchestrator workflow.

    Provides comprehensive results of the orchestrated registration workflow,
    including per-target success status and detailed execution metrics.

    Attributes:
        correlation_id: Correlation ID for distributed tracing.
        status: Overall workflow status - success, partial, or failed.
        consul_applied: Whether Consul registration succeeded.
        postgres_applied: Whether PostgreSQL registration succeeded.
        consul_error: Error message from Consul registration if any.
        postgres_error: Error message from PostgreSQL registration if any.
        intent_results: Detailed results for each executed intent.
        total_execution_time_ms: Total workflow execution time in milliseconds.
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
    intent_results: list[ModelIntentExecutionResult] = Field(
        default_factory=list,
        description="Results of each intent execution",
    )
    total_execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total workflow execution time in milliseconds",
    )


__all__ = ["ModelIntentExecutionResult", "ModelOrchestratorOutput"]
