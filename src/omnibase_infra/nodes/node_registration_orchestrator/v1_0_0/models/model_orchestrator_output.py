# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for registration orchestrator."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_intent_execution_result import (
    ModelIntentExecutionResult,
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


__all__ = ["ModelOrchestratorOutput"]
