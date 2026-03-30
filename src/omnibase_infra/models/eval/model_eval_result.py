# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval result model for autonomous off-peak evaluations.

Related:
    - OMN-6795: Define eval task models and enums
    - OMN-6796: Build eval runner service

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.eval.model_eval_finding import ModelEvalFinding


class ModelEvalResult(BaseModel):
    """Result of an evaluation task execution.

    Attributes:
        task_id: The ID of the completed task.
        findings: List of findings from the evaluation.
        high_count: Number of high-severity findings.
        medium_count: Number of medium-severity findings.
        low_count: Number of low-severity findings.
        tokens_used: Total tokens consumed.
        actual_cost_usd: Actual cost in USD.
        duration_ms: Execution duration in milliseconds.
        llm_model_label: Human-readable label of model used.
        llm_provider_label: Human-readable label of the provider that served the request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: UUID = Field(..., description="Task identifier.")
    findings: tuple[ModelEvalFinding, ...] = Field(
        default_factory=tuple, description="Findings from the evaluation."
    )
    high_count: int = Field(default=0, ge=0, description="High-severity count.")
    medium_count: int = Field(default=0, ge=0, description="Medium-severity count.")
    low_count: int = Field(default=0, ge=0, description="Low-severity count.")
    tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed.")
    actual_cost_usd: float = Field(default=0.0, ge=0.0, description="Actual cost.")
    duration_ms: int = Field(default=0, ge=0, description="Execution duration in ms.")
    llm_model_label: str = Field(default="", description="Model used.")
    llm_provider_label: str = Field(
        default="", description="Provider that served request."
    )


__all__: list[str] = ["ModelEvalResult"]
