# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inbound command model for the chain learning workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelChainLearnCommand(BaseModel):
    """Command to trigger the chain learning workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    prompt_text: str = Field(..., min_length=1, description="Prompt text to learn from")
    workflow_ref: str = Field(
        ...,
        min_length=1,
        description="Reference of the workflow that produced the chain",
    )
    context: dict[str, str] = Field(
        default_factory=dict, description="Additional context for retrieval"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for cache hit",
    )


__all__ = ["ModelChainLearnCommand"]
