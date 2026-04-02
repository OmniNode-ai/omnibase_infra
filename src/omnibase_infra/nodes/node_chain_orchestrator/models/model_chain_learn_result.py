# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Final result model for the chain learning workflow."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelChainLearnResult(BaseModel):
    """Final output of the chain learning workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    path_taken: Literal["hit_replay", "miss_explore", "fallback_explore"] = Field(
        ..., description="Which execution path was taken"
    )
    chain_ref: str = Field(
        default="", description="Reference to stored or reused chain (empty if none)"
    )
    success: bool = Field(..., description="Whether the workflow succeeded")
    error_message: str = Field(default="", description="Error details if failed")
    retrieval_latency_ms: int = Field(
        default=0, ge=0, description="Time spent on retrieval"
    )
    total_latency_ms: int = Field(
        default=0, ge=0, description="Total workflow duration"
    )


__all__ = ["ModelChainLearnResult"]
