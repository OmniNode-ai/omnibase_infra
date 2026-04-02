# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single step within a chain trajectory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelChainStep(BaseModel):
    """Single step within a chain trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(..., ge=0, description="Ordered position in the chain")
    node_ref: str = Field(
        ..., description="ONEX node reference that executed this step"
    )
    operation: str = Field(..., description="Operation performed")
    input_hash: str = Field(..., description="SHA-256 of input payload")
    output_hash: str = Field(..., description="SHA-256 of output payload")
    duration_ms: int = Field(..., ge=0, description="Step execution time in ms")
    event_topic: str = Field(..., description="Kafka topic this step produced")


__all__ = ["ModelChainStep"]
