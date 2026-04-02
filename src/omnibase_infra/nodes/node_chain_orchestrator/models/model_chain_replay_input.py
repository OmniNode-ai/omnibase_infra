# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input to the chain replay compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_entry import ModelChainEntry


class ModelChainReplayInput(BaseModel):
    """Input to the chain replay compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    cached_chain: ModelChainEntry = Field(..., description="Cached chain to replay")
    new_prompt_text: str = Field(..., description="New prompt to adapt chain for")
    new_context: dict[str, str] = Field(
        default_factory=dict, description="Context variables for adaptation"
    )


__all__ = ["ModelChainReplayInput"]
