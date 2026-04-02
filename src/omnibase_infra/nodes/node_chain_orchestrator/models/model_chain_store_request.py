# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Store request model for chain persistence."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_entry import ModelChainEntry


class ModelChainStoreRequest(BaseModel):
    """Request to store a verified chain in Qdrant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    chain_entry: ModelChainEntry = Field(..., description="Verified chain to store")
    prompt_embedding: list[float] = Field(
        ..., min_length=1, description="Embedding vector for the prompt"
    )


__all__ = ["ModelChainStoreRequest"]
