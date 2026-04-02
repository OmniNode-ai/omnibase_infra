# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verified trajectory entry stored in Qdrant."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_step import ModelChainStep


class ModelChainEntry(BaseModel):
    """A verified trajectory stored in Qdrant."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    chain_id: UUID = Field(..., description="Unique chain identifier")
    prompt_text: str = Field(
        ..., description="Original prompt that produced this chain"
    )
    prompt_hash: str = Field(..., description="SHA-256 of prompt for dedup")
    chain_steps: tuple[ModelChainStep, ...] = Field(
        ..., description="Ordered execution steps"
    )
    contract_hash: str = Field(
        ..., description="Hash of contract that validated this chain"
    )
    success_timestamp: datetime = Field(..., description="When the chain was verified")
    workflow_ref: str = Field(
        ..., description="Reference of the workflow that produced this chain"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold used for this entry",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )


__all__ = ["ModelChainEntry"]
