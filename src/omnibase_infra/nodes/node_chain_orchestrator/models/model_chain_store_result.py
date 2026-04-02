# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Store result model for chain persistence."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelChainStoreResult(BaseModel):
    """Result of storing a chain in Qdrant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    chain_id: UUID = Field(..., description="ID of the stored chain")
    success: bool = Field(..., description="Whether the store succeeded")
    error_message: str = Field(default="", description="Error details if failed")


__all__ = ["ModelChainStoreResult"]
