# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request for embedding and retrieving similar chains."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelChainRetrievalRequest(BaseModel):
    """Typed request for embedding and retrieving similar chains."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    prompt_text: str
    correlation_id: UUID
    similarity_threshold: float = 0.85
    limit: int = 5


__all__ = ["ModelChainRetrievalRequest"]
