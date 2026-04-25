# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stub invocation command model for the remote-agent invoke effect node.

Full model definition deferred to OMN-9625/9626/9627 core delegation stack.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInvocationCommand(BaseModel):
    """Command requesting remote-agent task submission and lifecycle watch."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ..., description="Unique identifier for the invocation."
    )
    agent_id: UUID = Field(..., description="Target agent identifier.")
    payload: dict[str, object] = Field(
        default_factory=dict, description="Invocation payload."
    )


__all__ = ["ModelInvocationCommand"]
