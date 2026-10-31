# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Request model to write the session index."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_session_state_effect.models.model_session_index import (
    ModelSessionIndex,
)


class ModelSessionIndexWriteRequest(BaseModel):
    """Write the session index.

    Attributes:
        index: The session index to persist.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    index: ModelSessionIndex = Field(
        ...,
        description="The session index to persist.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )


__all__: list[str] = ["ModelSessionIndexWriteRequest"]
