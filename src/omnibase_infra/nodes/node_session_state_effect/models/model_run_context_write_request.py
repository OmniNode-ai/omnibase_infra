# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Request model to write a run context."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_session_state_effect.models.model_run_context import (
    ModelRunContext,
)


class ModelRunContextWriteRequest(BaseModel):
    """Write a run context.

    Attributes:
        context: The run context to persist.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    context: ModelRunContext = Field(
        ...,
        description="The run context to persist.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )


__all__: list[str] = ["ModelRunContextWriteRequest"]
