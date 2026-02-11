# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for checkpoint validation.

Ticket: OMN-2143
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint


class ModelCheckpointValidateInput(BaseModel):
    """Input for the checkpoint validate compute handler."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    checkpoint: ModelCheckpoint = Field(
        ...,
        description="The checkpoint to validate structurally.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing.",
    )


__all__: list[str] = ["ModelCheckpointValidateInput"]
