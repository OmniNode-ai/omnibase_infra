# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of one in-progress probe hygiene sweep (OMN-17942)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_outcome import (
    ModelProbeHygieneOutcome,
)


class ModelProbeHygieneResult(BaseModel):
    """Run-level counts plus the per-ticket outcomes that produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Run correlation id.")
    dry_run: bool = Field(default=True, description="Whether writes were withheld.")
    tickets_scanned: int = Field(default=0, ge=0)
    tickets_with_probe: int = Field(default=0, ge=0)
    tickets_without_probe: int = Field(
        default=0,
        ge=0,
        description=(
            "THE FINDING. Tickets In Progress with no executable close probe "
            "anywhere — unreachable by the evidence closer and by every other "
            "mechanical closing path, whatever this run did about it."
        ),
    )
    tickets_commented: int = Field(default=0, ge=0)
    tickets_skipped: int = Field(default=0, ge=0)
    tickets_errored: int = Field(default=0, ge=0)
    outcomes: tuple[ModelProbeHygieneOutcome, ...] = Field(default=())
    success: bool = Field(default=True)
    error_message: str = Field(default="")


__all__ = ["ModelProbeHygieneResult"]
