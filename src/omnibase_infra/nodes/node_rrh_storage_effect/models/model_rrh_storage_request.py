# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input model for the RRH storage effect node."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.rrh.model_rrh_result import ModelRRHResult


class ModelRRHStorageRequest(BaseModel):
    """Request to persist an RRH result as a JSON artifact.

    Attributes:
        result: The RRH validation result to store.
        output_dir: Base directory for RRH artifacts.
        correlation_id: Distributed tracing correlation ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    result: ModelRRHResult = Field(..., description="RRH result to persist.")
    output_dir: str = Field(..., description="Base directory for artifacts.")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing."
    )


__all__: list[str] = ["ModelRRHStorageRequest"]
