# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Request model to garbage-collect stale run documents."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelStaleRunGcRequest(BaseModel):
    """Garbage-collect stale run documents.

    Attributes:
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )


__all__: list[str] = ["ModelStaleRunGcRequest"]
