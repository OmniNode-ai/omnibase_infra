# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Request model to read a run context."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRunContextReadRequest(BaseModel):
    """Read a run context.

    Attributes:
        run_id: The unique run identifier.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    run_id: Annotated[
        str,
        Field(description="The unique run identifier."),
    ]
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )


__all__: list[str] = ["ModelRunContextReadRequest"]
