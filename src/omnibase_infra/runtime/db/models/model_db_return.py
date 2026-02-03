# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database return type model for SQL operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbReturn(BaseModel):
    """Return type specification for a database operation.

    Attributes:
        model_ref: Reference to the model type for results (e.g., "User")
        many: If True, operation returns multiple rows; if False, single row
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_ref: str = Field(default="", description="Model reference for results")
    many: bool = Field(
        default=False, description="Whether operation returns multiple rows"
    )


__all__ = ["ModelDbReturn"]
