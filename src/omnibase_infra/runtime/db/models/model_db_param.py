# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database parameter model for SQL operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbParam(BaseModel):
    """Parameter definition for a database operation.

    Attributes:
        name: Parameter name (used for documentation/error messages)
        param_type: Parameter type (e.g., "integer", "string", "uuid")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Parameter name")
    param_type: str = Field(..., description="Parameter type")


__all__ = ["ModelDbParam"]
