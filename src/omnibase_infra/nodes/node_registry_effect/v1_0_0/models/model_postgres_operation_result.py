# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL operation result model for registry operations."""

from pydantic import BaseModel, ConfigDict


class ModelPostgresOperationResult(BaseModel):
    """Result of a PostgreSQL operation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool
    rows_affected: int = 0
    error: str | None = None
