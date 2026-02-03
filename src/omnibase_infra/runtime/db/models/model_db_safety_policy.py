# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database safety policy model for SQL operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbSafetyPolicy(BaseModel):
    """Safety policy constraints for database operations.

    Attributes:
        require_where_clause: If True, require WHERE clause for updates/deletes
        max_affected_rows: Maximum number of rows that can be affected
        allow_full_table_scan: If True, allow queries without index usage
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_where_clause: bool = Field(
        default=True, description="Require WHERE clause for updates/deletes"
    )
    max_affected_rows: int | None = Field(
        default=None, description="Maximum affected rows (None = unlimited)"
    )
    allow_full_table_scan: bool = Field(
        default=True, description="Allow queries without index usage"
    )
