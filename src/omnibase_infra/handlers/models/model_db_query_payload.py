# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Query Payload Model.

This module provides the Pydantic model for database query result payloads.

Note on row typing:
    Database rows are typed as list[dict[str, object]] because:
    1. Column names are dynamic (determined by SQL query)
    2. Column types are heterogeneous (str, int, float, datetime, etc.)
    3. The adapter returns generic rows that callers must interpret

    For strongly-typed domain models, callers should map these generic
    rows to their specific Pydantic models after retrieval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbQueryPayload(BaseModel):
    """Payload containing database query results.

    Attributes:
        rows: List of result rows as column->value dictionaries.
            Column types are preserved from database (str, int, float, etc.).
        row_count: Number of rows returned or affected.

    Example:
        >>> payload = ModelDbQueryPayload(
        ...     rows=[{"id": 1, "name": "test"}, {"id": 2, "name": "example"}],
        ...     row_count=2,
        ... )
        >>> print(len(payload.rows))
        2
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    rows: list[dict[str, object]] = Field(
        description="Result rows as column->value dictionaries",
    )
    row_count: int = Field(
        ge=0,
        description="Number of rows returned or affected",
    )


__all__: list[str] = ["ModelDbQueryPayload"]
