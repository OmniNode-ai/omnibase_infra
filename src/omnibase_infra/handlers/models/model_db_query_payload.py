# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Query Payload Model.

This module provides the Pydantic model for database query result payloads.

Note on row typing:
    Database rows are typed as ``list[dict[str, object]]`` because:

    1. Column names are dynamic (determined by SQL query)
    2. Column types are heterogeneous (str, int, float, datetime, etc.)
    3. The handler returns generic rows that callers must interpret

    For strongly-typed domain models, callers should map these generic
    rows to their specific Pydantic models after retrieval.

JsonType Recursion Fix (OMN-1274):
    This module uses ``dict[str, object]`` instead of the recursive ``JsonType``
    type alias from omnibase_core. Here is why:

    **The Original Problem:**
    ``JsonType`` was defined as a recursive type alias::

        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None

    Pydantic 2.x processes model fields at **class definition time** (not runtime).
    When building JSON schemas for validation, it recursively expands type aliases::

        JsonType -> dict[str, JsonType] | list[JsonType] | ...
                 -> dict[str, dict[str, JsonType] | list[JsonType] | ...] | ...
                 -> ... (infinite expansion)

    This causes ``RecursionError: maximum recursion depth exceeded`` during
    Pydantic's schema generation in ``_generate_schema.py``.

    **Why dict[str, object] is Correct Here:**
    For database rows, we specifically need dictionaries with string keys and
    heterogeneous values. We do NOT need to support arrays or primitives at the
    root level (unlike HTTP response bodies which can be any JSON value).

    Using ``dict[str, object]`` provides:
    - Correct semantics: DB rows are always dictionaries
    - Type safety: Pydantic validates the structure
    - No recursion: ``object`` is not a recursive type alias

    **Caveats:**
    - ``object`` provides no static type checking on values (use runtime validation)
    - Callers should map to strongly-typed models for domain logic
    - For fields needing full JSON support (any JSON value), use ``JsonType``
      from ``omnibase_core.types`` (now fixed via TypeAlias pattern)

See Also:
    - ADR: ``docs/decisions/adr-any-type-pydantic-workaround.md`` (historical)
    - Pydantic issue: https://github.com/pydantic/pydantic/issues/3278
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
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    rows: list[dict[str, object]] = Field(
        description="Result rows as column->value dictionaries",
    )
    row_count: int = Field(
        ge=0,
        description="Number of rows returned or affected",
    )


__all__: list[str] = ["ModelDbQueryPayload"]
