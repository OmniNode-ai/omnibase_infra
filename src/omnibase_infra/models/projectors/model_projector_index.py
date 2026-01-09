# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Index Model.

Defines the schema for database indexes on projection tables.
Used by ModelProjectorSchema to describe index requirements for
validation and migration SQL generation.

NOTE: This model is temporarily defined in omnibase_infra until omnibase_core
provides it at omnibase_core.models.projectors. Once available, this should
be moved there and re-exported from this module.

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelProjectorIndex(BaseModel):
    """Definition of a database index on a projection table.

    Describes the index name, columns, type, and optional partial index
    predicate. Used by ProjectorSchemaManager for schema validation and
    migration SQL generation.

    Attributes:
        name: Index name (must be unique within the database).
        columns: List of column names included in the index.
        index_type: PostgreSQL index type (btree, gin, hash).
        unique: Whether the index enforces uniqueness (default: False).
        where_clause: Optional partial index predicate (SQL expression).

    Example:
        >>> index = ModelProjectorIndex(
        ...     name="idx_registration_capability_tags",
        ...     columns=["capability_tags"],
        ...     index_type="gin",
        ... )
        >>> print(index.name)
        'idx_registration_capability_tags'
    """

    name: str = Field(
        ...,
        description="Index name (must be unique within the database)",
        min_length=1,
        max_length=128,
    )

    columns: list[str] = Field(
        ...,
        description="List of column names included in the index",
        min_length=1,
    )

    index_type: Literal["btree", "gin", "hash"] = Field(
        default="btree",
        description="PostgreSQL index type",
    )

    unique: bool = Field(
        default=False,
        description="Whether the index enforces uniqueness",
    )

    where_clause: str | None = Field(
        default=None,
        description="Optional partial index predicate (SQL expression)",
    )

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }

    @field_validator("columns")
    @classmethod
    def validate_columns_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that columns list is not empty and contains valid names."""
        if not v:
            raise ValueError("Index must have at least one column")
        for col in v:
            if not col or not col.strip():
                raise ValueError("Column name cannot be empty")
        return v

    def to_sql_definition(self, table_name: str) -> str:
        """Generate SQL CREATE INDEX statement.

        Args:
            table_name: Name of the table to create the index on.

        Returns:
            SQL CREATE INDEX statement.

        Example:
            >>> index = ModelProjectorIndex(
            ...     name="idx_registration_state",
            ...     columns=["current_state"],
            ...     index_type="btree",
            ... )
            >>> index.to_sql_definition("registration_projections")
            'CREATE INDEX IF NOT EXISTS idx_registration_state ON registration_projections (current_state)'
        """
        unique_clause = "UNIQUE " if self.unique else ""
        using_clause = (
            f"USING {self.index_type.upper()}" if self.index_type != "btree" else ""
        )
        columns_sql = ", ".join(self.columns)

        parts = [
            f"CREATE {unique_clause}INDEX IF NOT EXISTS {self.name}",
            f"ON {table_name}",
        ]

        if using_clause:
            parts.append(using_clause)

        parts.append(f"({columns_sql})")

        if self.where_clause:
            parts.append(f"WHERE {self.where_clause}")

        return " ".join(parts)


__all__ = ["ModelProjectorIndex"]
