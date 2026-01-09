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

from omnibase_infra.models.projectors.util_sql_identifiers import (
    IDENT_PATTERN,
    quote_identifier,
)


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

    @field_validator("name")
    @classmethod
    def validate_name_identifier(cls, v: str) -> str:
        """Validate that the index name is a valid PostgreSQL identifier.

        Prevents SQL injection by ensuring the name matches the safe identifier
        pattern (letters, digits, underscores, starting with letter or underscore).

        Args:
            v: Index name to validate.

        Returns:
            Validated index name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        if not IDENT_PATTERN.match(v):
            raise ValueError(
                f"Invalid index name '{v}': must match pattern "
                "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only, "
                "starting with letter or underscore)"
            )
        return v

    @field_validator("columns")
    @classmethod
    def validate_columns_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that columns list is not empty and contains valid names.

        Validates each column name against the PostgreSQL identifier pattern
        to prevent SQL injection.

        Args:
            v: List of column names.

        Returns:
            Validated list of column names.

        Raises:
            ValueError: If the list is empty or any column name is invalid.
        """
        if not v:
            raise ValueError("Index must have at least one column")
        for col in v:
            if not col or not col.strip():
                raise ValueError("Column name cannot be empty")
            if not IDENT_PATTERN.match(col):
                raise ValueError(
                    f"Invalid column name '{col}': must match pattern "
                    "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only, "
                    "starting with letter or underscore)"
                )
        return v

    def to_sql_definition(self, table_name: str) -> str:
        """Generate SQL CREATE INDEX statement.

        Uses quoted identifiers for index name, table name, and column names
        to prevent SQL injection.

        Args:
            table_name: Name of the table to create the index on.

        Returns:
            SQL CREATE INDEX statement with properly quoted identifiers.

        Example:
            >>> index = ModelProjectorIndex(
            ...     name="idx_registration_state",
            ...     columns=["current_state"],
            ...     index_type="btree",
            ... )
            >>> index.to_sql_definition("registration_projections")
            'CREATE INDEX IF NOT EXISTS "idx_registration_state" ON "registration_projections" ("current_state")'
        """
        unique_clause = "UNIQUE " if self.unique else ""
        using_clause = (
            f"USING {self.index_type.upper()}" if self.index_type != "btree" else ""
        )
        # Quote all column names to prevent SQL injection
        columns_sql = ", ".join(quote_identifier(col) for col in self.columns)

        # Quote index name and table name
        quoted_index_name = quote_identifier(self.name)
        quoted_table_name = quote_identifier(table_name)

        parts = [
            f"CREATE {unique_clause}INDEX IF NOT EXISTS {quoted_index_name}",
            f"ON {quoted_table_name}",
        ]

        if using_clause:
            parts.append(using_clause)

        parts.append(f"({columns_sql})")

        if self.where_clause:
            # Note: where_clause is trusted SQL expression from contract.yaml
            # It is raw SQL by design and should only come from trusted sources
            parts.append(f"WHERE {self.where_clause}")

        return " ".join(parts)


__all__ = ["ModelProjectorIndex"]
