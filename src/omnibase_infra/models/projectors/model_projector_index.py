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
    validate_identifier,
)


class ModelProjectorIndex(BaseModel):
    """Definition of a database index on a projection table.

    Describes the index name, columns, type, and optional partial index
    predicate. Used by ProjectorSchemaValidator for schema validation and
    migration SQL generation.

    Attributes:
        name: Index name (must be unique within the database).
        columns: List of column names included in the index.
        index_type: PostgreSQL index type (btree, gin, hash).
        unique: Whether the index enforces uniqueness (default: False).
        where_clause: Optional partial index predicate (SQL expression).
        description: Optional human-readable description for documentation.

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
        description=(
            "Optional partial index predicate (SQL expression). "
            "TRUST BOUNDARY: This field accepts raw SQL and must only come from "
            "trusted contract.yaml sources. Do NOT populate from user input. "
            "Line breaks are rejected to prevent multi-statement injection."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description for documentation purposes",
        max_length=1024,
    )

    model_config = {
        "extra": "forbid",
        # NOTE: frozen=True provides shallow immutability only. The `columns` field
        # is a list which is protected from reassignment, but list elements could
        # theoretically be modified via index access (e.g., idx.columns[0] = "x").
        # In practice, string elements are immutable, so this is not a concern.
        # For deep immutability, use tuple instead of list.
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

    @field_validator("where_clause")
    @classmethod
    def validate_where_clause(cls, v: str | None) -> str | None:
        """Validate where_clause for SQL safety.

        Prevents multi-statement injection by rejecting line breaks. The where_clause
        is a trust boundary that accepts raw SQL expressions by design (for partial
        index predicates), so contract sources must be trusted. This validator
        prevents accidental line breaks that could enable multi-statement injection.

        Args:
            v: Where clause to validate.

        Returns:
            Validated where clause.

        Raises:
            ValueError: If the where_clause contains line breaks.
        """
        if v is None:
            return v
        if "\n" in v or "\r" in v:
            raise ValueError("where_clause must not contain line breaks")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Validate description for SQL safety.

        Prevents potential injection by rejecting line breaks in description values.

        Args:
            v: Description value to validate.

        Returns:
            Validated description value.

        Raises:
            ValueError: If the description contains line breaks.
        """
        if v is None:
            return v
        if "\n" in v or "\r" in v:
            raise ValueError("description must not contain line breaks")
        return v

    def to_sql_definition(self, table_name: str) -> str:
        """Generate SQL CREATE INDEX statement.

        Uses quoted identifiers for index name, table name, and column names
        to prevent SQL injection. Validates table_name for defense-in-depth.

        Args:
            table_name: Name of the table to create the index on. Must be a valid
                PostgreSQL identifier (letters, digits, underscores only).

        Returns:
            SQL CREATE INDEX statement with properly quoted identifiers.

        Raises:
            ValueError: If table_name is not a valid PostgreSQL identifier.

        Example:
            >>> index = ModelProjectorIndex(
            ...     name="idx_registration_state",
            ...     columns=["current_state"],
            ...     index_type="btree",
            ... )
            >>> index.to_sql_definition("registration_projections")
            'CREATE INDEX IF NOT EXISTS "idx_registration_state" ON "registration_projections" USING BTREE ("current_state")'

        Security Notes:
            - All identifiers (index name, table name, columns) are validated
              against the PostgreSQL identifier pattern and quoted.
            - The where_clause field is a **TRUST BOUNDARY**: It accepts raw SQL
              expressions by design for partial index predicates. Only accept
              where_clause values from trusted sources (contract.yaml files).
              Untrusted input in where_clause could enable SQL injection.
        """
        # Defense-in-depth: validate table_name even though callers should validate
        validate_identifier(table_name, "table name")

        unique_clause = "UNIQUE " if self.unique else ""
        # Always emit USING clause for explicitness, even for btree (the default)
        # This makes the intent clear and prevents ambiguity in migrations
        using_clause = f"USING {self.index_type.upper()}"
        # Quote all column names to prevent SQL injection
        columns_sql = ", ".join(quote_identifier(col) for col in self.columns)

        # Quote index name and table name
        quoted_index_name = quote_identifier(self.name)
        quoted_table_name = quote_identifier(table_name)

        parts = [
            f"CREATE {unique_clause}INDEX IF NOT EXISTS {quoted_index_name}",
            f"ON {quoted_table_name}",
            using_clause,
            f"({columns_sql})",
        ]

        if self.where_clause:
            # TRUST BOUNDARY: where_clause is raw SQL from contract.yaml
            # This enables partial index predicates like "deleted_at IS NULL"
            # SECURITY: Only accept where_clause from trusted contract sources
            # Do NOT accept user input here - it would enable SQL injection
            parts.append(f"WHERE {self.where_clause}")

        return " ".join(parts)


__all__ = ["ModelProjectorIndex"]
