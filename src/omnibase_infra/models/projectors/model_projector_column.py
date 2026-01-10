# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Column Model.

Defines the schema for individual columns within a projection table.
Used by ModelProjectorSchema to describe table structure for validation
and migration SQL generation.

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


class ModelProjectorColumn(BaseModel):
    """Definition of a single column in a projection table.

    Describes the column name, data type, constraints, and default value.
    Used by ProjectorSchemaValidator for schema validation and migration
    SQL generation.

    Attributes:
        name: Column name (snake_case by convention).
        column_type: PostgreSQL data type for the column.
        nullable: Whether the column allows NULL values (default: True).
        default: Optional default value expression (SQL literal or expression).
        primary_key: Whether this column is part of the primary key (default: False).

    Example:
        >>> column = ModelProjectorColumn(
        ...     name="entity_id",
        ...     column_type="uuid",
        ...     nullable=False,
        ...     primary_key=True,
        ... )
        >>> print(column.name)
        'entity_id'
    """

    name: str = Field(
        ...,
        description="Column name (snake_case by convention)",
        min_length=1,
        max_length=128,
    )

    column_type: Literal[
        "uuid",
        "varchar",
        "text",
        "integer",
        "bigint",
        "timestamp",
        "timestamptz",
        "jsonb",
        "boolean",
    ] = Field(
        ...,
        description="PostgreSQL data type for the column",
    )

    nullable: bool = Field(
        default=True,
        description="Whether the column allows NULL values",
    )

    default: str | None = Field(
        default=None,
        description="Default value expression (SQL literal or expression)",
    )

    primary_key: bool = Field(
        default=False,
        description="Whether this column is part of the primary key",
    )

    length: int | None = Field(
        default=None,
        description="Length for varchar columns (e.g., 128 for VARCHAR(128))",
        ge=1,
        le=65535,
    )

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }

    @field_validator("name")
    @classmethod
    def validate_name_identifier(cls, v: str) -> str:
        """Validate that the column name is a valid PostgreSQL identifier.

        Prevents SQL injection by ensuring the name matches the safe identifier
        pattern (letters, digits, underscores, starting with letter or underscore).

        Args:
            v: Column name to validate.

        Returns:
            Validated column name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        if not IDENT_PATTERN.match(v):
            raise ValueError(
                f"Invalid column name '{v}': must match pattern "
                "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only, "
                "starting with letter or underscore)"
            )
        return v

    @field_validator("default")
    @classmethod
    def validate_default(cls, v: str | None) -> str | None:
        """Validate default value for SQL safety.

        Prevents SQL injection by rejecting line breaks in default values.
        Default values are raw SQL expressions by design (e.g., 'now()', 'true'),
        so contract sources must be trusted. This validator prevents accidental
        line breaks that could enable multi-statement injection.

        Args:
            v: Default value to validate.

        Returns:
            Validated default value.

        Raises:
            ValueError: If the default contains line breaks.
        """
        if v is None:
            return v
        if "\n" in v or "\r" in v:
            raise ValueError("default value must not contain line breaks")
        return v

    def to_sql_definition(self) -> str:
        """Generate SQL column definition for CREATE TABLE statement.

        Uses quoted identifiers to prevent SQL injection.

        Returns:
            SQL column definition string (e.g., '"entity_id" UUID NOT NULL').

        Example:
            >>> column = ModelProjectorColumn(
            ...     name="entity_id",
            ...     column_type="uuid",
            ...     nullable=False,
            ... )
            >>> column.to_sql_definition()
            '"entity_id" UUID NOT NULL'
        """
        # Map column_type to PostgreSQL type with length
        type_map: dict[str, str] = {
            "uuid": "UUID",
            "varchar": f"VARCHAR({self.length or 255})",
            "text": "TEXT",
            "integer": "INTEGER",
            "bigint": "BIGINT",
            "timestamp": "TIMESTAMP",
            "timestamptz": "TIMESTAMPTZ",
            "jsonb": "JSONB",
            "boolean": "BOOLEAN",
        }

        sql_type = type_map[self.column_type]
        # Quote the column name to prevent SQL injection
        quoted_name = quote_identifier(self.name)
        parts = [quoted_name, sql_type]

        if not self.nullable:
            parts.append("NOT NULL")

        if self.default is not None:
            # Note: default is trusted SQL expression from contract.yaml
            parts.append(f"DEFAULT {self.default}")

        return " ".join(parts)


__all__ = ["ModelProjectorColumn"]
