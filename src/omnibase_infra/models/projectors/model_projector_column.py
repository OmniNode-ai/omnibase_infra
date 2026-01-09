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

from pydantic import BaseModel, Field


class ModelProjectorColumn(BaseModel):
    """Definition of a single column in a projection table.

    Describes the column name, data type, constraints, and default value.
    Used by ProjectorSchemaManager for schema validation and migration
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

    def to_sql_definition(self) -> str:
        """Generate SQL column definition for CREATE TABLE statement.

        Returns:
            SQL column definition string (e.g., "entity_id UUID NOT NULL").

        Example:
            >>> column = ModelProjectorColumn(
            ...     name="entity_id",
            ...     column_type="uuid",
            ...     nullable=False,
            ... )
            >>> column.to_sql_definition()
            'entity_id UUID NOT NULL'
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
        parts = [self.name, sql_type]

        if not self.nullable:
            parts.append("NOT NULL")

        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")

        return " ".join(parts)


__all__ = ["ModelProjectorColumn"]
