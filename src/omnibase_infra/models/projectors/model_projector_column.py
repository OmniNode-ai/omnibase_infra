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

from pydantic import BaseModel, Field, field_validator, model_validator

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
        description=(
            "Default value expression (SQL literal or expression). "
            "TRUST BOUNDARY: This field accepts raw SQL expressions (e.g., 'now()', "
            "'true', 'gen_random_uuid()') and must only come from trusted contract.yaml "
            "sources. Do NOT populate from user input. Line breaks are rejected to "
            "prevent multi-statement injection."
        ),
    )

    primary_key: bool = Field(
        default=False,
        description="Whether this column is part of the primary key",
    )

    length: int | None = Field(
        default=None,
        description=(
            "Length for varchar columns (e.g., 128 for VARCHAR(128)). "
            "Required when column_type='varchar' to prevent schema drift. "
            "PostgreSQL maximum is 10485760, but values above 65535 are rarely needed."
        ),
        ge=1,
        le=10485760,  # PostgreSQL maximum varchar length
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description for SQL COMMENT",
        max_length=1024,
    )

    model_config = {
        "extra": "forbid",
        # NOTE: frozen=True provides shallow immutability only. Fields with mutable
        # types (e.g., list, dict) are protected from reassignment but their contents
        # can still be modified. For this model, all fields are primitives (str, bool,
        # int, None), so this limitation does not apply.
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

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Validate description for SQL safety.

        Prevents SQL injection by rejecting line breaks in description values.
        Descriptions are used in SQL COMMENT statements.

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

    @model_validator(mode="after")
    def validate_varchar_has_length(self) -> ModelProjectorColumn:
        """Validate that varchar columns specify an explicit length.

        Prevents schema drift by requiring explicit length specification for
        varchar columns. Without this, different environments might use
        different defaults, causing migration inconsistencies.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If column_type is 'varchar' but length is not specified.
        """
        if self.column_type == "varchar" and self.length is None:
            raise ValueError(
                f"Column '{self.name}' with type 'varchar' must specify an explicit "
                "length to prevent schema drift (e.g., length=255)"
            )
        return self

    def to_sql_definition(self) -> str:
        """Generate SQL column definition for CREATE TABLE statement.

        Uses quoted identifiers to prevent SQL injection. If a description
        is provided, appends an inline SQL comment.

        Returns:
            SQL column definition string (e.g., '"entity_id" UUID NOT NULL').
            If description is set, includes inline comment:
            '"entity_id" UUID NOT NULL  -- Unique identifier'

        Example:
            >>> column = ModelProjectorColumn(
            ...     name="entity_id",
            ...     column_type="uuid",
            ...     nullable=False,
            ... )
            >>> column.to_sql_definition()
            '"entity_id" UUID NOT NULL'

            >>> column_with_desc = ModelProjectorColumn(
            ...     name="name",
            ...     column_type="varchar",
            ...     length=255,
            ...     description="Display name for the entity",
            ... )
            >>> column_with_desc.to_sql_definition()
            '"name" VARCHAR(255)  -- Display name for the entity'

        Security Notes:
            - Column name is validated against PostgreSQL identifier pattern
              during model construction and quoted in SQL output.
            - The default field is a **TRUST BOUNDARY**: It accepts raw SQL
              expressions by design (e.g., 'now()', 'gen_random_uuid()').
              Only accept default values from trusted sources (contract.yaml).
            - The description field is validated to reject line breaks and
              is only used in SQL comments (safe context).
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

        definition = " ".join(parts)

        # Add inline comment if description is provided
        if self.description:
            definition = f"{definition}  -- {self.description}"

        return definition


__all__ = ["ModelProjectorColumn"]
