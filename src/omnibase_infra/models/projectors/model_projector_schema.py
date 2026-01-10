# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Schema Model.

Defines the complete schema for a projection table, including columns,
indexes, and constraints. Used by ProjectorSchemaValidator for schema
validation and migration SQL generation.

NOTE: This model is temporarily defined in omnibase_infra until omnibase_core
provides it at omnibase_core.models.projectors. Once available, this should
be moved there and re-exported from this module.

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from omnibase_infra.models.projectors.model_projector_column import (
    ModelProjectorColumn,
)
from omnibase_infra.models.projectors.model_projector_index import ModelProjectorIndex
from omnibase_infra.models.projectors.util_sql_identifiers import (
    IDENT_PATTERN,
    escape_sql_string,
    quote_identifier,
)


class ModelProjectorSchema(BaseModel):
    """Complete schema definition for a projection table.

    Describes the table name, columns, indexes, and constraints for a
    projection table. Used by ProjectorSchemaValidator for schema validation
    and migration SQL generation.

    Attributes:
        table_name: Name of the projection table (snake_case by convention).
        schema_name: PostgreSQL schema name (currently only 'public' supported).
        columns: List of column definitions.
        indexes: List of index definitions (optional).
        schema_version: Schema version string (semver format).

    Schema Limitation:
        Currently only the 'public' schema is supported. This is because:
        1. Most ONEX projections use the default public schema
        2. Multi-schema support requires additional migration complexity
        3. Schema-qualified identifiers need additional SQL generation changes

        Future versions may support custom schemas (see ticket backlog).

    Example:
        >>> from omnibase_infra.models.projectors import (
        ...     ModelProjectorSchema,
        ...     ModelProjectorColumn,
        ...     ModelProjectorIndex,
        ... )
        >>> schema = ModelProjectorSchema(
        ...     table_name="registration_projections",
        ...     columns=[
        ...         ModelProjectorColumn(
        ...             name="entity_id",
        ...             column_type="uuid",
        ...             nullable=False,
        ...             primary_key=True,
        ...         ),
        ...         ModelProjectorColumn(
        ...             name="current_state",
        ...             column_type="varchar",
        ...             length=64,
        ...             nullable=False,
        ...         ),
        ...     ],
        ...     indexes=[
        ...         ModelProjectorIndex(
        ...             name="idx_registration_state",
        ...             columns=["current_state"],
        ...         ),
        ...     ],
        ...     schema_version="1.0.0",
        ... )
    """

    table_name: str = Field(
        ...,
        description="Name of the projection table (snake_case by convention)",
        min_length=1,
        max_length=128,
    )

    schema_name: Literal["public"] = Field(
        default="public",
        description=(
            "PostgreSQL schema name. Currently only 'public' is supported. "
            "This limitation exists because multi-schema support requires "
            "additional migration tooling and schema-qualified SQL generation."
        ),
    )

    columns: list[ModelProjectorColumn] = Field(
        ...,
        description="List of column definitions",
        min_length=1,
    )

    indexes: list[ModelProjectorIndex] = Field(
        default_factory=list,
        description="List of index definitions",
    )

    schema_version: str = Field(
        default="1.0.0",
        description="Schema version string (semver format)",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description for SQL COMMENT ON TABLE",
        max_length=1024,
    )

    model_config = {
        "extra": "forbid",
        # NOTE: frozen=True provides shallow immutability only. The `columns` and
        # `indexes` fields are lists which are protected from reassignment, but list
        # contents could theoretically be modified via index access. However, since
        # the contained objects (ModelProjectorColumn, ModelProjectorIndex) are
        # themselves frozen, practical deep immutability is achieved. For strict
        # deep immutability, consider using tuple[ModelProjectorColumn, ...] instead.
        "frozen": True,
    }

    @field_validator("table_name")
    @classmethod
    def validate_table_name_identifier(cls, v: str) -> str:
        """Validate that the table name is a valid PostgreSQL identifier.

        Prevents SQL injection by ensuring the name matches the safe identifier
        pattern (letters, digits, underscores, starting with letter or underscore).

        Args:
            v: Table name to validate.

        Returns:
            Validated table name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        if not IDENT_PATTERN.match(v):
            raise ValueError(
                f"Invalid table name '{v}': must match pattern "
                "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only, "
                "starting with letter or underscore)"
            )
        return v

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Validate schema_version format and reject line breaks.

        Prevents SQL comment injection by rejecting line breaks and ensures
        the version follows semantic versioning format.

        Args:
            v: Schema version to validate.

        Returns:
            Validated schema version.

        Raises:
            ValueError: If the version contains line breaks or invalid format.
        """
        if "\n" in v or "\r" in v:
            raise ValueError("schema_version must not contain line breaks")
        if not re.fullmatch(r"\d+\.\d+\.\d+", v):
            raise ValueError("schema_version must match semver 'MAJOR.MINOR.PATCH'")
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

    @field_validator("columns")
    @classmethod
    def validate_columns_not_empty(
        cls, v: list[ModelProjectorColumn]
    ) -> list[ModelProjectorColumn]:
        """Validate that columns list is not empty."""
        if not v:
            raise ValueError("Schema must have at least one column")
        return v

    @model_validator(mode="after")
    def validate_primary_key_exists(self) -> ModelProjectorSchema:
        """Validate that at least one column is marked as primary key."""
        primary_keys = [col for col in self.columns if col.primary_key]
        if not primary_keys:
            raise ValueError("Schema must have at least one primary key column")
        return self

    @model_validator(mode="after")
    def validate_column_names_unique(self) -> ModelProjectorSchema:
        """Validate that column names are unique within the schema.

        Uses Counter for O(n) duplicate detection instead of O(n^2) list.count().
        """
        names = [col.name for col in self.columns]
        name_counts = Counter(names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate column names: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_index_columns_exist(self) -> ModelProjectorSchema:
        """Validate that all index columns reference existing column names."""
        column_names = {col.name for col in self.columns}
        for idx in self.indexes:
            for col in idx.columns:
                if col not in column_names:
                    raise ValueError(
                        f"Index '{idx.name}' references non-existent column: {col}"
                    )
        return self

    def get_primary_key_columns(self) -> list[str]:
        """Get list of primary key column names.

        Returns:
            List of column names that form the primary key.
        """
        return [col.name for col in self.columns if col.primary_key]

    def get_column_names(self) -> list[str]:
        """Get list of all column names.

        Returns:
            List of all column names in the schema.
        """
        return [col.name for col in self.columns]

    def to_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement.

        Uses quoted identifiers for table name and column names in the
        primary key constraint to prevent SQL injection.

        Returns:
            SQL CREATE TABLE statement including columns and primary key.

        Example:
            >>> schema.to_create_table_sql()
            'CREATE TABLE IF NOT EXISTS "registration_projections" (...)'
        """
        column_defs = [col.to_sql_definition() for col in self.columns]

        # Add primary key constraint with quoted column names
        pk_columns = self.get_primary_key_columns()
        if pk_columns:
            quoted_pk_columns = ", ".join(quote_identifier(col) for col in pk_columns)
            pk_clause = f"PRIMARY KEY ({quoted_pk_columns})"
            column_defs.append(pk_clause)

        columns_sql = ",\n    ".join(column_defs)

        # Quote table name to prevent SQL injection
        quoted_table_name = quote_identifier(self.table_name)

        return f"CREATE TABLE IF NOT EXISTS {quoted_table_name} (\n    {columns_sql}\n)"

    def to_create_indexes_sql(self) -> list[str]:
        """Generate CREATE INDEX SQL statements for all indexes.

        Returns:
            List of SQL CREATE INDEX statements.
        """
        return [idx.to_sql_definition(self.table_name) for idx in self.indexes]

    def to_comment_statements_sql(self) -> list[str]:
        """Generate SQL COMMENT statements for table and columns.

        Generates COMMENT ON TABLE if the schema has a description,
        and COMMENT ON COLUMN for each column that has a description.
        Uses proper SQL escaping to prevent injection.

        Returns:
            List of SQL COMMENT statements.

        Example:
            >>> schema = ModelProjectorSchema(
            ...     table_name="users",
            ...     description="Stores user data",
            ...     columns=[
            ...         ModelProjectorColumn(
            ...             name="id", column_type="uuid",
            ...             primary_key=True, nullable=False,
            ...             description="Unique identifier",
            ...         ),
            ...     ],
            ... )
            >>> schema.to_comment_statements_sql()
            ['COMMENT ON TABLE "users" IS \\'Stores user data\\'',
             'COMMENT ON COLUMN "users"."id" IS \\'Unique identifier\\'']
        """
        statements: list[str] = []
        quoted_table = quote_identifier(self.table_name)

        # Add table comment if description exists
        if self.description:
            escaped_desc = escape_sql_string(self.description)
            statements.append(f"COMMENT ON TABLE {quoted_table} IS '{escaped_desc}'")

        # Add column comments for columns with descriptions
        for col in self.columns:
            if col.description:
                quoted_col = quote_identifier(col.name)
                escaped_desc = escape_sql_string(col.description)
                statements.append(
                    f"COMMENT ON COLUMN {quoted_table}.{quoted_col} IS '{escaped_desc}'"
                )

        return statements

    def to_full_migration_sql(self) -> str:
        """Generate complete migration SQL including table, indexes, and comments.

        Generates a complete migration script including:
        1. CREATE TABLE statement with all columns (inline comments if descriptions)
        2. CREATE INDEX statements for all defined indexes
        3. COMMENT ON TABLE/COLUMN statements for documentation

        Returns:
            Complete SQL migration script.

        Example:
            >>> schema = ModelProjectorSchema(
            ...     table_name="users",
            ...     description="Stores user projection data",
            ...     columns=[
            ...         ModelProjectorColumn(
            ...             name="id", column_type="uuid",
            ...             primary_key=True, nullable=False,
            ...             description="Unique user identifier",
            ...         ),
            ...     ],
            ... )
            >>> print(schema.to_full_migration_sql())
            -- Migration for users (version 1.0.0)
            -- Generated by ProjectorSchemaValidator
            ...
            COMMENT ON TABLE "users" IS 'Stores user projection data';
            COMMENT ON COLUMN "users"."id" IS 'Unique user identifier';

        Security Notes:
            - All identifiers (table name, column names, index names) are validated
              during model construction and quoted in SQL output.
            - schema_version is validated to semver format and must not contain line
              breaks (prevents SQL comment injection).
            - description fields are validated to reject line breaks and are properly
              escaped for SQL string literals.
            - TRUST BOUNDARIES exist for: column default values, index where_clause.
              These accept raw SQL and must only come from trusted contract.yaml sources.
        """
        parts = [
            f"-- Migration for {self.table_name} (version {self.schema_version})",
            "-- Generated by ProjectorSchemaValidator",
            "",
            self.to_create_table_sql() + ";",
            "",
        ]

        index_statements = self.to_create_indexes_sql()
        if index_statements:
            parts.append("-- Indexes")
            for stmt in index_statements:
                parts.append(stmt + ";")
            parts.append("")

        comment_statements = self.to_comment_statements_sql()
        if comment_statements:
            parts.append("-- Comments")
            for stmt in comment_statements:
                parts.append(stmt + ";")

        return "\n".join(parts)


__all__ = ["ModelProjectorSchema"]
