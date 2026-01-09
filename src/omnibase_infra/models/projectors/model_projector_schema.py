# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Schema Model.

Defines the complete schema for a projection table, including columns,
indexes, and constraints. Used by ProjectorSchemaManager for schema
validation and migration SQL generation.

NOTE: This model is temporarily defined in omnibase_infra until omnibase_core
provides it at omnibase_core.models.projectors. Once available, this should
be moved there and re-exported from this module.

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from omnibase_infra.models.projectors.model_projector_column import (
    ModelProjectorColumn,
)
from omnibase_infra.models.projectors.model_projector_index import ModelProjectorIndex
from omnibase_infra.models.projectors.util_sql_identifiers import (
    IDENT_PATTERN,
    quote_identifier,
)


class ModelProjectorSchema(BaseModel):
    """Complete schema definition for a projection table.

    Describes the table name, columns, indexes, and constraints for a
    projection table. Used by ProjectorSchemaManager for schema validation
    and migration SQL generation.

    Attributes:
        table_name: Name of the projection table (snake_case by convention).
        columns: List of column definitions.
        indexes: List of index definitions (optional).
        schema_version: Schema version string (semver format).

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

    model_config = {
        "extra": "forbid",
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
        """Validate that column names are unique within the schema."""
        names = [col.name for col in self.columns]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
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

    def to_full_migration_sql(self) -> str:
        """Generate complete migration SQL including table and indexes.

        Returns:
            Complete SQL migration script.
        """
        parts = [
            f"-- Migration for {self.table_name} (version {self.schema_version})",
            "-- Generated by ProjectorSchemaManager",
            "",
            self.to_create_table_sql() + ";",
            "",
        ]

        index_statements = self.to_create_indexes_sql()
        if index_statements:
            parts.append("-- Indexes")
            for stmt in index_statements:
                parts.append(stmt + ";")

        return "\n".join(parts)


__all__ = ["ModelProjectorSchema"]
