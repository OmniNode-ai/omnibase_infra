# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Schema Validator.

Validates projection table schemas with validation-only semantics.
Does NOT auto-create tables - only validates schemas exist and
generates migration SQL for manual application.

This design follows the principle of explicit schema management:
- Production schemas should be managed via migration scripts
- Runtime should validate schemas exist, not create them
- Migration SQL is generated for review before application

NOTE: Schema models are currently defined in omnibase_infra.models.projectors.
Once omnibase_core provides them at omnibase_core.models.projectors, the imports
should be updated accordingly.

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)

# NOTE: Import from omnibase_infra.models.projectors until omnibase_core provides them.
# Future import path: from omnibase_core.models.projectors import ModelProjectorSchema
from omnibase_infra.models.projectors import ModelProjectorSchema

logger = logging.getLogger(__name__)


# =============================================================================
# PROJECTOR SCHEMA ERROR
# =============================================================================


class ProjectorSchemaError(RuntimeHostError):
    """Raised when projection schema validation fails.

    Used for:
    - Missing projection tables
    - Missing required columns
    - Schema mismatch errors

    The error message includes a hint about running migrations to resolve
    the issue.

    Example:
        >>> raise ProjectorSchemaError(
        ...     "Table 'registration_projections' does not exist. "
        ...     "Run migration first: psql -f schema_registration_projection.sql",
        ...     context=context,
        ...     table_name="registration_projections",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ProjectorSchemaError.

        Args:
            message: Human-readable error message with migration hint.
            context: Bundled infrastructure context.
            **extra_context: Additional context (table_name, missing_columns, etc.).
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            context=context,
            **extra_context,
        )


# =============================================================================
# TYPE MAPPING
# =============================================================================

# Mapping from ModelProjectorColumn types to PostgreSQL type strings
_POSTGRES_TYPE_MAP: dict[str, str] = {
    "uuid": "uuid",
    "varchar": "character varying",
    "text": "text",
    "integer": "integer",
    "bigint": "bigint",
    "timestamp": "timestamp without time zone",
    "timestamptz": "timestamp with time zone",
    "jsonb": "jsonb",
    "boolean": "boolean",
}


# =============================================================================
# PROJECTOR SCHEMA VALIDATOR
# =============================================================================


class ProjectorSchemaValidator:
    """Validates projection table schemas.

    This class implements ProtocolProjectorSchemaValidator from omnibase_infra.protocols.
    Protocol compliance is verified via duck typing (@runtime_checkable).

    NOTE: Auto-migration is disallowed in core runtime.
    This class validates schemas exist, does NOT auto-create.

    The validator provides:
    - Schema validation (ensure_schema_exists): Verifies table and columns exist
    - Migration generation (generate_migration): Creates SQL for manual application
    - Table existence checks (table_exists): Low-level table verification
    - Column introspection (_get_table_columns): Lists existing columns

    Protocol Compliance:
        Implements ProtocolProjectorSchemaValidator with:
        - ensure_schema_exists(schema, correlation_id) -> None
        - table_exists(table_name, correlation_id, schema_name=None) -> bool

    Design Philosophy:
        Production database schemas should be managed through explicit migration
        scripts that are reviewed and applied manually. This class supports that
        workflow by:
        1. Validating that required schemas exist at runtime
        2. Generating migration SQL when schemas are missing
        3. Providing clear error messages with actionable hints

    Schema Support:
        Currently defaults to 'public' schema. Both table_exists() and
        _get_table_columns() accept an optional schema_name parameter for
        custom schema support.

    Thread Safety:
        This class is coroutine-safe for concurrent async calls. Uses asyncpg
        connection pool for connection management. Not thread-safe - for
        multi-threaded access, additional synchronization would be required.

    Example:
        >>> pool = await asyncpg.create_pool(dsn)
        >>> validator = ProjectorSchemaValidator(pool)
        >>>
        >>> # Validate schema exists
        >>> try:
        ...     await validator.ensure_schema_exists(schema)
        ... except ProjectorSchemaError as e:
        ...     print(f"Schema missing: {e}")
        ...     print(await validator.generate_migration(schema))
        >>>
        >>> # Generate migration for new schema
        >>> migration_sql = await validator.generate_migration(schema)
        >>> print(migration_sql)
    """

    def __init__(self, db_pool: asyncpg.Pool) -> None:
        """Initialize schema manager with database connection pool.

        Args:
            db_pool: asyncpg connection pool for database access.
                     Pool should be created by the caller (e.g., from HandlerDb).
        """
        self._pool = db_pool

    async def ensure_schema_exists(
        self,
        schema: ModelProjectorSchema,
        correlation_id: UUID,
    ) -> None:
        """Verify that the projection table schema exists.

        Checks that:
        1. The table exists in the database
        2. All required columns exist in the table

        This method does NOT auto-create missing schemas. If the schema is
        missing or incomplete, it raises ProjectorSchemaError with a helpful
        message including migration hints.

        Args:
            schema: Projector schema to validate.
            correlation_id: Correlation ID for distributed tracing. Required
                to ensure proper observability across service boundaries.

        Raises:
            ProjectorSchemaError: If table does not exist or required columns
                are missing. Error message includes migration command hint.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If validation query times out.

        Example:
            >>> try:
            ...     await manager.ensure_schema_exists(schema, correlation_id)
            ...     print("Schema valid")
            ... except ProjectorSchemaError as e:
            ...     print(f"Migration needed: {e}")
        """
        corr_id = str(correlation_id)
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="ensure_schema_exists",
            target_name=f"schema.{schema.table_name}",
            correlation_id=corr_id,
        )

        # Check table exists
        exists = await self.table_exists(schema.table_name, correlation_id)
        if not exists:
            migration_hint = (
                "Run migration first. Generate migration SQL with:\n"
                "  manager.generate_migration(schema)\n"
                "Or apply manually with psql."
            )
            raise ProjectorSchemaError(
                f"Table '{schema.table_name}' does not exist. {migration_hint}",
                context=ctx,
                table_name=schema.table_name,
            )

        # Check columns exist
        existing_columns = await self._get_table_columns(
            schema.table_name, correlation_id
        )
        required_columns = schema.get_column_names()
        missing_columns = [
            col for col in required_columns if col not in existing_columns
        ]

        if missing_columns:
            migration_hint = (
                f"Table '{schema.table_name}' is missing required columns. "
                f"Run ALTER TABLE migration to add missing columns."
            )
            raise ProjectorSchemaError(
                f"Table '{schema.table_name}' is missing columns: {missing_columns}. "
                f"{migration_hint}",
                context=ctx,
                table_name=schema.table_name,
                missing_columns=missing_columns,
                existing_columns=existing_columns,
            )

        logger.debug(
            "Schema validation passed",
            extra={
                "table_name": schema.table_name,
                "column_count": len(required_columns),
                "correlation_id": corr_id,
            },
        )

    def validate_schema_deeply(
        self,
        schema: ModelProjectorSchema,
    ) -> list[str]:
        """Perform deep validation of schema (column types, indexes, constraints).

        Provides additional validation beyond basic Pydantic model validation.
        Checks for:
        - Column types are valid PostgreSQL types
        - Index columns reference existing table columns
        - Primary key columns are not nullable
        - Varchar columns have reasonable lengths
        - Index names follow naming convention

        This method is synchronous as it only validates the schema model
        without making database calls. Use ensure_schema_exists() for
        database-level validation.

        Args:
            schema: Projector schema to validate deeply.

        Returns:
            List of validation warnings (empty if all valid).
            Warnings are informational - they don't block migration generation
            but may indicate potential issues.

        Example:
            >>> warnings = validator.validate_schema_deeply(schema)
            >>> if warnings:
            ...     for warning in warnings:
            ...         print(f"Warning: {warning}")
            ... else:
            ...     print("Schema passes deep validation")
        """
        warnings: list[str] = []

        # Validate column types are valid PostgreSQL types
        # Note: ModelProjectorColumn already uses Literal for column_type,
        # so this is a redundant safety check. The valid types match those
        # defined in _POSTGRES_TYPE_MAP.
        valid_types = {
            "uuid",
            "varchar",
            "text",
            "jsonb",
            "timestamp",
            "timestamptz",
            "integer",
            "bigint",
            "boolean",
        }
        for col in schema.columns:
            # column_type is already constrained by Literal, but we check anyway
            # for defense-in-depth (e.g., if model is constructed via dict)
            col_type = col.column_type.lower()
            if col_type not in valid_types:
                warnings.append(
                    f"Column '{col.name}' has potentially unsupported type: "
                    f"{col.column_type}"
                )

        # Validate index columns exist in the table
        # Note: This is also validated by ModelProjectorSchema.validate_index_columns_exist
        # but we include it here for completeness when validating external schemas.
        column_names = {col.name for col in schema.columns}
        for idx in schema.indexes:
            for idx_col in idx.columns:
                if idx_col not in column_names:
                    warnings.append(
                        f"Index '{idx.name}' references non-existent column: {idx_col}"
                    )

        # Validate varchar columns have reasonable length
        for col in schema.columns:
            if col.column_type == "varchar":
                if col.length is not None and col.length > 10000:
                    warnings.append(
                        f"Column '{col.name}' has very large varchar length "
                        f"({col.length}). Consider using TEXT type instead."
                    )

        # Validate primary key columns are not nullable
        for col in schema.columns:
            if col.primary_key and col.nullable:
                warnings.append(
                    f"Primary key column '{col.name}' is marked as nullable. "
                    "Primary key columns should be NOT NULL."
                )

        # Validate index names follow naming convention
        for idx in schema.indexes:
            if not idx.name.startswith("idx_"):
                warnings.append(
                    f"Index '{idx.name}' does not follow 'idx_' naming convention"
                )

        logger.debug(
            "Deep schema validation completed",
            extra={
                "table_name": schema.table_name,
                "warning_count": len(warnings),
            },
        )

        return warnings

    def generate_migration(
        self,
        schema: ModelProjectorSchema,
        correlation_id: UUID,
    ) -> str:
        """Generate CREATE TABLE SQL for manual application.

        Generates a complete migration script including:
        1. CREATE TABLE statement with all columns
        2. CREATE INDEX statements for all defined indexes

        The generated SQL uses IF NOT EXISTS clauses to be idempotent.
        This SQL should be reviewed before applying to production.

        Note:
            This method is synchronous as it only performs string generation
            without any I/O operations.

        Args:
            schema: Projector schema to generate migration for.
            correlation_id: Correlation ID for distributed tracing (for logging).

        Returns:
            Complete SQL migration script as a string.

        Example:
            >>> migration_sql = manager.generate_migration(schema, correlation_id)
            >>> print(migration_sql)
            -- Migration for registration_projections (version 1.0.0)
            -- Generated by ProjectorSchemaValidator
            ...
        """
        corr_id = str(correlation_id)

        logger.debug(
            "Generating migration SQL",
            extra={
                "table_name": schema.table_name,
                "schema_version": schema.schema_version,
                "correlation_id": corr_id,
            },
        )

        # Use the schema's built-in SQL generation
        return schema.to_full_migration_sql()

    async def table_exists(
        self,
        table_name: str,
        correlation_id: UUID,
        schema_name: str | None = None,
    ) -> bool:
        """Check if a table exists in the database.

        Queries the PostgreSQL information_schema to verify table existence.

        Args:
            table_name: Name of the table to check.
            correlation_id: Correlation ID for distributed tracing. Required
                to ensure proper observability across service boundaries.
            schema_name: Optional database schema name. Defaults to 'public'.

        Returns:
            True if table exists, False otherwise.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.
        """
        corr_id = str(correlation_id)
        effective_schema = schema_name or "public"
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="table_exists",
            target_name=f"{effective_schema}.{table_name}",
            correlation_id=corr_id,
        )

        query = """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = $1
                AND table_name = $2
            )
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(query, effective_schema, table_name)
                return bool(result)

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Failed to connect to database for table existence check",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="table_exists",
                target_name=f"{effective_schema}.{table_name}",
                correlation_id=correlation_id,
                # timeout_seconds omitted - database timeout is connection-pool level, not available here
            )
            raise InfraTimeoutError(
                "Table existence check timed out",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to check table existence: {type(e).__name__}",
                context=ctx,
            ) from e

    async def _get_table_columns(
        self,
        table_name: str,
        correlation_id: UUID,
        schema_name: str | None = None,
    ) -> list[str]:
        """Get list of existing column names for a table.

        Queries the PostgreSQL information_schema to retrieve all column
        names for the specified table.

        Args:
            table_name: Name of the table to inspect.
            correlation_id: Correlation ID for distributed tracing. Required
                to ensure proper observability across service boundaries.
            schema_name: Optional database schema name. Defaults to 'public'.

        Returns:
            List of column names in the table. Empty list if table doesn't exist.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.
        """
        corr_id = str(correlation_id)
        effective_schema = schema_name or "public"
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_table_columns",
            target_name=f"{effective_schema}.{table_name}",
            correlation_id=corr_id,
        )

        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = $1
            AND table_name = $2
            ORDER BY ordinal_position
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, effective_schema, table_name)
                return [row["column_name"] for row in rows]

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Failed to connect to database for column introspection",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_table_columns",
                target_name=f"{effective_schema}.{table_name}",
                correlation_id=correlation_id,
                # timeout_seconds omitted - database timeout is connection-pool level, not available here
            )
            raise InfraTimeoutError(
                "Column introspection timed out",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to get table columns: {type(e).__name__}",
                context=ctx,
            ) from e


__all__ = [
    "ProjectorSchemaError",
    "ProjectorSchemaValidator",
]
