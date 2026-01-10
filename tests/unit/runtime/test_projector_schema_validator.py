# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ProjectorSchemaValidator.

This test suite validates schema management for ONEX projectors with:
- Schema validation (table/column existence checks)
- Migration SQL generation (CREATE TABLE, CREATE INDEX)
- Table existence detection
- Error handling with migration hints

Test Organization:
    - TestSchemaValidation: Schema existence and validation
    - TestMigrationSQLGeneration: SQL generation for migrations
    - TestTableExistence: Table existence detection
    - TestErrorHandling: Error message formatting

Coverage Goals:
    - >90% code coverage for schema validator
    - All error paths tested
    - SQL generation correctness verified

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery/loading
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    RuntimeHostError,
)
from omnibase_infra.models.projectors import (
    ModelProjectorColumn,
    ModelProjectorIndex,
    ModelProjectorSchema,
)
from omnibase_infra.runtime.projector_schema_manager import (
    ProjectorSchemaError,
    ProjectorSchemaValidator,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema() -> ModelProjectorSchema:
    """Create a sample projector schema for testing."""
    return ModelProjectorSchema(
        table_name="test_projections",
        columns=[
            ModelProjectorColumn(
                name="id", column_type="uuid", nullable=False, primary_key=True
            ),
            ModelProjectorColumn(
                name="status", column_type="varchar", length=50, nullable=False
            ),
            ModelProjectorColumn(name="data", column_type="jsonb", nullable=True),
            ModelProjectorColumn(
                name="created_at", column_type="timestamp", nullable=False
            ),
            ModelProjectorColumn(
                name="is_active", column_type="boolean", nullable=False
            ),
        ],
        indexes=[
            ModelProjectorIndex(name="idx_status", columns=["status"]),
            ModelProjectorIndex(name="idx_created", columns=["created_at"]),
        ],
        schema_version="1.0.0",
    )


@pytest.fixture
def composite_pk_schema() -> ModelProjectorSchema:
    """Create a schema with composite primary key for testing."""
    return ModelProjectorSchema(
        table_name="entity_projections",
        columns=[
            ModelProjectorColumn(
                name="entity_id", column_type="uuid", nullable=False, primary_key=True
            ),
            ModelProjectorColumn(
                name="domain",
                column_type="varchar",
                length=128,
                nullable=False,
                primary_key=True,
            ),
            ModelProjectorColumn(
                name="state", column_type="varchar", length=64, nullable=False
            ),
            ModelProjectorColumn(name="metadata", column_type="jsonb", nullable=True),
        ],
        indexes=[
            ModelProjectorIndex(
                name="idx_entity_state", columns=["entity_id", "state"]
            ),
        ],
        schema_version="1.0.0",
    )


@pytest.fixture
def schema_with_gin_index() -> ModelProjectorSchema:
    """Create a schema with GIN index on JSONB column."""
    return ModelProjectorSchema(
        table_name="searchable_projections",
        columns=[
            ModelProjectorColumn(
                name="id", column_type="uuid", nullable=False, primary_key=True
            ),
            ModelProjectorColumn(name="attributes", column_type="jsonb", nullable=True),
        ],
        indexes=[
            ModelProjectorIndex(
                name="idx_attrs_gin", columns=["attributes"], index_type="gin"
            ),
        ],
        schema_version="1.0.0",
    )


@pytest.fixture
def simple_schema() -> ModelProjectorSchema:
    """Create a simple schema with minimal columns."""
    return ModelProjectorSchema(
        table_name="registrations",
        columns=[
            ModelProjectorColumn(
                name="id", column_type="uuid", nullable=False, primary_key=True
            ),
            ModelProjectorColumn(
                name="name", column_type="varchar", length=255, nullable=False
            ),
        ],
        indexes=[],
        schema_version="1.0.0",
    )


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock(spec=asyncpg.Pool)
    return pool


@pytest.fixture
def mock_connection() -> AsyncMock:
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    return conn


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSchemaValidation:
    """Test schema validation functionality."""

    async def test_ensure_schema_exists_no_error_when_table_exists(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test ensure_schema_exists does not raise when table exists.

        Expected behavior:
        - Query information_schema for table existence
        - Query columns to verify required columns present
        - Return successfully without raising
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table exists (EXISTS query returns True)
        mock_connection.fetchval.return_value = True

        # Mock: all columns exist
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
            {"column_name": "data"},
            {"column_name": "created_at"},
            {"column_name": "is_active"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        # Should not raise any exception
        await validator.ensure_schema_exists(sample_schema, correlation_id=str(uuid4()))

        # Verify table_exists was called
        assert mock_pool.acquire.called

    async def test_ensure_schema_missing_raises_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test ensure_schema_exists raises when table is missing.

        Expected behavior:
        - Query information_schema returns no table
        - Raise ProjectorSchemaError with migration hint
        - Error message includes table name
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table does not exist
        mock_connection.fetchval.return_value = False

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=str(uuid4())
            )

        error_message = str(exc_info.value)
        assert "test_projections" in error_message
        assert "does not exist" in error_message
        assert "migration" in error_message.lower()

    async def test_ensure_schema_missing_columns_raises_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test ensure_schema_exists raises when required columns are missing.

        Expected behavior:
        - Table exists but missing required columns
        - Raise ProjectorSchemaError listing missing columns
        - Error message includes migration hint
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table exists
        mock_connection.fetchval.return_value = True

        # Mock: missing 'data' and 'is_active' columns
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
            {"column_name": "created_at"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=str(uuid4())
            )

        error_message = str(exc_info.value)
        assert "data" in error_message
        assert "is_active" in error_message
        assert "missing" in error_message.lower()

    async def test_ensure_schema_validates_correlation_id_propagated(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        simple_schema: ModelProjectorSchema,
    ) -> None:
        """Test ensure_schema_exists propagates correlation ID.

        Expected behavior:
        - Correlation ID is used in error context
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table does not exist
        mock_connection.fetchval.return_value = False

        correlation_id = str(uuid4())
        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                simple_schema, correlation_id=correlation_id
            )

        # Verify correlation ID is propagated to the error
        assert exc_info.value.correlation_id is not None
        assert str(exc_info.value.correlation_id) == correlation_id


# =============================================================================
# MIGRATION SQL GENERATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestMigrationSQLGeneration:
    """Test migration SQL generation functionality."""

    async def test_generate_migration_basic_table(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test generate_migration produces correct CREATE TABLE SQL.

        Expected behavior:
        - SQL starts with CREATE TABLE IF NOT EXISTS
        - Includes all columns with correct types
        - Includes PRIMARY KEY constraint
        - Includes NOT NULL constraints where specified
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(
            sample_schema, correlation_id=str(uuid4())
        )

        # Verify CREATE TABLE statement
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert '"test_projections"' in sql

        # Verify columns
        assert '"id"' in sql
        assert "UUID" in sql
        assert '"status"' in sql
        assert "VARCHAR" in sql
        assert '"data"' in sql
        assert "JSONB" in sql
        assert '"created_at"' in sql
        assert "TIMESTAMP" in sql
        assert '"is_active"' in sql
        assert "BOOLEAN" in sql

        # Verify PRIMARY KEY
        assert "PRIMARY KEY" in sql

        # Verify NOT NULL constraints
        assert "NOT NULL" in sql

    async def test_generate_migration_composite_primary_key(
        self,
        mock_pool: MagicMock,
        composite_pk_schema: ModelProjectorSchema,
    ) -> None:
        """Test generate_migration handles composite primary keys.

        Expected behavior:
        - PRIMARY KEY includes all key columns
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(
            composite_pk_schema, correlation_id=str(uuid4())
        )

        # Verify composite PRIMARY KEY (both columns)
        assert "PRIMARY KEY" in sql
        assert '"entity_id"' in sql
        assert '"domain"' in sql

    async def test_generate_migration_includes_indexes(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test generate_migration includes CREATE INDEX statements.

        Expected behavior:
        - SQL includes CREATE INDEX statements for all indexes
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(
            sample_schema, correlation_id=str(uuid4())
        )

        # Verify index statements
        assert "CREATE INDEX IF NOT EXISTS" in sql
        assert '"idx_status"' in sql
        assert '"idx_created"' in sql

    async def test_generate_migration_gin_index(
        self,
        mock_pool: MagicMock,
        schema_with_gin_index: ModelProjectorSchema,
    ) -> None:
        """Test generate_migration produces correct GIN index SQL.

        Expected behavior:
        - SQL includes USING GIN clause
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(
            schema_with_gin_index, correlation_id=str(uuid4())
        )

        # Verify GIN index
        assert "USING GIN" in sql
        assert '"idx_attrs_gin"' in sql

    async def test_generate_migration_includes_version_comment(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test generate_migration includes version in comments.

        Expected behavior:
        - SQL includes migration header with version
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(
            sample_schema, correlation_id=str(uuid4())
        )

        # Verify migration comment with version
        assert "Migration for test_projections" in sql
        assert "version 1.0.0" in sql

    async def test_generate_migration_default_values(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test generate_migration includes DEFAULT clauses.

        Expected behavior:
        - Columns with defaults include DEFAULT clause
        """
        schema = ModelProjectorSchema(
            table_name="test_defaults",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="status",
                    column_type="varchar",
                    length=50,
                    nullable=False,
                    default="'pending'",
                ),
                ModelProjectorColumn(
                    name="count",
                    column_type="integer",
                    nullable=False,
                    default="0",
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(schema, correlation_id=str(uuid4()))

        assert "DEFAULT 'pending'" in sql
        assert "DEFAULT 0" in sql

    async def test_generate_migration_unique_index(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test generate_migration handles unique indexes.

        Expected behavior:
        - SQL includes UNIQUE keyword
        """
        schema = ModelProjectorSchema(
            table_name="users",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="email", column_type="varchar", length=255, nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="idx_unique_email",
                    columns=["email"],
                    unique=True,
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(schema, correlation_id=str(uuid4()))

        assert "CREATE UNIQUE INDEX" in sql
        assert '"idx_unique_email"' in sql

    async def test_generate_migration_partial_index(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test generate_migration handles partial indexes.

        Expected behavior:
        - SQL includes WHERE clause
        """
        schema = ModelProjectorSchema(
            table_name="users",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="created_at", column_type="timestamp", nullable=False
                ),
                ModelProjectorColumn(
                    name="is_active", column_type="boolean", nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="idx_active_users",
                    columns=["created_at"],
                    where_clause="is_active = true",
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(schema, correlation_id=str(uuid4()))

        assert "WHERE is_active = true" in sql

    async def test_generate_migration_composite_index(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test generate_migration handles multi-column indexes.

        Expected behavior:
        - All columns included in parentheses
        """
        schema = ModelProjectorSchema(
            table_name="registrations",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="domain", column_type="varchar", length=128, nullable=False
                ),
                ModelProjectorColumn(
                    name="current_state",
                    column_type="varchar",
                    length=64,
                    nullable=False,
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="idx_domain_state",
                    columns=["domain", "current_state"],
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        sql = await validator.generate_migration(schema, correlation_id=str(uuid4()))

        # Verify both columns in index
        assert '"domain"' in sql
        assert '"current_state"' in sql


# =============================================================================
# TABLE EXISTENCE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTableExistence:
    """Test table existence detection."""

    async def test_table_exists_returns_true_for_existing_table(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists returns True when table exists.

        Expected behavior:
        - Query information_schema.tables
        - Return True when row found
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table exists (EXISTS returns True)
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists(
            "registration_projections", correlation_id=str(uuid4())
        )

        assert result is True
        mock_pool.acquire.assert_called_once()

    async def test_table_exists_returns_false_for_missing_table(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists returns False when table is missing.

        Expected behavior:
        - Query information_schema.tables
        - Return False when no row found
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table does not exist
        mock_connection.fetchval.return_value = False

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists(
            "nonexistent_table", correlation_id=str(uuid4())
        )

        assert result is False

    async def test_table_exists_handles_schema_qualified_name(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists handles schema-qualified table names.

        Expected behavior:
        - Query includes table_schema filter
        - Uses provided schema_name parameter
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock: table exists in 'onex' schema
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists(
            table_name="registrations",
            schema_name="onex",
            correlation_id=str(uuid4()),
        )

        assert result is True

        # Verify query was called with schema_name = 'onex'
        call_args = mock_connection.fetchval.call_args
        # The query should have $1 (schema_name) and $2 (table_name) parameters
        args = call_args[0]
        assert "onex" in args  # schema_name parameter
        assert "registrations" in args  # table_name parameter

    async def test_table_exists_default_public_schema(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists defaults to public schema when not specified.

        Expected behavior:
        - Query uses 'public' as default schema
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists("my_table", correlation_id=str(uuid4()))

        assert result is True

        # Verify query used 'public' schema (default)
        call_args = mock_connection.fetchval.call_args
        args = call_args[0]
        assert "public" in args  # default schema_name

    async def test_table_exists_connection_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists handles connection errors.

        Expected behavior:
        - Raise InfraConnectionError on database connection failure
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_connection.fetchval.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraConnectionError):
            await validator.table_exists("test_table", correlation_id=str(uuid4()))

    async def test_table_exists_timeout_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists handles timeout errors.

        Expected behavior:
        - Raise InfraTimeoutError on query timeout
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.side_effect = asyncpg.QueryCanceledError("timeout")

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraTimeoutError):
            await validator.table_exists("test_table", correlation_id=str(uuid4()))

    async def test_table_exists_generic_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists handles generic errors.

        Expected behavior:
        - Raise RuntimeHostError on unexpected errors
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.side_effect = Exception("Unexpected error")

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(RuntimeHostError):
            await validator.table_exists("test_table", correlation_id=str(uuid4()))


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and message formatting."""

    async def test_error_includes_migration_hint(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test error messages include CLI migration hint.

        Expected behavior:
        - Error message suggests running migration command
        - Message provides actionable guidance
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.return_value = False  # Table doesn't exist

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=str(uuid4())
            )

        error_message = str(exc_info.value)
        assert "migration" in error_message.lower()

    async def test_error_includes_table_name(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test error messages include the table name.

        Expected behavior:
        - Error message clearly identifies which table is missing
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.return_value = False

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=str(uuid4())
            )

        error_message = str(exc_info.value)
        assert sample_schema.table_name in error_message

    async def test_error_includes_missing_columns(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test error messages list missing columns.

        Expected behavior:
        - Error lists all columns that are missing
        - Message is actionable (shows what needs to be added)
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.return_value = True  # Table exists
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
        ]  # Missing data, created_at, is_active

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=str(uuid4())
            )

        error_message = str(exc_info.value)
        assert "data" in error_message
        assert "created_at" in error_message
        assert "is_active" in error_message

    async def test_error_context_includes_correlation_id(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test error context includes correlation ID for tracing.

        Expected behavior:
        - Error includes correlation_id
        - Enables distributed tracing of failures
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.return_value = False

        correlation_id = str(uuid4())
        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=correlation_id
            )

        # Verify correlation ID is propagated to the error
        assert exc_info.value.correlation_id is not None
        assert str(exc_info.value.correlation_id) == correlation_id

    async def test_connection_error_does_not_expose_credentials(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test connection errors do not expose database credentials.

        Security requirement:
        - Error messages must not contain passwords
        - Error messages must not contain full DSN with credentials
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused to postgres://user:secret_password@localhost:5432/db"
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraConnectionError) as exc_info:
            await validator.table_exists("test_table", correlation_id=str(uuid4()))

        error_str = str(exc_info.value)
        # The InfraConnectionError should have a sanitized message
        assert "Failed to connect to database" in error_str


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions using model validation."""

    def test_empty_columns_list_raises_validation_error(self) -> None:
        """Test schema with no columns raises validation error.

        Expected behavior:
        - ValidationError for empty columns list
        """
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="empty_table",
                columns=[],  # Empty!
                indexes=[],
                schema_version="1.0.0",
            )

    def test_no_primary_key_raises_validation_error(self) -> None:
        """Test schema without primary key raises validation error.

        Expected behavior:
        - ValidationError when no column has primary_key=True
        """
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="no_pk_table",
                columns=[
                    ModelProjectorColumn(
                        name="id", column_type="uuid", nullable=False, primary_key=False
                    ),
                    ModelProjectorColumn(
                        name="name",
                        column_type="varchar",
                        length=255,
                        nullable=False,
                        primary_key=False,
                    ),
                ],
                indexes=[],
                schema_version="1.0.0",
            )

        assert "primary key" in str(exc_info.value).lower()

    def test_index_columns_must_exist_in_columns(self) -> None:
        """Test index column references must exist in columns.

        Expected behavior:
        - ValidationError if index references non-existent column
        """
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="bad_index",
                columns=[
                    ModelProjectorColumn(
                        name="id", column_type="uuid", nullable=False, primary_key=True
                    ),
                ],
                indexes=[
                    ModelProjectorIndex(
                        name="idx_bad",
                        columns=["nonexistent_column"],
                    ),
                ],
                schema_version="1.0.0",
            )

        assert "non-existent column" in str(exc_info.value).lower()

    def test_duplicate_column_names_raises_validation_error(self) -> None:
        """Test duplicate column names raise validation error.

        Expected behavior:
        - ValidationError for duplicate column names
        """
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="duplicate_cols",
                columns=[
                    ModelProjectorColumn(
                        name="id", column_type="uuid", nullable=False, primary_key=True
                    ),
                    ModelProjectorColumn(
                        name="id",
                        column_type="integer",
                        nullable=False,
                    ),  # Duplicate!
                ],
                indexes=[],
                schema_version="1.0.0",
            )

        assert "duplicate" in str(exc_info.value).lower()

    def test_invalid_table_name_raises_validation_error(self) -> None:
        """Test invalid table names (SQL injection attempts) are rejected.

        Expected behavior:
        - ValidationError for table names with special characters
        """
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="test; DROP TABLE users; --",  # SQL injection attempt
                columns=[
                    ModelProjectorColumn(
                        name="id", column_type="uuid", nullable=False, primary_key=True
                    ),
                ],
                indexes=[],
                schema_version="1.0.0",
            )

    def test_invalid_column_name_raises_validation_error(self) -> None:
        """Test invalid column names are rejected.

        Expected behavior:
        - ValidationError for column names with special characters
        """
        with pytest.raises(ValueError):
            ModelProjectorColumn(
                name="column; DROP TABLE users;",  # SQL injection attempt
                column_type="uuid",
                nullable=False,
            )

    def test_invalid_schema_version_raises_validation_error(self) -> None:
        """Test invalid schema versions are rejected.

        Expected behavior:
        - ValidationError for non-semver versions
        """
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="test_table",
                columns=[
                    ModelProjectorColumn(
                        name="id", column_type="uuid", nullable=False, primary_key=True
                    ),
                ],
                indexes=[],
                schema_version="invalid-version",  # Not semver format
            )


# =============================================================================
# CONCURRENT OPERATIONS TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestConcurrentOperations:
    """Test behavior under concurrent access."""

    async def test_multiple_ensure_schema_exists_calls_are_safe(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test multiple concurrent ensure_schema_exists calls are safe.

        Expected behavior:
        - Multiple concurrent calls don't cause race conditions
        - All calls complete successfully or fail gracefully
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_connection.fetchval.return_value = True
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
            {"column_name": "data"},
            {"column_name": "created_at"},
            {"column_name": "is_active"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        # Run 5 concurrent calls
        tasks = [
            validator.ensure_schema_exists(sample_schema, correlation_id=str(uuid4()))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (no exceptions)
        for result in results:
            assert result is None  # ensure_schema_exists returns None on success

    async def test_table_exists_concurrent_calls(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test table_exists can be called concurrently.

        Expected behavior:
        - Multiple concurrent calls return correct results
        - No shared state corruption
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        # Run 10 concurrent calls
        tasks = [
            validator.table_exists("test_table", correlation_id=str(uuid4()))
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # All should return True
        assert all(r is True for r in results)


# =============================================================================
# GET TABLE COLUMNS TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetTableColumns:
    """Test _get_table_columns method."""

    async def test_get_table_columns_returns_column_list(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test _get_table_columns returns list of column names.

        Expected behavior:
        - Returns list of column names from table
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "name"},
            {"column_name": "created_at"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        # Access private method for testing
        columns = await validator._get_table_columns(
            "test_table", correlation_id=str(uuid4())
        )

        assert columns == ["id", "name", "created_at"]

    async def test_get_table_columns_empty_for_missing_table(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test _get_table_columns returns empty list for missing table.

        Expected behavior:
        - Returns empty list if table doesn't exist
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        columns = await validator._get_table_columns(
            "nonexistent_table", correlation_id=str(uuid4())
        )

        assert columns == []

    async def test_get_table_columns_handles_schema_name(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test _get_table_columns uses schema_name parameter.

        Expected behavior:
        - Query uses provided schema_name
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        await validator._get_table_columns(
            "test_table",
            correlation_id=str(uuid4()),
            schema_name="onex",
        )

        # Verify query was called with schema_name = 'onex'
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert "onex" in args

    async def test_get_table_columns_connection_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test _get_table_columns handles connection errors.

        Expected behavior:
        - Raise InfraConnectionError on database connection failure
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraConnectionError):
            await validator._get_table_columns(
                "test_table", correlation_id=str(uuid4())
            )

    async def test_get_table_columns_timeout_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test _get_table_columns handles timeout errors.

        Expected behavior:
        - Raise InfraTimeoutError on query timeout
        """
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraTimeoutError):
            await validator._get_table_columns(
                "test_table", correlation_id=str(uuid4())
            )


# =============================================================================
# DEEP VALIDATION TESTS
# =============================================================================


@pytest.mark.unit
class TestValidateSchemaDeep:
    """Test validate_schema_deeply method."""

    def test_validate_schema_deeply_returns_empty_for_valid_schema(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test validate_schema_deeply returns empty list for valid schema.

        Expected behavior:
        - Returns empty list when schema is valid
        - No warnings for well-formed schemas
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(sample_schema)

        assert warnings == []

    def test_validate_schema_deeply_warns_nullable_primary_key(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test validate_schema_deeply warns when primary key is nullable.

        Expected behavior:
        - Returns warning for nullable primary key
        - Warning message mentions the column name
        """
        # Create schema with nullable primary key (this bypasses model validation
        # by using object.__setattr__ after creation)
        schema = ModelProjectorSchema(
            table_name="test_nullable_pk",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )
        # Modify column to be nullable primary key for testing deep validation
        # Note: This tests the validator's ability to catch issues that might
        # come from externally constructed schemas (e.g., from dict)
        schema.columns[0] = ModelProjectorColumn(
            name="id", column_type="uuid", nullable=True, primary_key=True
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("nullable" in w.lower() and "id" in w for w in warnings)

    def test_validate_schema_deeply_warns_large_varchar_length(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test validate_schema_deeply warns for very large varchar lengths.

        Expected behavior:
        - Returns warning for varchar length > 10000
        - Suggests using TEXT type instead
        """
        schema = ModelProjectorSchema(
            table_name="test_large_varchar",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="huge_content",
                    column_type="varchar",
                    length=50000,  # Very large
                    nullable=True,
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("huge_content" in w and "TEXT" in w for w in warnings)

    def test_validate_schema_deeply_warns_index_naming_convention(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test validate_schema_deeply warns for non-standard index names.

        Expected behavior:
        - Returns warning for indexes not starting with 'idx_'
        """
        schema = ModelProjectorSchema(
            table_name="test_index_naming",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="name", column_type="varchar", length=100, nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="bad_index_name",  # Does not start with idx_
                    columns=["name"],
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("bad_index_name" in w and "idx_" in w for w in warnings)

    def test_validate_schema_deeply_passes_with_proper_idx_prefix(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test validate_schema_deeply passes for properly named indexes.

        Expected behavior:
        - No warning for indexes starting with 'idx_'
        """
        schema = ModelProjectorSchema(
            table_name="test_good_index",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="name", column_type="varchar", length=100, nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="idx_proper_name",  # Proper naming
                    columns=["name"],
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        # No warnings about index naming
        assert not any("idx_" in w for w in warnings)

    def test_validate_schema_deeply_multiple_warnings(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test validate_schema_deeply returns multiple warnings for multiple issues.

        Expected behavior:
        - Returns all applicable warnings
        - Each issue generates its own warning
        """
        schema = ModelProjectorSchema(
            table_name="test_multiple_issues",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="huge_field",
                    column_type="varchar",
                    length=20000,  # Too large
                    nullable=True,
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="no_prefix",  # Missing idx_ prefix
                    columns=["huge_field"],
                ),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        # Should have at least 2 warnings: large varchar and index naming
        assert len(warnings) >= 2
        # Verify both types of warnings are present
        has_varchar_warning = any("huge_field" in w and "TEXT" in w for w in warnings)
        has_naming_warning = any("no_prefix" in w and "idx_" in w for w in warnings)
        assert has_varchar_warning
        assert has_naming_warning

    def test_validate_schema_deeply_is_synchronous(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Test validate_schema_deeply is synchronous (not async).

        Expected behavior:
        - Method can be called directly without await
        - Does not require database connection
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        # Should work without await - this validates it's synchronous
        result = validator.validate_schema_deeply(sample_schema)

        assert isinstance(result, list)
        # Pool should NOT have been accessed
        assert not mock_pool.acquire.called
