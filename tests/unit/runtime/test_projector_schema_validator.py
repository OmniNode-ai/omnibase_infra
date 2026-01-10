# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ProjectorSchemaValidator.

This test suite validates the ProjectorSchemaValidator's behavior:
- Error handling and exception wrapping
- Correlation ID propagation
- Schema validation coordination
- Deep validation rules

Test Organization:
    - TestSchemaValidationCoordination: Tests ensure_schema_exists behavior
    - TestTableExistenceQueries: Tests table_exists database queries
    - TestColumnIntrospection: Tests _get_table_columns behavior
    - TestDeepValidation: Tests validate_schema_deeply rules
    - TestCorrelationIdHandling: Tests correlation ID propagation
    - TestErrorWrapping: Tests exception type mapping

Note: SQL generation tests belong in model tests (ModelProjectorSchema).
The validator delegates SQL generation to the schema model.

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
        ],
        indexes=[
            ModelProjectorIndex(name="idx_status", columns=["status"]),
        ],
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


def _setup_pool_with_connection(
    mock_pool: MagicMock, mock_connection: AsyncMock
) -> None:
    """Configure mock pool to return mock connection via async context manager."""
    mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
    mock_pool.acquire.return_value.__aexit__.return_value = None


# =============================================================================
# SCHEMA VALIDATION COORDINATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSchemaValidationCoordination:
    """Test ensure_schema_exists behavior and error conditions."""

    async def test_passes_when_table_and_columns_exist(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Validator passes silently when table and all columns exist."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True  # table exists
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
            {"column_name": "data"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        # Should complete without raising
        await validator.ensure_schema_exists(sample_schema, correlation_id=uuid4())

    async def test_raises_schema_error_when_table_missing(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Validator raises ProjectorSchemaError when table does not exist."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = False  # table missing

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(sample_schema, correlation_id=uuid4())

        # Verify error message contains table name and migration hint
        error_msg = str(exc_info.value)
        assert sample_schema.table_name in error_msg
        assert "does not exist" in error_msg
        assert "migration" in error_msg.lower()

    async def test_raises_schema_error_when_columns_missing(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Validator raises ProjectorSchemaError listing missing columns."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True  # table exists
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            # Missing: status, data
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(sample_schema, correlation_id=uuid4())

        error_msg = str(exc_info.value)
        assert "status" in error_msg
        assert "data" in error_msg
        assert "missing" in error_msg.lower()


# =============================================================================
# TABLE EXISTENCE QUERY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTableExistenceQueries:
    """Test table_exists query behavior and result handling."""

    async def test_returns_true_when_query_finds_table(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """table_exists returns True when EXISTS query returns True."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists("some_table", correlation_id=uuid4())

        assert result is True

    async def test_returns_false_when_query_finds_no_table(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """table_exists returns False when EXISTS query returns False."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = False

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = await validator.table_exists("nonexistent", correlation_id=uuid4())

        assert result is False

    async def test_uses_public_schema_by_default(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """table_exists defaults to public schema when not specified."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        await validator.table_exists("test_table", correlation_id=uuid4())

        # Verify query was called with public schema
        call_args = mock_connection.fetchval.call_args
        assert call_args is not None
        args = call_args[0]
        assert "public" in args

    async def test_uses_custom_schema_when_specified(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """table_exists uses provided schema_name parameter."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        await validator.table_exists(
            table_name="test_table",
            schema_name="custom_schema",
            correlation_id=uuid4(),
        )

        call_args = mock_connection.fetchval.call_args
        assert call_args is not None
        args = call_args[0]
        assert "custom_schema" in args


# =============================================================================
# ERROR WRAPPING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestErrorWrapping:
    """Test that database exceptions are wrapped in appropriate error types."""

    async def test_connection_error_wraps_postgres_connection_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """PostgresConnectionError is wrapped as InfraConnectionError."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraConnectionError):
            await validator.table_exists("test", correlation_id=uuid4())

    async def test_timeout_error_wraps_query_canceled(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """QueryCanceledError is wrapped as InfraTimeoutError."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.side_effect = asyncpg.QueryCanceledError("timeout")

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraTimeoutError):
            await validator.table_exists("test", correlation_id=uuid4())

    async def test_generic_error_wraps_as_runtime_host_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Unknown exceptions are wrapped as RuntimeHostError."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.side_effect = Exception("Unexpected")

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(RuntimeHostError):
            await validator.table_exists("test", correlation_id=uuid4())

    async def test_column_introspection_wraps_connection_errors(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """_get_table_columns also wraps connection errors correctly."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetch.side_effect = asyncpg.PostgresConnectionError(
            "Connection lost"
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        with pytest.raises(InfraConnectionError):
            await validator._get_table_columns("test", correlation_id=uuid4())


# =============================================================================
# CORRELATION ID HANDLING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCorrelationIdHandling:
    """Test correlation ID propagation to error context."""

    async def test_correlation_id_propagated_to_schema_error(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Provided correlation_id is accessible in ProjectorSchemaError."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = False  # table missing

        correlation_id = uuid4()
        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        with pytest.raises(ProjectorSchemaError) as exc_info:
            await validator.ensure_schema_exists(
                sample_schema, correlation_id=correlation_id
            )

        # Verify correlation ID is in the error
        assert exc_info.value.correlation_id is not None
        assert str(exc_info.value.correlation_id) == str(correlation_id)


# =============================================================================
# DEEP VALIDATION TESTS
# =============================================================================


@pytest.mark.unit
class TestDeepValidation:
    """Test validate_schema_deeply rules (synchronous, no DB access)."""

    def test_valid_schema_returns_empty_warnings(
        self, mock_pool: MagicMock, sample_schema: ModelProjectorSchema
    ) -> None:
        """Well-formed schema produces no warnings."""
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(sample_schema)
        assert warnings == []

    def test_warns_on_nullable_primary_key(self, mock_pool: MagicMock) -> None:
        """Warning produced when primary key column is nullable."""
        schema = ModelProjectorSchema(
            table_name="test",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )
        # Modify to have nullable PK (bypasses model validation for testing)
        schema.columns[0] = ModelProjectorColumn(
            name="id", column_type="uuid", nullable=True, primary_key=True
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("nullable" in w.lower() and "id" in w for w in warnings)

    def test_warns_on_large_varchar_length(self, mock_pool: MagicMock) -> None:
        """Warning produced for varchar length exceeding 10000."""
        schema = ModelProjectorSchema(
            table_name="test",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="huge", column_type="varchar", length=50000, nullable=True
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("huge" in w and "TEXT" in w for w in warnings)

    def test_warns_on_index_missing_idx_prefix(self, mock_pool: MagicMock) -> None:
        """Warning produced for index names not starting with idx_."""
        schema = ModelProjectorSchema(
            table_name="test",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="name", column_type="varchar", length=100, nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(name="bad_name", columns=["name"]),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert len(warnings) >= 1
        assert any("bad_name" in w and "idx_" in w for w in warnings)

    def test_no_warning_for_proper_idx_prefix(self, mock_pool: MagicMock) -> None:
        """No index naming warning when index starts with idx_."""
        schema = ModelProjectorSchema(
            table_name="test",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
                ModelProjectorColumn(
                    name="name", column_type="varchar", length=100, nullable=False
                ),
            ],
            indexes=[
                ModelProjectorIndex(name="idx_name", columns=["name"]),
            ],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        warnings = validator.validate_schema_deeply(schema)

        assert not any("idx_" in w for w in warnings)

    def test_is_synchronous_no_db_access(self, mock_pool: MagicMock) -> None:
        """validate_schema_deeply is synchronous and doesn't use DB."""
        schema = ModelProjectorSchema(
            table_name="test",
            columns=[
                ModelProjectorColumn(
                    name="id", column_type="uuid", nullable=False, primary_key=True
                ),
            ],
            indexes=[],
            schema_version="1.0.0",
        )

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = validator.validate_schema_deeply(schema)

        # Should return a list without await (duck-type check via iteration)
        assert hasattr(result, "__iter__"), "Result should be iterable"
        assert hasattr(result, "__len__"), "Result should have length"
        # Pool should NOT have been accessed
        assert not mock_pool.acquire.called


# =============================================================================
# MODEL VALIDATION TESTS (Pydantic Constraints)
# =============================================================================


@pytest.mark.unit
class TestModelValidation:
    """Test Pydantic model validation rules for schema models.

    These tests verify the model constraints work correctly.
    They don't test the validator class directly.
    """

    def test_empty_columns_rejected(self) -> None:
        """Schema with empty columns list raises ValidationError."""
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="test",
                columns=[],
                indexes=[],
                schema_version="1.0.0",
            )

    def test_no_primary_key_rejected(self) -> None:
        """Schema without any primary key column raises ValidationError."""
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="test",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        nullable=False,
                        primary_key=False,
                    ),
                ],
                indexes=[],
                schema_version="1.0.0",
            )
        assert "primary key" in str(exc_info.value).lower()

    def test_index_referencing_nonexistent_column_rejected(self) -> None:
        """Index referencing non-existent column raises ValidationError."""
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="test",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        nullable=False,
                        primary_key=True,
                    ),
                ],
                indexes=[
                    ModelProjectorIndex(name="idx_bad", columns=["nonexistent"]),
                ],
                schema_version="1.0.0",
            )
        assert "non-existent column" in str(exc_info.value).lower()

    def test_duplicate_column_names_rejected(self) -> None:
        """Schema with duplicate column names raises ValidationError."""
        with pytest.raises(ValueError) as exc_info:
            ModelProjectorSchema(
                table_name="test",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        nullable=False,
                        primary_key=True,
                    ),
                    ModelProjectorColumn(
                        name="id",
                        column_type="integer",
                        nullable=False,
                    ),
                ],
                indexes=[],
                schema_version="1.0.0",
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_sql_injection_in_table_name_rejected(self) -> None:
        """Table name with SQL injection characters raises ValidationError."""
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="test; DROP TABLE users; --",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        nullable=False,
                        primary_key=True,
                    ),
                ],
                indexes=[],
                schema_version="1.0.0",
            )

    def test_invalid_semver_version_rejected(self) -> None:
        """Invalid schema_version format raises ValidationError."""
        with pytest.raises(ValueError):
            ModelProjectorSchema(
                table_name="test",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        nullable=False,
                        primary_key=True,
                    ),
                ],
                indexes=[],
                schema_version="not-a-version",
            )


# =============================================================================
# CONCURRENT SAFETY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestConcurrentSafety:
    """Test validator behavior under concurrent access."""

    async def test_multiple_concurrent_table_exists_calls(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Multiple concurrent table_exists calls complete without errors."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        tasks = [
            validator.table_exists("test", correlation_id=uuid4()) for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert all(r is True for r in results)

    async def test_multiple_concurrent_ensure_schema_calls(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """Multiple concurrent ensure_schema_exists calls complete without errors."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetchval.return_value = True
        mock_connection.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "status"},
            {"column_name": "data"},
        ]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        tasks = [
            validator.ensure_schema_exists(sample_schema, correlation_id=uuid4())
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        for result in results:
            assert result is None


# =============================================================================
# GENERATE MIGRATION DELEGATION TEST
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateMigration:
    """Test generate_migration method.

    Note: SQL generation logic is tested in model tests.
    This only verifies the validator correctly delegates to the schema model.
    """

    async def test_delegates_to_schema_model(
        self,
        mock_pool: MagicMock,
        sample_schema: ModelProjectorSchema,
    ) -> None:
        """generate_migration returns result from schema.to_full_migration_sql().

        Note: generate_migration is synchronous as it only performs string
        generation without I/O operations.
        """
        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        result = validator.generate_migration(sample_schema, correlation_id=uuid4())

        # Should contain CREATE TABLE for the schema
        assert "CREATE TABLE" in result
        assert sample_schema.table_name in result
