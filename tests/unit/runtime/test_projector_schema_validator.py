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
    """Configure mock pool to return mock connection via async context manager.

    Note on Concurrency Tests:
        This setup shares a single mock_connection across all concurrent callers.
        This is safe for our concurrency tests because:
        1. The tests verify behavior (results, no exceptions), not call counts
        2. AsyncMock is thread-safe for return_value access
        3. We're testing the validator logic, not connection pool behavior

        If testing connection pool behavior or call ordering, use separate
        mock connections per acquire() call via side_effect.
    """
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
        # NOTE: Validation Bypass - Direct column mutation to create invalid state.
        # Pydantic's model_validator would normally reject nullable=True on a
        # primary_key column. We bypass this by mutating the schema after initial
        # construction to test that validate_schema_deeply() produces warnings
        # for edge cases that might arise from schema migrations or external data.
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
    """Test validator behavior under concurrent access.

    Note on Mock Strategy:
        These tests use a shared mock connection via _setup_pool_with_connection.
        This is intentional - we're testing that the validator itself doesn't have
        race conditions or state corruption when called concurrently, NOT testing
        the connection pool's concurrency behavior.

        The tests verify:
        - All concurrent calls return expected results
        - No exceptions are raised due to validator-internal race conditions
        - Results are deterministic regardless of scheduling order

        They do NOT verify:
        - Call ordering or timing
        - Connection pool acquire/release semantics
        - Specific call counts (which could vary with scheduling)
    """

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


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


@pytest.mark.unit
class TestProtocolCompliance:
    """Test that ProjectorSchemaValidator implements ProtocolProjectorSchemaValidator.

    The protocol is @runtime_checkable, enabling isinstance checks.
    This verifies duck typing compliance at test time.
    """

    def test_validator_implements_protocol(self, mock_pool: MagicMock) -> None:
        """ProjectorSchemaValidator passes runtime protocol check."""
        from omnibase_infra.protocols import ProtocolProjectorSchemaValidator

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        # @runtime_checkable protocol enables isinstance check
        assert isinstance(validator, ProtocolProjectorSchemaValidator)

    def test_validator_has_ensure_schema_exists_method(
        self, mock_pool: MagicMock
    ) -> None:
        """Validator has ensure_schema_exists method with correct signature."""
        import inspect

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        assert hasattr(validator, "ensure_schema_exists")
        assert callable(validator.ensure_schema_exists)

        # Check signature matches protocol
        sig = inspect.signature(validator.ensure_schema_exists)
        params = list(sig.parameters.keys())
        assert "schema" in params
        assert "correlation_id" in params

    def test_validator_has_table_exists_method(self, mock_pool: MagicMock) -> None:
        """Validator has table_exists method with correct signature."""
        import inspect

        validator = ProjectorSchemaValidator(db_pool=mock_pool)

        assert hasattr(validator, "table_exists")
        assert callable(validator.table_exists)

        # Check signature matches protocol
        sig = inspect.signature(validator.table_exists)
        params = list(sig.parameters.keys())
        assert "table_name" in params
        assert "correlation_id" in params
        assert "schema_name" in params


# =============================================================================
# COLUMN INTROSPECTION SCHEMA TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestColumnIntrospectionSchema:
    """Test _get_table_columns schema_name parameter behavior."""

    async def test_uses_public_schema_by_default(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """_get_table_columns defaults to public schema when not specified."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetch.return_value = [{"column_name": "id"}]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        await validator._get_table_columns("test_table", correlation_id=uuid4())

        # Verify query was called with public schema
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert "public" in args

    async def test_uses_custom_schema_when_specified(
        self,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """_get_table_columns uses provided schema_name parameter."""
        _setup_pool_with_connection(mock_pool, mock_connection)
        mock_connection.fetch.return_value = [{"column_name": "id"}]

        validator = ProjectorSchemaValidator(db_pool=mock_pool)
        await validator._get_table_columns(
            table_name="test_table",
            schema_name="analytics",
            correlation_id=uuid4(),
        )

        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert "analytics" in args
