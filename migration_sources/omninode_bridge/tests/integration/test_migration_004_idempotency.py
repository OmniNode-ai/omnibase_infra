"""
Comprehensive tests for migration 004 idempotency and enum validation.

Tests cover:
1. Fresh migration (enum doesn't exist)
2. Re-running migration (enum exists with correct values) - true idempotency
3. Migration with existing enum with wrong values (should fail)
4. Downgrade migration with proper cleanup
5. Full migration cycle (upgrade -> downgrade -> upgrade)
6. Edge cases and partial states
"""

import logging
import os
from collections.abc import Generator

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text

from alembic import command
from alembic.config import Config

# Configure logging for test debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test constants
EXPECTED_ENUM_VALUES = ["effect", "compute", "reducer", "orchestrator"]
MIGRATION_REVISION = "004"


@pytest.fixture(scope="module")
def alembic_config() -> Config:
    """Create Alembic configuration for testing."""
    # Path to alembic.ini
    alembic_ini_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "alembic.ini"
    )

    config = Config(alembic_ini_path)

    # Override database URL for testing
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/test_omninode_bridge",
    )
    config.set_main_option("sqlalchemy.url", test_db_url)

    return config


@pytest.fixture(scope="function")
def db_connection(
    alembic_config: Config,
) -> Generator[sa.engine.Connection, None, None]:
    """
    Create a database connection for testing.

    This fixture provides a fresh database connection for each test,
    ensuring test isolation.
    """
    # Get database URL from alembic config
    db_url = alembic_config.get_main_option("sqlalchemy.url")

    # Create engine
    engine = sa.create_engine(db_url)

    # Create connection
    connection = engine.connect()

    yield connection

    # Cleanup
    connection.close()
    engine.dispose()


def _enum_exists(conn: sa.engine.Connection, enum_name: str) -> bool:
    """Check if enum type exists."""
    result = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_type WHERE typname = :enum_name
            )
            """
        ),
        {"enum_name": enum_name},
    )
    return result.scalar()


def _get_enum_values(conn: sa.engine.Connection, enum_name: str) -> list[str]:
    """Get values from an enum type."""
    result = conn.execute(
        text(
            """
            SELECT enumlabel
            FROM pg_enum
            WHERE enumtypid = (
                SELECT oid
                FROM pg_type
                WHERE typname = :enum_name
            )
            ORDER BY enumsortorder
            """
        ),
        {"enum_name": enum_name},
    )
    return [row[0] for row in result.fetchall()]


def _table_exists(conn: sa.engine.Connection, table_name: str) -> bool:
    """Check if table exists."""
    inspector = sa.inspect(conn)
    return table_name in inspector.get_table_names()


def _create_corrupt_enum(conn: sa.engine.Connection, enum_name: str, values: list[str]):
    """Create an enum with wrong values for testing failure scenarios."""
    # Drop if exists
    conn.execute(text(f"DROP TYPE IF EXISTS {enum_name} CASCADE"))
    conn.commit()

    # Create with wrong values
    values_str = ", ".join(f"'{v}'" for v in values)
    conn.execute(text(f"CREATE TYPE {enum_name} AS ENUM ({values_str})"))
    conn.commit()


def _cleanup_migration_state(conn: sa.engine.Connection):
    """Clean up all migration artifacts for fresh testing."""
    # Drop table if exists
    conn.execute(text("DROP TABLE IF EXISTS node_registrations CASCADE"))

    # Drop enum if exists
    conn.execute(text("DROP TYPE IF EXISTS node_type_enum CASCADE"))

    conn.commit()


@pytest.fixture(scope="function")
def clean_database(db_connection: sa.engine.Connection) -> sa.engine.Connection:
    """Ensure database is in clean state before each test."""
    _cleanup_migration_state(db_connection)
    return db_connection


class TestMigration004FreshInstall:
    """Test migration 004 on fresh database (no existing enum or table)."""

    def test_fresh_migration_creates_enum(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that fresh migration creates enum with correct values."""
        # Verify clean state
        assert not _enum_exists(clean_database, "node_type_enum")
        assert not _table_exists(clean_database, "node_registrations")

        # Run upgrade migration
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify enum was created with correct values
        assert _enum_exists(clean_database, "node_type_enum")
        enum_values = _get_enum_values(clean_database, "node_type_enum")
        assert set(enum_values) == set(EXPECTED_ENUM_VALUES)

        # Verify table was created
        assert _table_exists(clean_database, "node_registrations")

    def test_fresh_migration_creates_table(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that fresh migration creates table with correct schema."""
        # Run upgrade migration
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify table structure
        inspector = sa.inspect(clean_database)
        columns = {
            col["name"]: col for col in inspector.get_columns("node_registrations")
        }

        # Check required columns exist
        required_columns = [
            "id",
            "node_id",
            "node_type",
            "capabilities",
            "endpoints",
            "metadata",
            "health_endpoint",
            "last_heartbeat",
            "registered_at",
            "updated_at",
        ]
        for col_name in required_columns:
            assert col_name in columns, f"Missing required column: {col_name}"

        # Check column types
        assert isinstance(columns["id"]["type"], postgresql.UUID)
        assert isinstance(columns["node_type"]["type"], sa.Enum)
        assert isinstance(columns["capabilities"]["type"], postgresql.JSONB)

    def test_fresh_migration_creates_indexes(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that fresh migration creates all required indexes."""
        # Run upgrade migration
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify indexes exist
        inspector = sa.inspect(clean_database)
        indexes = {
            idx["name"]: idx for idx in inspector.get_indexes("node_registrations")
        }

        expected_indexes = [
            "idx_node_registrations_node_id",
            "idx_node_registrations_node_type",
            "idx_node_registrations_last_heartbeat",
            "idx_node_registrations_updated_at",
        ]

        for idx_name in expected_indexes:
            assert idx_name in indexes, f"Missing index: {idx_name}"


class TestMigration004Idempotency:
    """Test migration 004 idempotency (re-running migration)."""

    def test_idempotent_migration_with_existing_enum(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that re-running migration with existing enum succeeds."""
        # Run migration first time
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify enum exists
        assert _enum_exists(clean_database, "node_type_enum")

        # Run migration second time (should succeed without changes)
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify enum still exists with correct values
        assert _enum_exists(clean_database, "node_type_enum")
        enum_values = _get_enum_values(clean_database, "node_type_enum")
        assert set(enum_values) == set(EXPECTED_ENUM_VALUES)

    def test_idempotent_migration_with_existing_table(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that re-running migration with existing table succeeds."""
        # Run migration first time
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Insert test data
        clean_database.execute(
            text(
                """
                INSERT INTO node_registrations (node_id, node_type, capabilities, endpoints, metadata)
                VALUES ('test-node-1', 'effect', '{}', '{}', '{}')
                """
            )
        )
        clean_database.commit()

        # Run migration second time (should succeed without data loss)
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify data is preserved
        result = clean_database.execute(
            text(
                "SELECT COUNT(*) FROM node_registrations WHERE node_id = 'test-node-1'"
            )
        )
        assert result.scalar() == 1

    def test_idempotent_migration_with_existing_indexes(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that re-running migration with existing indexes succeeds."""
        # Run migration first time
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify indexes exist
        inspector = sa.inspect(clean_database)
        initial_indexes = {
            idx["name"] for idx in inspector.get_indexes("node_registrations")
        }

        # Run migration second time
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify indexes still exist (no duplicates or errors)
        final_indexes = {
            idx["name"] for idx in inspector.get_indexes("node_registrations")
        }
        assert initial_indexes == final_indexes


class TestMigration004EnumValidation:
    """Test migration 004 enum validation (should fail with wrong values)."""

    def test_migration_fails_with_wrong_enum_values(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that migration fails if enum exists with wrong values."""
        # Create enum with wrong values
        wrong_values = ["effect", "compute", "wrong_value"]
        _create_corrupt_enum(clean_database, "node_type_enum", wrong_values)

        # Verify corrupt enum exists
        assert _enum_exists(clean_database, "node_type_enum")
        assert set(_get_enum_values(clean_database, "node_type_enum")) != set(
            EXPECTED_ENUM_VALUES
        )

        # Run migration (should fail with RuntimeError)
        with pytest.raises(RuntimeError) as exc_info:
            command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "MIGRATION VALIDATION FAILED" in error_msg
        assert "node_type_enum" in error_msg
        assert "incompatible values" in error_msg

    def test_migration_fails_with_missing_enum_values(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that migration fails if enum is missing required values."""
        # Create enum with missing values
        incomplete_values = [
            "effect",
            "compute",
        ]  # Missing 'reducer' and 'orchestrator'
        _create_corrupt_enum(clean_database, "node_type_enum", incomplete_values)

        # Run migration (should fail)
        with pytest.raises(RuntimeError) as exc_info:
            command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify error message mentions missing values
        error_msg = str(exc_info.value)
        assert "Missing values" in error_msg
        assert "reducer" in error_msg or "orchestrator" in error_msg

    def test_migration_fails_with_extra_enum_values(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that migration fails if enum has extra values."""
        # Create enum with extra values
        extended_values = EXPECTED_ENUM_VALUES + ["extra_value_1", "extra_value_2"]
        _create_corrupt_enum(clean_database, "node_type_enum", extended_values)

        # Run migration (should fail)
        with pytest.raises(RuntimeError) as exc_info:
            command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify error message mentions extra values
        error_msg = str(exc_info.value)
        assert "Extra values" in error_msg
        assert "extra_value_1" in error_msg or "extra_value_2" in error_msg


class TestMigration004Downgrade:
    """Test migration 004 downgrade (rollback)."""

    def test_downgrade_removes_table(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that downgrade removes the table."""
        # Run upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)
        assert _table_exists(clean_database, "node_registrations")

        # Run downgrade
        command.downgrade(alembic_config, "003")

        # Verify table is removed
        assert not _table_exists(clean_database, "node_registrations")

    def test_downgrade_removes_enum(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that downgrade removes the enum."""
        # Run upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)
        assert _enum_exists(clean_database, "node_type_enum")

        # Run downgrade
        command.downgrade(alembic_config, "003")

        # Verify enum is removed
        assert not _enum_exists(clean_database, "node_type_enum")

    def test_downgrade_removes_indexes(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that downgrade removes all indexes."""
        # Run upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify indexes exist
        inspector = sa.inspect(clean_database)
        indexes_before = {
            idx["name"] for idx in inspector.get_indexes("node_registrations")
        }
        assert len(indexes_before) > 0

        # Run downgrade
        command.downgrade(alembic_config, "003")

        # Verify indexes are removed (table doesn't exist)
        assert not _table_exists(clean_database, "node_registrations")

    def test_downgrade_with_data_warns(
        self, clean_database: sa.engine.Connection, alembic_config: Config, caplog
    ):
        """Test that downgrade warns when table contains data."""
        # Run upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Insert test data
        clean_database.execute(
            text(
                """
                INSERT INTO node_registrations (node_id, node_type, capabilities, endpoints, metadata)
                VALUES ('test-node-1', 'effect', '{}', '{}', '{}')
                """
            )
        )
        clean_database.commit()

        # Run downgrade (should warn about data loss)
        with caplog.at_level(logging.WARNING):
            command.downgrade(alembic_config, "003")

        # Verify warning was logged
        assert any(
            "MIGRATION DOWNGRADE WARNING" in record.message for record in caplog.records
        )
        assert any("permanent data loss" in record.message for record in caplog.records)

    def test_downgrade_idempotent(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that running downgrade multiple times is safe."""
        # Run upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Run downgrade first time
        command.downgrade(alembic_config, "003")
        assert not _table_exists(clean_database, "node_registrations")
        assert not _enum_exists(clean_database, "node_type_enum")

        # Run downgrade second time (should succeed without errors)
        command.downgrade(alembic_config, "003")

        # Verify state is still clean
        assert not _table_exists(clean_database, "node_registrations")
        assert not _enum_exists(clean_database, "node_type_enum")


class TestMigration004FullCycle:
    """Test migration 004 full cycle (upgrade -> downgrade -> upgrade)."""

    def test_full_cycle_preserves_functionality(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that full migration cycle works correctly."""
        # Initial upgrade
        command.upgrade(alembic_config, MIGRATION_REVISION)
        assert _table_exists(clean_database, "node_registrations")
        assert _enum_exists(clean_database, "node_type_enum")

        # Downgrade
        command.downgrade(alembic_config, "003")
        assert not _table_exists(clean_database, "node_registrations")
        assert not _enum_exists(clean_database, "node_type_enum")

        # Upgrade again
        command.upgrade(alembic_config, MIGRATION_REVISION)
        assert _table_exists(clean_database, "node_registrations")
        assert _enum_exists(clean_database, "node_type_enum")

        # Verify enum values are correct
        enum_values = _get_enum_values(clean_database, "node_type_enum")
        assert set(enum_values) == set(EXPECTED_ENUM_VALUES)

    def test_full_cycle_with_data_insertion(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that data can be inserted after full migration cycle."""
        # Full cycle
        command.upgrade(alembic_config, MIGRATION_REVISION)
        command.downgrade(alembic_config, "003")
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Insert test data
        clean_database.execute(
            text(
                """
                INSERT INTO node_registrations (node_id, node_type, capabilities, endpoints, metadata)
                VALUES ('test-node-1', 'effect', '{}', '{}', '{}')
                """
            )
        )
        clean_database.commit()

        # Verify data was inserted successfully
        result = clean_database.execute(text("SELECT COUNT(*) FROM node_registrations"))
        assert result.scalar() == 1


@pytest.mark.slow
class TestMigration004EdgeCases:
    """Test migration 004 edge cases and unusual scenarios."""

    def test_migration_with_concurrent_enum_usage(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test migration behavior when enum is in use by other tables."""
        # Run migration
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Create another table using the enum
        clean_database.execute(
            text(
                """
                CREATE TABLE test_table (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    test_type node_type_enum NOT NULL
                )
                """
            )
        )
        clean_database.commit()

        # Try to downgrade (should fail due to enum dependency)
        with pytest.raises(Exception):  # Could be RuntimeError or psycopg2 error
            command.downgrade(alembic_config, "003")

        # Cleanup
        clean_database.execute(text("DROP TABLE test_table"))
        clean_database.commit()

    def test_migration_validates_enum_order_independent(
        self, clean_database: sa.engine.Connection, alembic_config: Config
    ):
        """Test that enum validation is order-independent."""
        # Create enum with same values but different order
        reordered_values = ["orchestrator", "reducer", "compute", "effect"]
        _create_corrupt_enum(clean_database, "node_type_enum", reordered_values)

        # Migration should succeed (order doesn't matter for validation)
        command.upgrade(alembic_config, MIGRATION_REVISION)

        # Verify enum exists with correct values (regardless of order)
        assert _enum_exists(clean_database, "node_type_enum")
        enum_values = _get_enum_values(clean_database, "node_type_enum")
        assert set(enum_values) == set(EXPECTED_ENUM_VALUES)
