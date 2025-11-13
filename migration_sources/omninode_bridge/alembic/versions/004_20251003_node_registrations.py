"""Add node_registrations table for dynamic node discovery

Revision ID: 004
Revises: 003
Create Date: 2025-10-03 12:00:00.000000

"""

import logging
import os

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text

from alembic import op

# Configure logger for migration warnings
logger = logging.getLogger("alembic.runtime.migration")

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

# Expected enum values for validation
EXPECTED_NODE_TYPE_ENUM_VALUES = ["effect", "compute", "reducer", "orchestrator"]


def _validate_enum_values(conn, enum_name: str, expected_values: list[str]) -> bool:
    """
    Validate that an existing enum type has the expected values.

    Args:
        conn: Database connection
        enum_name: Name of the enum type to validate
        expected_values: List of expected enum values

    Returns:
        True if enum values match expected values, False otherwise

    Raises:
        RuntimeError: If enum exists with different values (migration should fail)
    """
    result = conn.execute(
        text(
            """
            SELECT enumlabel
            FROM pg_enum
            WHERE enumtypid = to_regtype(:enum_name)::oid
            ORDER BY enumsortorder
            """
        ),
        {"enum_name": enum_name},
    )

    existing_values = [row[0] for row in result.fetchall()]

    if not existing_values:
        # Enum doesn't exist, which is fine
        return False

    # Sort both lists for comparison (order shouldn't matter for validation)
    existing_set = set(existing_values)
    expected_set = set(expected_values)

    if existing_set != expected_set:
        missing_values = expected_set - existing_set
        extra_values = existing_set - expected_set

        error_msg = f"MIGRATION VALIDATION FAILED: Enum '{enum_name}' exists with incompatible values!\n"
        error_msg += f"  Expected values: {sorted(expected_values)}\n"
        error_msg += f"  Existing values: {sorted(existing_values)}\n"

        if missing_values:
            error_msg += f"  Missing values: {sorted(missing_values)}\n"
        if extra_values:
            error_msg += f"  Extra values: {sorted(extra_values)}\n"

        error_msg += "\nTo fix this issue:\n"
        error_msg += f"  1. Manually drop the enum: DROP TYPE {enum_name} CASCADE;\n"
        error_msg += "  2. Re-run the migration\n"
        error_msg += "  OR\n"
        error_msg += "  3. Manually alter the enum to match expected values\n"

        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"✓ Enum '{enum_name}' exists with correct values: {existing_values}")
    return True


def _create_enum_safe(conn, enum_name: str, values: list[str]):
    """
    Create enum type with validation for idempotency.

    Args:
        conn: Database connection
        enum_name: Name of the enum type
        values: List of enum values

    Raises:
        RuntimeError: If enum exists with different values
        ValueError: If enum_name is not in the allowed whitelist (SQL injection protection)
    """
    # SQL injection protection: Validate enum_name against whitelist
    ALLOWED_ENUM_NAMES = {"node_type_enum"}
    if enum_name not in ALLOWED_ENUM_NAMES:
        raise ValueError(
            f"SQL injection protection: enum_name '{enum_name}' is not in allowed whitelist: {ALLOWED_ENUM_NAMES}"
        )

    # Check if enum exists and validate values
    enum_exists = _validate_enum_values(conn, enum_name, values)

    if enum_exists:
        logger.info(
            f"Enum '{enum_name}' already exists with correct values, skipping creation"
        )
        return

    # Create the enum (safe to use f-string after whitelist validation)
    values_str = ", ".join(f"'{v}'" for v in values)
    conn.execute(text(f"CREATE TYPE {enum_name} AS ENUM ({values_str})"))
    logger.info(f"✓ Created enum '{enum_name}' with values: {values}")


def upgrade():
    """Add node_registrations table for dynamic node discovery and registration."""

    conn = op.get_bind()

    # Create ENUM type with validation for idempotency
    _create_enum_safe(conn, "node_type_enum", EXPECTED_NODE_TYPE_ENUM_VALUES)

    # Create node_registrations table (check existence first for idempotency)
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "node_registrations" not in inspector.get_table_names():
        # Ensure uuid-ossp extension exists before using uuid_generate_v4()
        op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        logger.info("✓ Ensured uuid-ossp extension is available")

        op.create_table(
            "node_registrations",
            sa.Column(
                "id",
                postgresql.UUID(),
                nullable=False,
                server_default=sa.text("uuid_generate_v4()"),
            ),
            sa.Column("node_id", sa.VARCHAR(255), nullable=False),
            sa.Column(
                "node_type",
                postgresql.ENUM(
                    "effect",
                    "compute",
                    "reducer",
                    "orchestrator",
                    name="node_type_enum",
                    create_type=False,
                ),
                nullable=False,
            ),
            sa.Column(
                "capabilities",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "endpoints",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "metadata",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("health_endpoint", sa.VARCHAR(500), nullable=True),
            sa.Column("last_heartbeat", sa.TIMESTAMP(timezone=True), nullable=True),
            sa.Column(
                "registered_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("id", name="node_registrations_pkey"),
            sa.UniqueConstraint("node_id", name="node_registrations_node_id_key"),
        )

    # Create indexes for performance (using IF NOT EXISTS for idempotency)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_node_id ON node_registrations (node_id)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_node_type ON node_registrations (node_type)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_last_heartbeat ON node_registrations (last_heartbeat DESC)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_updated_at ON node_registrations (updated_at DESC)"
    )

    # Create GIN indexes for JSONB fields for efficient querying (with IF NOT EXISTS)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_capabilities_gin ON node_registrations USING GIN(capabilities)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_endpoints_gin ON node_registrations USING GIN(endpoints)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_registrations_metadata_gin ON node_registrations USING GIN(metadata)"
    )


def _check_enum_dependencies(conn, enum_name: str) -> list[str]:
    """
    Check if enum type has any dependencies (columns using it).

    Args:
        conn: Database connection
        enum_name: Name of the enum type

    Returns:
        List of table.column names that depend on this enum
    """
    result = conn.execute(
        text(
            """
            SELECT
                n.nspname AS schema_name,
                c.relname AS table_name,
                a.attname AS column_name
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE a.atttypid = to_regtype(:enum_name)::oid
            GROUP BY n.nspname, c.relname, a.attname
            ORDER BY n.nspname, c.relname, a.attname
            """
        ),
        {"enum_name": enum_name},
    )

    dependencies = [f"{row[0]}.{row[1]}.{row[2]}" for row in result.fetchall()]
    return dependencies


def _drop_enum_safe(conn, enum_name: str, cascade: bool = False):
    """
    Drop enum type safely with dependency checking.

    Args:
        conn: Database connection
        enum_name: Name of the enum type
        cascade: Whether to use CASCADE (drops dependent objects)

    Raises:
        RuntimeError: If enum has dependencies and cascade is False
    """
    # Check if enum exists (schema-aware using to_regtype)
    result = conn.execute(
        text(
            """
            SELECT to_regtype(:enum_name) IS NOT NULL
            """
        ),
        {"enum_name": enum_name},
    )

    enum_exists = result.scalar()

    if not enum_exists:
        logger.info(f"Enum '{enum_name}' does not exist, skipping drop")
        return

    # Check for dependencies
    dependencies = _check_enum_dependencies(conn, enum_name)

    if dependencies and not cascade:
        error_msg = f"Cannot drop enum '{enum_name}' - it has dependencies:\n"
        for dep in dependencies:
            error_msg += f"  - {dep}\n"
        error_msg += "\nDrop dependent objects first or use cascade=True"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Drop the enum
    cascade_clause = " CASCADE" if cascade else ""
    conn.execute(text(f"DROP TYPE {enum_name}{cascade_clause}"))
    logger.info(f"✓ Dropped enum '{enum_name}'{cascade_clause}")


def downgrade():
    """Remove node_registrations table and related indexes (idempotent with IF EXISTS)."""

    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Issue #4 fix: Early safety check before any destructive operations
    # Check if table exists and contains data
    if "node_registrations" in inspector.get_table_names():
        result = conn.execute(text("SELECT COUNT(*) FROM node_registrations"))
        count = result.scalar()

        if count > 0:
            # Check for production safety flag
            if os.getenv("ALLOW_DESTRUCTIVE_MIGRATION", "false").lower() != "true":
                error_msg = (
                    f"⚠️  DESTRUCTIVE MIGRATION BLOCKED: node_registrations table contains {count} records.\n"
                    "   This downgrade would result in permanent data loss of all registered nodes.\n"
                    "   To proceed with this destructive operation, set environment variable:\n"
                    "   ALLOW_DESTRUCTIVE_MIGRATION=true"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Log warning if explicitly allowed
            logger.warning(
                f"⚠️  DESTRUCTIVE MIGRATION ALLOWED: Will drop node_registrations table with {count} existing records!"
            )
            logger.warning(
                "   This will result in permanent data loss of all registered nodes."
            )

    # Check if enum has any dependencies beyond node_registrations
    dependencies = _check_enum_dependencies(conn, "node_type_enum")
    if dependencies:
        # Filter out node_registrations since we're about to drop it
        other_dependencies = [
            dep
            for dep in dependencies
            if not dep.endswith("node_registrations.node_type")
        ]
        if other_dependencies:
            error_msg = "⚠️  CANNOT DROP ENUM: 'node_type_enum' has dependencies beyond node_registrations:\n"
            for dep in other_dependencies:
                error_msg += f"   - {dep}\n"
            error_msg += (
                "   Drop these dependent objects first before running this downgrade."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    # Drop all indexes first (using IF EXISTS for idempotency)
    indexes_to_drop = [
        "idx_node_registrations_metadata_gin",
        "idx_node_registrations_endpoints_gin",
        "idx_node_registrations_capabilities_gin",
        "idx_node_registrations_updated_at",
        "idx_node_registrations_last_heartbeat",
        "idx_node_registrations_node_type",
        "idx_node_registrations_node_id",
    ]

    for index_name in indexes_to_drop:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
        logger.info(f"✓ Dropped index (if exists): {index_name}")

    # Drop table if it exists (safety check already done at function start)
    if "node_registrations" in inspector.get_table_names():
        op.drop_table("node_registrations")
        logger.info("✓ Dropped table: node_registrations")

    # Drop the ENUM type safely (will fail if dependencies exist)
    # Using try/except to handle case where table was already dropped manually
    try:
        _drop_enum_safe(conn, "node_type_enum", cascade=False)
    except RuntimeError as e:
        # If enum has dependencies that weren't cleaned up, this is a real problem
        logger.error(f"Failed to drop enum type: {e}")
        raise
