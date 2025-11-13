"""Initial OmniNode Bridge schema

Revision ID: 001
Revises:
Create Date: 2025-01-01 12:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema - Create initial OmniNode Bridge tables."""

    # Enable required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"')

    # Service sessions table with enhanced security
    op.create_table(
        "service_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("service_name", sa.String(length=255), nullable=False),
        sa.Column("instance_id", sa.String(length=255), nullable=True),
        sa.Column(
            "session_start",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("session_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.String(length=50),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'ended', 'terminated', 'failed')",
            name="service_sessions_status_check",
        ),
        sa.CheckConstraint("session_start <= NOW()", name="session_start_valid"),
        sa.CheckConstraint(
            "session_end IS NULL OR session_end >= session_start",
            name="session_end_valid",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Hook events table for persistence and debugging with security enhancements
    op.create_table(
        "hook_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=255), nullable=False),
        sa.Column("resource", sa.String(length=255), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "processed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("processing_errors", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "retry_count >= 0 AND retry_count <= 10",
            name="retry_count_valid",
        ),
        sa.CheckConstraint(
            "processed_at IS NULL OR processed_at >= created_at",
            name="processing_time_valid",
        ),
        sa.CheckConstraint("retry_count <= 10", name="retry_count_limit"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Event processing metrics with enhanced constraints
    op.create_table(
        "event_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("kafka_publish_success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "processing_time_ms >= 0",
            name="processing_time_non_negative",
        ),
        sa.CheckConstraint(
            "processing_time_ms < 300000",
            name="processing_time_reasonable",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["hook_events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Security audit table for connection and authentication events
    op.create_table(
        "security_audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column(
            "client_info",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Performance monitoring table
    op.create_table(
        "connection_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pool_size", sa.Integer(), nullable=False),
        sa.Column("active_connections", sa.Integer(), nullable=False),
        sa.Column("idle_connections", sa.Integer(), nullable=False),
        sa.Column(
            "total_queries",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "failed_queries",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("avg_query_time_ms", sa.Float(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Workflows table for workflow coordinator
    op.create_table(
        "workflows",
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "definition",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("input_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "output_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("workflow_id"),
    )

    # Workflow tasks table for workflow coordinator
    op.create_table(
        "workflow_tasks",
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("task_type", sa.String(length=50), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "dependencies",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.workflow_id"],
        ),
        sa.PrimaryKeyConstraint("task_id"),
    )

    # Create indexes for performance
    op.create_index(
        "idx_service_sessions_service_status",
        "service_sessions",
        ["service_name", "status"],
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "idx_service_sessions_created_at",
        "service_sessions",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_service_sessions_cleanup",
        "service_sessions",
        ["session_end"],
        postgresql_where=sa.text("session_end IS NOT NULL"),
    )

    op.create_index(
        "idx_hook_events_source_action",
        "hook_events",
        ["source", "action"],
    )
    op.create_index(
        "idx_hook_events_processed",
        "hook_events",
        ["processed", sa.text("created_at DESC")],
        postgresql_where=sa.text("NOT processed"),
    )
    op.create_index(
        "idx_hook_events_created_at",
        "hook_events",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_hook_events_retry",
        "hook_events",
        ["retry_count", "created_at"],
        postgresql_where=sa.text("retry_count > 0"),
    )

    op.create_index("idx_event_metrics_event_id", "event_metrics", ["event_id"])
    op.create_index(
        "idx_event_metrics_created_at",
        "event_metrics",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_event_metrics_performance",
        "event_metrics",
        ["processing_time_ms", "kafka_publish_success"],
    )

    op.create_index(
        "idx_security_audit_event_type",
        "security_audit_log",
        ["event_type", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_security_audit_failed",
        "security_audit_log",
        [sa.text("created_at DESC")],
        postgresql_where=sa.text("NOT success"),
    )

    op.create_index(
        "idx_connection_metrics_recorded_at",
        "connection_metrics",
        [sa.text("recorded_at DESC")],
    )

    op.create_index("idx_workflows_status", "workflows", ["status"])
    op.create_index("idx_workflow_tasks_workflow_id", "workflow_tasks", ["workflow_id"])
    op.create_index("idx_workflow_tasks_status", "workflow_tasks", ["status"])

    # Create data retention function for automatic cleanup
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cleanup_old_data() RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER := 0;
            temp_count INTEGER;
        BEGIN
            -- Clean up old processed hook events (older than 30 days)
            DELETE FROM hook_events
            WHERE processed = TRUE
            AND created_at < NOW() - INTERVAL '30 days';
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;

            -- Clean up old event metrics (older than 90 days)
            DELETE FROM event_metrics
            WHERE created_at < NOW() - INTERVAL '90 days';
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;

            -- Clean up old security audit logs (older than 365 days)
            DELETE FROM security_audit_log
            WHERE created_at < NOW() - INTERVAL '365 days';
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;

            -- Clean up old connection metrics (older than 30 days)
            DELETE FROM connection_metrics
            WHERE recorded_at < NOW() - INTERVAL '30 days';
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;

            -- Clean up ended service sessions (older than 7 days)
            DELETE FROM service_sessions
            WHERE session_end IS NOT NULL
            AND session_end < NOW() - INTERVAL '7 days';
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;

            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """,
    )


def downgrade() -> None:
    """Downgrade database schema - Drop all OmniNode Bridge tables."""

    # Drop cleanup function
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_data()")

    # Drop indexes
    op.drop_index("idx_workflow_tasks_status", table_name="workflow_tasks")
    op.drop_index("idx_workflow_tasks_workflow_id", table_name="workflow_tasks")
    op.drop_index("idx_workflows_status", table_name="workflows")

    op.drop_index("idx_connection_metrics_recorded_at", table_name="connection_metrics")
    op.drop_index("idx_security_audit_failed", table_name="security_audit_log")
    op.drop_index("idx_security_audit_event_type", table_name="security_audit_log")

    op.drop_index("idx_event_metrics_performance", table_name="event_metrics")
    op.drop_index("idx_event_metrics_created_at", table_name="event_metrics")
    op.drop_index("idx_event_metrics_event_id", table_name="event_metrics")

    op.drop_index("idx_hook_events_retry", table_name="hook_events")
    op.drop_index("idx_hook_events_created_at", table_name="hook_events")
    op.drop_index("idx_hook_events_processed", table_name="hook_events")
    op.drop_index("idx_hook_events_source_action", table_name="hook_events")

    op.drop_index("idx_service_sessions_cleanup", table_name="service_sessions")
    op.drop_index("idx_service_sessions_created_at", table_name="service_sessions")
    op.drop_index("idx_service_sessions_service_status", table_name="service_sessions")

    # Drop tables in reverse dependency order
    op.drop_table("workflow_tasks")
    op.drop_table("workflows")
    op.drop_table("connection_metrics")
    op.drop_table("security_audit_log")
    op.drop_table("event_metrics")
    op.drop_table("hook_events")
    op.drop_table("service_sessions")
