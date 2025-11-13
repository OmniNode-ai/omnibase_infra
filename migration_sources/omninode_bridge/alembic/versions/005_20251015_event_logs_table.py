"""Add event_logs table for code generation event tracing

Revision ID: 005
Revises: 004
Create Date: 2025-10-15 00:00:00.000000

This migration creates the event_logs table for storing events from the
autonomous code generation infrastructure (PR #25). Supports tracing
events across 13 Kafka topics with 9 event types.

Key Features:
- Event tracing by session_id and correlation_id
- Performance metrics calculation (response times, percentiles)
- Bottleneck identification
- JSONB payload storage for flexible event data
- Optimized indexes for dashboard queries

Related Files:
- src/omninode_bridge/events/codegen_schemas.py (9 event schemas)
- src/omninode_bridge/dashboard/codegen_event_tracer.py (tracer implementation)

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade():
    """Create event_logs table with optimized indexes for event tracing."""

    # Ensure UUID generation function is available
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Create event_logs table
    op.create_table(
        "event_logs",
        # Primary key
        sa.Column(
            "event_id",
            postgresql.UUID(),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
            comment="Unique event identifier",
        ),
        # Session and correlation tracking
        sa.Column(
            "session_id",
            postgresql.UUID(),
            nullable=False,
            comment="Code generation session ID",
        ),
        sa.Column(
            "correlation_id",
            postgresql.UUID(),
            nullable=True,
            comment="Request/response correlation ID (null for status events)",
        ),
        # Event classification
        sa.Column(
            "event_type",
            sa.VARCHAR(50),
            nullable=False,
            comment="Event type: request, response, status, error",
        ),
        sa.Column(
            "topic",
            sa.VARCHAR(255),
            nullable=False,
            comment="Kafka topic name",
        ),
        # Temporal tracking
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Event timestamp (UTC)",
        ),
        # Event status and performance
        sa.Column(
            "status",
            sa.VARCHAR(50),
            nullable=False,
            comment="Event status: sent, received, failed, processing",
        ),
        sa.Column(
            "processing_time_ms",
            sa.INTEGER(),
            nullable=True,
            comment="Processing time in milliseconds (for responses)",
        ),
        # Event data
        sa.Column(
            "payload",
            postgresql.JSONB(),
            nullable=False,
            comment="Event data (full event schema)",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Additional context (source, destination, retries, etc.)",
        ),
        # Table configuration
        comment="Event logs for autonomous code generation event tracing and debugging",
    )

    # Create indexes for query performance
    # Index 1: Session-based queries (trace_session_events)
    op.create_index(
        "idx_event_logs_session_id",
        "event_logs",
        ["session_id"],
    )

    # Index 2: Correlation-based queries (find_correlated_events)
    op.create_index(
        "idx_event_logs_correlation_id",
        "event_logs",
        ["correlation_id"],
    )

    # Index 3: Temporal queries and sorting (explicit SQL for DESC ordering)
    op.execute('CREATE INDEX idx_event_logs_timestamp ON event_logs ("timestamp" DESC)')

    # Index 4: Event type filtering
    op.create_index(
        "idx_event_logs_event_type",
        "event_logs",
        ["event_type"],
    )

    # Index 5: Topic grouping and analysis
    op.create_index(
        "idx_event_logs_topic",
        "event_logs",
        ["topic"],
    )

    # Index 6: Composite index for optimal session + time queries (explicit SQL for DESC ordering)
    op.execute(
        'CREATE INDEX idx_event_logs_session_timestamp ON event_logs (session_id, "timestamp" DESC)'
    )

    # Index 7: GIN index for JSONB payload queries
    op.execute(
        "CREATE INDEX idx_event_logs_payload_gin ON event_logs USING GIN(payload)"
    )

    # Index 8: GIN index for JSONB metadata queries
    op.execute(
        "CREATE INDEX idx_event_logs_metadata_gin ON event_logs USING GIN(metadata)"
    )

    # Add CHECK constraints for data validation
    op.create_check_constraint(
        "ck_event_logs_event_type",
        "event_logs",
        "event_type IN ('request', 'response', 'status', 'error')",
    )

    op.create_check_constraint(
        "ck_event_logs_status",
        "event_logs",
        "status IN ('sent', 'received', 'failed', 'processing', 'completed')",
    )

    op.create_check_constraint(
        "ck_event_logs_processing_time_positive",
        "event_logs",
        "processing_time_ms IS NULL OR processing_time_ms >= 0",
    )


def downgrade():
    """Drop event_logs table and all indexes."""

    # Drop CHECK constraints first
    op.drop_constraint("ck_event_logs_processing_time_positive", "event_logs")
    op.drop_constraint("ck_event_logs_status", "event_logs")
    op.drop_constraint("ck_event_logs_event_type", "event_logs")

    # Drop GIN indexes
    op.drop_index("idx_event_logs_metadata_gin", table_name="event_logs")
    op.drop_index("idx_event_logs_payload_gin", table_name="event_logs")

    # Drop B-tree indexes (using keyword args for clarity)
    op.drop_index("idx_event_logs_session_timestamp", table_name="event_logs")
    op.drop_index("idx_event_logs_topic", table_name="event_logs")
    op.drop_index("idx_event_logs_event_type", table_name="event_logs")
    op.drop_index("idx_event_logs_timestamp", table_name="event_logs")
    op.drop_index("idx_event_logs_correlation_id", table_name="event_logs")
    op.drop_index("idx_event_logs_session_id", table_name="event_logs")

    # Drop table
    op.drop_table("event_logs")
