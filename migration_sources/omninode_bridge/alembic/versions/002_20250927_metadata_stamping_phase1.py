"""Phase 1 MetadataStampingService schema

Revision ID: 002
Revises: 001
Create Date: 2025-09-27 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    """Create metadata stamping tables and indexes."""

    # Create metadata_stamps table
    op.create_table(
        "metadata_stamps",
        sa.Column(
            "id",
            postgresql.UUID(),
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("file_hash", sa.VARCHAR(64), nullable=False),
        sa.Column("file_path", sa.TEXT(), nullable=False),
        sa.Column("file_size", sa.BIGINT(), nullable=False),
        sa.Column("content_type", sa.VARCHAR(255), nullable=True),
        sa.Column("stamp_data", postgresql.JSONB(), nullable=False),
        sa.Column(
            "protocol_version", sa.VARCHAR(10), nullable=False, server_default="1.0"
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_hash"),
        sa.CheckConstraint("file_size >= 0", name="check_file_size_positive"),
    )

    # Create protocol_handlers table
    op.create_table(
        "protocol_handlers",
        sa.Column(
            "id",
            postgresql.UUID(),
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("handler_type", sa.VARCHAR(100), nullable=False),
        sa.Column("file_extensions", postgresql.ARRAY(sa.TEXT()), nullable=False),
        sa.Column("handler_config", postgresql.JSONB(), nullable=True),
        sa.Column("is_active", sa.BOOLEAN(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("handler_type"),
    )

    # Create hash_metrics table (partitioned)
    op.execute(
        """
        CREATE TABLE hash_metrics (
            id UUID DEFAULT uuid_generate_v4(),
            operation_type VARCHAR(50) NOT NULL,
            execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0),
            file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
            cpu_usage_percent DECIMAL(5,2) CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
            memory_usage_mb INTEGER CHECK (memory_usage_mb >= 0),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        ) PARTITION BY RANGE (timestamp)
    """
    )

    # Create initial partitions
    op.execute(
        """
        CREATE TABLE hash_metrics_y2025m09 PARTITION OF hash_metrics
        FOR VALUES FROM ('2025-09-01') TO ('2025-10-01')
    """
    )

    op.execute(
        """
        CREATE TABLE hash_metrics_y2025m10 PARTITION OF hash_metrics
        FOR VALUES FROM ('2025-10-01') TO ('2025-11-01')
    """
    )

    # Create indexes for metadata_stamps
    op.create_index("idx_metadata_stamps_file_hash", "metadata_stamps", ["file_hash"])
    op.create_index(
        "idx_metadata_stamps_created_at",
        "metadata_stamps",
        ["created_at"],
        postgresql_using="btree",
    )
    op.create_index(
        "idx_metadata_stamps_protocol_version", "metadata_stamps", ["protocol_version"]
    )

    # Create JSONB index for stamp_data
    op.execute(
        "CREATE INDEX idx_metadata_stamps_stamp_data_gin ON metadata_stamps USING GIN(stamp_data)"
    )

    # Create partial index for large files
    op.execute(
        "CREATE INDEX idx_metadata_stamps_file_size ON metadata_stamps(file_size) WHERE file_size > 1048576"
    )

    # Create partial index for content_type
    op.execute(
        "CREATE INDEX idx_metadata_stamps_content_type ON metadata_stamps(content_type) WHERE content_type IS NOT NULL"
    )

    # Create indexes for protocol_handlers
    op.create_index("idx_protocol_handlers_type", "protocol_handlers", ["handler_type"])
    op.execute(
        "CREATE INDEX idx_protocol_handlers_active ON protocol_handlers(is_active) WHERE is_active = true"
    )
    op.execute(
        "CREATE INDEX idx_protocol_handlers_extensions ON protocol_handlers USING GIN(file_extensions)"
    )

    # Create indexes for hash_metrics
    op.execute(
        "CREATE INDEX idx_hash_metrics_timestamp ON hash_metrics(timestamp DESC)"
    )
    op.execute(
        "CREATE INDEX idx_hash_metrics_operation_type ON hash_metrics(operation_type, timestamp)"
    )
    op.execute(
        "CREATE INDEX idx_hash_metrics_execution_time ON hash_metrics(execution_time_ms) WHERE execution_time_ms > 2"
    )

    # Create update trigger function
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    """
    )

    # Create triggers for timestamp updates
    op.execute(
        """
        CREATE TRIGGER update_metadata_stamps_updated_at
        BEFORE UPDATE ON metadata_stamps
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """
    )

    op.execute(
        """
        CREATE TRIGGER update_protocol_handlers_updated_at
        BEFORE UPDATE ON protocol_handlers
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """
    )


def downgrade():
    """Drop metadata stamping tables."""

    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS update_metadata_stamps_updated_at ON metadata_stamps"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_protocol_handlers_updated_at ON protocol_handlers"
    )

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop tables
    op.drop_table("protocol_handlers")
    op.drop_table("metadata_stamps")

    # Drop partitioned tables
    op.execute("DROP TABLE IF EXISTS hash_metrics_y2025m10")
    op.execute("DROP TABLE IF EXISTS hash_metrics_y2025m09")
    op.execute("DROP TABLE IF EXISTS hash_metrics")
