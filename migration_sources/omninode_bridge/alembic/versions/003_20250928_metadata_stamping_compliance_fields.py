"""Add compliance fields to metadata_stamps table

Revision ID: 003
Revises: 002
Create Date: 2025-09-28 18:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    """Add compliance fields to metadata_stamps table."""

    # Add compliance fields to metadata_stamps table
    op.add_column(
        "metadata_stamps",
        sa.Column(
            "intelligence_data",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    op.add_column(
        "metadata_stamps",
        sa.Column("version", sa.INTEGER(), nullable=False, server_default=sa.text("1")),
    )

    op.add_column(
        "metadata_stamps",
        sa.Column(
            "op_id",
            postgresql.UUID(),
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
    )

    op.add_column(
        "metadata_stamps",
        sa.Column(
            "namespace",
            sa.VARCHAR(255),
            nullable=False,
            server_default="omninode.services.metadata",
        ),
    )

    op.add_column(
        "metadata_stamps",
        sa.Column(
            "metadata_version", sa.VARCHAR(10), nullable=False, server_default="0.1"
        ),
    )

    # Create indexes for new compliance fields
    op.create_index("idx_metadata_stamps_namespace", "metadata_stamps", ["namespace"])

    op.create_index("idx_metadata_stamps_op_id", "metadata_stamps", ["op_id"])

    op.create_index("idx_metadata_stamps_version", "metadata_stamps", ["version"])

    op.create_index(
        "idx_metadata_stamps_metadata_version", "metadata_stamps", ["metadata_version"]
    )

    # Create GIN index for intelligence_data JSONB field
    op.execute(
        "CREATE INDEX idx_metadata_stamps_intelligence_data_gin ON metadata_stamps USING GIN(intelligence_data)"
    )


def downgrade():
    """Remove compliance fields from metadata_stamps table."""

    # Drop indexes first
    op.drop_index("idx_metadata_stamps_intelligence_data_gin", "metadata_stamps")
    op.drop_index("idx_metadata_stamps_metadata_version", "metadata_stamps")
    op.drop_index("idx_metadata_stamps_version", "metadata_stamps")
    op.drop_index("idx_metadata_stamps_op_id", "metadata_stamps")
    op.drop_index("idx_metadata_stamps_namespace", "metadata_stamps")

    # Drop columns
    op.drop_column("metadata_stamps", "metadata_version")
    op.drop_column("metadata_stamps", "namespace")
    op.drop_column("metadata_stamps", "op_id")
    op.drop_column("metadata_stamps", "version")
    op.drop_column("metadata_stamps", "intelligence_data")
