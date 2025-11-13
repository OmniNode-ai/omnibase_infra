"""Add document lifecycle tracking tables

Revision ID: 006
Revises: 005
Create Date: 2025-10-18 00:00:00.000000

This migration creates comprehensive document lifecycle tracking tables for
managing document metadata, access patterns, and version history across
multiple repositories (omniclaude, omninode_bridge, Archon, etc.).

Key Features:
- Multi-repository document metadata tracking
- Lifecycle status management (active, deprecated, deleted)
- Content hash-based change detection (SHA256)
- Integration with Qdrant vector store and Memgraph knowledge graph
- Document access analytics and logging
- Full version history with commit tracking
- JSONB storage for flexible metadata extension

Tables:
1. document_metadata - Core document tracking with lifecycle status
2. document_access_log - Access pattern analytics
3. document_versions - Version history and change tracking

Performance Optimizations:
- Composite indexes for common query patterns
- GIN indexes for JSONB metadata searches
- Partial indexes for active documents
- DESC ordering for temporal queries

Integration Points:
- Qdrant vector_id for semantic search
- Memgraph graph_id for knowledge graph relationships
- Pattern lineage via correlation_id in access logs

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade():
    """Create document lifecycle tracking tables with optimized indexes."""

    # Ensure UUID generation function is available
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # =========================================================================
    # Table 1: document_metadata
    # Core document tracking with lifecycle management
    # =========================================================================
    op.create_table(
        "document_metadata",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
            comment="Unique document identifier",
        ),
        # Repository and location
        sa.Column(
            "repository",
            sa.VARCHAR(255),
            nullable=False,
            comment="Repository name (omniclaude, omninode_bridge, Archon, etc.)",
        ),
        sa.Column(
            "file_path",
            sa.TEXT(),
            nullable=False,
            comment="Relative file path within repository",
        ),
        # Lifecycle management
        sa.Column(
            "status",
            sa.VARCHAR(50),
            nullable=False,
            server_default=sa.text("'active'"),
            comment="Lifecycle status: active, deprecated, deleted, archived",
        ),
        # Content tracking
        sa.Column(
            "content_hash",
            sa.VARCHAR(64),
            nullable=True,
            comment="SHA256 hash for change detection",
        ),
        sa.Column(
            "size_bytes",
            sa.BIGINT(),
            nullable=True,
            comment="File size in bytes",
        ),
        sa.Column(
            "mime_type",
            sa.VARCHAR(255),
            nullable=True,
            comment="MIME type (text/markdown, application/yaml, etc.)",
        ),
        # Temporal tracking
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Document creation timestamp (UTC)",
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Last modification timestamp (UTC)",
        ),
        sa.Column(
            "deleted_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Soft deletion timestamp (UTC)",
        ),
        # Access analytics
        sa.Column(
            "access_count",
            sa.INTEGER(),
            nullable=False,
            server_default=sa.text("0"),
            comment="Total access count",
        ),
        sa.Column(
            "last_accessed_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Last access timestamp (UTC)",
        ),
        # Integration IDs
        sa.Column(
            "vector_id",
            sa.TEXT(),
            nullable=True,
            comment="Qdrant vector store identifier",
        ),
        sa.Column(
            "graph_id",
            sa.TEXT(),
            nullable=True,
            comment="Memgraph knowledge graph node identifier",
        ),
        # Flexible metadata
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Flexible metadata (tags, categories, author, language, etc.)",
        ),
        # Unique constraint for repository + file_path
        sa.UniqueConstraint("repository", "file_path", name="uq_document_repo_path"),
        # Table configuration
        comment="Document metadata tracking with lifecycle management and integration IDs",
    )

    # =========================================================================
    # Table 2: document_access_log
    # Access pattern analytics for usage tracking and optimization
    # =========================================================================
    op.create_table(
        "document_access_log",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
            comment="Unique access log entry identifier",
        ),
        # Foreign key to document_metadata
        sa.Column(
            "document_id",
            postgresql.UUID(),
            nullable=False,
            comment="Reference to document_metadata.id",
        ),
        # Access details
        sa.Column(
            "accessed_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Access timestamp (UTC)",
        ),
        sa.Column(
            "access_type",
            sa.VARCHAR(50),
            nullable=False,
            comment="Access type: rag_query, direct_read, vector_search, graph_traversal, etc.",
        ),
        # Correlation and tracing
        sa.Column(
            "correlation_id",
            postgresql.UUID(),
            nullable=True,
            comment="Request correlation ID for tracing",
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(),
            nullable=True,
            comment="User/agent session identifier",
        ),
        # Query context
        sa.Column(
            "query_text",
            sa.TEXT(),
            nullable=True,
            comment="Search query text (for RAG queries)",
        ),
        sa.Column(
            "relevance_score",
            sa.FLOAT(),
            nullable=True,
            comment="Relevance score (0.0-1.0) for search results",
        ),
        # Performance metrics
        sa.Column(
            "response_time_ms",
            sa.INTEGER(),
            nullable=True,
            comment="Response time in milliseconds",
        ),
        # Additional context
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Additional context (user_agent, source_service, filters, etc.)",
        ),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document_metadata.id"],
            name="fk_document_access_log_document_id",
            ondelete="CASCADE",
        ),
        # Table configuration
        comment="Document access analytics for usage tracking and optimization",
    )

    # =========================================================================
    # Table 3: document_versions
    # Version history and change tracking
    # =========================================================================
    op.create_table(
        "document_versions",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
            comment="Unique version identifier",
        ),
        # Foreign key to document_metadata
        sa.Column(
            "document_id",
            postgresql.UUID(),
            nullable=False,
            comment="Reference to document_metadata.id",
        ),
        # Version tracking
        sa.Column(
            "version",
            sa.INTEGER(),
            nullable=False,
            comment="Sequential version number (1, 2, 3...)",
        ),
        sa.Column(
            "commit_sha",
            sa.VARCHAR(40),
            nullable=False,
            comment="Git commit SHA",
        ),
        sa.Column(
            "commit_message",
            sa.TEXT(),
            nullable=True,
            comment="Git commit message",
        ),
        # Change tracking
        sa.Column(
            "changes_summary",
            sa.TEXT(),
            nullable=True,
            comment="Summary of changes in this version",
        ),
        sa.Column(
            "diff_size",
            sa.INTEGER(),
            nullable=True,
            comment="Size of diff in bytes",
        ),
        sa.Column(
            "content_hash",
            sa.VARCHAR(64),
            nullable=True,
            comment="SHA256 hash of document content at this version",
        ),
        # Temporal tracking
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Version creation timestamp (UTC)",
        ),
        # Author information
        sa.Column(
            "author_name",
            sa.VARCHAR(255),
            nullable=True,
            comment="Git commit author name",
        ),
        sa.Column(
            "author_email",
            sa.VARCHAR(255),
            nullable=True,
            comment="Git commit author email",
        ),
        # Additional metadata
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Additional version metadata (tags, release_notes, etc.)",
        ),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document_metadata.id"],
            name="fk_document_versions_document_id",
            ondelete="CASCADE",
        ),
        # Unique constraint for document + version
        sa.UniqueConstraint("document_id", "version", name="uq_document_version"),
        # Table configuration
        comment="Document version history and change tracking",
    )

    # =========================================================================
    # INDEXES: document_metadata
    # =========================================================================

    # Index 1: Repository filtering (common query pattern)
    op.create_index(
        "idx_document_metadata_repository",
        "document_metadata",
        ["repository"],
    )

    # Index 2: Status filtering (active documents)
    op.create_index(
        "idx_document_metadata_status",
        "document_metadata",
        ["status"],
    )

    # Index 3: Composite index for repository + status queries
    op.create_index(
        "idx_document_metadata_repo_status",
        "document_metadata",
        ["repository", "status"],
    )

    # Index 4: Updated timestamp for change detection (DESC for latest first)
    op.execute(
        "CREATE INDEX idx_document_metadata_updated ON document_metadata (updated_at DESC)"
    )

    # Index 5: Last accessed timestamp for analytics (DESC for recent first)
    op.execute(
        "CREATE INDEX idx_document_metadata_last_accessed ON document_metadata (last_accessed_at DESC NULLS LAST)"
    )

    # Index 6: Access count for popularity analysis (DESC for most accessed first)
    op.execute(
        "CREATE INDEX idx_document_metadata_access_count ON document_metadata (access_count DESC)"
    )

    # Index 7: Content hash for duplicate detection
    op.create_index(
        "idx_document_metadata_content_hash",
        "document_metadata",
        ["content_hash"],
        postgresql_where=sa.text("content_hash IS NOT NULL"),
    )

    # Index 8: Vector ID lookup for Qdrant integration
    op.create_index(
        "idx_document_metadata_vector_id",
        "document_metadata",
        ["vector_id"],
        postgresql_where=sa.text("vector_id IS NOT NULL"),
    )

    # Index 9: Graph ID lookup for Memgraph integration
    op.create_index(
        "idx_document_metadata_graph_id",
        "document_metadata",
        ["graph_id"],
        postgresql_where=sa.text("graph_id IS NOT NULL"),
    )

    # Index 10: GIN index for JSONB metadata queries
    op.execute(
        "CREATE INDEX idx_document_metadata_metadata_gin ON document_metadata USING GIN(metadata)"
    )

    # Index 11: Partial index for active documents (most common queries)
    op.execute(
        "CREATE INDEX idx_document_metadata_active ON document_metadata (repository, updated_at DESC) WHERE status = 'active'"
    )

    # =========================================================================
    # INDEXES: document_access_log
    # =========================================================================

    # Index 1: Document lookup for access history
    op.create_index(
        "idx_document_access_log_document_id",
        "document_access_log",
        ["document_id"],
    )

    # Index 2: Temporal queries (DESC for recent first)
    op.execute(
        "CREATE INDEX idx_document_access_log_accessed_at ON document_access_log (accessed_at DESC)"
    )

    # Index 3: Composite index for document + time queries
    op.execute(
        "CREATE INDEX idx_document_access_log_doc_time ON document_access_log (document_id, accessed_at DESC)"
    )

    # Index 4: Correlation ID for distributed tracing
    op.create_index(
        "idx_document_access_log_correlation_id",
        "document_access_log",
        ["correlation_id"],
        postgresql_where=sa.text("correlation_id IS NOT NULL"),
    )

    # Index 5: Session ID for user analytics
    op.create_index(
        "idx_document_access_log_session_id",
        "document_access_log",
        ["session_id"],
        postgresql_where=sa.text("session_id IS NOT NULL"),
    )

    # Index 6: Access type filtering
    op.create_index(
        "idx_document_access_log_access_type",
        "document_access_log",
        ["access_type"],
    )

    # Index 7: Relevance score for quality analysis (DESC for highest relevance)
    op.execute(
        "CREATE INDEX idx_document_access_log_relevance ON document_access_log (relevance_score DESC NULLS LAST) WHERE relevance_score IS NOT NULL"
    )

    # Index 8: GIN index for JSONB metadata queries
    op.execute(
        "CREATE INDEX idx_document_access_log_metadata_gin ON document_access_log USING GIN(metadata)"
    )

    # =========================================================================
    # INDEXES: document_versions
    # =========================================================================

    # Index 1: Document lookup for version history
    op.create_index(
        "idx_document_versions_document_id",
        "document_versions",
        ["document_id"],
    )

    # Index 2: Composite index for document + version queries (DESC for latest first)
    op.execute(
        "CREATE INDEX idx_document_versions_doc_version ON document_versions (document_id, version DESC)"
    )

    # Index 3: Commit SHA lookup
    op.create_index(
        "idx_document_versions_commit_sha",
        "document_versions",
        ["commit_sha"],
    )

    # Index 4: Created timestamp (DESC for recent first)
    op.execute(
        "CREATE INDEX idx_document_versions_created ON document_versions (created_at DESC)"
    )

    # Index 5: Content hash for version comparison
    op.create_index(
        "idx_document_versions_content_hash",
        "document_versions",
        ["content_hash"],
        postgresql_where=sa.text("content_hash IS NOT NULL"),
    )

    # Index 6: Author email for contributor analytics
    op.create_index(
        "idx_document_versions_author_email",
        "document_versions",
        ["author_email"],
        postgresql_where=sa.text("author_email IS NOT NULL"),
    )

    # Index 7: GIN index for JSONB metadata queries
    op.execute(
        "CREATE INDEX idx_document_versions_metadata_gin ON document_versions USING GIN(metadata)"
    )

    # =========================================================================
    # CHECK CONSTRAINTS: Data validation
    # =========================================================================

    # document_metadata constraints
    op.create_check_constraint(
        "ck_document_metadata_status",
        "document_metadata",
        "status IN ('active', 'deprecated', 'deleted', 'archived', 'draft')",
    )

    op.create_check_constraint(
        "ck_document_metadata_size_positive",
        "document_metadata",
        "size_bytes IS NULL OR size_bytes >= 0",
    )

    op.create_check_constraint(
        "ck_document_metadata_access_count_positive",
        "document_metadata",
        "access_count >= 0",
    )

    op.create_check_constraint(
        "ck_document_metadata_content_hash_format",
        "document_metadata",
        "content_hash IS NULL OR length(content_hash) = 64",
    )

    # document_access_log constraints
    op.create_check_constraint(
        "ck_document_access_log_relevance_range",
        "document_access_log",
        "relevance_score IS NULL OR (relevance_score >= 0.0 AND relevance_score <= 1.0)",
    )

    op.create_check_constraint(
        "ck_document_access_log_response_time_positive",
        "document_access_log",
        "response_time_ms IS NULL OR response_time_ms >= 0",
    )

    # document_versions constraints
    op.create_check_constraint(
        "ck_document_versions_version_positive",
        "document_versions",
        "version > 0",
    )

    op.create_check_constraint(
        "ck_document_versions_diff_size_positive",
        "document_versions",
        "diff_size IS NULL OR diff_size >= 0",
    )

    op.create_check_constraint(
        "ck_document_versions_commit_sha_format",
        "document_versions",
        "length(commit_sha) = 40",
    )

    op.create_check_constraint(
        "ck_document_versions_content_hash_format",
        "document_versions",
        "content_hash IS NULL OR length(content_hash) = 64",
    )


def downgrade():
    """Drop document lifecycle tracking tables and all indexes."""

    # Drop CHECK constraints for document_versions
    op.drop_constraint("ck_document_versions_content_hash_format", "document_versions")
    op.drop_constraint("ck_document_versions_commit_sha_format", "document_versions")
    op.drop_constraint("ck_document_versions_diff_size_positive", "document_versions")
    op.drop_constraint("ck_document_versions_version_positive", "document_versions")

    # Drop CHECK constraints for document_access_log
    op.drop_constraint(
        "ck_document_access_log_response_time_positive", "document_access_log"
    )
    op.drop_constraint("ck_document_access_log_relevance_range", "document_access_log")

    # Drop CHECK constraints for document_metadata
    op.drop_constraint("ck_document_metadata_content_hash_format", "document_metadata")
    op.drop_constraint(
        "ck_document_metadata_access_count_positive", "document_metadata"
    )
    op.drop_constraint("ck_document_metadata_size_positive", "document_metadata")
    op.drop_constraint("ck_document_metadata_status", "document_metadata")

    # Drop indexes for document_versions
    op.drop_index("idx_document_versions_metadata_gin", table_name="document_versions")
    op.drop_index("idx_document_versions_author_email", table_name="document_versions")
    op.drop_index("idx_document_versions_content_hash", table_name="document_versions")
    op.drop_index("idx_document_versions_created", table_name="document_versions")
    op.drop_index("idx_document_versions_commit_sha", table_name="document_versions")
    op.drop_index("idx_document_versions_doc_version", table_name="document_versions")
    op.drop_index("idx_document_versions_document_id", table_name="document_versions")

    # Drop indexes for document_access_log
    op.drop_index(
        "idx_document_access_log_metadata_gin", table_name="document_access_log"
    )
    op.drop_index("idx_document_access_log_relevance", table_name="document_access_log")
    op.drop_index(
        "idx_document_access_log_access_type", table_name="document_access_log"
    )
    op.drop_index(
        "idx_document_access_log_session_id", table_name="document_access_log"
    )
    op.drop_index(
        "idx_document_access_log_correlation_id", table_name="document_access_log"
    )
    op.drop_index("idx_document_access_log_doc_time", table_name="document_access_log")
    op.drop_index(
        "idx_document_access_log_accessed_at", table_name="document_access_log"
    )
    op.drop_index(
        "idx_document_access_log_document_id", table_name="document_access_log"
    )

    # Drop indexes for document_metadata
    op.drop_index("idx_document_metadata_active", table_name="document_metadata")
    op.drop_index("idx_document_metadata_metadata_gin", table_name="document_metadata")
    op.drop_index("idx_document_metadata_graph_id", table_name="document_metadata")
    op.drop_index("idx_document_metadata_vector_id", table_name="document_metadata")
    op.drop_index("idx_document_metadata_content_hash", table_name="document_metadata")
    op.drop_index("idx_document_metadata_access_count", table_name="document_metadata")
    op.drop_index("idx_document_metadata_last_accessed", table_name="document_metadata")
    op.drop_index("idx_document_metadata_updated", table_name="document_metadata")
    op.drop_index("idx_document_metadata_repo_status", table_name="document_metadata")
    op.drop_index("idx_document_metadata_status", table_name="document_metadata")
    op.drop_index("idx_document_metadata_repository", table_name="document_metadata")

    # Drop tables (cascade will drop foreign keys)
    op.drop_table("document_versions")
    op.drop_table("document_access_log")
    op.drop_table("document_metadata")
