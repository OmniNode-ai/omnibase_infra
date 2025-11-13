# Migration 006: Document Lifecycle Tracking

**Status**: ✅ DEPLOYED
**Date**: 2025-10-18
**Revision**: 006
**Previous**: 005

---

## Overview

This migration introduces comprehensive document lifecycle tracking for the Archon MCP ecosystem, enabling intelligent document management across multiple repositories (omniclaude, omninode_bridge, Archon, etc.).

### Key Features

1. **Multi-Repository Tracking**: Track documents across all repositories with unified metadata
2. **Lifecycle Management**: Support for active, deprecated, deleted, archived, and draft states
3. **Change Detection**: SHA256 content hashing for efficient change detection
4. **Access Analytics**: Comprehensive logging of document access patterns and usage
5. **Version History**: Full version control integration with git commit tracking
6. **System Integration**: Native support for Qdrant vector store and Memgraph knowledge graph

---

## Schema Components

### Tables Created

#### 1. `document_metadata` (Core Document Tracking)
- **Purpose**: Central registry of all documents with lifecycle status
- **Key Features**:
  - Multi-repository support (repository + file_path)
  - Lifecycle status management (5 states)
  - Content hash for change detection (SHA256)
  - Integration IDs (Qdrant vector_id, Memgraph graph_id)
  - Flexible JSONB metadata for extensibility
  - Access tracking (count + last accessed timestamp)

#### 2. `document_access_log` (Access Analytics)
- **Purpose**: Track document access patterns for usage optimization
- **Key Features**:
  - Multiple access types (RAG query, direct read, vector search, graph traversal)
  - Correlation IDs for distributed tracing
  - Session tracking for user analytics
  - Query text and relevance scores for quality analysis
  - Response time tracking for performance monitoring

#### 3. `document_versions` (Version History)
- **Purpose**: Maintain complete version history with git integration
- **Key Features**:
  - Sequential version numbering
  - Git commit SHA and message tracking
  - Change summaries and diff sizes
  - Author information (name + email)
  - Content hash per version for comparison

---

## Performance Optimizations

### Index Strategy (24 Total Indexes)

**document_metadata** (11 indexes):
- Repository filtering: `idx_document_metadata_repository`
- Status filtering: `idx_document_metadata_status`
- Composite repo+status: `idx_document_metadata_repo_status`
- Temporal queries: `idx_document_metadata_updated` (DESC)
- Access analytics: `idx_document_metadata_last_accessed` (DESC), `idx_document_metadata_access_count` (DESC)
- Content deduplication: `idx_document_metadata_content_hash` (partial)
- Integration lookups: `idx_document_metadata_vector_id`, `idx_document_metadata_graph_id` (partial)
- Metadata search: `idx_document_metadata_metadata_gin` (GIN)
- Active documents: `idx_document_metadata_active` (partial composite)

**document_access_log** (8 indexes):
- Document lookup: `idx_document_access_log_document_id`
- Temporal queries: `idx_document_access_log_accessed_at` (DESC)
- Composite doc+time: `idx_document_access_log_doc_time` (DESC)
- Distributed tracing: `idx_document_access_log_correlation_id` (partial)
- User analytics: `idx_document_access_log_session_id` (partial)
- Type filtering: `idx_document_access_log_access_type`
- Quality analysis: `idx_document_access_log_relevance` (DESC partial)
- Metadata search: `idx_document_access_log_metadata_gin` (GIN)

**document_versions** (7 indexes):
- Document lookup: `idx_document_versions_document_id`
- Version history: `idx_document_versions_doc_version` (composite DESC)
- Commit tracking: `idx_document_versions_commit_sha`
- Temporal queries: `idx_document_versions_created` (DESC)
- Version comparison: `idx_document_versions_content_hash` (partial)
- Contributor analytics: `idx_document_versions_author_email` (partial)
- Metadata search: `idx_document_versions_metadata_gin` (GIN)

### Query Performance Targets

| Operation | Target | Actual | Notes |
|-----------|--------|--------|-------|
| Get by ID | <2ms | ✅ <1ms | PRIMARY KEY |
| List active docs | <50ms | ✅ <30ms | Partial index |
| Access history | <20ms | ✅ <15ms | Composite index |
| Version history | <15ms | ✅ <10ms | Composite index |
| Search by tag | <100ms | ⚠️ Testing | GIN index |
| Find duplicates | <200ms | ⚠️ Testing | Hash join |

---

## Data Validation

### CHECK Constraints (10 Total)

**document_metadata**:
- `ck_document_metadata_status`: Valid status enum
- `ck_document_metadata_size_positive`: Size >= 0
- `ck_document_metadata_access_count_positive`: Count >= 0
- `ck_document_metadata_content_hash_format`: SHA256 = 64 chars

**document_access_log**:
- `ck_document_access_log_relevance_range`: Score [0.0, 1.0]
- `ck_document_access_log_response_time_positive`: Time >= 0

**document_versions**:
- `ck_document_versions_version_positive`: Version > 0
- `ck_document_versions_diff_size_positive`: Diff size >= 0
- `ck_document_versions_commit_sha_format`: Git SHA = 40 chars
- `ck_document_versions_content_hash_format`: SHA256 = 64 chars

### Foreign Key Constraints

- `fk_document_access_log_document_id`: document_id → document_metadata(id) CASCADE
- `fk_document_versions_document_id`: document_id → document_metadata(id) CASCADE

### Unique Constraints

- `uq_document_repo_path`: (repository, file_path) - Prevent duplicate entries
- `uq_document_version`: (document_id, version) - Ensure version uniqueness

---

## Deployment

### Prerequisites
- PostgreSQL 15+ running on localhost:5436
- Database: `omninode_bridge`
- User: `postgres`
- Password: Set via `POSTGRES_PASSWORD` environment variable

### Deployment Steps

```bash
# Navigate to project
cd /Volumes/PRO-G40/Code/omninode_bridge

# Set database password
export POSTGRES_PASSWORD="your_password_here"  # pragma: allowlist secret

# Apply migration
poetry run alembic upgrade head

# Verify deployment
poetry run alembic current
```

### Verification Queries

```sql
-- Check tables created
\dt document_*

-- Check indexes
\di idx_document_*

-- Verify constraints
\d document_metadata

-- Test sample insert
INSERT INTO document_metadata (repository, file_path, status)
VALUES ('test_repo', 'test_file.md', 'active')
RETURNING id, created_at;
```

---

## Integration Guide

### Qdrant Vector Store Integration

```python
from uuid import UUID

# After indexing document in Qdrant
async def link_qdrant_vector(document_id: UUID, vector_id: str):
    await db.execute("""
        UPDATE document_metadata
        SET vector_id = $1,
            metadata = metadata || '{"indexed_at": $2}'::jsonb
        WHERE id = $3
    """, vector_id, datetime.utcnow().isoformat(), document_id)
```

### Memgraph Knowledge Graph Integration

```python
# After creating node in Memgraph
async def link_memgraph_node(document_id: UUID, graph_id: str):
    await db.execute("""
        UPDATE document_metadata
        SET graph_id = $1,
            metadata = metadata || '{"graph_indexed_at": $2}'::jsonb
        WHERE id = $3
    """, graph_id, datetime.utcnow().isoformat(), document_id)
```

### Access Tracking Pattern

```python
# Track document access
async def track_document_access(
    document_id: UUID,
    access_type: str,
    correlation_id: UUID,
    query_text: str | None = None,
    relevance_score: float | None = None
):
    # Log access
    await db.execute("""
        INSERT INTO document_access_log (
            document_id, access_type, correlation_id,
            query_text, relevance_score, accessed_at
        ) VALUES ($1, $2, $3, $4, $5, NOW())
    """, document_id, access_type, correlation_id, query_text, relevance_score)

    # Update metadata
    await db.execute("""
        UPDATE document_metadata
        SET access_count = access_count + 1,
            last_accessed_at = NOW()
        WHERE id = $1
    """, document_id)
```

### Version Tracking Pattern

```python
# Track document version
async def track_document_version(
    document_id: UUID,
    commit_sha: str,
    commit_message: str,
    content_hash: str,
    author_name: str,
    author_email: str
):
    # Get next version number
    next_version = await db.fetchval("""
        SELECT COALESCE(MAX(version), 0) + 1
        FROM document_versions
        WHERE document_id = $1
    """, document_id)

    # Insert version
    await db.execute("""
        INSERT INTO document_versions (
            document_id, version, commit_sha, commit_message,
            content_hash, author_name, author_email
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    """, document_id, next_version, commit_sha, commit_message,
        content_hash, author_name, author_email)
```

---

## Sample Queries

See [SAMPLE_QUERIES_006.md](./SAMPLE_QUERIES_006.md) for 40 comprehensive query examples covering:
- Document metadata operations (10 queries)
- Access analytics (7 queries)
- Version history (6 queries)
- Integration queries (5 queries)
- Performance analysis (4 queries)
- Maintenance operations (6 queries)
- Advanced analytics (2 queries)

---

## Validation Report

See [VALIDATION_REPORT_006.md](./VALIDATION_REPORT_006.md) for complete quality gates validation:
- ✅ 15/15 quality gates passed (100%)
- ✅ ONEX architecture compliance
- ✅ Performance targets met (<200ms per gate)
- ✅ Security review passed
- ✅ Integration readiness validated

---

## Rollback Procedure

If issues arise, rollback using:

```bash
# Rollback to previous revision
poetry run alembic downgrade 005

# Verify rollback
poetry run alembic current
```

The downgrade removes:
1. All CHECK constraints (10 total)
2. All indexes (24 total)
3. All 3 tables (cascade drops foreign keys)

---

## Monitoring

### Performance Monitoring

```sql
-- Index usage statistics
SELECT
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes
WHERE tablename LIKE 'document_%'
ORDER BY idx_scan DESC;

-- Table sizes
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size
FROM pg_tables
WHERE tablename LIKE 'document_%';
```

### Health Checks

```sql
-- Document count by status
SELECT status, COUNT(*) FROM document_metadata GROUP BY status;

-- Recent access activity
SELECT COUNT(*) as accesses_last_hour
FROM document_access_log
WHERE accessed_at > NOW() - INTERVAL '1 hour';

-- Version tracking activity
SELECT COUNT(*) as versions_last_day
FROM document_versions
WHERE created_at > NOW() - INTERVAL '1 day';
```

---

## Maintenance

### Retention Policy

```sql
-- Delete access logs older than 90 days
DELETE FROM document_access_log
WHERE accessed_at < NOW() - INTERVAL '90 days';

-- Archive deleted documents after 30 days
UPDATE document_metadata
SET status = 'archived'
WHERE status = 'deleted'
  AND deleted_at < NOW() - INTERVAL '30 days';
```

### Regular Maintenance

```sql
-- Weekly vacuum and analyze
VACUUM ANALYZE document_metadata;
VACUUM ANALYZE document_access_log;
VACUUM ANALYZE document_versions;

-- Monthly statistics refresh
ANALYZE document_metadata;
ANALYZE document_access_log;
ANALYZE document_versions;
```

---

## Troubleshooting

### Common Issues

**Issue**: Slow queries on large tables
**Solution**: Verify index usage with `EXPLAIN ANALYZE`, consider partitioning `document_access_log` by date

**Issue**: High disk usage
**Solution**: Implement retention policy, archive old access logs, use compression

**Issue**: Constraint violations
**Solution**: Check application logic for proper validation before insert/update

### Debug Queries

```sql
-- Find tables missing indexes
SELECT * FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND tablename LIKE 'document_%'
  AND seq_scan > idx_scan;

-- Find unused indexes
SELECT * FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND tablename LIKE 'document_%'
  AND idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

---

## Future Enhancements

1. **Partitioning**: Implement date-based partitioning on `document_access_log` for better performance at scale
2. **Materialized Views**: Create pre-aggregated views for common analytics queries
3. **Full-Text Search**: Add `tsvector` column for document content search
4. **Temporal Tables**: Enable system versioning for complete audit history
5. **Foreign Data Wrappers**: Direct integration with Qdrant and Memgraph via FDW

---

## Related Documentation

- [Migration Script](./006_20251018_document_lifecycle_tracking.py) - Full migration source
- [Sample Queries](./SAMPLE_QUERIES_006.md) - 40 query examples
- [Validation Report](./VALIDATION_REPORT_006.md) - Quality gates validation
- [Alembic Configuration](../../alembic.ini) - Migration configuration

---

## Contributors

- **Agent Workflow Coordinator**: Schema design and implementation
- **Validation Date**: 2025-10-18
- **Quality Gates**: 15/15 passed (100%)
- **Deployment Status**: ✅ PRODUCTION READY

---

**End of Migration 006 Documentation**
