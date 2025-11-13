# Sample Queries - Document Lifecycle Tracking (Migration 006)

This document provides comprehensive examples for common operations using the document lifecycle tracking schema.

## Table of Contents
- [Document Metadata Operations](#document-metadata-operations)
- [Access Analytics](#access-analytics)
- [Version History](#version-history)
- [Integration Queries](#integration-queries)
- [Performance Analysis](#performance-analysis)
- [Maintenance Operations](#maintenance-operations)

---

## Document Metadata Operations

### 1. Register New Document
```sql
-- Insert new document with metadata
INSERT INTO document_metadata (
    repository,
    file_path,
    status,
    content_hash,
    size_bytes,
    mime_type,
    metadata
) VALUES (
    'omniclaude',
    'agents/AGENT_FRAMEWORK.md',
    'active',
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',  -- pragma: allowlist secret
    12345,
    'text/markdown',
    '{"tags": ["agent", "framework"], "language": "en", "author": "system"}'::jsonb
)
RETURNING id, created_at;
```

### 2. Update Document Content Hash (After Change Detection)
```sql
-- Update content hash and timestamp after file modification
UPDATE document_metadata
SET
    content_hash = 'new_sha256_hash_here',
    updated_at = NOW(),
    size_bytes = 13500
WHERE repository = 'omniclaude'
  AND file_path = 'agents/AGENT_FRAMEWORK.md'
RETURNING id, updated_at;
```

### 3. Find All Active Documents in Repository
```sql
-- Get all active documents for a specific repository
-- Uses idx_document_metadata_active partial index
SELECT
    id,
    file_path,
    content_hash,
    size_bytes,
    updated_at,
    access_count,
    last_accessed_at
FROM document_metadata
WHERE repository = 'omniclaude'
  AND status = 'active'
ORDER BY updated_at DESC;
```

### 4. Detect Modified Documents (Change Detection)
```sql
-- Find documents modified in the last 24 hours
-- Uses idx_document_metadata_updated DESC index
SELECT
    repository,
    file_path,
    content_hash,
    updated_at,
    size_bytes
FROM document_metadata
WHERE updated_at > NOW() - INTERVAL '24 hours'
  AND status = 'active'
ORDER BY updated_at DESC;
```

### 5. Find Duplicate Content (By Hash)
```sql
-- Identify documents with identical content across repositories
-- Uses idx_document_metadata_content_hash
SELECT
    content_hash,
    array_agg(repository || '/' || file_path) as file_locations,
    COUNT(*) as duplicate_count
FROM document_metadata
WHERE content_hash IS NOT NULL
  AND status = 'active'
GROUP BY content_hash
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;
```

### 6. Soft Delete Document
```sql
-- Mark document as deleted (soft delete)
UPDATE document_metadata
SET
    status = 'deleted',
    deleted_at = NOW()
WHERE repository = 'omniclaude'
  AND file_path = 'deprecated/old_file.md'
RETURNING id, deleted_at;
```

### 7. Search Documents by Metadata Tags
```sql
-- Find all documents with specific tags using JSONB
-- Uses idx_document_metadata_metadata_gin GIN index
SELECT
    repository,
    file_path,
    metadata->>'tags' as tags,
    updated_at
FROM document_metadata
WHERE status = 'active'
  AND metadata @> '{"tags": ["agent"]}'::jsonb
ORDER BY updated_at DESC;
```

### 8. Get Most Popular Documents
```sql
-- Find most accessed documents
-- Uses idx_document_metadata_access_count DESC index
SELECT
    repository,
    file_path,
    access_count,
    last_accessed_at,
    ROUND(access_count::numeric / EXTRACT(EPOCH FROM (NOW() - created_at)) * 86400, 2) as avg_daily_accesses
FROM document_metadata
WHERE status = 'active'
  AND created_at < NOW() - INTERVAL '7 days'
ORDER BY access_count DESC
LIMIT 20;
```

### 9. Link Document to Qdrant Vector
```sql
-- Update document with Qdrant vector ID after indexing
UPDATE document_metadata
SET
    vector_id = 'qdrant_uuid_or_id_here',
    metadata = metadata || '{"indexed_at": "2025-10-18T10:30:00Z"}'::jsonb
WHERE id = 'document_uuid_here'
RETURNING id, vector_id;
```

### 10. Link Document to Memgraph Node
```sql
-- Update document with Memgraph graph ID after knowledge graph creation
UPDATE document_metadata
SET
    graph_id = 'memgraph_node_id_here',
    metadata = metadata || '{"graph_indexed_at": "2025-10-18T10:35:00Z"}'::jsonb
WHERE id = 'document_uuid_here'
RETURNING id, graph_id;
```

---

## Access Analytics

### 11. Log Document Access
```sql
-- Record document access event
INSERT INTO document_access_log (
    document_id,
    access_type,
    correlation_id,
    session_id,
    query_text,
    relevance_score,
    response_time_ms,
    metadata
) VALUES (
    'document_uuid_here',
    'rag_query',
    'correlation_uuid_here',
    'session_uuid_here',
    'how to implement agent workflow',
    0.92,
    150,
    '{"source_service": "archon_mcp", "user_agent": "claude_code"}'::jsonb
);

-- Increment access count on document_metadata
UPDATE document_metadata
SET
    access_count = access_count + 1,
    last_accessed_at = NOW()
WHERE id = 'document_uuid_here';
```

### 12. Get Document Access History
```sql
-- View all accesses for a specific document
-- Uses idx_document_access_log_doc_time composite index
SELECT
    accessed_at,
    access_type,
    correlation_id,
    query_text,
    relevance_score,
    response_time_ms
FROM document_access_log
WHERE document_id = 'document_uuid_here'
ORDER BY accessed_at DESC
LIMIT 50;
```

### 13. Analyze Access Patterns by Type
```sql
-- Aggregate access statistics by access type
-- Uses idx_document_access_log_access_type
SELECT
    access_type,
    COUNT(*) as total_accesses,
    AVG(relevance_score) as avg_relevance,
    AVG(response_time_ms) as avg_response_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time_ms
FROM document_access_log
WHERE accessed_at > NOW() - INTERVAL '7 days'
GROUP BY access_type
ORDER BY total_accesses DESC;
```

### 14. Find High-Relevance Searches
```sql
-- Identify searches that returned highly relevant results
-- Uses idx_document_access_log_relevance partial index
SELECT
    dal.query_text,
    dm.repository,
    dm.file_path,
    dal.relevance_score,
    dal.accessed_at
FROM document_access_log dal
JOIN document_metadata dm ON dal.document_id = dm.id
WHERE dal.relevance_score >= 0.85
  AND dal.query_text IS NOT NULL
  AND dal.accessed_at > NOW() - INTERVAL '24 hours'
ORDER BY dal.relevance_score DESC, dal.accessed_at DESC
LIMIT 20;
```

### 15. Track User Session Activity
```sql
-- View all document accesses in a user session
-- Uses idx_document_access_log_session_id partial index
SELECT
    dal.accessed_at,
    dm.repository,
    dm.file_path,
    dal.access_type,
    dal.query_text,
    dal.relevance_score
FROM document_access_log dal
JOIN document_metadata dm ON dal.document_id = dm.id
WHERE dal.session_id = 'session_uuid_here'
ORDER BY dal.accessed_at ASC;
```

### 16. Identify Slow Access Patterns
```sql
-- Find access patterns with high response times
SELECT
    access_type,
    AVG(response_time_ms) as avg_response_time,
    MAX(response_time_ms) as max_response_time,
    COUNT(*) as access_count,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY response_time_ms) as p90_response_time
FROM document_access_log
WHERE response_time_ms IS NOT NULL
  AND accessed_at > NOW() - INTERVAL '7 days'
GROUP BY access_type
HAVING AVG(response_time_ms) > 200
ORDER BY avg_response_time DESC;
```

### 17. Correlate Accesses Across Services
```sql
-- Trace document accesses by correlation ID
-- Uses idx_document_access_log_correlation_id partial index
SELECT
    dal.accessed_at,
    dm.repository,
    dm.file_path,
    dal.access_type,
    dal.response_time_ms,
    dal.metadata->>'source_service' as source_service
FROM document_access_log dal
JOIN document_metadata dm ON dal.document_id = dm.id
WHERE dal.correlation_id = 'correlation_uuid_here'
ORDER BY dal.accessed_at ASC;
```

---

## Version History

### 18. Create Document Version
```sql
-- Record new version after document modification
INSERT INTO document_versions (
    document_id,
    version,
    commit_sha,
    commit_message,
    changes_summary,
    diff_size,
    content_hash,
    author_name,
    author_email,
    metadata
) VALUES (
    'document_uuid_here',
    (SELECT COALESCE(MAX(version), 0) + 1 FROM document_versions WHERE document_id = 'document_uuid_here'),
    '1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t',  -- pragma: allowlist secret
    'Update agent framework documentation',
    'Added new workflow patterns and quality gates',
    2048,
    'new_content_hash_here',
    'Claude Code',
    'claude@anthropic.com',
    '{"release_notes": "Phase 8 updates", "breaking_changes": false}'::jsonb
)
RETURNING id, version, created_at;
```

### 19. Get Latest Version
```sql
-- Retrieve the most recent version of a document
-- Uses idx_document_versions_doc_version composite index
SELECT
    version,
    commit_sha,
    commit_message,
    changes_summary,
    content_hash,
    created_at,
    author_name
FROM document_versions
WHERE document_id = 'document_uuid_here'
ORDER BY version DESC
LIMIT 1;
```

### 20. Get Full Version History
```sql
-- List all versions for a document with details
-- Uses idx_document_versions_doc_version composite index
SELECT
    dv.version,
    dv.commit_sha,
    dv.commit_message,
    dv.changes_summary,
    dv.diff_size,
    dv.created_at,
    dv.author_name,
    dv.author_email,
    dm.file_path
FROM document_versions dv
JOIN document_metadata dm ON dv.document_id = dm.id
WHERE dv.document_id = 'document_uuid_here'
ORDER BY dv.version DESC;
```

### 21. Find Versions by Commit SHA
```sql
-- Lookup all document versions for a specific commit
-- Uses idx_document_versions_commit_sha
SELECT
    dm.repository,
    dm.file_path,
    dv.version,
    dv.changes_summary,
    dv.diff_size,
    dv.created_at
FROM document_versions dv
JOIN document_metadata dm ON dv.document_id = dm.id
WHERE dv.commit_sha = '1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t'  -- pragma: allowlist secret
ORDER BY dm.repository, dm.file_path;
```

### 22. Compare Two Versions
```sql
-- Get details for version comparison
SELECT
    v1.version as version_from,
    v2.version as version_to,
    v1.content_hash as content_hash_from,
    v2.content_hash as content_hash_to,
    v2.diff_size,
    v1.created_at as date_from,
    v2.created_at as date_to,
    EXTRACT(EPOCH FROM (v2.created_at - v1.created_at))/3600 as hours_between
FROM document_versions v1
JOIN document_versions v2 ON v1.document_id = v2.document_id
WHERE v1.document_id = 'document_uuid_here'
  AND v1.version = 5
  AND v2.version = 8;
```

### 23. Find Active Contributors
```sql
-- Identify most active document contributors
-- Uses idx_document_versions_author_email partial index
SELECT
    author_email,
    author_name,
    COUNT(*) as total_versions,
    SUM(diff_size) as total_changes_bytes,
    MIN(created_at) as first_contribution,
    MAX(created_at) as last_contribution
FROM document_versions
WHERE author_email IS NOT NULL
  AND created_at > NOW() - INTERVAL '90 days'
GROUP BY author_email, author_name
ORDER BY total_versions DESC
LIMIT 10;
```

---

## Integration Queries

### 24. Find Documents by Qdrant Vector ID
```sql
-- Lookup document metadata using Qdrant vector ID
-- Uses idx_document_metadata_vector_id partial index
SELECT
    id,
    repository,
    file_path,
    content_hash,
    updated_at,
    metadata
FROM document_metadata
WHERE vector_id = 'qdrant_vector_id_here'
  AND status = 'active';
```

### 25. Find Documents by Memgraph Node ID
```sql
-- Lookup document metadata using Memgraph graph ID
-- Uses idx_document_metadata_graph_id partial index
SELECT
    id,
    repository,
    file_path,
    content_hash,
    updated_at,
    metadata
FROM document_metadata
WHERE graph_id = 'memgraph_node_id_here'
  AND status = 'active';
```

### 26. List Documents Needing Vector Indexing
```sql
-- Find active documents not yet indexed in Qdrant
SELECT
    id,
    repository,
    file_path,
    updated_at,
    size_bytes
FROM document_metadata
WHERE status = 'active'
  AND vector_id IS NULL
ORDER BY updated_at DESC;
```

### 27. List Documents Needing Graph Indexing
```sql
-- Find active documents not yet added to knowledge graph
SELECT
    id,
    repository,
    file_path,
    updated_at,
    size_bytes
FROM document_metadata
WHERE status = 'active'
  AND graph_id IS NULL
ORDER BY updated_at DESC;
```

### 28. Sync Status Report
```sql
-- Generate sync status report across systems
SELECT
    status,
    COUNT(*) as total_documents,
    COUNT(vector_id) as indexed_in_qdrant,
    COUNT(graph_id) as indexed_in_memgraph,
    COUNT(*) FILTER (WHERE vector_id IS NOT NULL AND graph_id IS NOT NULL) as fully_synced,
    COUNT(*) FILTER (WHERE vector_id IS NULL OR graph_id IS NULL) as needs_sync
FROM document_metadata
GROUP BY status
ORDER BY status;
```

---

## Performance Analysis

### 29. Document Access Frequency Analysis
```sql
-- Analyze document access patterns over time
WITH access_stats AS (
    SELECT
        dm.repository,
        dm.file_path,
        dm.access_count,
        COUNT(dal.id) as recent_accesses,
        AVG(dal.relevance_score) as avg_relevance,
        MAX(dal.accessed_at) as last_access
    FROM document_metadata dm
    LEFT JOIN document_access_log dal ON dm.id = dal.document_id
        AND dal.accessed_at > NOW() - INTERVAL '7 days'
    WHERE dm.status = 'active'
    GROUP BY dm.id, dm.repository, dm.file_path, dm.access_count
)
SELECT
    repository,
    file_path,
    access_count as total_accesses,
    recent_accesses as last_7_days,
    ROUND(avg_relevance::numeric, 3) as avg_relevance_score,
    last_access,
    CASE
        WHEN recent_accesses > 10 THEN 'hot'
        WHEN recent_accesses > 5 THEN 'warm'
        WHEN recent_accesses > 0 THEN 'cool'
        ELSE 'cold'
    END as temperature
FROM access_stats
ORDER BY recent_accesses DESC, total_accesses DESC
LIMIT 50;
```

### 30. Repository Size and Activity Report
```sql
-- Aggregate statistics by repository
SELECT
    repository,
    COUNT(*) as total_documents,
    COUNT(*) FILTER (WHERE status = 'active') as active_documents,
    SUM(size_bytes) as total_size_bytes,
    ROUND(AVG(size_bytes)::numeric, 0) as avg_size_bytes,
    MAX(updated_at) as most_recent_update,
    SUM(access_count) as total_accesses
FROM document_metadata
GROUP BY repository
ORDER BY total_documents DESC;
```

### 31. Stale Document Detection
```sql
-- Find documents that haven't been accessed or updated recently
SELECT
    repository,
    file_path,
    updated_at,
    last_accessed_at,
    access_count,
    EXTRACT(DAYS FROM (NOW() - updated_at)) as days_since_update,
    EXTRACT(DAYS FROM (NOW() - COALESCE(last_accessed_at, created_at))) as days_since_access
FROM document_metadata
WHERE status = 'active'
  AND (
      updated_at < NOW() - INTERVAL '180 days'
      OR last_accessed_at < NOW() - INTERVAL '90 days'
      OR (last_accessed_at IS NULL AND created_at < NOW() - INTERVAL '90 days')
  )
ORDER BY
    LEAST(updated_at, COALESCE(last_accessed_at, created_at)) ASC
LIMIT 50;
```

### 32. Response Time Performance by Repository
```sql
-- Analyze response times grouped by repository
SELECT
    dm.repository,
    dal.access_type,
    COUNT(*) as total_accesses,
    ROUND(AVG(dal.response_time_ms)::numeric, 2) as avg_response_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY dal.response_time_ms) as p50_response_ms,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY dal.response_time_ms) as p90_response_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY dal.response_time_ms) as p99_response_ms
FROM document_access_log dal
JOIN document_metadata dm ON dal.document_id = dm.id
WHERE dal.response_time_ms IS NOT NULL
  AND dal.accessed_at > NOW() - INTERVAL '7 days'
GROUP BY dm.repository, dal.access_type
ORDER BY dm.repository, avg_response_ms DESC;
```

---

## Maintenance Operations

### 33. Cleanup Old Access Logs (Retention Policy)
```sql
-- Delete access logs older than 90 days (retention policy)
DELETE FROM document_access_log
WHERE accessed_at < NOW() - INTERVAL '90 days'
RETURNING COUNT(*);
```

### 34. Archive Deleted Documents
```sql
-- Move deleted documents to archived status after retention period
UPDATE document_metadata
SET status = 'archived'
WHERE status = 'deleted'
  AND deleted_at < NOW() - INTERVAL '30 days'
RETURNING id, repository, file_path, deleted_at;
```

### 35. Recalculate Access Counts
```sql
-- Recalculate access counts from access log (data repair)
UPDATE document_metadata dm
SET access_count = (
    SELECT COUNT(*)
    FROM document_access_log dal
    WHERE dal.document_id = dm.id
)
WHERE dm.id IN (
    SELECT DISTINCT document_id
    FROM document_access_log
);
```

### 36. Vacuum and Analyze Tables
```sql
-- Maintenance operations for query performance
VACUUM ANALYZE document_metadata;
VACUUM ANALYZE document_access_log;
VACUUM ANALYZE document_versions;
```

### 37. Index Health Check
```sql
-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND tablename IN ('document_metadata', 'document_access_log', 'document_versions')
ORDER BY tablename, idx_scan DESC;
```

### 38. Table Size Report
```sql
-- Check table and index sizes
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS indexes_size
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('document_metadata', 'document_access_log', 'document_versions')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Advanced Queries

### 39. Document Popularity Trending
```sql
-- Calculate trending score based on recent vs historical access patterns
WITH recent_stats AS (
    SELECT
        document_id,
        COUNT(*) as recent_accesses,
        AVG(relevance_score) as recent_relevance
    FROM document_access_log
    WHERE accessed_at > NOW() - INTERVAL '7 days'
    GROUP BY document_id
),
historical_stats AS (
    SELECT
        document_id,
        COUNT(*) as historical_accesses
    FROM document_access_log
    WHERE accessed_at BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'
    GROUP BY document_id
)
SELECT
    dm.repository,
    dm.file_path,
    COALESCE(rs.recent_accesses, 0) as last_7_days,
    COALESCE(hs.historical_accesses, 0) as previous_23_days,
    ROUND((COALESCE(rs.recent_accesses, 0)::numeric / NULLIF(hs.historical_accesses, 0)) * 100, 2) as trend_percentage,
    ROUND(rs.recent_relevance::numeric, 3) as avg_relevance,
    CASE
        WHEN COALESCE(rs.recent_accesses, 0)::numeric / NULLIF(hs.historical_accesses, 0) > 1.5 THEN 'trending_up'
        WHEN COALESCE(rs.recent_accesses, 0)::numeric / NULLIF(hs.historical_accesses, 0) < 0.5 THEN 'trending_down'
        ELSE 'stable'
    END as trend_status
FROM document_metadata dm
LEFT JOIN recent_stats rs ON dm.id = rs.document_id
LEFT JOIN historical_stats hs ON dm.id = hs.document_id
WHERE dm.status = 'active'
  AND (rs.recent_accesses IS NOT NULL OR hs.historical_accesses IS NOT NULL)
ORDER BY trend_percentage DESC NULLS LAST
LIMIT 30;
```

### 40. Document Lifecycle Dashboard
```sql
-- Comprehensive dashboard query for monitoring
SELECT
    dm.repository,
    dm.status,
    COUNT(*) as document_count,
    SUM(dm.size_bytes) as total_size_bytes,
    SUM(dm.access_count) as total_accesses,
    COUNT(dm.vector_id) as qdrant_indexed,
    COUNT(dm.graph_id) as graph_indexed,
    COUNT(DISTINCT dv.id) as total_versions,
    MAX(dm.updated_at) as most_recent_update,
    COUNT(dal.id) FILTER (WHERE dal.accessed_at > NOW() - INTERVAL '7 days') as recent_accesses
FROM document_metadata dm
LEFT JOIN document_versions dv ON dm.id = dv.document_id
LEFT JOIN document_access_log dal ON dm.id = dal.document_id
GROUP BY dm.repository, dm.status
ORDER BY dm.repository, dm.status;
```

---

## Notes

### Index Usage
- All queries are designed to utilize the 24 indexes created in migration 006
- GIN indexes enable fast JSONB queries on metadata fields
- Partial indexes optimize queries for common filters (status='active', NOT NULL conditions)
- Composite indexes support multi-column queries with proper ordering

### Performance Considerations
- Use `EXPLAIN ANALYZE` to verify index usage
- Consider partitioning `document_access_log` by date for large-scale deployments
- Implement retention policies to prevent unbounded table growth
- Use `VACUUM ANALYZE` regularly to maintain query performance

### Best Practices
- Always filter by `status = 'active'` for production queries
- Use correlation IDs for distributed tracing across services
- Leverage JSONB metadata for flexible schema extensions
- Implement proper cascading deletes via foreign key constraints
- Use soft deletes (status='deleted') instead of hard deletes for audit trails

---

**Migration Version**: 006
**Created**: 2025-10-18
**Database**: omninode_bridge
**Schema**: public
