# OnexTree Integration Design

**Document Version:** 1.0
**Created:** 2025-10-24
**Status:** Design Complete - Ready for Implementation
**Author:** Polymorphic Agent

## Overview

This document outlines the complete integration of OnexTree service with the omninode_bridge metadata stamping workflow. The integration provides automated project structure intelligence, pattern enrichment, and incremental tree updates to enhance metadata quality.

**Design Goals:**
- Automated tree generation triggered by repository changes
- Efficient tree storage with PostgreSQL persistence
- Intelligence enrichment using Archon patterns
- Graceful degradation with circuit breaker protection
- Sub-3s generation, <500ms queries, <100ms storage overhead

---

## 1. Architecture Overview

### When to Generate Trees

**Primary Trigger: Repository Change Detection**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tree Generation Triggers                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. On-Demand (Immediate)                                        │
│    - POST /tree/generate → Manual trigger via API              │
│    - CLI command: bridge-tree-generate                          │
│    - Use case: Initial setup, major refactoring                 │
│                                                                  │
│ 2. Scheduled (Daily)                                            │
│    - Cron: 02:00 UTC daily                                      │
│    - Background task via APScheduler                            │
│    - Use case: Regular synchronization                          │
│                                                                  │
│ 3. Event-Driven (Automatic)                                     │
│    - Git post-commit hook → Kafka event                         │
│    - Kafka consumer triggers incremental update                 │
│    - Use case: Real-time updates on commit                      │
│    - Debounced: 5-second window to batch commits               │
│                                                                  │
│ 4. Pre-Stamping (Conditional)                                   │
│    - Check tree age before stamping workflow                    │
│    - If tree_age > 24 hours → Trigger regeneration             │
│    - Use case: Ensure fresh intelligence                        │
└─────────────────────────────────────────────────────────────────┘
```

**Recommended Default: Hybrid Approach**
- **Event-driven** for development (real-time updates)
- **Daily scheduled** for production (controlled load)
- **Pre-stamping check** as fallback (ensure freshness)

### Where to Store Trees

**PostgreSQL Table: `onextree_snapshots`**

Trees are stored in PostgreSQL for:
- ACID guarantees (concurrent access safety)
- Integration with existing workflow_executions
- Efficient querying via JSONB indexes
- Audit trail with versioning support
- Backup/restore with standard pg_dump

**Alternative Considered: File Storage** ❌
- Pros: Simple, low overhead
- Cons: No transactional guarantees, difficult versioning, no concurrent access control

### Integration with Stamping Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│              Metadata Stamping Workflow with OnexTree                │
└──────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ Stamp Request   │
    │ (user/service)  │
    └────────┬────────┘
             │
             v
    ┌────────────────────────┐
    │ NodeBridgeOrchestrator │ ◄─── Main workflow coordinator
    └────────┬───────────────┘
             │
             ├─────────────────────────────────────────────────────┐
             │                                                     │
             v                                                     v
    ┌────────────────────┐                            ┌──────────────────┐
    │ Check Tree Status  │                            │ BLAKE3 Hash Gen  │
    │ - Age < 24h?       │                            │ (parallel)       │
    │ - Exists for proj? │                            └──────────────────┘
    └────────┬───────────┘                                       │
             │                                                   │
             v                                                   │
    ┌────────────────────┐                                       │
    │ Tree Current?      │                                       │
    └────────┬───────────┘                                       │
             │ No                                                │
             v                                                   │
    ┌────────────────────┐                                       │
    │ Generate/Update    │                                       │
    │ Tree (async)       │                                       │
    └────────┬───────────┘                                       │
             │                                                   │
             v                                                   │
    ┌────────────────────┐                                       │
    │ Enrich with Tree   │ ◄─────────────────────────────────────┤
    │ Intelligence       │                                       │
    │ - Structure data   │                                       │
    │ - Pattern analysis │                                       │
    │ - Related files    │                                       │
    └────────┬───────────┘                                       │
             │                                                   │
             v                                                   v
    ┌────────────────────────────────────────────────────────────┐
    │ Create Stamp with Enhanced Metadata                        │
    │ - BLAKE3 hash                                              │
    │ - Tree intelligence                                        │
    │ - Namespace context                                        │
    └────────┬───────────────────────────────────────────────────┘
             │
             v
    ┌────────────────────┐
    │ Publish Kafka      │
    │ Events             │
    └────────┬───────────┘
             │
             v
    ┌────────────────────┐
    │ Return Response    │
    └────────────────────┘
```

**Key Decision: Async Tree Generation**
- Tree generation happens asynchronously (non-blocking)
- Stamping workflow continues with cached/existing tree if available
- Fresh tree used in next request cycle
- Prevents timeout on large repositories (>10k files)

---

## 2. HTTP Client Design

### Existing Client Enhancement

The `AsyncOnexTreeClient` already exists with robust features. **No rewrite needed** - only minor enhancements:

**Existing Features (Keep):**
- ✅ Circuit breaker with configurable thresholds
- ✅ Exponential backoff retry logic
- ✅ In-memory caching with TTL (5 minutes default)
- ✅ Correlation ID propagation
- ✅ Health check interface
- ✅ Comprehensive metrics

**Enhancements Needed:**

```python
# src/omninode_bridge/clients/onextree_client.py

from typing import Optional
from uuid import UUID
from .base_client import BaseServiceClient, ClientError

class AsyncOnexTreeClient(BaseServiceClient):
    """Enhanced OnexTree client with tree generation support."""

    # ==================== NEW METHODS ====================

    async def generate_tree(
        self,
        project_root: str,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Generate new tree for project.

        Args:
            project_root: Absolute path to project root
            correlation_id: Request correlation ID

        Returns:
            {
                "success": bool,
                "total_files": int,
                "total_directories": int,
                "total_size_mb": float,
                "generation_time_ms": float,
                "tree_hash": str  # SHA256 of tree structure
            }

        Raises:
            ClientError: If generation fails
            ServiceUnavailableError: If OnexTree service unavailable
        """
        try:
            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/generate",
                correlation_id=correlation_id,
                json={"project_root": project_root},
                timeout=5.0,  # Tree generation can take up to 5s
            )

            result = response.json()

            if result.get("status") == "success":
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Tree generation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Tree generation failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "project_root": project_root,
                },
            )
            raise

    async def enrich_tree(
        self,
        project_root: str,
        pattern_sources: list[str] = None,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Enrich existing tree with pattern intelligence.

        Args:
            project_root: Project root path
            pattern_sources: Sources for patterns (e.g., ["archon", "local"])
            correlation_id: Request correlation ID

        Returns:
            {
                "success": bool,
                "patterns_added": int,
                "enrichment_time_ms": float,
                "intelligence_sources": list[str]
            }

        Raises:
            ClientError: If enrichment fails
        """
        try:
            request_body = {
                "project_root": project_root,
                "pattern_sources": pattern_sources or ["archon"],
            }

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/enrich",
                correlation_id=correlation_id,
                json=request_body,
                timeout=2.0,  # Enrichment faster than generation
            )

            result = response.json()

            if result.get("status") == "success":
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Tree enrichment failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Tree enrichment failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "project_root": project_root,
                },
            )
            raise

    async def get_tree_stats(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Get current tree statistics.

        Returns:
            {
                "total_files": int,
                "total_directories": int,
                "total_size_mb": float,
                "last_generated": str (ISO timestamp),
                "tree_hash": str
            }
        """
        try:
            response = await self._make_request_with_retry(
                method="GET",
                endpoint="/stats",
                correlation_id=correlation_id,
            )

            result = response.json()

            if result.get("status") == "success":
                return result.get("data", {})
            else:
                return {}  # Empty stats if not available

        except Exception as e:
            logger.debug(f"Tree stats unavailable: {e}")
            return {}

    # ==================== EXISTING METHODS (UNCHANGED) ====================
    # - get_intelligence(context, ...)
    # - query_knowledge(query, ...)
    # - navigate_tree(start_node, ...)
    # - health_check()
    # - get_metrics()
```

**Configuration Updates:**

```python
# src/omninode_bridge/clients/circuit_breaker.py

# OnexTree-specific circuit breaker config
ONEXTREE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,        # Open after 3 consecutive failures
    recovery_timeout=30,        # Try recovery after 30s
    timeout=5.0,                # 5s timeout for tree generation
    half_open_max_calls=1,      # Test with 1 call when half-open
)
```

---

## 3. Tree Storage Schema

### PostgreSQL Table Design

```sql
-- Migration: 012_create_onextree_snapshots.sql
-- Description: Store OnexTree snapshots with versioning and metadata
-- Dependencies: uuid-ossp extension (already exists)
-- Created: 2025-10-24

-- OnexTree snapshot storage
CREATE TABLE IF NOT EXISTS onextree_snapshots (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id VARCHAR(255) NOT NULL,              -- Project identifier (repo name or path hash)
    project_root TEXT NOT NULL,                     -- Absolute path to project root

    -- Tree data
    tree_structure JSONB NOT NULL,                  -- Complete tree structure
    tree_hash VARCHAR(64) NOT NULL,                 -- SHA256 hash of tree (change detection)

    -- Metadata
    total_files INTEGER NOT NULL CHECK (total_files >= 0),
    total_directories INTEGER NOT NULL CHECK (total_directories >= 0),
    total_size_bytes BIGINT NOT NULL CHECK (total_size_bytes >= 0),

    -- Intelligence enrichment
    enrichment_data JSONB DEFAULT '{}',             -- Pattern analysis, architectural insights
    enrichment_sources TEXT[] DEFAULT ARRAY[]::TEXT[], -- Sources: ['archon', 'local', ...]
    is_enriched BOOLEAN DEFAULT false,

    -- Performance tracking
    generation_time_ms INTEGER CHECK (generation_time_ms >= 0),
    enrichment_time_ms INTEGER CHECK (enrichment_time_ms >= 0),

    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    parent_snapshot_id UUID REFERENCES onextree_snapshots(id) ON DELETE SET NULL,

    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,            -- Auto-cleanup old snapshots

    -- Constraints
    CONSTRAINT tree_hash_format CHECK (tree_hash ~ '^[a-f0-9]{64}$')
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_project_id
    ON onextree_snapshots(project_id);

CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_tree_hash
    ON onextree_snapshots(tree_hash);

CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_created_at
    ON onextree_snapshots(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_expires_at
    ON onextree_snapshots(expires_at)
    WHERE expires_at IS NOT NULL;

-- JSONB indexes for fast intelligence queries
CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_tree_structure
    ON onextree_snapshots USING GIN (tree_structure);

CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_enrichment_data
    ON onextree_snapshots USING GIN (enrichment_data);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_onextree_snapshots_project_created
    ON onextree_snapshots(project_id, created_at DESC);

-- Comments for documentation
COMMENT ON TABLE onextree_snapshots IS 'Stores OnexTree project structure snapshots with versioning';
COMMENT ON COLUMN onextree_snapshots.project_id IS 'Unique project identifier (repo name or path hash)';
COMMENT ON COLUMN onextree_snapshots.tree_hash IS 'SHA256 hash of tree structure for change detection';
COMMENT ON COLUMN onextree_snapshots.enrichment_data IS 'Intelligence data from Archon patterns and analysis';
COMMENT ON COLUMN onextree_snapshots.version IS 'Snapshot version number for incremental updates';
COMMENT ON COLUMN onextree_snapshots.parent_snapshot_id IS 'Reference to previous snapshot for diff tracking';

-- Auto-update trigger for updated_at
CREATE OR REPLACE FUNCTION update_onextree_snapshots_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_onextree_snapshots_updated_at
    BEFORE UPDATE ON onextree_snapshots
    FOR EACH ROW
    EXECUTE FUNCTION update_onextree_snapshots_updated_at();

-- Cleanup function for expired snapshots
CREATE OR REPLACE FUNCTION cleanup_expired_onextree_snapshots()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM onextree_snapshots
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Scheduled cleanup (call from application or pg_cron)
-- Example: SELECT cleanup_expired_onextree_snapshots();
```

**Rollback Migration:**

```sql
-- Migration: 012_rollback_onextree_snapshots.sql

DROP TRIGGER IF EXISTS trigger_onextree_snapshots_updated_at ON onextree_snapshots;
DROP FUNCTION IF EXISTS update_onextree_snapshots_updated_at();
DROP FUNCTION IF EXISTS cleanup_expired_onextree_snapshots();

DROP INDEX IF EXISTS idx_onextree_snapshots_project_id;
DROP INDEX IF EXISTS idx_onextree_snapshots_tree_hash;
DROP INDEX IF EXISTS idx_onextree_snapshots_created_at;
DROP INDEX IF EXISTS idx_onextree_snapshots_expires_at;
DROP INDEX IF EXISTS idx_onextree_snapshots_tree_structure;
DROP INDEX IF EXISTS idx_onextree_snapshots_enrichment_data;
DROP INDEX IF EXISTS idx_onextree_snapshots_project_created;

DROP TABLE IF EXISTS onextree_snapshots;
```

### Storage Overhead Analysis

**Typical Tree Size:**
- 1,000 files: ~500 KB JSONB
- 10,000 files: ~5 MB JSONB
- 100,000 files: ~50 MB JSONB

**Retention Strategy:**
- Keep last 10 versions per project
- Expire snapshots older than 30 days
- Auto-cleanup runs daily via cron

**Estimated Storage for 100 Projects:**
- Average 5,000 files per project
- ~2.5 MB per snapshot
- 10 versions × 100 projects = 2.5 GB total

---

## 4. Generation Workflow

### End-to-End Tree Generation Flow

```
┌───────────────────────────────────────────────────────────────┐
│                   Tree Generation Workflow                     │
└───────────────────────────────────────────────────────────────┘

Step 1: Trigger Detection
─────────────────────────
    [Git Commit] OR [API Call] OR [Scheduled Task]
                    │
                    v
    ┌─────────────────────────────────┐
    │ Publish: TreeGenerationRequested│
    │ Topic: onextree.generation.req │
    │ Payload: {                      │
    │   project_id,                   │
    │   project_root,                 │
    │   trigger_type,                 │
    │   correlation_id                │
    │ }                               │
    └────────────┬────────────────────┘
                 │
                 v

Step 2: Tree Generation (OnexTree Service)
───────────────────────────────────────────
    ┌──────────────────────────────┐
    │ POST /generate                │
    │ {                            │
    │   "project_root": "/path"    │
    │ }                            │
    └────────────┬─────────────────┘
                 │
                 v
    ┌──────────────────────────────┐
    │ OnexTreeGenerator.generate() │
    │ - Scan filesystem             │
    │ - Build tree structure        │
    │ - Calculate statistics        │
    │ - Compute tree_hash           │
    └────────────┬─────────────────┘
                 │ (1-3s for 10k files)
                 v
    ┌──────────────────────────────┐
    │ Response: {                  │
    │   success: true,              │
    │   total_files: 8432,          │
    │   total_directories: 1203,    │
    │   total_size_mb: 145.3,       │
    │   generation_time_ms: 1850,   │
    │   tree_hash: "abc123..."      │
    │ }                             │
    └────────────┬─────────────────┘
                 │
                 v

Step 3: Store Tree (PostgreSQL)
────────────────────────────────
    ┌──────────────────────────────┐
    │ Query: Get latest snapshot   │
    │ WHERE project_id = ?          │
    └────────────┬─────────────────┘
                 │
                 v
    ┌──────────────────────────────┐
    │ Check tree_hash changed?     │
    │ - Same hash? → Skip insert   │
    │ - Different? → Insert new    │
    └────────────┬─────────────────┘
                 │ Changed
                 v
    ┌──────────────────────────────┐
    │ INSERT INTO onextree_snapshots│
    │ - tree_structure (JSONB)      │
    │ - tree_hash (SHA256)          │
    │ - version = prev_version + 1  │
    │ - parent_snapshot_id = prev.id│
    │ - expires_at = NOW() + 30d    │
    └────────────┬─────────────────┘
                 │ (<100ms)
                 v

Step 4: Enrich Tree (Optional)
───────────────────────────────
    ┌──────────────────────────────┐
    │ POST /enrich                 │
    │ {                            │
    │   "project_root": "/path",   │
    │   "pattern_sources": [       │
    │     "archon"                 │
    │   ]                          │
    │ }                            │
    └────────────┬─────────────────┘
                 │
                 v
    ┌──────────────────────────────┐
    │ Query Archon MCP for patterns│
    │ - Architectural patterns      │
    │ - Code quality insights       │
    │ - Related file relationships  │
    └────────────┬─────────────────┘
                 │ (<1s)
                 v
    ┌──────────────────────────────┐
    │ UPDATE onextree_snapshots    │
    │ SET enrichment_data = ?,      │
    │     is_enriched = true,       │
    │     enrichment_time_ms = ?    │
    │ WHERE id = ?                  │
    └────────────┬─────────────────┘
                 │
                 v

Step 5: Publish Events
──────────────────────
    ┌──────────────────────────────┐
    │ Publish: TreeGenerationCompleted│
    │ Topic: onextree.generation.completed│
    │ Payload: {                    │
    │   snapshot_id,                │
    │   project_id,                 │
    │   tree_hash,                  │
    │   total_files,                │
    │   generation_time_ms,         │
    │   correlation_id              │
    │ }                             │
    └────────────┬─────────────────┘
                 │
                 v
    ┌──────────────────────────────┐
    │ Update metrics dashboard      │
    │ - Tree generation rate        │
    │ - Average generation time     │
    │ - Storage growth              │
    └───────────────────────────────┘
```

### Request/Response Models

```python
# src/omninode_bridge/models/onextree_models.py

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

class TreeGenerationRequest(BaseModel):
    """Request to generate tree."""
    project_id: str = Field(..., description="Unique project identifier")
    project_root: str = Field(..., description="Absolute path to project root")
    trigger_type: str = Field(..., description="Trigger: manual, scheduled, event")
    correlation_id: UUID = Field(..., description="Request correlation ID")

class TreeGenerationResponse(BaseModel):
    """Tree generation result."""
    success: bool
    snapshot_id: UUID
    tree_hash: str
    total_files: int
    total_directories: int
    total_size_mb: float
    generation_time_ms: float
    correlation_id: UUID

class TreeEnrichmentRequest(BaseModel):
    """Request to enrich tree with intelligence."""
    snapshot_id: UUID
    pattern_sources: list[str] = Field(default=["archon"])
    correlation_id: UUID

class TreeEnrichmentResponse(BaseModel):
    """Tree enrichment result."""
    success: bool
    patterns_added: int
    enrichment_time_ms: float
    intelligence_sources: list[str]
    correlation_id: UUID
```

---

## 5. Enrichment Strategy

### When to Enrich

**Two-Phase Enrichment:**

1. **Immediate Basic Enrichment** (During generation)
   - File type classification
   - Directory purpose inference
   - Basic pattern detection
   - **Time:** <500ms
   - **Coverage:** 100% of files

2. **Deferred Advanced Enrichment** (Async after generation)
   - Archon pattern analysis
   - Cross-file relationship mapping
   - Architectural pattern detection
   - **Time:** <3s
   - **Coverage:** Key files only (entry points, configs, main modules)

**Enrichment Triggers:**

```
┌─────────────────────────────────────────────────────────────┐
│              Enrichment Decision Matrix                      │
├─────────────────────────────────────────────────────────────┤
│ Condition                    │ Action                       │
├──────────────────────────────┼──────────────────────────────┤
│ Tree just generated          │ Queue async enrichment       │
│ Tree age < 24h               │ Skip (use cached)            │
│ Tree age 24h-7d              │ Enrich if accessed           │
│ Tree age > 7d                │ Regenerate + enrich          │
│ User requests fresh intel    │ Force enrichment             │
│ Archon MCP unavailable       │ Use basic enrichment only    │
└─────────────────────────────────────────────────────────────┘
```

### Intelligence Sources

**Primary: Archon MCP Service**

```python
# src/omninode_bridge/intelligence/archon_intelligence.py

from typing import Any, Optional
from uuid import UUID

class ArchonIntelligenceClient:
    """Client for Archon MCP intelligence queries."""

    async def get_architectural_patterns(
        self,
        project_root: str,
        file_paths: list[str],
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Query Archon for architectural patterns.

        Returns:
            {
                "patterns": [
                    {
                        "name": "Repository Pattern",
                        "confidence": 0.92,
                        "files": ["src/repositories/user_repo.py"],
                        "description": "Data access abstraction layer"
                    }
                ],
                "relationships": [
                    {
                        "source": "api/routes.py",
                        "target": "services/user_service.py",
                        "type": "dependency"
                    }
                ],
                "quality_insights": {
                    "onex_compliance": 0.85,
                    "anti_patterns": ["god_object"]
                }
            }
        """
        # Use existing Archon MCP tools:
        # - perform_rag_query for pattern analysis
        # - assess_code_quality for ONEX compliance
        # - search_code_examples for related patterns
        pass

    async def get_file_relationships(
        self,
        project_root: str,
        target_file: str,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, list[str]]:
        """
        Get related files for target file.

        Returns:
            {
                "imports": ["module1.py", "module2.py"],
                "imported_by": ["caller1.py", "caller2.py"],
                "similar_files": ["sibling1.py"]
            }
        """
        pass
```

### Enrichment Payload Format

```json
{
  "enrichment_data": {
    "version": "1.0",
    "enriched_at": "2025-10-24T10:30:00Z",
    "sources": ["archon", "local"],

    "architectural_patterns": [
      {
        "name": "Repository Pattern",
        "confidence": 0.92,
        "affected_files": ["src/repositories/*.py"],
        "description": "Data access abstraction layer"
      }
    ],

    "file_relationships": {
      "src/api/routes.py": {
        "imports": ["src/services/user_service.py"],
        "imported_by": ["src/main.py"],
        "similar_files": ["src/api/auth_routes.py"]
      }
    },

    "quality_insights": {
      "onex_compliance_score": 0.85,
      "anti_patterns_detected": ["god_object", "circular_dependency"],
      "recommendations": [
        "Split UserService into smaller services",
        "Resolve circular dependency between module A and B"
      ]
    },

    "statistics": {
      "total_files_analyzed": 8432,
      "patterns_detected": 12,
      "relationships_mapped": 1520
    }
  }
}
```

### Caching Strategy

**Three-Level Cache:**

1. **In-Memory (AsyncOnexTreeClient):**
   - TTL: 5 minutes
   - Size: 1000 entries (LRU eviction)
   - Use: Hot path queries

2. **Database (onextree_snapshots):**
   - TTL: 30 days (expires_at)
   - Size: Last 10 versions per project
   - Use: Recent snapshots

3. **OnexTree Service:**
   - TTL: Until tree regeneration
   - Size: Single active tree in memory
   - Use: Fast lookups during service uptime

**Cache Invalidation:**
- Tree regenerated → Invalidate all caches
- Project files modified → Invalidate in-memory cache
- Manual flush → API endpoint `/cache/clear`

---

## 6. Event Schema

### Kafka Events for Tree Operations

```python
# src/omninode_bridge/models/events.py

from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field

class TreeEventType(str, Enum):
    """OnexTree event types."""
    GENERATION_REQUESTED = "generation_requested"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    ENRICHMENT_STARTED = "enrichment_started"
    ENRICHMENT_COMPLETED = "enrichment_completed"
    ENRICHMENT_FAILED = "enrichment_failed"

class TreeGenerationRequestedEvent(BaseEvent):
    """Event: Tree generation requested."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: Literal[TreeEventType.GENERATION_REQUESTED] = TreeEventType.GENERATION_REQUESTED

    # Payload fields
    project_id: str
    project_root: str
    trigger_type: str  # manual, scheduled, event
    requested_by: Optional[str] = None

    def to_kafka_topic(self) -> str:
        return "dev.omninode_bridge.onex.evt.tree-generation-requested.v1"

class TreeGenerationStartedEvent(BaseEvent):
    """Event: Tree generation started."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: Literal[TreeEventType.GENERATION_STARTED] = TreeEventType.GENERATION_STARTED

    project_id: str
    project_root: str
    started_at: datetime

    def to_kafka_topic(self) -> str:
        return "dev.omninode_bridge.onex.evt.tree-generation-started.v1"

class TreeGenerationCompletedEvent(BaseEvent):
    """Event: Tree generation completed successfully."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: Literal[TreeEventType.GENERATION_COMPLETED] = TreeEventType.GENERATION_COMPLETED

    snapshot_id: UUID
    project_id: str
    tree_hash: str
    total_files: int
    total_directories: int
    total_size_mb: float
    generation_time_ms: float

    def to_kafka_topic(self) -> str:
        return "dev.omninode_bridge.onex.evt.tree-generation-completed.v1"

class TreeGenerationFailedEvent(BaseEvent):
    """Event: Tree generation failed."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: Literal[TreeEventType.GENERATION_FAILED] = TreeEventType.GENERATION_FAILED

    project_id: str
    project_root: str
    error_message: str
    error_code: Optional[str] = None
    retry_count: int = 0

    def to_kafka_topic(self) -> str:
        return "dev.omninode_bridge.onex.evt.tree-generation-failed.v1"

class TreeEnrichmentCompletedEvent(BaseEvent):
    """Event: Tree enrichment completed."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: Literal[TreeEventType.ENRICHMENT_COMPLETED] = TreeEventType.ENRICHMENT_COMPLETED

    snapshot_id: UUID
    patterns_added: int
    enrichment_time_ms: float
    intelligence_sources: list[str]

    def to_kafka_topic(self) -> str:
        return "dev.omninode_bridge.onex.evt.tree-enrichment-completed.v1"
```

### Event Publishing Integration

```python
# src/omninode_bridge/services/onextree_event_publisher.py

from uuid import UUID
from typing import Optional
from .kafka_client import KafkaClient
from ..models.events import (
    TreeGenerationRequestedEvent,
    TreeGenerationCompletedEvent,
    TreeGenerationFailedEvent,
    TreeEnrichmentCompletedEvent,
)

class OnexTreeEventPublisher:
    """Publishes OnexTree lifecycle events to Kafka."""

    def __init__(self, kafka_client: KafkaClient):
        self.kafka_client = kafka_client

    async def publish_generation_requested(
        self,
        project_id: str,
        project_root: str,
        trigger_type: str,
        correlation_id: UUID,
    ) -> None:
        """Publish tree generation requested event."""
        event = TreeGenerationRequestedEvent(
            service="onextree_service",
            correlation_id=correlation_id,
            project_id=project_id,
            project_root=project_root,
            trigger_type=trigger_type,
        )

        await self.kafka_client.publish_event(
            topic=event.to_kafka_topic(),
            event=event,
        )

    async def publish_generation_completed(
        self,
        snapshot_id: UUID,
        project_id: str,
        tree_hash: str,
        total_files: int,
        total_directories: int,
        total_size_mb: float,
        generation_time_ms: float,
        correlation_id: UUID,
    ) -> None:
        """Publish tree generation completed event."""
        event = TreeGenerationCompletedEvent(
            service="onextree_service",
            correlation_id=correlation_id,
            snapshot_id=snapshot_id,
            project_id=project_id,
            tree_hash=tree_hash,
            total_files=total_files,
            total_directories=total_directories,
            total_size_mb=total_size_mb,
            generation_time_ms=generation_time_ms,
        )

        await self.kafka_client.publish_event(
            topic=event.to_kafka_topic(),
            event=event,
        )

    # Similar methods for other events...
```

---

## 7. API Endpoints

### FastAPI Endpoints for Tree Access

```python
# src/omninode_bridge/api/routes/onextree.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from uuid import UUID
from typing import Optional
from ...models.onextree_models import (
    TreeGenerationRequest,
    TreeGenerationResponse,
    TreeEnrichmentRequest,
    TreeEnrichmentResponse,
)
from ...services.onextree_service import OnexTreeService
from ...services.metadata_stamping.models.responses import UnifiedResponse

router = APIRouter(prefix="/tree", tags=["OnexTree"])

# ==================== Tree Generation ====================

@router.post("/generate", response_model=UnifiedResponse)
async def generate_tree(
    request: TreeGenerationRequest,
    background_tasks: BackgroundTasks,
    service: OnexTreeService = Depends(),
) -> UnifiedResponse:
    """
    Generate OnexTree for project.

    **Performance:** <3s for 10k files
    **Trigger:** Manual, scheduled, or event-driven

    Args:
        request: Generation request with project_id and project_root
        background_tasks: FastAPI background tasks for async enrichment

    Returns:
        {
            "status": "success",
            "data": {
                "snapshot_id": "uuid",
                "tree_hash": "sha256",
                "total_files": 8432,
                "generation_time_ms": 1850
            }
        }
    """
    try:
        # Generate tree synchronously
        result = await service.generate_tree(
            project_id=request.project_id,
            project_root=request.project_root,
            correlation_id=request.correlation_id,
        )

        # Enrich tree asynchronously in background
        background_tasks.add_task(
            service.enrich_tree,
            snapshot_id=result["snapshot_id"],
            pattern_sources=["archon"],
            correlation_id=request.correlation_id,
        )

        return UnifiedResponse(
            status="success",
            data=result,
            metadata={
                "correlation_id": str(request.correlation_id),
                "enrichment": "queued_for_background_processing",
            },
        )

    except Exception as e:
        return UnifiedResponse(
            status="error",
            error=str(e),
            metadata={"correlation_id": str(request.correlation_id)},
        )

# ==================== Tree Retrieval ====================

@router.get("/{project_id}", response_model=UnifiedResponse)
async def get_tree(
    project_id: str,
    version: Optional[int] = None,
    include_enrichment: bool = True,
    service: OnexTreeService = Depends(),
) -> UnifiedResponse:
    """
    Get tree for project.

    **Performance:** <100ms (from PostgreSQL cache)

    Args:
        project_id: Project identifier
        version: Specific version (default: latest)
        include_enrichment: Include intelligence data

    Returns:
        {
            "status": "success",
            "data": {
                "snapshot_id": "uuid",
                "tree_structure": {...},
                "enrichment_data": {...},
                "metadata": {
                    "total_files": 8432,
                    "created_at": "2025-10-24T10:30:00Z",
                    "is_enriched": true
                }
            }
        }
    """
    try:
        snapshot = await service.get_tree(
            project_id=project_id,
            version=version,
            include_enrichment=include_enrichment,
        )

        if not snapshot:
            return UnifiedResponse(
                status="error",
                error="Tree not found for project",
                metadata={"project_id": project_id},
            )

        return UnifiedResponse(
            status="success",
            data=snapshot,
        )

    except Exception as e:
        return UnifiedResponse(
            status="error",
            error=str(e),
            metadata={"project_id": project_id},
        )

# ==================== Tree Enrichment ====================

@router.post("/{project_id}/enrich", response_model=UnifiedResponse)
async def enrich_tree(
    project_id: str,
    pattern_sources: list[str] = ["archon"],
    service: OnexTreeService = Depends(),
) -> UnifiedResponse:
    """
    Enrich existing tree with intelligence.

    **Performance:** <3s for full enrichment

    Args:
        project_id: Project identifier
        pattern_sources: Intelligence sources (archon, local)

    Returns:
        {
            "status": "success",
            "data": {
                "patterns_added": 12,
                "enrichment_time_ms": 2450,
                "intelligence_sources": ["archon"]
            }
        }
    """
    try:
        # Get latest snapshot
        snapshot = await service.get_tree(project_id=project_id)
        if not snapshot:
            raise ValueError("Tree not found")

        # Enrich synchronously
        result = await service.enrich_tree(
            snapshot_id=snapshot["snapshot_id"],
            pattern_sources=pattern_sources,
            correlation_id=uuid4(),
        )

        return UnifiedResponse(
            status="success",
            data=result,
        )

    except Exception as e:
        return UnifiedResponse(
            status="error",
            error=str(e),
            metadata={"project_id": project_id},
        )

# ==================== Tree Statistics ====================

@router.get("/{project_id}/stats", response_model=UnifiedResponse)
async def get_tree_stats(
    project_id: str,
    service: OnexTreeService = Depends(),
) -> UnifiedResponse:
    """
    Get tree statistics.

    **Performance:** <50ms

    Returns:
        {
            "status": "success",
            "data": {
                "total_files": 8432,
                "total_directories": 1203,
                "total_size_mb": 145.3,
                "last_generated": "2025-10-24T10:30:00Z",
                "is_enriched": true,
                "version": 5
            }
        }
    """
    try:
        stats = await service.get_tree_stats(project_id=project_id)

        if not stats:
            return UnifiedResponse(
                status="error",
                error="Tree not found",
                metadata={"project_id": project_id},
            )

        return UnifiedResponse(
            status="success",
            data=stats,
        )

    except Exception as e:
        return UnifiedResponse(
            status="error",
            error=str(e),
        )

# ==================== Cache Management ====================

@router.post("/cache/clear", response_model=UnifiedResponse)
async def clear_cache(
    project_id: Optional[str] = None,
    service: OnexTreeService = Depends(),
) -> UnifiedResponse:
    """
    Clear OnexTree cache.

    Args:
        project_id: Clear specific project (default: all)

    Returns:
        {
            "status": "success",
            "data": {
                "cleared_entries": 150,
                "scope": "all" | "project"
            }
        }
    """
    try:
        cleared = await service.clear_cache(project_id=project_id)

        return UnifiedResponse(
            status="success",
            data={
                "cleared_entries": cleared,
                "scope": "project" if project_id else "all",
            },
        )

    except Exception as e:
        return UnifiedResponse(
            status="error",
            error=str(e),
        )
```

---

## 8. Incremental Updates

### Incremental Update Strategy

**Goal:** Update only changed files instead of full regeneration

**Change Detection:**

```python
# src/omninode_bridge/services/onextree_change_detector.py

from typing import Optional
import hashlib
from pathlib import Path

class OnexTreeChangeDetector:
    """Detect file changes for incremental updates."""

    async def detect_changes(
        self,
        project_root: str,
        previous_tree_hash: str,
        modified_files: list[str],
    ) -> dict[str, list[str]]:
        """
        Detect changes between current state and previous snapshot.

        Args:
            project_root: Project root path
            previous_tree_hash: Hash of previous tree
            modified_files: List of modified file paths (from git)

        Returns:
            {
                "added": ["new_file.py"],
                "modified": ["existing_file.py"],
                "deleted": ["removed_file.py"],
                "renamed": [{"old": "old.py", "new": "new.py"}]
            }
        """
        # Implementation:
        # 1. Load previous tree structure from database
        # 2. Scan modified_files and categorize changes
        # 3. Detect renames (same content hash, different path)
        # 4. Return structured change summary
        pass

    async def apply_incremental_update(
        self,
        snapshot_id: UUID,
        changes: dict[str, list[str]],
    ) -> dict[str, Any]:
        """
        Apply incremental changes to existing tree.

        Args:
            snapshot_id: ID of snapshot to update
            changes: Change summary from detect_changes()

        Returns:
            {
                "updated_snapshot_id": UUID,
                "files_updated": int,
                "update_time_ms": float,
                "tree_hash": str  # New hash after update
            }
        """
        # Implementation:
        # 1. Load tree structure from database
        # 2. Apply changes (add/modify/delete nodes)
        # 3. Recalculate tree_hash
        # 4. If hash unchanged, skip insert
        # 5. If hash changed, insert new version
        pass
```

### Git Hook Integration

**Post-Commit Hook:**

```bash
#!/bin/bash
# .git/hooks/post-commit

# Get list of changed files
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

# Get project root
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Publish Kafka event for incremental update
curl -X POST http://localhost:8053/tree/incremental-update \
  -H "Content-Type: application/json" \
  -d "{
    \"project_root\": \"$PROJECT_ROOT\",
    \"modified_files\": $(echo $CHANGED_FILES | jq -R -s -c 'split(\"\n\")'),
    \"trigger_type\": \"git_commit\"
  }"
```

**Debouncing Logic:**

```python
# src/omninode_bridge/services/onextree_debouncer.py

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable

class OnexTreeUpdateDebouncer:
    """Debounce rapid tree update requests."""

    def __init__(self, debounce_window_seconds: int = 5):
        self.debounce_window = timedelta(seconds=debounce_window_seconds)
        self.pending_updates: defaultdict[str, list[str]] = defaultdict(list)
        self.last_trigger: dict[str, datetime] = {}

    async def queue_update(
        self,
        project_id: str,
        modified_files: list[str],
        update_callback: Callable,
    ) -> None:
        """
        Queue incremental update with debouncing.

        Args:
            project_id: Project identifier
            modified_files: Files changed in this commit
            update_callback: Function to call when debounce window expires
        """
        # Add files to pending queue
        self.pending_updates[project_id].extend(modified_files)
        self.last_trigger[project_id] = datetime.now()

        # Wait for debounce window
        await asyncio.sleep(self.debounce_window.total_seconds())

        # Check if more updates came in during wait
        if (datetime.now() - self.last_trigger[project_id]) >= self.debounce_window:
            # No new updates, process batched changes
            files_to_update = self.pending_updates.pop(project_id, [])
            if files_to_update:
                await update_callback(project_id, files_to_update)
```

### Performance Optimization

**Full vs Incremental Update Decision:**

```
┌─────────────────────────────────────────────────────────┐
│         Full Regeneration vs Incremental Update         │
├─────────────────────────────────────────────────────────┤
│ Condition                  │ Action                     │
├────────────────────────────┼────────────────────────────┤
│ < 10 files changed         │ Incremental update         │
│ 10-100 files changed       │ Incremental update         │
│ > 100 files changed        │ Full regeneration          │
│ Directory structure change │ Full regeneration          │
│ .gitignore modified        │ Full regeneration          │
│ First-time generation      │ Full regeneration          │
└─────────────────────────────────────────────────────────┘
```

**Expected Performance:**
- Incremental update (<10 files): <200ms
- Incremental update (10-100 files): <1s
- Full regeneration (10k files): <3s

---

## 9. Circuit Breaker Configuration

### Circuit Breaker Thresholds

```python
# src/omninode_bridge/clients/circuit_breaker_config.py

from .circuit_breaker import CircuitBreakerConfig

# OnexTree Service Circuit Breaker
ONEXTREE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    # Failure threshold
    failure_threshold=3,         # Open circuit after 3 consecutive failures
    success_threshold=2,         # Close circuit after 2 consecutive successes

    # Timeout configuration
    timeout=5.0,                 # 5s timeout for tree generation
    recovery_timeout=30,         # 30s before attempting recovery

    # Half-open state
    half_open_max_calls=1,       # Test with 1 call when half-open

    # Monitoring
    window_size=100,             # Track last 100 calls
    failure_rate_threshold=0.5,  # Open at 50% failure rate
)

# Apply to AsyncOnexTreeClient
class AsyncOnexTreeClient(BaseServiceClient):
    def __init__(self, ...):
        super().__init__(
            base_url=base_url,
            service_name="OnexTreeService",
            circuit_breaker_config=ONEXTREE_CIRCUIT_BREAKER_CONFIG,
        )
```

### Fallback Behavior

**When OnexTree Service Unavailable:**

```python
# src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py

class NodeBridgeOrchestrator:
    async def execute_orchestration(self, contract):
        """Execute stamping workflow with OnexTree fallback."""

        # Try to get OnexTree intelligence
        try:
            intelligence = await self.onextree_client.get_intelligence(
                context=contract.file_path,
                correlation_id=contract.correlation_id,
            )
        except (ClientError, ServiceUnavailableError) as e:
            logger.warning(
                f"OnexTree unavailable, falling back to basic stamping: {e}",
                extra={"correlation_id": str(contract.correlation_id)},
            )

            # FALLBACK: Use cached tree from database
            intelligence = await self._get_cached_tree_intelligence(
                project_id=contract.metadata.get("project_id"),
            )

            if not intelligence:
                # FALLBACK: Generate basic metadata without tree
                intelligence = {
                    "exists_in_tree": False,
                    "status": "onextree_unavailable",
                    "fallback_mode": "basic_stamping",
                }

        # Continue with stamping workflow
        stamp_result = await self._create_stamp(
            file_path=contract.file_path,
            intelligence=intelligence,
            correlation_id=contract.correlation_id,
        )

        return stamp_result

    async def _get_cached_tree_intelligence(
        self,
        project_id: str,
    ) -> Optional[dict]:
        """Get cached tree intelligence from database."""
        # Query onextree_snapshots for latest snapshot
        # Extract basic intelligence even if enrichment incomplete
        # Return partial intelligence for degraded mode
        pass
```

### Monitoring and Alerts

**Circuit Breaker Metrics:**

```python
# Prometheus metrics for circuit breaker state

from prometheus_client import Gauge, Counter

onextree_circuit_state = Gauge(
    "onextree_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
)

onextree_circuit_failures = Counter(
    "onextree_circuit_breaker_failures_total",
    "Total failures detected by circuit breaker",
)

onextree_fallback_invocations = Counter(
    "onextree_fallback_invocations_total",
    "Total fallback invocations when OnexTree unavailable",
)
```

**Alert Rules:**

```yaml
# Prometheus alert rules

groups:
  - name: onextree_alerts
    interval: 30s
    rules:
      - alert: OnexTreeCircuitBreakerOpen
        expr: onextree_circuit_breaker_state == 2
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "OnexTree circuit breaker is OPEN"
          description: "OnexTree service is unavailable. Stamping workflow using fallback mode."

      - alert: OnexTreeHighFailureRate
        expr: rate(onextree_circuit_breaker_failures_total[5m]) > 0.5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "OnexTree service high failure rate"
          description: "OnexTree failures exceed 50% over 5 minutes."

      - alert: OnexTreeFallbackModeActive
        expr: rate(onextree_fallback_invocations_total[5m]) > 10
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "OnexTree fallback mode active"
          description: "Stamping workflows using fallback mode frequently."
```

---

## 10. Performance Targets

### Detailed Performance Requirements

```
┌────────────────────────────────────────────────────────────────┐
│                 OnexTree Performance Targets                    │
├────────────────────────────────────────────────────────────────┤
│ Operation                    │ Target    │ P99      │ Notes    │
├──────────────────────────────┼───────────┼──────────┼──────────┤
│ Tree Generation (1k files)   │ <500ms    │ <750ms   │ Small    │
│ Tree Generation (10k files)  │ <2s       │ <3s      │ Medium   │
│ Tree Generation (100k files) │ <10s      │ <15s     │ Large    │
│                              │           │          │          │
│ Tree Query (file lookup)     │ <5ms      │ <10ms    │ Memory   │
│ Tree Query (directory scan)  │ <50ms     │ <100ms   │ JSONB    │
│ Tree Query (pattern search)  │ <200ms    │ <500ms   │ Complex  │
│                              │           │          │          │
│ Enrichment (basic)           │ <500ms    │ <1s      │ Local    │
│ Enrichment (Archon patterns) │ <3s       │ <5s      │ Remote   │
│                              │           │          │          │
│ Storage (INSERT snapshot)    │ <100ms    │ <200ms   │ Postgres │
│ Storage (UPDATE enrichment)  │ <50ms     │ <100ms   │ JSONB    │
│                              │           │          │          │
│ Incremental Update (<10)     │ <200ms    │ <500ms   │ Fast     │
│ Incremental Update (10-100)  │ <1s       │ <2s      │ Medium   │
│                              │           │          │          │
│ Cache Hit (in-memory)        │ <1ms      │ <5ms     │ Dict     │
│ Cache Hit (PostgreSQL)       │ <50ms     │ <100ms   │ JSONB    │
└────────────────────────────────────────────────────────────────┘
```

### Load Testing Requirements

**Concurrent Tree Generations:**
- Target: 10 concurrent generations
- Expected: <5s per generation
- Total throughput: ~120 trees/minute

**Concurrent Queries:**
- Target: 1000 req/s
- Expected: <50ms average latency
- Success rate: >99%

**Storage Scalability:**
- 100 projects × 10 versions = 1000 snapshots
- Average size: 2.5 MB per snapshot
- Total storage: ~2.5 GB
- Query performance: <100ms (JSONB indexes)

### Benchmarking Script

```python
# tests/performance/test_onextree_performance.py

import asyncio
import time
from uuid import uuid4

async def benchmark_tree_generation():
    """Benchmark tree generation performance."""
    client = AsyncOnexTreeClient()

    # Test small project (1k files)
    start = time.monotonic()
    result = await client.generate_tree(
        project_root="/path/to/small_project",
        correlation_id=uuid4(),
    )
    duration_ms = (time.monotonic() - start) * 1000

    assert duration_ms < 500, f"Small project generation took {duration_ms}ms (target: <500ms)"

    # Test medium project (10k files)
    start = time.monotonic()
    result = await client.generate_tree(
        project_root="/path/to/medium_project",
        correlation_id=uuid4(),
    )
    duration_ms = (time.monotonic() - start) * 1000

    assert duration_ms < 2000, f"Medium project generation took {duration_ms}ms (target: <2s)"

async def benchmark_concurrent_queries():
    """Benchmark concurrent query performance."""
    client = AsyncOnexTreeClient()

    # 1000 concurrent queries
    tasks = [
        client.query_knowledge(
            query=f"find files matching pattern_{i}",
            correlation_id=uuid4(),
        )
        for i in range(1000)
    ]

    start = time.monotonic()
    results = await asyncio.gather(*tasks)
    duration_s = time.monotonic() - start

    throughput = len(tasks) / duration_s
    assert throughput > 1000, f"Query throughput {throughput:.0f} req/s (target: >1000 req/s)"
```

---

## 11. Error Handling

### Error Categories and Responses

```python
# src/omninode_bridge/services/onextree_error_handler.py

from enum import Enum
from typing import Optional

class OnexTreeErrorCode(str, Enum):
    """OnexTree error codes."""
    # Tree Generation Errors
    TREE_GENERATION_FAILED = "TREE_GENERATION_FAILED"
    PROJECT_ROOT_NOT_FOUND = "PROJECT_ROOT_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TREE_TOO_LARGE = "TREE_TOO_LARGE"

    # Storage Errors
    DATABASE_ERROR = "DATABASE_ERROR"
    TREE_NOT_FOUND = "TREE_NOT_FOUND"
    DUPLICATE_TREE_HASH = "DUPLICATE_TREE_HASH"

    # Service Errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    TIMEOUT = "TIMEOUT"

    # Enrichment Errors
    ENRICHMENT_FAILED = "ENRICHMENT_FAILED"
    ARCHON_UNAVAILABLE = "ARCHON_UNAVAILABLE"

class OnexTreeError(Exception):
    """Base exception for OnexTree operations."""

    def __init__(
        self,
        message: str,
        error_code: OnexTreeErrorCode,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class OnexTreeErrorHandler:
    """Handle OnexTree errors with graceful degradation."""

    async def handle_generation_error(
        self,
        error: Exception,
        project_id: str,
        correlation_id: UUID,
    ) -> dict[str, Any]:
        """
        Handle tree generation errors.

        Returns fallback response with degraded mode info.
        """
        if isinstance(error, FileNotFoundError):
            # Project root doesn't exist
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.PROJECT_ROOT_NOT_FOUND,
                "message": "Project root directory not found",
                "fallback": "use_empty_tree",
                "correlation_id": str(correlation_id),
            }

        elif isinstance(error, PermissionError):
            # Permission denied
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.PERMISSION_DENIED,
                "message": "Insufficient permissions to read project files",
                "fallback": "skip_protected_files",
                "correlation_id": str(correlation_id),
            }

        elif isinstance(error, MemoryError):
            # Project too large
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.TREE_TOO_LARGE,
                "message": "Project exceeds maximum tree size",
                "fallback": "use_sampling_strategy",
                "correlation_id": str(correlation_id),
            }

        else:
            # Generic failure
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.TREE_GENERATION_FAILED,
                "message": str(error),
                "fallback": "retry_with_backoff",
                "correlation_id": str(correlation_id),
            }

    async def handle_enrichment_error(
        self,
        error: Exception,
        snapshot_id: UUID,
    ) -> dict[str, Any]:
        """
        Handle enrichment errors.

        Returns partial enrichment or basic tree.
        """
        if "Archon" in str(error):
            # Archon MCP unavailable
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.ARCHON_UNAVAILABLE,
                "message": "Archon MCP service unavailable",
                "fallback": "use_basic_enrichment",
                "partial_enrichment": True,
            }

        else:
            # Generic enrichment failure
            return {
                "success": False,
                "error_code": OnexTreeErrorCode.ENRICHMENT_FAILED,
                "message": str(error),
                "fallback": "skip_enrichment",
                "basic_tree_available": True,
            }
```

### Stale Tree Detection

**Problem:** Tree becomes outdated after repository changes

**Solution:**

```python
# src/omninode_bridge/services/onextree_freshness_checker.py

from datetime import datetime, timedelta, UTC

class OnexTreeFreshnessChecker:
    """Check if tree is fresh enough for use."""

    MAX_TREE_AGE_HOURS = 24

    async def is_tree_fresh(
        self,
        snapshot: dict,
        project_root: str,
    ) -> dict[str, Any]:
        """
        Check if tree is fresh enough.

        Returns:
            {
                "is_fresh": bool,
                "age_hours": float,
                "recommendation": str
            }
        """
        created_at = datetime.fromisoformat(snapshot["created_at"])
        age = datetime.now(UTC) - created_at
        age_hours = age.total_seconds() / 3600

        if age_hours < self.MAX_TREE_AGE_HOURS:
            return {
                "is_fresh": True,
                "age_hours": age_hours,
                "recommendation": "use_existing_tree",
            }
        else:
            return {
                "is_fresh": False,
                "age_hours": age_hours,
                "recommendation": "regenerate_tree",
                "reason": f"tree_age_{age_hours:.1f}h_exceeds_limit_{self.MAX_TREE_AGE_HOURS}h",
            }
```

### Retry Strategy

**Exponential Backoff with Jitter:**

```python
# Built into BaseServiceClient via tenacity

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

retry_policy = AsyncRetrying(
    stop=stop_after_attempt(3),              # Max 3 retries
    wait=wait_exponential(
        multiplier=1,                        # Base delay: 1s
        min=1,                               # Min delay: 1s
        max=10,                              # Max delay: 10s
    ),
    retry=retry_if_exception_type((
        TimeoutError,
        ConnectionError,
        ServiceUnavailableError,
    )),
    reraise=True,
)
```

**Retry Timeline:**
1. Initial attempt: Fails
2. Wait 1s → Retry 1: Fails
3. Wait 2s → Retry 2: Fails
4. Wait 4s → Retry 3: Fails
5. Total time: ~7s before giving up

---

## 12. Integration Steps

### Implementation Roadmap

**Phase 1: Foundation (Week 1)**

```
✅ Task 1.1: Database Schema
   - Create migration 012_create_onextree_snapshots.sql
   - Run migration on dev database
   - Verify indexes and constraints

✅ Task 1.2: HTTP Client Enhancement
   - Add generate_tree() method to AsyncOnexTreeClient
   - Add enrich_tree() method
   - Add get_tree_stats() method
   - Update circuit breaker config

✅ Task 1.3: Event Models
   - Add TreeEventType enum
   - Add TreeGeneration*Event classes
   - Add TreeEnrichment*Event classes
   - Update to_kafka_topic() methods
```

**Phase 2: Core Integration (Week 2)**

```
✅ Task 2.1: OnexTreeService
   - Create OnexTreeService class
   - Implement generate_tree() with database persistence
   - Implement enrich_tree() with Archon MCP integration
   - Implement get_tree() with caching

✅ Task 2.2: Event Publisher
   - Create OnexTreeEventPublisher
   - Wire into KafkaClient
   - Test event publishing end-to-end

✅ Task 2.3: API Endpoints
   - Add /tree/generate endpoint
   - Add /tree/{project_id} endpoint
   - Add /tree/{project_id}/enrich endpoint
   - Add /tree/{project_id}/stats endpoint
   - Add /tree/cache/clear endpoint
```

**Phase 3: Workflow Integration (Week 3)**

```
✅ Task 3.1: Orchestrator Integration
   - Update NodeBridgeOrchestrator.execute_orchestration()
   - Add tree freshness check before stamping
   - Add fallback behavior for OnexTree unavailable
   - Add async tree generation trigger

✅ Task 3.2: Incremental Updates
   - Create OnexTreeChangeDetector
   - Create OnexTreeUpdateDebouncer
   - Add /tree/incremental-update endpoint
   - Add git post-commit hook example

✅ Task 3.3: Enrichment Integration
   - Create ArchonIntelligenceClient
   - Implement get_architectural_patterns()
   - Implement get_file_relationships()
   - Wire into OnexTreeService.enrich_tree()
```

**Phase 4: Testing & Optimization (Week 4)**

```
✅ Task 4.1: Unit Tests
   - Test AsyncOnexTreeClient methods
   - Test OnexTreeService CRUD operations
   - Test OnexTreeEventPublisher
   - Test error handling and fallbacks

✅ Task 4.2: Integration Tests
   - Test end-to-end tree generation
   - Test enrichment workflow
   - Test stamping with OnexTree integration
   - Test circuit breaker behavior

✅ Task 4.3: Performance Tests
   - Benchmark tree generation (1k, 10k, 100k files)
   - Benchmark concurrent queries (1000 req/s)
   - Benchmark storage overhead
   - Optimize JSONB queries

✅ Task 4.4: Load Tests
   - Test 10 concurrent tree generations
   - Test database under load (1000 snapshots)
   - Test cache eviction behavior
   - Verify circuit breaker under failure
```

**Phase 5: Production Readiness (Week 5)**

```
✅ Task 5.1: Monitoring
   - Add Prometheus metrics for tree operations
   - Add circuit breaker state metrics
   - Add alert rules for OnexTree failures
   - Dashboard for tree generation stats

✅ Task 5.2: Documentation
   - Update ARCHITECTURE.md with OnexTree integration
   - Add OnexTree section to API_REFERENCE.md
   - Create ONEXTREE_OPERATIONS.md runbook
   - Add examples to CLIENT_INTEGRATION_GUIDE.md

✅ Task 5.3: Deployment
   - Update docker-compose.yml with OnexTree service
   - Add environment variables for configuration
   - Test in staging environment
   - Create rollback plan

✅ Task 5.4: Training
   - Document common operations (generate, query, enrich)
   - Document troubleshooting (circuit breaker, stale trees)
   - Document performance tuning
   - Share with team
```

### Quick Start Commands

```bash
# 1. Apply database migration
psql -h localhost -U postgres -d omninode_bridge -f migrations/012_create_onextree_snapshots.sql

# 2. Start OnexTree service
docker compose up -d omninode-bridge-onextree

# 3. Generate tree for project
curl -X POST http://localhost:8053/tree/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "omninode_bridge",
    "project_root": "/Volumes/PRO-G40/Code/omninode_bridge",
    "trigger_type": "manual",
    "correlation_id": "'"$(uuidgen)"'"
  }'

# 4. Get tree for project
curl http://localhost:8053/tree/omninode_bridge

# 5. Enrich tree with Archon patterns
curl -X POST http://localhost:8053/tree/omninode_bridge/enrich \
  -H "Content-Type: application/json" \
  -d '{"pattern_sources": ["archon"]}'

# 6. Monitor circuit breaker
curl http://localhost:8053/metrics | grep onextree_circuit
```

---

## Summary

This design provides:

✅ **Automated Tree Generation** - Event-driven, scheduled, and on-demand triggers
✅ **Efficient Storage** - PostgreSQL with JSONB indexes, 30-day retention
✅ **Intelligence Enrichment** - Archon MCP integration for pattern analysis
✅ **Graceful Degradation** - Circuit breaker with fallback to cached trees
✅ **Performance Targets** - Sub-3s generation, <500ms queries, <100ms storage
✅ **Incremental Updates** - Change detection and partial tree updates
✅ **Comprehensive Events** - 7 Kafka event types for observability
✅ **Production Ready** - Monitoring, alerts, error handling, and rollback plans

**Next Steps:**
1. Review design with team
2. Approve implementation roadmap
3. Begin Phase 1: Foundation
4. Iterate based on performance testing

**Questions for Review:**
- Tree retention: 30 days appropriate?
- Enrichment timing: Async background vs synchronous?
- Circuit breaker thresholds: Adjust based on SLAs?
- Incremental update triggers: Git hooks vs API?
