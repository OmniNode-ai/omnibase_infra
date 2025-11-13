# Workstream 1B Completion Summary

**Agent ID**: poly-foundation-1b
**Duration**: ~2 hours
**Status**: ✅ COMPLETE
**Date**: 2025-10-21

## Deliverables

### 1. Migration File ✅
**File**: `migrations/011_projection_and_watermarks.sql`

- Created `workflow_projection` table with 8 columns
- Created `projection_watermarks` table with 3 columns
- Added 5 indexes for workflow_projection (namespace, tag, combined, version, GIN for JSONB)
- Added comprehensive comments for documentation
- **Note**: Used quoted "offset" column name to avoid PostgreSQL reserved keyword conflict

**Rollback File**: `migrations/011_rollback_projection_and_watermarks.sql`

### 2. Pydantic Models ✅

**File**: `src/omninode_bridge/infrastructure/entities/model_workflow_projection.py`
- ModelWorkflowProjection with 8 fields
- ONEX v2.0 compliant (suffix-based naming)
- Pydantic v2 with strict validation
- JSONB field validation for indices and extras

**File**: `src/omninode_bridge/infrastructure/entities/model_projection_watermark.py`
- ModelProjectionWatermark with 3 fields
- Partition-based progress tracking
- Offset validation (>= 0)
- ONEX v2.0 compliant

### 3. Unit Tests ✅

**File**: `tests/unit/infrastructure/entities/test_projection_models.py`
- 24 comprehensive tests (13 for projection, 11 for watermark)
- All tests passing (100%)
- Coverage includes:
  - Model instantiation (all fields, required only)
  - Validation (missing fields, invalid values, boundaries)
  - Serialization/deserialization
  - JSON roundtrip
  - Boundary conditions (BIGINT max for version and offset)
  - Watermark lag calculation pattern
  - Multi-partition pattern

## Acceptance Criteria

- ✅ Migration runs successfully
- ✅ Both Pydantic models validate correctly
- ✅ All unit tests pass (24/24)
- ✅ Watermark offset increments correctly (0 → BIGINT max tested)
- ✅ Rollback migration works correctly
- ✅ Database operations verified (INSERT, UPDATE with GREATEST)

## Database Schema

### workflow_projection
```sql
workflow_key (TEXT, PK)
version (BIGINT, NOT NULL)
tag (TEXT, NOT NULL)
last_action (TEXT, NULL)
namespace (TEXT, NOT NULL)
updated_at (TIMESTAMPTZ, NOT NULL, DEFAULT NOW())
indices (JSONB, NULL)
extras (JSONB, NULL)
```

**Indexes**:
- idx_projection_namespace (namespace)
- idx_projection_tag (tag)
- idx_projection_namespace_tag (namespace, tag)
- idx_projection_version (version DESC)
- idx_projection_indices_gin (indices USING GIN)

### projection_watermarks
```sql
partition_id (TEXT, PK)
"offset" (BIGINT, NOT NULL, DEFAULT 0)
updated_at (TIMESTAMPTZ, NOT NULL, DEFAULT NOW())
```

**Constraints**:
- watermark_offset_non_negative CHECK ("offset" >= 0)

## Test Results

```bash
$ poetry run pytest tests/unit/infrastructure/entities/test_projection_models.py -v --no-cov
======================== 24 passed, 1 warning in 0.32s =========================

Tests:
✅ test_model_instantiation_with_all_fields (x2)
✅ test_model_instantiation_with_required_fields_only (x2)
✅ test_model_validation_missing_required_fields (x2)
✅ test_model_validation_invalid_workflow_key
✅ test_model_validation_invalid_version
✅ test_model_validation_invalid_tag
✅ test_model_validation_invalid_namespace
✅ test_model_validation_invalid_last_action
✅ test_model_validation_invalid_partition_id
✅ test_model_validation_invalid_offset
✅ test_model_serialization (x2)
✅ test_model_deserialization (x2)
✅ test_model_roundtrip (x2)
✅ test_model_jsonb_validation
✅ test_model_version_boundary_conditions
✅ test_model_offset_boundary_conditions
✅ test_watermark_lag_calculation
✅ test_multiple_partitions
```

## Implementation Notes

### Reserved Keyword Handling
PostgreSQL treats `offset` as a reserved keyword. Solution: Quoted column name `"offset"` in:
- Table definition
- CHECK constraint
- Comments
- All SQL queries must use quoted identifier

### Eventual Consistency Pattern
The schema supports the eventual consistency pattern described in the refactor plan:
1. Canonical store writes to `workflow_state` (Workstream 1A)
2. Projection materializer reads from Kafka `StateCommitted` events
3. Writes to `workflow_projection` atomically with watermark update
4. Projection store can gate reads by version or fallback to canonical

### JSONB Optimization
- `indices` field: Custom query indexes for future optimization (e.g., `{"priority": "high"}`)
- `extras` field: Projection-specific metadata (e.g., `{"retry_count": 3}`)
- GIN index on `indices` for fast JSONB queries (WHERE clause filters empty objects)

### Performance Characteristics
- `workflow_projection` table: Optimized for reads (5 indexes)
- `projection_watermarks` table: Minimal (primary key only)
- Watermark updates use `GREATEST()` for idempotent offset advancement
- Combined namespace+tag index for common query patterns

## Next Steps (Wave 2)

This workstream provides the schema foundation for:
- **Workstream 2B**: ProjectionStoreService (get_state with version gating)
- **Workstream 2C**: ProjectionMaterializer (subscribe to StateCommitted, update projection+watermark)

## Files Modified

- `migrations/011_projection_and_watermarks.sql` (NEW)
- `migrations/011_rollback_projection_and_watermarks.sql` (NEW)
- `src/omninode_bridge/infrastructure/entities/model_workflow_projection.py` (NEW)
- `src/omninode_bridge/infrastructure/entities/model_projection_watermark.py` (NEW)
- `tests/unit/infrastructure/entities/test_projection_models.py` (NEW)

**Total Lines of Code**: ~800 lines (migration: 110, models: 150, tests: 540)

## Dependencies

None - This workstream has no dependencies and can be integrated immediately.

## Migration Path

1. Apply migration: `psql -f migrations/011_projection_and_watermarks.sql`
2. Rollback: `psql -f migrations/011_rollback_projection_and_watermarks.sql`
3. Re-apply: `psql -f migrations/011_projection_and_watermarks.sql`

All operations tested and verified on PostgreSQL 12+.
