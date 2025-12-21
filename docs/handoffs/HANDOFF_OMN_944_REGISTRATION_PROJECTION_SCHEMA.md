# Handoff: OMN-944 - F1: Implement Registration Projection Schema

**Created**: 2025-12-21
**Ticket**: OMN-944
**Branch**: `jonah/omn-944-f1-implement-registration-projection-schema`
**Status**: IN PROGRESS - Core components implemented, tests pending

---

## Executive Summary

This ticket implements the PostgreSQL schema and infrastructure for registration projections, enabling orchestrators to query materialized registration state for workflow decisions. This is a key component of the F0 projector execution model.

**What's Done**:
- ✅ EnumRegistrationState FSM enum
- ✅ ModelSequenceInfo for ordering
- ✅ ModelRegistrationProjection Pydantic model
- ✅ SQL schema with indexes
- ✅ ProjectorRegistration implementation
- ✅ ProjectionReaderRegistration implementation

**What's Remaining**:
- ⏳ Unit tests for all components
- ⏳ Integration tests with testcontainers
- ⏳ Update models/__init__.py exports
- ⏳ Run linting and type checks
- ⏳ Create PR

---

## Files Created

### Enums

| File | Purpose |
|------|---------|
| `src/omnibase_infra/enums/enum_registration_state.py` | FSM state enum with 8 states and helper methods |

**FSM States**:
- `PENDING_REGISTRATION` → Initial state
- `ACCEPTED` → Orchestrator accepted
- `AWAITING_ACK` → Waiting for node ack
- `REJECTED` → Terminal (rejected)
- `ACK_TIMED_OUT` → Retriable
- `ACK_RECEIVED` → Ack received
- `ACTIVE` → Fully operational
- `LIVENESS_EXPIRED` → Terminal (dead)

### Models

| File | Purpose |
|------|---------|
| `src/omnibase_infra/models/projection/__init__.py` | Package exports |
| `src/omnibase_infra/models/projection/model_sequence_info.py` | Sequence info for ordering/idempotency |
| `src/omnibase_infra/models/projection/model_registration_projection.py` | Main projection Pydantic model |

**Key Fields in ModelRegistrationProjection**:
- Identity: `entity_id`, `domain`
- State: `current_state` (EnumRegistrationState)
- Deadlines: `ack_deadline`, `liveness_deadline`
- Timeout markers: `ack_timeout_emitted_at`, `liveness_timeout_emitted_at`
- Idempotency: `last_applied_event_id`, `last_applied_offset`, `last_applied_sequence`

### SQL Schema

| File | Purpose |
|------|---------|
| `src/omnibase_infra/schemas/schema_registration_projection.sql` | DDL for registration_projections table |

**Indexes Created**:
1. `idx_registration_ack_deadline` - Partial index for ack deadline scanning
2. `idx_registration_liveness_deadline` - Partial index for liveness scanning
3. `idx_registration_current_state` - State filtering
4. `idx_registration_domain_state` - Domain + state queries
5. `idx_registration_last_event_id` - Idempotency checks
6. `idx_registration_capabilities` - GIN index on JSONB
7. `idx_registration_ack_timeout_scan` - Composite for C2 timeout scans
8. `idx_registration_liveness_timeout_scan` - Composite for C2 liveness scans

### Projectors

| File | Purpose |
|------|---------|
| `src/omnibase_infra/projectors/__init__.py` | Package exports |
| `src/omnibase_infra/projectors/projector_registration.py` | Projection persistence with ordering |
| `src/omnibase_infra/projectors/projection_reader_registration.py` | Projection queries for orchestrators |

**ProjectorRegistration Methods**:
- `initialize_schema()` - Create table/indexes
- `persist()` - Upsert with sequence validation
- `is_stale()` - Check if sequence is outdated

**ProjectionReaderRegistration Methods**:
- `get_entity_state()` - Point lookup
- `get_registration_status()` - Get current FSM state
- `get_by_state()` - Query by state
- `get_overdue_ack_registrations()` - C2 timeout scan
- `get_overdue_liveness_registrations()` - C2 liveness scan
- `count_by_state()` - Aggregation for metrics

---

## Files Modified

| File | Change |
|------|--------|
| `src/omnibase_infra/enums/__init__.py` | Added EnumRegistrationState export |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary Key | `(entity_id, domain)` | Multi-domain support; registration is default |
| Sequence Storage | Both offset and sequence | Kafka uses offset; generic transports use sequence |
| Schema Init | Explicit `initialize_schema()` | Ops control; idempotent DDL |
| DB Enum | PostgreSQL ENUM type | Type safety at DB level |
| Capabilities | JSONB with GIN index | Flexible schema for evolving capabilities |
| Circuit Breaker | 5 failures, 60s reset | Standard resilience pattern |

---

## Remaining Work

### 1. Unit Tests (Priority: High)

Create `tests/unit/projectors/` directory with:

```python
# tests/unit/projectors/__init__.py
# tests/unit/projectors/test_enum_registration_state.py
# tests/unit/projectors/test_model_sequence_info.py
# tests/unit/projectors/test_model_registration_projection.py
# tests/unit/projectors/test_projector_registration.py
# tests/unit/projectors/test_projection_reader_registration.py
```

**Test Scenarios**:
- Enum state transitions and helper methods
- ModelSequenceInfo staleness comparison
- ModelRegistrationProjection validation and helpers
- Projector persist with mock DB (stale rejection, idempotency)
- Reader queries with mock DB

### 2. Integration Tests (Priority: Medium)

Create `tests/integration/projectors/test_projector_registration_integration.py`:
- Use testcontainers for real PostgreSQL
- Test schema creation
- Test upsert ordering
- Test deadline queries

### 3. Package Exports (Priority: High)

Update `src/omnibase_infra/models/__init__.py` to export projection subpackage.

### 4. Validation (Priority: High)

```bash
# Run linting
poetry run ruff check src/omnibase_infra/enums/enum_registration_state.py
poetry run ruff check src/omnibase_infra/models/projection/
poetry run ruff check src/omnibase_infra/projectors/
poetry run ruff check src/omnibase_infra/schemas/

# Run type checks
poetry run mypy src/omnibase_infra/enums/enum_registration_state.py
poetry run mypy src/omnibase_infra/models/projection/
poetry run mypy src/omnibase_infra/projectors/

# Run tests
poetry run pytest tests/unit/projectors/ -v
```

---

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| OMN-940 (F0) | ✅ Done | Projector execution model defined in omnibase_spi |
| omnibase_spi ^0.4.0 | ✅ Available | Contains ProtocolProjector, ProtocolProjectionReader |
| asyncpg ^0.29.0 | ✅ Available | PostgreSQL async driver |
| testcontainers ^4.9.0 | ✅ Available | For integration tests |

---

## Architecture Context

```
Events → Handler → Runtime → Reducer → Projector → PostgreSQL
                                                        ↓
                                              ProjectionReader ← Orchestrator
```

**Key Invariants**:
1. Projections are persisted to storage, NOT published to Kafka
2. Per-entity monotonic ordering via `(partition, offset)` or `sequence`
3. Stale updates rejected with WHERE clause in upsert
4. Orchestrators read projections for state decisions (never scan Kafka)

---

## Related Tickets

| Ticket | Title | Relationship |
|--------|-------|--------------|
| OMN-940 (F0) | Projector Execution Model | Dependency (Done) |
| OMN-932 (C2) | Durable Timeout Handling | Uses deadline fields |
| OMN-930 (C0) | Projection Reader | Protocol definition |
| OMN-888 (C1) | Registration Orchestrator | Consumer of projections |
| OMN-947 (F2) | Snapshot Publishing | Future extension |

---

## Testing Checklist

- [ ] EnumRegistrationState helper methods
- [ ] ModelSequenceInfo staleness comparison
- [ ] ModelRegistrationProjection validation
- [ ] ModelRegistrationProjection.needs_ack_timeout_event()
- [ ] ModelRegistrationProjection.needs_liveness_timeout_event()
- [ ] ProjectorRegistration.persist() - success case
- [ ] ProjectorRegistration.persist() - stale rejection
- [ ] ProjectorRegistration.is_stale()
- [ ] ProjectorRegistration.initialize_schema()
- [ ] ProjectionReaderRegistration.get_entity_state()
- [ ] ProjectionReaderRegistration.get_overdue_ack_registrations()
- [ ] ProjectionReaderRegistration.get_overdue_liveness_registrations()
- [ ] Integration test with real PostgreSQL

---

## Quick Start for Next Developer

```bash
# Switch to branch
git checkout jonah/omn-944-f1-implement-registration-projection-schema

# Install dependencies
poetry install

# Run existing tests to verify no regressions
poetry run pytest tests/ -x

# Create test directory
mkdir -p tests/unit/projectors

# Run linting on new files
poetry run ruff check src/omnibase_infra/enums/enum_registration_state.py
poetry run ruff check src/omnibase_infra/models/projection/
poetry run ruff check src/omnibase_infra/projectors/
```

---

## Notes

- The SQL schema uses PostgreSQL ENUM type for registration_state, which must match EnumRegistrationState values exactly
- Projector uses atomic upsert with sequence comparison to reject stale updates
- Both ProjectorRegistration and ProjectionReaderRegistration use MixinAsyncCircuitBreaker for resilience
- All queries use parameterized statements for SQL injection protection
