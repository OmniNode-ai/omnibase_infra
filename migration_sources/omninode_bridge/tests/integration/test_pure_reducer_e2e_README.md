# Pure Reducer E2E Integration Tests - Setup Guide

## Overview

This directory contains comprehensive end-to-end integration tests for the Pure Reducer architecture (Wave 6A).

## Test Coverage

The integration test suite (`test_pure_reducer_e2e.py`) validates:

1. **Happy Path** - Complete workflow from action to projection
2. **Conflict Resolution** - Concurrent actions with optimistic concurrency control
3. **Action Deduplication** - Duplicate action handling with idempotency
4. **Projection Lag** - Fallback to canonical store when projection lags
5. **Reducer Gave Up** - Max retries exceeded escalation
6. **Performance Validation** - Performance targets validation
7. **Edge Cases** - Empty payload, invalid workflow keys, cleanup

## Prerequisites

### 1. Install omnibase_core Dependencies

The tests require `omnibase_core` which is not installed by default. Install it:

```bash
# Option 1: Install from PyPI (if available)
pip install omnibase-core

# Option 2: Install from local clone
git clone https://github.com/omnibase/omnibase-core
cd omnibase-core
pip install -e .
```

### 2. Database Setup

Ensure PostgreSQL is running with the test database:

```bash
# Start PostgreSQL (port 5436 for test database)
docker compose -f deployment/docker-compose.yml up -d postgres

# Run migrations
alembic upgrade head
```

### 3. Kafka Setup (Optional)

Kafka is mocked in tests but can be enabled for full integration:

```bash
# Start Kafka/Redpanda
docker compose -f deployment/docker-compose.yml up -d redpanda
```

## Running Tests

### Run All Integration Tests

```bash
# Run all Pure Reducer E2E tests
pytest tests/integration/test_pure_reducer_e2e.py -v

# Run with coverage
pytest tests/integration/test_pure_reducer_e2e.py --cov=omninode_bridge.services --cov-report=html
```

### Run Specific Test Scenarios

```bash
# Happy path test
pytest tests/integration/test_pure_reducer_e2e.py::test_happy_path_action_to_projection -v

# Conflict resolution test
pytest tests/integration/test_pure_reducer_e2e.py::test_concurrent_actions_conflict_resolution -v

# Deduplication test
pytest tests/integration/test_pure_reducer_e2e.py::test_action_deduplication -v

# Projection lag test
pytest tests/integration/test_pure_reducer_e2e.py::test_projection_lag_fallback -v

# Reducer gave up test
pytest tests/integration/test_pure_reducer_e2e.py::test_reducer_gave_up_escalation -v

# Performance validation
pytest tests/integration/test_pure_reducer_e2e.py::test_reducer_service_performance_targets -v
```

## Test Fixtures

### Database Fixtures

- `postgres_client` - PostgreSQL client connected to test database
- `setup_test_workflow` - Helper to create initial workflow state
- `cleanup_test_data` - Automatic cleanup of test data after each test

### Service Fixtures

- `canonical_store` - CanonicalStoreService for version-controlled state
- `projection_store` - ProjectionStoreService for read-optimized projections
- `action_dedup` - ActionDedupService for idempotent action processing
- `kafka_client` - Mocked Kafka client for event publishing
- `mock_reducer` - Mock reducer for testing without ONEX dependencies
- `reducer_service` - Complete ReducerService with all dependencies

## Performance Targets

The tests validate these performance targets:

| Metric | Target | Test Environment Target |
|--------|--------|------------------------|
| Deduplication check | <5ms | <10ms |
| State read | <5ms | <10ms |
| Reducer invocation | <50ms | <100ms |
| Commit attempt | <10ms | <20ms |
| Total latency (no conflicts) | <70ms p95 | <140ms p95 |
| Average latency | <50ms | <100ms |

**Note**: Test environment targets are 2x production targets to account for overhead.

## Troubleshooting

### ModuleNotFoundError: No module named 'omnibase_core'

**Cause**: The `omnibase_core` package is not installed.

**Solution**:
```bash
# Install omnibase_core from local clone
git clone https://github.com/omnibase/omnibase-core
cd omnibase-core
pip install -e .
```

### Database Connection Errors

**Cause**: PostgreSQL is not running or migrations not applied.

**Solution**:
```bash
# Start PostgreSQL
docker compose -f deployment/docker-compose.yml up -d postgres

# Run migrations
alembic upgrade head

# Verify connection
psql -h localhost -p 5436 -U postgres -d omninode_bridge_test -c "SELECT 1"
```

### Table Does Not Exist Errors

**Cause**: Migrations for Pure Reducer tables not applied.

**Solution**:
```bash
# Check which migrations are applied
alembic current

# Apply missing migrations
alembic upgrade head

# Verify tables exist
psql -h localhost -p 5436 -U postgres -d omninode_bridge_test -c "\dt"
```

Expected tables:
- `workflow_state` - Canonical workflow state
- `workflow_projection` - Read-optimized projections
- `projection_watermarks` - Projection lag tracking
- `action_dedup_log` - Action deduplication records

### Performance Test Failures

**Cause**: Test environment overhead or database not optimized.

**Solution**:
1. Ensure PostgreSQL has adequate resources
2. Run tests on dedicated test environment
3. Adjust performance targets in test if needed (documented as test environment targets)

### Cleanup Errors

**Cause**: Test data not properly cleaned up between runs.

**Solution**:
```bash
# Manual cleanup
psql -h localhost -p 5436 -U postgres -d omninode_bridge_test <<EOF
DELETE FROM workflow_state WHERE workflow_key LIKE 'test-workflow-%';
DELETE FROM workflow_projection WHERE workflow_key LIKE 'test-workflow-%';
DELETE FROM action_dedup_log WHERE workflow_key LIKE 'test-workflow-%';
EOF
```

## Test Scenarios

### 1. Happy Path (test_happy_path_action_to_projection)

**Workflow**:
1. Create initial workflow state (version 1)
2. Create action with payload
3. Process action through ReducerService
4. Verify state committed to canonical store (version 2)
5. Verify action recorded in dedup log
6. Verify metrics updated correctly

**Expected Results**:
- Version incremented from 1 → 2
- Action processing latency <100ms
- Successful action count = 1
- Duplicate action count = 0

### 2. Conflict Resolution (test_concurrent_actions_conflict_resolution)

**Workflow**:
1. Create initial workflow state
2. Launch 10 concurrent actions on same workflow_key
3. ReducerService handles conflicts with retry + backoff
4. Verify eventual consistency (all actions processed)
5. Verify conflict metrics tracked

**Expected Results**:
- Final version = 1 + num_actions
- All actions successful
- Conflict attempts > 0
- Avg conflicts per action tracked

### 3. Action Deduplication (test_action_deduplication)

**Workflow**:
1. Process action first time
2. Process same action_id again (duplicate)
3. Verify second attempt skipped
4. Verify state not modified
5. Verify dedup metrics updated

**Expected Results**:
- Version remains unchanged (duplicate skipped)
- Dedup hit rate = 50%
- Successful actions = 1
- Duplicate actions skipped = 1

### 4. Projection Lag (test_projection_lag_fallback)

**Workflow**:
1. Create workflow state in canonical store
2. Query projection before materialization
3. ProjectionStore falls back to canonical
4. Verify fallback worked correctly
5. Verify metrics tracked

**Expected Results**:
- Projection version matches canonical
- Fallback count = 1
- Fallback rate = 100%

### 5. Reducer Gave Up (test_reducer_gave_up_escalation)

**Workflow**:
1. Mock canonical_store.try_commit to always return conflict
2. Process action (will exhaust retries)
3. Verify exactly 3 retry attempts
4. Verify ReducerGaveUp event published
5. Verify failure metrics updated

**Expected Results**:
- Failed actions = 1
- Conflict attempts = 3 (max_attempts)
- ReducerGaveUp event published to Kafka

### 6. Performance Validation (test_reducer_service_performance_targets)

**Workflow**:
1. Process 100 actions sequentially
2. Measure latency for each action
3. Calculate p95 and average latency
4. Verify against targets

**Expected Results**:
- P95 latency < 140ms (test environment)
- Average latency < 100ms
- Successful actions = 100
- Failed actions = 0

## Coverage Targets

Target: **>90% coverage** for critical paths

Critical paths include:
- ReducerService.handle_action
- CanonicalStoreService.try_commit
- ActionDedupService.should_process
- ProjectionStoreService.get_state
- Conflict resolution retry loop
- Event publishing flows

Run coverage report:
```bash
pytest tests/integration/test_pure_reducer_e2e.py \
  --cov=omninode_bridge.services.reducer_service \
  --cov=omninode_bridge.services.canonical_store \
  --cov=omninode_bridge.services.action_dedup \
  --cov=omninode_bridge.services.projection_store \
  --cov-report=html \
  --cov-report=term-missing
```

## Success Criteria

✅ All 15-20 integration tests passing
✅ Test coverage >90% for critical paths
✅ All 5 test scenarios validated (happy path, conflicts, dedup, lag, gave up)
✅ Performance targets met (with test environment adjustments)
✅ Test execution time <5 minutes total

## Documentation

For complete Pure Reducer architecture details, see:
- `docs/planning/PURE_REDUCER_REFACTOR_PLAN.md` - Overall architecture plan
- `src/omninode_bridge/services/reducer_service.py` - ReducerService implementation
- `src/omninode_bridge/services/canonical_store.py` - CanonicalStoreService implementation
- `src/omninode_bridge/services/projection_store.py` - ProjectionStoreService implementation
- `src/omninode_bridge/services/action_dedup.py` - ActionDedupService implementation

## Contact

For questions or issues with integration tests, contact the OmniNode Bridge team or file an issue in the repository.
