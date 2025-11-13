# State Recovery Mechanisms Design

**Document Version:** 1.0
**Created:** 2025-10-15
**Status:** Implementation Ready

## Overview

This document outlines the design for state recovery mechanisms in the Bridge Nodes (Orchestrator and Reducer) to handle crash recovery, restart scenarios, and partial workflow execution.

## Goals

1. **Crash Recovery**: Automatically recover workflows after node crashes
2. **State Consistency**: Maintain consistent state across restarts
3. **Idempotency**: Support safe re-execution of partial workflows
4. **Performance**: Recover 1000+ workflows in <5 seconds
5. **Observability**: Track recovery metrics and stuck workflow detection

## Database Schema

### Existing Tables

1. **workflow_executions**: Primary workflow tracking
   - `id`, `correlation_id`, `workflow_type`, `current_state`
   - `namespace`, `started_at`, `completed_at`, `execution_time_ms`
   - `error_message`, `metadata`

2. **workflow_steps**: Step-level execution tracking
   - `workflow_id` (FK to workflow_executions)
   - `step_name`, `step_order`, `status`, `execution_time_ms`
   - `step_data`, `error_message`

3. **fsm_transitions**: State transition history
   - `entity_id`, `entity_type`, `from_state`, `to_state`
   - `transition_event`, `transition_data`

4. **bridge_states**: Aggregation state tracking
   - `bridge_id`, `namespace`, `total_workflows_processed`
   - `total_items_aggregated`, `aggregation_metadata`
   - `current_fsm_state`, `last_aggregation_timestamp`

## Recovery Mechanisms

### 1. Orchestrator Recovery (`NodeBridgeOrchestrator`)

#### Recovery Process

**Startup Flow:**
```
1. Query workflow_executions for in-progress workflows
   WHERE current_state IN ('PENDING', 'PROCESSING')
   AND updated_at > NOW() - INTERVAL '24 hours'

2. For each workflow:
   a. Load workflow context from metadata
   b. Query workflow_steps to find last completed step
   c. Check if workflow is stuck (timeout detection)
   d. Resume from next step or mark as FAILED

3. Restore FSM state cache:
   - workflow_fsm_states dict
   - workflow_correlation_ids dict

4. Publish recovery metrics to Kafka
```

**Stuck Workflow Detection:**
- Workflows in PROCESSING state for >1 hour
- Workflows with no step updates for >30 minutes
- Action: Transition to FAILED state, log error

#### Idempotent Operations

**Hash Generation:**
- Check if file_hash already exists in workflow context
- Skip MetadataStampingService call if hash present
- Resume from next step

**Stamp Creation:**
- Check if stamp_id exists in workflow metadata
- Skip stamp creation if already completed
- Resume from next step

**OnexTree Intelligence:**
- Check intelligence_data in workflow metadata
- Use cached intelligence if available
- Skip OnexTree call for faster recovery

#### Recovery Methods

```python
class NodeBridgeOrchestrator:
    async def recover_workflows(self) -> dict[str, int]:
        """
        Recover in-progress workflows on startup.

        Returns:
            Recovery statistics:
            - recovered: Successfully recovered workflows
            - failed: Failed recovery attempts
            - stuck: Stuck workflows transitioned to FAILED
            - total: Total workflows processed
        """
```

**PostgreSQL Queries:**

```sql
-- Find in-progress workflows
SELECT * FROM workflow_executions
WHERE current_state IN ('PENDING', 'PROCESSING')
  AND updated_at > NOW() - INTERVAL '24 hours'
ORDER BY started_at ASC;

-- Find last completed step
SELECT * FROM workflow_steps
WHERE workflow_id = $1
  AND status = 'COMPLETED'
ORDER BY step_order DESC
LIMIT 1;

-- Check for stuck workflows
SELECT * FROM workflow_executions
WHERE current_state = 'PROCESSING'
  AND updated_at < NOW() - INTERVAL '1 hour';
```

### 2. Reducer Recovery (`NodeBridgeReducer`)

#### Recovery Process

**Startup Flow:**
```
1. Query bridge_states for active namespaces

2. Restore FSM state manager cache:
   - Load workflow states from workflow_executions
   - Rebuild transition history from fsm_transitions
   - Restore aggregation buffer from bridge_states

3. Query for incomplete aggregations:
   WHERE current_fsm_state = 'AGGREGATING'
   AND last_aggregation_timestamp < NOW() - INTERVAL '10 minutes'

4. Resume aggregations from last checkpoint

5. Publish recovery metrics
```

**Aggregation Checkpoint Strategy:**
- Save aggregation state every 1000 items
- Store namespace-level aggregations in bridge_states
- Use UPSERT for incremental updates

#### Recovery Methods

```python
class NodeBridgeReducer:
    async def recover_aggregations(self) -> dict[str, int]:
        """
        Recover incomplete aggregations on startup.

        Returns:
            Recovery statistics:
            - recovered_namespaces: Namespaces with restored state
            - recovered_workflows: Workflows with restored FSM state
            - failed: Failed recovery attempts
            - total: Total recovery operations
        """
```

**PostgreSQL Queries:**

```sql
-- Find active bridge states
SELECT * FROM bridge_states
WHERE current_fsm_state IN ('AGGREGATING', 'PROCESSING')
  AND last_aggregation_timestamp > NOW() - INTERVAL '24 hours';

-- Restore workflow FSM states
SELECT we.correlation_id, we.current_state, we.metadata
FROM workflow_executions we
WHERE we.current_state IN ('PENDING', 'PROCESSING', 'COMPLETED')
  AND we.updated_at > NOW() - INTERVAL '24 hours';

-- Restore FSM transition history
SELECT * FROM fsm_transitions
WHERE entity_id = $1
  AND entity_type = 'workflow_execution'
ORDER BY created_at ASC;
```

### 3. FSM State Manager Recovery

The `FSMStateManager` class (in reducer) handles FSM state recovery:

```python
async def recover_states(self) -> dict[str, int]:
    """
    Recover FSM states from PostgreSQL on startup.

    Steps:
    1. Query workflow_executions for active workflows
    2. Query fsm_transitions for transition history
    3. Rebuild _state_cache dict
    4. Rebuild _transition_history dict
    5. Validate state consistency

    Returns:
        Recovery statistics
    """
```

## Persistence Layer

### PostgreSQL Client Integration

**Connection Manager:**
- Use asyncpg connection pool (existing in container)
- Circuit breaker pattern for resilience
- Retry logic with exponential backoff

**Transaction Management:**
```python
async with container.get_service('postgresql_client') as db:
    async with db.transaction():
        # Persist workflow execution
        await db.upsert_workflow_execution(...)

        # Persist workflow steps
        await db.insert_workflow_step(...)

        # Record FSM transition
        await db.insert_fsm_transition(...)
```

### CRUD Operations

**Workflow Execution:**
- `create_workflow_execution(workflow_data)` → UUID
- `get_workflow_execution(correlation_id)` → WorkflowExecution
- `update_workflow_state(correlation_id, state, metadata)` → bool
- `list_in_progress_workflows(time_window)` → List[WorkflowExecution]

**Workflow Steps:**
- `insert_workflow_step(workflow_id, step_data)` → UUID
- `get_workflow_steps(workflow_id)` → List[WorkflowStep]
- `get_last_completed_step(workflow_id)` → Optional[WorkflowStep]

**FSM Transitions:**
- `insert_fsm_transition(entity_id, from_state, to_state, event)` → UUID
- `get_transition_history(entity_id, entity_type)` → List[FSMTransition]

**Bridge States:**
- `upsert_bridge_state(namespace, aggregation_data)` → UUID
- `get_bridge_state(namespace)` → Optional[BridgeState]
- `list_active_bridge_states()` → List[BridgeState]

## Timeout and Stuck Workflow Detection

### Timeout Configuration

```python
RECOVERY_TIMEOUTS = {
    'workflow_stuck_threshold_minutes': 60,  # 1 hour
    'step_timeout_minutes': 30,              # 30 minutes
    'aggregation_timeout_minutes': 10,       # 10 minutes
    'recovery_query_timeout_seconds': 30,    # Query timeout
}
```

### Stuck Workflow Detection

**Conditions for "Stuck" Classification:**
1. Workflow in PROCESSING state for >1 hour
2. No step updates for >30 minutes
3. Workflow started >24 hours ago and incomplete

**Actions on Stuck Detection:**
1. Transition workflow to FAILED state
2. Log error with correlation_id and context
3. Publish `WORKFLOW_STUCK` event to Kafka
4. Increment stuck_workflow_count metric

## Recovery Metrics

### Metrics Collection

```python
recovery_metrics = {
    'orchestrator': {
        'total_workflows_recovered': int,
        'successful_recoveries': int,
        'failed_recoveries': int,
        'stuck_workflows_failed': int,
        'recovery_duration_ms': float,
    },
    'reducer': {
        'total_namespaces_recovered': int,
        'total_workflows_recovered': int,
        'aggregations_resumed': int,
        'recovery_duration_ms': float,
    }
}
```

### Observability

**Logging:**
- Log recovery start/completion at INFO level
- Log each workflow recovery at DEBUG level
- Log recovery failures at ERROR level

**Kafka Events:**
- `RECOVERY_STARTED` event on startup
- `WORKFLOW_RECOVERED` event per workflow
- `RECOVERY_COMPLETED` event with metrics
- `WORKFLOW_STUCK` event for stuck workflows

## Performance Requirements

### Recovery Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Query in-progress workflows | <500ms | p95 query time |
| Recover single workflow | <100ms | p95 recovery time |
| Recover 1000 workflows | <5 seconds | Total recovery time |
| FSM state restore | <1 second | Total restore time |
| Bridge state restore | <500ms | Total restore time |

### Optimization Strategies

1. **Batch Queries**: Fetch workflows in batches of 100
2. **Parallel Recovery**: Recover workflows concurrently (limit: 50)
3. **Lazy Loading**: Load step history only when needed
4. **Connection Pooling**: Reuse database connections
5. **Query Optimization**: Use prepared statements

## Error Handling

### Recovery Failure Scenarios

**Scenario 1: Database Connection Failure**
- Action: Retry with exponential backoff (3 attempts)
- Fallback: Start in degraded mode, skip recovery
- Log: ERROR with connection details

**Scenario 2: Corrupted Workflow State**
- Action: Mark workflow as FAILED, log error
- Fallback: Continue recovery for other workflows
- Log: WARNING with workflow_id and state

**Scenario 3: Missing Workflow Steps**
- Action: Restart workflow from beginning
- Fallback: Mark workflow as FAILED if restart impossible
- Log: INFO with recovery strategy

**Scenario 4: Recovery Timeout**
- Action: Transition to FAILED after timeout
- Fallback: Schedule retry in background task
- Log: WARNING with timeout duration

## Testing Strategy

### Unit Tests

1. **Test Recovery Query Construction**
   - Verify SQL queries for in-progress workflows
   - Test timeout threshold calculations
   - Validate query parameter binding

2. **Test State Restoration**
   - Mock database responses
   - Verify FSM state cache rebuilding
   - Test workflow context restoration

3. **Test Idempotent Operations**
   - Verify skip logic for completed steps
   - Test duplicate detection
   - Validate metadata checking

### Integration Tests

1. **Test End-to-End Recovery**
   - Create workflow → simulate crash → restart → verify recovery
   - Test multiple workflow recovery
   - Validate state consistency

2. **Test Database Persistence**
   - Test CRUD operations
   - Verify transaction rollback
   - Test connection pool exhaustion

3. **Test Stuck Workflow Detection**
   - Create stuck workflows
   - Verify detection logic
   - Test FAILED state transition

### Recovery Scenario Tests

1. **Test Crash During Workflow Execution**
   ```python
   async def test_recovery_from_crash_mid_workflow():
       # Start workflow
       # Complete 2 of 3 steps
       # Simulate crash (kill node)
       # Restart node
       # Verify workflow resumes from step 3
       # Verify final state is COMPLETED
   ```

2. **Test Crash During State Transition**
   ```python
   async def test_recovery_from_crash_during_state_transition():
       # Start state transition
       # Crash before transition completes
       # Restart node
       # Verify state is consistent
       # Verify workflow continues
   ```

3. **Test Multiple Workflow Recovery**
   ```python
   async def test_recover_1000_concurrent_workflows():
       # Create 1000 in-progress workflows
       # Simulate crash
       # Restart node
       # Verify all 1000 workflows recovered
       # Measure recovery time (<5s target)
   ```

### Performance Tests

1. **Test Recovery Throughput**
   - Create 1000 workflows
   - Measure recovery time
   - Target: <5 seconds

2. **Test Query Performance**
   - Query 1000 in-progress workflows
   - Measure query time
   - Target: <500ms

3. **Test Concurrent Recovery**
   - Recover 50 workflows concurrently
   - Measure total time
   - Target: <2 seconds

## Implementation Checklist

- [ ] PostgreSQL client wrapper in `src/omninode_bridge/services/postgresql_client.py`
- [ ] CRUD operations for workflow_executions
- [ ] CRUD operations for workflow_steps
- [ ] CRUD operations for fsm_transitions
- [ ] CRUD operations for bridge_states
- [ ] `recover_workflows()` method in NodeBridgeOrchestrator
- [ ] `recover_aggregations()` method in NodeBridgeReducer
- [ ] `recover_states()` implementation in FSMStateManager
- [ ] Stuck workflow detection logic
- [ ] Idempotent operation checks
- [ ] Recovery metrics collection
- [ ] Recovery event publishing
- [ ] Unit tests (target: 100 tests)
- [ ] Integration tests (target: 20 tests)
- [ ] Recovery scenario tests (target: 10 tests)
- [ ] Performance tests (target: 5 tests)
- [ ] Test coverage validation (target: >90%)

## Success Criteria

1. **Functionality**
   - All workflows recover successfully after crash
   - Stuck workflows detected and transitioned to FAILED
   - State consistency maintained across restarts
   - Idempotent operations prevent duplicate work

2. **Performance**
   - 1000 workflows recovered in <5 seconds
   - Recovery queries complete in <500ms
   - No performance degradation after recovery

3. **Testing**
   - >90% test coverage for recovery code
   - All recovery scenarios pass
   - Performance tests meet targets
   - Integration tests verify end-to-end recovery

4. **Observability**
   - Recovery metrics logged and published
   - Stuck workflows tracked
   - Recovery failures reported
   - Performance metrics collected

## References

- PostgreSQL migration files: `migrations/001-004_*.sql`
- NodeBridgeOrchestrator: `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
- NodeBridgeReducer: `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`
- FSMStateManager: Embedded in reducer node
