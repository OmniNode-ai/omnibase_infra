# Pure Reducer Refactor - Implementation Plan

**Status**: Planning
**Target**: Transform omninode_bridge to follow pure OmniNode 4-node architecture
**Branch**: `feature/pure-reducer-refactor`

## Architecture Goals

1. **Pure Reducer**: Stateless function `(state, action) → (state', intents[])`
2. **Reducer Service**: Wrapper with retry loop, reads canonical/projection
3. **Projection Materializer**: Event-driven projection builder from `StateCommitted`
4. **Store Effect Node**: Handles all persistence, publishes confirmations
5. **Event-Driven Coordination**: No direct node-to-node references

## Implementation Waves

Execute in order. Each wave has parallel workstreams.

---

## Wave 1: Foundation (Database Schema & Models)

**Duration**: 1-2 days
**Dependencies**: None
**Parallelizable**: Yes (3 workstreams)

### Workstream 1A: Canonical State Schema

**Agent**: `poly-foundation-1a`

**Tasks**:
1. Create migration `006_canonical_workflow_state.sql`
2. Schema definition:
   ```sql
   CREATE TABLE workflow_state (
       workflow_key TEXT PRIMARY KEY,
       version BIGINT NOT NULL DEFAULT 1,
       state JSONB NOT NULL,
       updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
       schema_version INT NOT NULL DEFAULT 1,
       provenance JSONB NOT NULL
   );

   CREATE INDEX idx_workflow_state_version ON workflow_state(workflow_key, version);
   CREATE INDEX idx_workflow_state_updated ON workflow_state(updated_at DESC);
   ```

3. Create Pydantic model:
   ```python
   # src/omninode_bridge/infrastructure/entities/model_workflow_state.py
   class ModelWorkflowState(BaseModel):
       workflow_key: str
       version: int
       state: dict[str, Any]
       updated_at: datetime
       schema_version: int
       provenance: dict[str, Any]
   ```

**Deliverables**:
- `migrations/006_canonical_workflow_state.sql`
- `src/omninode_bridge/infrastructure/entities/model_workflow_state.py`
- Unit tests for model validation

---

### Workstream 1B: Projection Schema & Watermarks

**Agent**: `poly-foundation-1b`

**Tasks**:
1. Create migration `007_projection_and_watermarks.sql`
2. Projection schema:
   ```sql
   CREATE TABLE workflow_projection (
       workflow_key TEXT PRIMARY KEY,
       version BIGINT NOT NULL,
       tag TEXT NOT NULL,
       last_action TEXT,
       namespace TEXT NOT NULL,
       updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
       indices JSONB,
       extras JSONB
   );

   CREATE INDEX idx_projection_namespace ON workflow_projection(namespace);
   CREATE INDEX idx_projection_tag ON workflow_projection(tag);
   ```

3. Watermark schema:
   ```sql
   CREATE TABLE projection_watermarks (
       partition_id TEXT PRIMARY KEY,
       offset BIGINT NOT NULL DEFAULT 0,
       updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
   );
   ```

4. Create Pydantic models:
   ```python
   # src/omninode_bridge/infrastructure/entities/model_workflow_projection.py
   class ModelWorkflowProjection(BaseModel):
       workflow_key: str
       version: int
       tag: str
       last_action: Optional[str]
       namespace: str
       updated_at: datetime
       indices: Optional[dict[str, Any]]
       extras: Optional[dict[str, Any]]
   ```

**Deliverables**:
- `migrations/007_projection_and_watermarks.sql`
- `src/omninode_bridge/infrastructure/entities/model_workflow_projection.py`
- `src/omninode_bridge/infrastructure/entities/model_projection_watermark.py`

---

### Workstream 1C: Action Deduplication Schema

**Agent**: `poly-foundation-1c`

**Tasks**:
1. Create migration `008_action_deduplication.sql`
2. Schema definition:
   ```sql
   CREATE TABLE action_dedup_log (
       workflow_key TEXT NOT NULL,
       action_id UUID NOT NULL,
       result_hash TEXT NOT NULL,
       processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
       expires_at TIMESTAMPTZ NOT NULL,
       PRIMARY KEY (workflow_key, action_id)
   );

   CREATE INDEX idx_action_dedup_expires ON action_dedup_log(expires_at);

   -- Cleanup job: DELETE FROM action_dedup_log WHERE expires_at < NOW()
   ```

3. Create Pydantic model:
   ```python
   # src/omninode_bridge/infrastructure/entities/model_action_dedup.py
   class ModelActionDedup(BaseModel):
       workflow_key: str
       action_id: UUID
       result_hash: str
       processed_at: datetime
       expires_at: datetime
   ```

**Deliverables**:
- `migrations/008_action_deduplication.sql`
- `src/omninode_bridge/infrastructure/entities/model_action_dedup.py`

---

## Wave 2: Core Service Implementations

**Duration**: 3-5 days
**Dependencies**: Wave 1 complete
**Parallelizable**: Yes (4 workstreams)

### Workstream 2A: Canonical Store Service

**Agent**: `poly-core-2a`

**Tasks**:
1. Create `src/omninode_bridge/services/canonical_store.py`
2. Implement interface:
   ```python
   class CanonicalStoreService:
       async def get_state(self, workflow_key: str) -> ModelWorkflowState
       async def try_commit(
           self,
           workflow_key: str,
           expected_version: int,
           state_prime: dict[str, Any],
           provenance: dict[str, Any]
       ) -> Union[StateCommitted, StateConflict]
   ```

3. Optimistic concurrency control:
   ```sql
   UPDATE workflow_state
   SET state = $1, version = version + 1, updated_at = NOW(), provenance = $2
   WHERE workflow_key = $3 AND version = $4
   RETURNING workflow_key, version;
   ```

4. Emit `StateCommitted` or `StateConflict` events

**Deliverables**:
- `src/omninode_bridge/services/canonical_store.py`
- Unit tests with version conflict scenarios
- Integration tests with PostgreSQL

---

### Workstream 2B: Projection Store Service

**Agent**: `poly-core-2b`

**Tasks**:
1. Create `src/omninode_bridge/services/projection_store.py`
2. Implement interface:
   ```python
   class ProjectionStoreService:
       async def get_state(
           self,
           workflow_key: str,
           required_version: Optional[int] = None,
           max_wait_ms: int = 100
       ) -> ModelWorkflowProjection

       async def get_version(self, workflow_key: str) -> int
   ```

3. Implement gating with fallback:
   ```python
   async def get_state(self, key, required_version, max_wait_ms=100):
       deadline = now() + max_wait_ms
       while now() < deadline:
           rec = await self._read_projection(key)
           if not required_version or rec.version >= required_version:
               return rec
           await asyncio.sleep(0.005)  # 5ms poll

       # Fallback to canonical
       canonical = await self.canonical_store.get_state(key)
       return self._to_projection(canonical)
   ```

**Deliverables**:
- `src/omninode_bridge/services/projection_store.py`
- Unit tests for gating logic
- Metrics for `projection_fallback_count`

---

### Workstream 2C: Projection Materializer Service

**Agent**: `poly-core-2c`

**Tasks**:
1. Create `src/omninode_bridge/services/projection_materializer.py`
2. Subscribe to `StateCommitted` Kafka events
3. Implement atomic projection + watermark update:
   ```python
   async def on_state_committed(self, event: StateCommitted):
       async with self.db.transaction():
           # Upsert projection
           await self.db.execute("""
               INSERT INTO workflow_projection (workflow_key, version, tag, ...)
               VALUES ($1, $2, $3, ...)
               ON CONFLICT (workflow_key) DO UPDATE
               SET version = EXCLUDED.version, ...
           """)

           # Advance watermark atomically
           await self.db.execute("""
               UPDATE projection_watermarks
               SET offset = GREATEST(offset, $1), updated_at = NOW()
               WHERE partition_id = $2
           """)
   ```

4. Idempotence check (optional):
   ```python
   # Check if already applied
   exists = await self.db.fetchval(
       "SELECT 1 FROM proj_apply_log WHERE partition_id=$1 AND offset=$2",
       partition_id, origin_offset
   )
   if exists:
       return  # Already processed
   ```

**Deliverables**:
- `src/omninode_bridge/services/projection_materializer.py`
- Kafka consumer for `StateCommitted` events
- Metrics for `projection_wm_lag_ms`, `wm_regressions_total`

---

### Workstream 2D: Action Deduplication Service

**Agent**: `poly-core-2d`

**Tasks**:
1. Create `src/omninode_bridge/services/action_dedup.py`
2. Implement interface:
   ```python
   class ActionDedupService:
       async def should_process(
           self,
           workflow_key: str,
           action_id: UUID
       ) -> bool

       async def remember(
           self,
           workflow_key: str,
           action_id: UUID,
           result_hash: str,
           ttl_hours: int = 6
       ) -> None
   ```

3. Postgres-backed dedup:
   ```python
   async def should_process(self, workflow_key, action_id):
       exists = await self.db.fetchval(
           "SELECT 1 FROM action_dedup_log WHERE workflow_key=$1 AND action_id=$2",
           workflow_key, action_id
       )
       return not exists

   async def remember(self, workflow_key, action_id, result_hash, ttl_hours=6):
       await self.db.execute("""
           INSERT INTO action_dedup_log (workflow_key, action_id, result_hash, expires_at)
           VALUES ($1, $2, $3, NOW() + INTERVAL '%s hours')
           ON CONFLICT (workflow_key, action_id) DO NOTHING
       """, workflow_key, action_id, result_hash, ttl_hours)
   ```

4. Cleanup job (cron or background task):
   ```python
   async def cleanup_expired(self):
       deleted = await self.db.execute(
           "DELETE FROM action_dedup_log WHERE expires_at < NOW()"
       )
       logger.info(f"Cleaned up {deleted} expired dedup entries")
   ```

**Deliverables**:
- `src/omninode_bridge/services/action_dedup.py`
- Cleanup background task
- Metrics for `action_dedup_hits_total`

---

## Wave 3: Reducer Service & Conflict Handling

**Duration**: 2-3 days
**Dependencies**: Wave 2 complete
**Parallelizable**: Partially (2 workstreams)

### Workstream 3A: Pure Reducer Function Refactor

**Agent**: `poly-reducer-3a`

**Tasks**:
1. Refactor `NodeBridgeReducer` to be pure:
   ```python
   class NodeBridgeReducer(NodeReducer):
       async def process(
           self,
           input_data: ModelReducerInput
       ) -> ModelReducerOutput:
           # Pure aggregation logic only
           # NO database writes
           # NO Kafka publishing
           # Return intents instead

           aggregated_data = self._aggregate_metadata(input_data.data)

           return ModelReducerOutput(
               result=aggregated_data,
               intents=[
                   ModelIntent(
                       intent_type="PersistState",
                       target="store_effect",
                       payload={
                           "aggregated_data": aggregated_data,
                           "version": input_data.metadata.get("version")
                       }
                   )
               ],
               processing_time_ms=processing_time,
               items_processed=len(input_data.data)
           )
   ```

2. Remove FSMStateManager (move to Store Effect)
3. Remove `_persist_state()` method
4. Remove `_publish_event()` calls
5. Keep only pure aggregation logic

**Deliverables**:
- Refactored `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`
- Unit tests with mocked state (no DB)
- Performance benchmarks (should be faster without I/O)

---

### Workstream 3B: Reducer Service Wrapper

**Agent**: `poly-reducer-3b`

**Tasks**:
1. Create `src/omninode_bridge/services/reducer_service.py`
2. Implement retry loop with jittered backoff:
   ```python
   class ReducerService:
       def __init__(
           self,
           reducer: NodeBridgeReducer,
           canonical_store: CanonicalStoreService,
           projection_store: ProjectionStoreService,
           action_dedup: ActionDedupService,
           event_bus: EventBus
       ):
           self.reducer = reducer
           self.canonical_store = canonical_store
           self.projection_store = projection_store
           self.action_dedup = action_dedup
           self.event_bus = event_bus

           # Config
           self.max_attempts = 3
           self.backoff_base_ms = 10
           self.backoff_cap_ms = 250

       async def handle_action(self, action: Action):
           # 1. Dedup check
           if not await self.action_dedup.should_process(
               action.workflow_key, action.action_id
           ):
               logger.info(f"Skipping duplicate action {action.action_id}")
               return

           # 2. Retry loop
           for attempt in range(1, self.max_attempts + 1):
               try:
                   # Read state (canonical for now)
                   state_record = await self.canonical_store.get_state(
                       action.workflow_key
                   )

                   # Call pure reducer
                   result = await self.reducer.process(
                       ModelReducerInput(
                           data=state_record.state,
                           metadata={"action": action, "version": state_record.version}
                       )
                   )

                   # Emit PersistState intent as event
                   await self.event_bus.publish("PersistState", {
                       "workflow_key": action.workflow_key,
                       "expected_version": state_record.version,
                       "state_prime": result.result,
                       "action_id": action.action_id
                   })

                   # Wait for confirmation
                   event = await self.event_bus.wait_any(
                       ["StateCommitted", "StateConflict"],
                       key=action.workflow_key,
                       timeout_ms=5000
                   )

                   if event.type == "StateCommitted":
                       # Success - publish other intents
                       for intent in result.intents:
                           if intent.intent_type != "PersistState":
                               await self.event_bus.publish(
                                   intent.intent_type,
                                   intent.payload
                               )

                       # Remember action
                       result_hash = self._hash_result(result.result)
                       await self.action_dedup.remember(
                           action.workflow_key,
                           action.action_id,
                           result_hash
                       )
                       return

                   elif event.type == "StateConflict":
                       # Conflict - retry with backoff
                       backoff_ms = self._backoff_delay(attempt)
                       logger.warning(
                           f"Version conflict on {action.workflow_key}, "
                           f"attempt {attempt}/{self.max_attempts}, "
                           f"backoff {backoff_ms}ms"
                       )
                       await asyncio.sleep(backoff_ms / 1000)
                       continue

               except asyncio.TimeoutError:
                   logger.error(
                       f"Timeout waiting for StateCommitted on {action.workflow_key}"
                   )
                   break

           # 3. Max retries exceeded
           await self.event_bus.publish("ReducerGaveUp", {
               "workflow_key": action.workflow_key,
               "action_id": action.action_id,
               "attempts": self.max_attempts
           })

       def _backoff_delay(self, attempt: int) -> int:
           """Decorrelated jitter backoff"""
           import random
           max_ms = min(self.backoff_cap_ms, self.backoff_base_ms * (2 ** attempt))
           return random.randint(self.backoff_base_ms, max_ms)
   ```

**Deliverables**:
- `src/omninode_bridge/services/reducer_service.py`
- Unit tests with mocked stores
- Integration tests with conflict scenarios
- Metrics: `conflict_attempts_total`, `backoff_ms_histogram`

---

## Wave 4: Store Effect Node Refactor

**Duration**: 2-3 days
**Dependencies**: Wave 3 complete
**Parallelizable**: No (sequential)

### Workstream 4A: Store Effect Node

**Agent**: `poly-effect-4a`

**Tasks**:
1. Refactor `NodeBridgeDatabaseAdapterEffect` to be Store Effect Node
2. Subscribe to `PersistState` events
3. Delegate to `CanonicalStoreService.try_commit()`
4. Publish `StateCommitted` or `StateConflict` events
5. Move FSMStateManager from Reducer to here

**Implementation**:
```python
class NodeBridgeStoreEffect(NodeEffect):
    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        self.canonical_store = container.canonical_store()
        self.event_bus = container.event_bus()
        self.fsm_manager = FSMStateManager(container)

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        operation = contract.input_data.get("operation")

        if operation == "persist_state":
            return await self._handle_persist_state(contract.input_data)
        elif operation == "fsm_transition":
            return await self._handle_fsm_transition(contract.input_data)
        # ... other operations

    async def _handle_persist_state(self, data: dict[str, Any]):
        workflow_key = data["workflow_key"]
        expected_version = data["expected_version"]
        state_prime = data["state_prime"]

        # Build provenance
        provenance = {
            "effect_id": str(self.node_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "action_id": data.get("action_id")
        }

        # Try commit
        result = await self.canonical_store.try_commit(
            workflow_key=workflow_key,
            expected_version=expected_version,
            state_prime=state_prime,
            provenance=provenance
        )

        # Publish event
        if isinstance(result, StateCommitted):
            await self.event_bus.publish("StateCommitted", {
                "workflow_key": workflow_key,
                "version": result.version,
                "commit_offset": result.commit_offset,
                "provenance": provenance
            })
        elif isinstance(result, StateConflict):
            await self.event_bus.publish("StateConflict", {
                "workflow_key": workflow_key,
                "expected_version": expected_version,
                "actual_version": result.actual_version
            })

        return result

    async def _handle_fsm_transition(self, data: dict[str, Any]):
        # FSM state management moved here from Reducer
        workflow_id = data["workflow_id"]
        from_state = data["from_state"]
        to_state = data["to_state"]

        await self.fsm_manager.transition_state(
            workflow_id, from_state, to_state
        )

        # Persist via canonical store
        # Publish FSM_STATE_TRANSITIONED event
```

**Deliverables**:
- Refactored `src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/node.py`
- Event subscription for `PersistState`
- Integration tests with full event flow
- Metrics: `state_commits_total`, `state_conflicts_total`

---

## Wave 5: Orchestrator Refactor

**Duration**: 2 days
**Dependencies**: Wave 4 complete
**Parallelizable**: No

### Workstream 5A: Event-Driven Orchestrator

**Agent**: `poly-orchestrator-5a`

**Tasks**:
1. Remove direct DB writes from `NodeBridgeOrchestrator`
2. Remove direct calls to Reducer
3. Publish `Action` events instead
4. Subscribe to `StateCommitted`, `ReducerGaveUp` events
5. Implement DAG policy for failures

**Implementation**:
```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    async def orchestrate_workflow(self, contract: ModelContractOrchestrator):
        workflow_id = uuid4()

        # 1. Publish Action event (instead of calling reducer directly)
        await self.event_bus.publish("Action", {
            "action_id": uuid4(),
            "workflow_key": str(workflow_id),
            "epoch": 1,
            "lease_id": str(uuid4()),
            "payload": contract.input_data
        })

        # 2. Wait for StateCommitted or ReducerGaveUp
        event = await self.event_bus.wait_any(
            ["StateCommitted", "ReducerGaveUp"],
            key=str(workflow_id),
            timeout_ms=30000
        )

        if event.type == "StateCommitted":
            # Success - continue workflow
            return await self._handle_success(workflow_id, event)
        elif event.type == "ReducerGaveUp":
            # Failure - apply backoff/compensation policy
            return await self._handle_failure(workflow_id, event)
```

**Deliverables**:
- Refactored `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
- Event-driven coordination
- Integration tests with full workflow

---

## Wave 6: Testing & Validation

**Duration**: 2-3 days
**Dependencies**: Wave 5 complete
**Parallelizable**: Yes (3 workstreams)

### Workstream 6A: Integration Tests

**Agent**: `poly-test-6a`

**Tasks**:
1. Create `tests/integration/test_pure_reducer_e2e.py`
2. Test scenarios:
   - Happy path: Action → Reduce → Commit → Projection
   - Conflict resolution: Multiple concurrent actions on same key
   - Deduplication: Duplicate action ignored
   - Projection lag: Fallback to canonical
   - Reducer gave up: Escalation to orchestrator

**Deliverables**:
- Comprehensive E2E tests
- Test coverage > 90%

---

### Workstream 6B: Performance Tests

**Agent**: `poly-test-6b`

**Tasks**:
1. Create `tests/performance/test_hot_key_contention.py`
2. Simulate hot key scenarios (100+ concurrent reducers on same key)
3. Measure:
   - Conflict rate
   - Retry distribution
   - Throughput (actions/sec)
   - Latency (p50, p95, p99)
4. Validate SLAs:
   - Projection lag < 250ms (p99)
   - Conflict rate < 0.5% (p99)

**Deliverables**:
- Performance test suite
- Benchmark results baseline

---

### Workstream 6C: Metrics & Observability

**Agent**: `poly-test-6c`

**Tasks**:
1. Add Prometheus metrics to all services:
   - `conflict_attempts_total{workflow_key}`
   - `backoff_ms_histogram`
   - `projection_wm_lag_ms{workflow_key}`
   - `projection_fallback_count`
   - `action_dedup_hits_total`
   - `state_commits_total`
   - `state_conflicts_total`
   - `reducer_gaveup_total`

2. Create Grafana dashboard JSON
3. Add structured logging with correlation IDs

**Deliverables**:
- Metrics instrumentation
- Grafana dashboard: `dashboards/pure_reducer_monitoring.json`
- Runbook: `docs/runbooks/PURE_REDUCER_OPERATIONS.md`

---

## Wave 7: Documentation & Cleanup

**Duration**: 1 day
**Dependencies**: Wave 6 complete
**Parallelizable**: Yes (2 workstreams)

### Workstream 7A: Architecture Documentation

**Agent**: `poly-docs-7a`

**Tasks**:
1. Update `docs/architecture/ARCHITECTURE.md`
2. Create sequence diagrams for:
   - Action → Reduce → Commit flow
   - Conflict retry loop
   - Projection materialization
3. Document event contracts
4. Add troubleshooting guide

**Deliverables**:
- Updated architecture docs
- Sequence diagrams (Mermaid format)
- API documentation

---

### Workstream 7B: Migration Guide & Cleanup

**Agent**: `poly-docs-7b`

**Tasks**:
1. Create `docs/guides/PURE_REDUCER_MIGRATION.md`
2. Document breaking changes
3. Provide migration steps from old architecture
4. Remove dead code:
   - Old `_persist_state()` methods
   - Direct Kafka publishing from Reducer
   - In-memory FSMStateManager caches

**Deliverables**:
- Migration guide
- Cleanup PR
- Changelog entry

---

## Execution Strategy

### Phase 1: Foundation
```bash
# Fire off Wave 1 in parallel (3 agents)
poly-foundation-1a &  # Canonical schema
poly-foundation-1b &  # Projection schema
poly-foundation-1c &  # Dedup schema
wait
```

### Phase 2: Core Services
```bash
# Fire off Wave 2 in parallel (4 agents)
poly-core-2a &  # Canonical store
poly-core-2b &  # Projection store
poly-core-2c &  # Materializer
poly-core-2d &  # Action dedup
wait
```

### Phase 3: Reducer Layer
```bash
# Fire off Wave 3 in parallel (2 agents)
poly-reducer-3a &  # Pure reducer refactor
poly-reducer-3b &  # Reducer service wrapper
wait
```

### Phase 4-5: Integration (Sequential)
```bash
poly-effect-4a      # Store effect node
poly-orchestrator-5a  # Orchestrator refactor
```

### Phase 6: Validation
```bash
# Fire off Wave 6 in parallel (3 agents)
poly-test-6a &  # Integration tests
poly-test-6b &  # Performance tests
poly-test-6c &  # Metrics
wait
```

### Phase 7: Documentation
```bash
# Fire off Wave 7 in parallel (2 agents)
poly-docs-7a &  # Architecture docs
poly-docs-7b &  # Migration guide
wait
```

## Success Criteria

- ✅ All migrations applied successfully
- ✅ Reducer is pure (no I/O, no side effects)
- ✅ ReducerService handles retries with bounded attempts
- ✅ Projection materializer runs with <250ms lag (p99)
- ✅ Conflict rate <0.5% under load
- ✅ Action deduplication prevents duplicate processing
- ✅ All tests passing (>90% coverage)
- ✅ Performance benchmarks meet SLAs
- ✅ Metrics dashboards operational
- ✅ Documentation complete

## Rollback Plan

If issues arise:
1. Feature flag: `ENABLE_PURE_REDUCER=false` (use old path)
2. Database rollback: `alembic downgrade -1` (reverse migrations)
3. Code rollback: `git revert feature/pure-reducer-refactor`

## Timeline Estimate

- Wave 1: 1-2 days
- Wave 2: 3-5 days
- Wave 3: 2-3 days
- Wave 4: 2-3 days
- Wave 5: 2 days
- Wave 6: 2-3 days
- Wave 7: 1 day

**Total**: 13-19 days (with parallelization, ~2-3 weeks)

## Dependencies

### External:
- PostgreSQL 12+ (for GREATEST() function)
- Kafka (for event bus)
- Redis (optional, for faster action dedup)

### Internal:
- omnibase_core v2.0+ (ModelIntent support)
- All migrations from Wave 1

## Risk Mitigation

1. **Data loss**: All migrations are additive, old tables remain
2. **Performance regression**: Feature flag allows instant rollback
3. **Integration issues**: Comprehensive E2E tests before deployment
4. **Operational complexity**: Runbooks and dashboards ready before launch

---

**Next Steps**: Create feature branch and begin Wave 1 execution
