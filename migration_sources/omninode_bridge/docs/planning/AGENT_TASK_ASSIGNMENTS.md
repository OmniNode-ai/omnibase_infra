# Polymorphic Agent Task Assignments

**Plan**: Pure Reducer Refactor
**Status**: Ready for execution
**Main Plan**: [PURE_REDUCER_REFACTOR_PLAN.md](./PURE_REDUCER_REFACTOR_PLAN.md)

## Quick Dispatch Reference

### Wave 1: Foundation (Parallel Execution)

```bash
# Execute all 3 workstreams in parallel
```

#### Agent 1A: Canonical Schema
```yaml
Agent ID: poly-foundation-1a
Duration: 4-6 hours
Files:
  - migrations/006_canonical_workflow_state.sql
  - src/omninode_bridge/infrastructure/entities/model_workflow_state.py
  - tests/unit/infrastructure/entities/test_model_workflow_state.py
Dependencies: None
Blockers: None
```

#### Agent 1B: Projection Schema
```yaml
Agent ID: poly-foundation-1b
Duration: 4-6 hours
Files:
  - migrations/007_projection_and_watermarks.sql
  - src/omninode_bridge/infrastructure/entities/model_workflow_projection.py
  - src/omninode_bridge/infrastructure/entities/model_projection_watermark.py
  - tests/unit/infrastructure/entities/test_projection_models.py
Dependencies: None
Blockers: None
```

#### Agent 1C: Dedup Schema
```yaml
Agent ID: poly-foundation-1c
Duration: 3-4 hours
Files:
  - migrations/008_action_deduplication.sql
  - src/omninode_bridge/infrastructure/entities/model_action_dedup.py
  - tests/unit/infrastructure/entities/test_model_action_dedup.py
Dependencies: None
Blockers: None
```

---

### Wave 2: Core Services (Parallel Execution)

**Depends on**: Wave 1 complete

```bash
# Execute all 4 workstreams in parallel after Wave 1
```

#### Agent 2A: Canonical Store
```yaml
Agent ID: poly-core-2a
Duration: 8-12 hours
Files:
  - src/omninode_bridge/services/canonical_store.py
  - tests/unit/services/test_canonical_store.py
  - tests/integration/services/test_canonical_store_integration.py
Dependencies: Wave 1 complete
Blockers: Needs model_workflow_state.py
```

#### Agent 2B: Projection Store
```yaml
Agent ID: poly-core-2b
Duration: 8-12 hours
Files:
  - src/omninode_bridge/services/projection_store.py
  - tests/unit/services/test_projection_store.py
  - tests/integration/services/test_projection_store_integration.py
Dependencies: Wave 1 complete, Agent 2A (for canonical fallback)
Blockers: Needs model_workflow_projection.py
```

#### Agent 2C: Projection Materializer
```yaml
Agent ID: poly-core-2c
Duration: 10-14 hours
Files:
  - src/omninode_bridge/services/projection_materializer.py
  - tests/unit/services/test_projection_materializer.py
  - tests/integration/services/test_materializer_kafka.py
Dependencies: Wave 1 complete
Blockers: Needs Kafka consumer setup
```

#### Agent 2D: Action Dedup Service
```yaml
Agent ID: poly-core-2d
Duration: 6-8 hours
Files:
  - src/omninode_bridge/services/action_dedup.py
  - src/omninode_bridge/background/dedup_cleanup.py
  - tests/unit/services/test_action_dedup.py
Dependencies: Wave 1 complete
Blockers: Needs model_action_dedup.py
```

---

### Wave 3: Reducer Layer (Parallel Execution)

**Depends on**: Wave 2 complete

```bash
# Execute both workstreams in parallel after Wave 2
```

#### Agent 3A: Pure Reducer Refactor
```yaml
Agent ID: poly-reducer-3a
Duration: 12-16 hours
Files:
  - src/omninode_bridge/nodes/reducer/v1_0_0/node.py (REFACTOR)
  - tests/unit/nodes/reducer/test_pure_reducer.py
  - tests/performance/test_reducer_benchmarks.py
Dependencies: Wave 2 complete
Blockers: Major refactor - remove all I/O
Critical: DO NOT delete existing tests, update them
```

#### Agent 3B: Reducer Service Wrapper
```yaml
Agent ID: poly-reducer-3b
Duration: 14-18 hours
Files:
  - src/omninode_bridge/services/reducer_service.py
  - tests/unit/services/test_reducer_service.py
  - tests/integration/services/test_reducer_service_retry.py
Dependencies: Wave 2 complete, Agent 3A
Blockers: Needs pure reducer interface
Critical: Implement jittered backoff correctly
```

---

### Wave 4: Store Effect (Sequential Execution)

**Depends on**: Wave 3 complete

```bash
# Execute after Wave 3 - single agent, not parallelizable
```

#### Agent 4A: Store Effect Node
```yaml
Agent ID: poly-effect-4a
Duration: 12-16 hours
Files:
  - src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/node.py (REFACTOR)
  - src/omninode_bridge/nodes/fsm_state_manager.py (MOVE from reducer)
  - tests/integration/test_store_effect_e2e.py
Dependencies: Wave 3 complete
Blockers: Needs CanonicalStoreService
Critical: Move FSMStateManager from Reducer to here
```

---

### Wave 5: Orchestrator (Sequential Execution)

**Depends on**: Wave 4 complete

```bash
# Execute after Wave 4 - single agent
```

#### Agent 5A: Event-Driven Orchestrator
```yaml
Agent ID: poly-orchestrator-5a
Duration: 10-12 hours
Files:
  - src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py (REFACTOR)
  - tests/integration/test_event_driven_orchestrator.py
Dependencies: Wave 4 complete
Blockers: Needs Store Effect Node operational
Critical: Remove all direct DB writes and Reducer calls
```

---

### Wave 6: Testing & Validation (Parallel Execution)

**Depends on**: Wave 5 complete

```bash
# Execute all 3 workstreams in parallel after Wave 5
```

#### Agent 6A: Integration Tests
```yaml
Agent ID: poly-test-6a
Duration: 10-14 hours
Files:
  - tests/integration/test_pure_reducer_e2e.py
  - tests/integration/test_conflict_resolution.py
  - tests/integration/test_projection_lag.py
Dependencies: Wave 5 complete
Blockers: Needs full system operational
Target: >90% coverage
```

#### Agent 6B: Performance Tests
```yaml
Agent ID: poly-test-6b
Duration: 8-12 hours
Files:
  - tests/performance/test_hot_key_contention.py
  - tests/performance/test_projection_lag_load.py
  - scripts/benchmark_pure_reducer.py
Dependencies: Wave 5 complete
Blockers: Needs full system operational
Critical: Validate SLAs (lag <250ms, conflict <0.5%)
```

#### Agent 6C: Metrics & Observability
```yaml
Agent ID: poly-test-6c
Duration: 8-10 hours
Files:
  - src/omninode_bridge/metrics/pure_reducer_metrics.py
  - dashboards/pure_reducer_monitoring.json
  - docs/runbooks/PURE_REDUCER_OPERATIONS.md
Dependencies: Wave 5 complete
Blockers: None
Critical: All metrics listed in plan must be instrumented
```

---

### Wave 7: Documentation (Parallel Execution)

**Depends on**: Wave 6 complete

```bash
# Execute both workstreams in parallel after Wave 6
```

#### Agent 7A: Architecture Documentation
```yaml
Agent ID: poly-docs-7a
Duration: 6-8 hours
Files:
  - docs/architecture/ARCHITECTURE.md (UPDATE)
  - docs/architecture/PURE_REDUCER_ARCHITECTURE.md (NEW)
  - docs/diagrams/pure_reducer_flow.mmd (Mermaid)
Dependencies: Wave 6 complete
Blockers: None
```

#### Agent 7B: Migration Guide
```yaml
Agent ID: poly-docs-7b
Duration: 6-8 hours
Files:
  - docs/guides/PURE_REDUCER_MIGRATION.md
  - CHANGELOG.md (UPDATE)
  - docs/BREAKING_CHANGES.md
Dependencies: Wave 6 complete
Blockers: None
Critical: Document all breaking changes
```

---

## Execution Commands

### Wave 1 (Parallel)
```bash
# In separate terminals or via Task tool
Task(poly-foundation-1a, "Implement canonical workflow_state schema and model")
Task(poly-foundation-1b, "Implement projection and watermark schemas")
Task(poly-foundation-1c, "Implement action deduplication schema")
```

### Wave 2 (Parallel, after Wave 1)
```bash
Task(poly-core-2a, "Implement CanonicalStoreService with optimistic concurrency")
Task(poly-core-2b, "Implement ProjectionStoreService with gating and fallback")
Task(poly-core-2c, "Implement ProjectionMaterializer with atomic watermarks")
Task(poly-core-2d, "Implement ActionDedupService with TTL cleanup")
```

### Wave 3 (Parallel, after Wave 2)
```bash
Task(poly-reducer-3a, "Refactor NodeBridgeReducer to pure function")
Task(poly-reducer-3b, "Implement ReducerService with retry loop and backoff")
```

### Wave 4 (Sequential, after Wave 3)
```bash
Task(poly-effect-4a, "Refactor DatabaseAdapter to Store Effect Node")
```

### Wave 5 (Sequential, after Wave 4)
```bash
Task(poly-orchestrator-5a, "Refactor Orchestrator to event-driven coordination")
```

### Wave 6 (Parallel, after Wave 5)
```bash
Task(poly-test-6a, "Create comprehensive E2E integration tests")
Task(poly-test-6b, "Create performance tests and validate SLAs")
Task(poly-test-6c, "Instrument metrics and create monitoring dashboard")
```

### Wave 7 (Parallel, after Wave 6)
```bash
Task(poly-docs-7a, "Update architecture documentation and diagrams")
Task(poly-docs-7b, "Create migration guide and document breaking changes")
```

---

## Progress Tracking

### Wave 1 Foundation
- [ ] Agent 1A: Canonical Schema (poly-foundation-1a)
- [ ] Agent 1B: Projection Schema (poly-foundation-1b)
- [ ] Agent 1C: Dedup Schema (poly-foundation-1c)

### Wave 2 Core Services
- [ ] Agent 2A: Canonical Store (poly-core-2a)
- [ ] Agent 2B: Projection Store (poly-core-2b)
- [ ] Agent 2C: Materializer (poly-core-2c)
- [ ] Agent 2D: Action Dedup (poly-core-2d)

### Wave 3 Reducer Layer
- [ ] Agent 3A: Pure Reducer (poly-reducer-3a)
- [ ] Agent 3B: Reducer Service (poly-reducer-3b)

### Wave 4 Store Effect
- [ ] Agent 4A: Store Effect Node (poly-effect-4a)

### Wave 5 Orchestrator
- [ ] Agent 5A: Event-Driven Orchestrator (poly-orchestrator-5a)

### Wave 6 Testing
- [ ] Agent 6A: Integration Tests (poly-test-6a)
- [ ] Agent 6B: Performance Tests (poly-test-6b)
- [ ] Agent 6C: Metrics (poly-test-6c)

### Wave 7 Documentation
- [ ] Agent 7A: Architecture Docs (poly-docs-7a)
- [ ] Agent 7B: Migration Guide (poly-docs-7b)

---

## Success Metrics

Track progress with:
```bash
# Files created/modified
git diff --stat origin/main

# Tests passing
pytest tests/ -v --cov

# Performance benchmarks
python scripts/benchmark_pure_reducer.py

# Coverage
pytest --cov-report=term-missing
```

**Target**: All checkboxes above completed, all tests passing, coverage >90%
