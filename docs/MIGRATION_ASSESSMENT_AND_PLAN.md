# Migration Assessment & Execution Plan
**Date:** 2025-11-14
**Status:** Ready for Execution
**Goal:** Extract omninode_bridge functionality and refactor to proper ONEX node architecture

---

## 🔍 Current State Assessment

### Repository Status
```
✅ All content ARCHIVED - src/ directory does NOT exist
✅ pyproject.toml expects: packages = [{include = "omnibase_infra", from = "src"}]
✅ Dependencies aligned with omninode_bridge (tagged v0.1.0 versions)
✅ Comprehensive observability stack installed
✅ All infrastructure in: archive/src_implementation/
```

### Archived Content Inventory

#### 1. Nodes (10 total)
```
archive/src_implementation/omnibase_infra/nodes/
├── node_postgres_adapter_effect/v1_0_0/          (72KB node.py - COMPLETE)
├── kafka_adapter/v1_0_0/
├── consul/v1_0_0/                                (Consul adapter)
├── consul_projector/v1_0_0/                      (Consul state projection)
├── postgres_adapter/v1_0_0/                      (Alternative postgres adapter)
├── node_infrastructure_observability_compute/v1_0_0/
├── node_distributed_tracing_compute/v1_0_0/
├── node_event_bus_circuit_breaker_compute/v1_0_0/
├── node_infrastructure_health_monitor_orchestrator/v1_0_0/  (NOT proper orchestrator!)
└── hook_node/v1_0_0/
```

**Base Classes Used:**
- `NodeEffectService` - Used by: postgres, kafka, consul adapters (✅ CORRECT)
- `NodeComputeService` - Used by: observability, tracing, circuit breaker (✅ CORRECT)
- **MISSING**: `NodeOrchestratorService` for workflow coordination
- **MISSING**: `NodeReducerService` for state aggregation

#### 2. Infrastructure Utilities (8 files, 3,974 lines)
```
archive/src_implementation/omnibase_infra/infrastructure/
├── postgres_connection_manager.py      (634 lines - connection pooling)
├── container.py                        (802 lines - dependency injection)
├── kafka_producer_pool.py             (459 lines - Kafka producer pooling)
├── infrastructure_observability.py    (557 lines - metrics & monitoring)
├── event_bus_circuit_breaker.py       (550 lines - circuit breaker)
├── distributed_tracing.py             (543 lines - OpenTelemetry tracing)
├── infrastructure_health_monitor.py   (429 lines - health monitoring)
└── __init__.py
```

#### 3. Shared Models (Comprehensive DRY Pattern)
```
archive/src_implementation/omnibase_infra/models/
├── postgres/          (15+ models: connection, query, health, metrics)
├── kafka/             (10+ models: producer, consumer, message, config)
├── consul/            (models for KV, service registration)
├── infrastructure/    (4 models: health metrics, config, circuit breaker)
├── event_publishing/  (event envelope models)
├── circuit_breaker/   (circuit breaker config models)
├── observability/     (tracing, metrics models)
├── security/          (security config models)
├── health/            (health check models)
└── [many more categories]
```

---

## 🎯 Migration Strategy

### Phase 1: Foundation Setup (Create Active Structure)

**Goal:** Establish clean src/ directory structure following INFRASTRUCTURE_PREPARATION_PLAN.md

```
src/omnibase_infra/
├── infrastructure/           # Foundation utilities (hand-written/migrated)
│   ├── postgres/
│   │   ├── connection_pool.py         # Extract from postgres_connection_manager
│   │   ├── health_monitor.py          # NEW or extract
│   │   └── query_metrics.py           # NEW or extract
│   ├── kafka/
│   │   ├── producer_pool.py           # Migrate kafka_producer_pool
│   │   ├── consumer_factory.py        # NEW or extract
│   │   └── topic_registry.py          # NEW
│   ├── resilience/
│   │   ├── circuit_breaker_factory.py # Migrate event_bus_circuit_breaker
│   │   ├── retry_policy.py            # NEW (using tenacity)
│   │   └── rate_limiter.py            # NEW (using slowapi)
│   └── observability/
│       ├── structured_logger.py       # Extract from nodes
│       ├── tracer_factory.py          # Migrate distributed_tracing
│       └── metrics_registry.py        # Migrate infrastructure_observability
│
├── models/                   # Shared models (MIGRATE as-is with import updates)
│   ├── postgres/             # ✅ Already comprehensive
│   ├── kafka/                # ✅ Already comprehensive
│   ├── consul/               # ✅ Already exists
│   ├── infrastructure/       # ✅ Already exists
│   └── [all other categories] # ✅ Migrate all
│
├── nodes/                    # Contract-driven nodes (REFACTOR + NEW)
│   ├── postgres_adapter/v1_0_0/      # MIGRATE + UPDATE imports
│   ├── kafka_adapter/v1_0_0/         # MIGRATE + UPDATE imports
│   ├── consul_adapter/v1_0_0/        # MIGRATE + UPDATE imports
│   ├── consul_projector/v1_0_0/      # MIGRATE + UPDATE imports
│   ├── infrastructure_reducer/v1_0_0/          # ⚠️ CREATE NEW
│   └── infrastructure_orchestrator/v1_0_0/    # ⚠️ CREATE NEW
│
└── registry/                 # Dependency injection
    ├── container.py          # MIGRATE + UPDATE
    ├── protocol_registry.py  # NEW or extract
    └── service_registry.py   # NEW or extract
```

### Phase 2: Extract & Migrate Nodes

**Priority Order (from CLAUDE.md):**

1. ✅ **postgres_adapter** - Already complete, just needs import updates
2. ✅ **kafka_adapter** - Exists, needs migration
3. ✅ **consul_adapter** - Exists, needs migration
4. ✅ **consul_projector** - Exists, needs migration
5. ⚠️ **infrastructure_reducer** - CREATE NEW (critical missing component!)
6. ⚠️ **infrastructure_orchestrator** - CREATE NEW (critical missing component!)

**Compute Nodes (Evaluate if needed):**
- `node_infrastructure_observability_compute` - May fold into infrastructure utilities
- `node_distributed_tracing_compute` - May fold into observability infrastructure
- `node_event_bus_circuit_breaker_compute` - May fold into resilience infrastructure

### Phase 3: Create Missing Infrastructure Nodes

#### 3.1 Infrastructure Reducer (NEW - CRITICAL)

**Purpose:** Aggregate and consolidate infrastructure state across all services

**Contract:** `infrastructure_reducer/v1_0_0/contract.yaml`
```yaml
node_name: "infrastructure_reducer"
node_type: "REDUCER"
description: "Infrastructure state aggregation and consolidation"

capabilities:
  - name: "health_aggregation"
    description: "Aggregate health status from all infrastructure nodes"
  - name: "metrics_consolidation"
    description: "Consolidate metrics from postgres, kafka, consul adapters"
  - name: "circuit_breaker_coordination"
    description: "Coordinate circuit breaker states across services"
  - name: "connection_pool_monitoring"
    description: "Monitor connection pool health across all adapters"

input_model: "ModelInfrastructureReducerInput"
output_model: "ModelInfrastructureReducerOutput"

dependencies:
  - name: "protocol_event_bus"
    type: "protocol"
  - name: "protocol_state_store"
    type: "protocol"
```

**Responsibilities:**
- Consume events from all infrastructure adapters
- Aggregate health status across postgres, kafka, consul
- Consolidate performance metrics
- Track circuit breaker states
- Monitor connection pool health
- Publish consolidated state events

#### 3.2 Infrastructure Orchestrator (NEW - CRITICAL)

**Purpose:** Coordinate workflows and operations across infrastructure services

**Contract:** `infrastructure_orchestrator/v1_0_0/contract.yaml`
```yaml
node_name: "infrastructure_orchestrator"
node_type: "ORCHESTRATOR"
description: "Infrastructure workflow coordination and service orchestration"

capabilities:
  - name: "health_check_orchestration"
    description: "Coordinate health checks across all infrastructure nodes"
  - name: "failover_coordination"
    description: "Coordinate service failover and recovery"
  - name: "connection_management"
    description: "Orchestrate connection pool initialization and cleanup"
  - name: "event_routing"
    description: "Route infrastructure events to appropriate handlers"

input_model: "ModelInfrastructureOrchestratorInput"
output_model: "ModelInfrastructureOrchestratorOutput"

dependencies:
  - name: "protocol_event_bus"
    type: "protocol"
  - name: "infrastructure_reducer"
    type: "node"
```

**Responsibilities:**
- Coordinate initialization of all infrastructure adapters
- Orchestrate health check sequences
- Manage failover and recovery workflows
- Route infrastructure events
- Coordinate connection pool lifecycle
- Handle graceful shutdown sequences

---

## 📋 Execution Checklist

### Stage 1: Foundation (Week 1)
- [ ] Create `src/omnibase_infra/` directory structure
- [ ] Migrate all shared models to `src/omnibase_infra/models/`
- [ ] Update all model imports: `omnibase.` → `omnibase_core.`
- [ ] Migrate infrastructure utilities to `src/omnibase_infra/infrastructure/`
- [ ] Update container.py for proper ONEXContainer integration
- [ ] Validate: All imports resolve, no `omnibase.` references remain

### Stage 2: Adapter Nodes (Week 1-2)
- [ ] Migrate postgres_adapter to `src/omnibase_infra/nodes/postgres_adapter/v1_0_0/`
- [ ] Update postgres_adapter imports and test
- [ ] Migrate kafka_adapter with import updates
- [ ] Migrate consul_adapter with import updates
- [ ] Migrate consul_projector with import updates
- [ ] Validate: All adapter nodes initialize without errors

### Stage 3: Infrastructure Reducer (Week 2)
- [ ] Design infrastructure reducer contract
- [ ] Create reducer models (input/output)
- [ ] Implement NodeReducerService subclass
- [ ] Implement health aggregation logic
- [ ] Implement metrics consolidation
- [ ] Implement circuit breaker coordination
- [ ] Write comprehensive tests
- [ ] Validate: Reducer receives and processes events from adapters

### Stage 4: Infrastructure Orchestrator (Week 2)
- [ ] Design infrastructure orchestrator contract
- [ ] Create orchestrator models (input/output)
- [ ] Implement NodeOrchestratorService subclass
- [ ] Implement health check orchestration
- [ ] Implement failover coordination
- [ ] Implement connection lifecycle management
- [ ] Write comprehensive tests
- [ ] Validate: Orchestrator coordinates all infrastructure operations

### Stage 5: Integration & Testing (Week 3)
- [ ] Integration tests: All nodes → Reducer → Orchestrator
- [ ] Event flow validation: postgres → kafka → reducer → orchestrator
- [ ] Health check workflow end-to-end
- [ ] Failover scenarios
- [ ] Load testing with multiple concurrent operations
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## 🚨 Critical Missing Components

### 1. Infrastructure Reducer (HIGHEST PRIORITY)
**Status:** Does NOT exist
**Impact:** No state aggregation across infrastructure services
**Required:** Must be created from scratch following NodeReducerService pattern

### 2. Infrastructure Orchestrator (HIGHEST PRIORITY)
**Status:** Health monitor "orchestrator" exists but doesn't use NodeOrchestratorService
**Impact:** No proper workflow coordination
**Required:** Must be created from scratch following NodeOrchestratorService pattern

### 3. Vault Adapter (MEDIUM PRIORITY)
**Status:** Not found in archive
**Impact:** No secret management integration
**Required:** Should be created after core adapters are migrated

---

## 🔄 Import Migration Pattern

**Every file must be updated:**
```python
# BEFORE (old omnibase references)
from omnibase.core.errors.onex_error import OnexError, CoreErrorCode
from omnibase.core.node_effect_service import NodeEffectService
from omnibase.protocol.protocol_event_bus import ProtocolEventBus

# AFTER (omnibase_core references)
from omnibase_core.core.errors.onex_error import OnexError, CoreErrorCode
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
```

**Automated approach:**
```bash
# Find all Python files with old imports
find src/ -name "*.py" -exec grep -l "from omnibase\." {} \;

# Replace omnibase. with omnibase_core. (careful with omnibase_infra!)
find src/ -name "*.py" -exec sed -i 's/from omnibase\.core/from omnibase_core.core/g' {} \;
find src/ -name "*.py" -exec sed -i 's/from omnibase\.protocol/from omnibase_core.protocol/g' {} \;
find src/ -name "*.py" -exec sed -i 's/from omnibase\.enums/from omnibase_core.enums/g' {} \;
find src/ -name "*.py" -exec sed -i 's/from omnibase\.models/from omnibase_core.models/g' {} \;
```

---

## 📊 Success Metrics

### Phase 1 Complete When:
✅ All models migrated with updated imports
✅ All infrastructure utilities operational
✅ Container provides proper dependency injection
✅ No `omnibase.` import references remain

### Phase 2 Complete When:
✅ All 4 adapter nodes (postgres, kafka, consul, consul_projector) functional
✅ All adapters publish events to event bus
✅ All adapters report health status
✅ Integration tests pass

### Phase 3 Complete When:
✅ Infrastructure reducer aggregates state from all adapters
✅ Reducer publishes consolidated state events
✅ Reducer tracks circuit breaker states
✅ Reducer monitors connection pool health

### Phase 4 Complete When:
✅ Infrastructure orchestrator coordinates all infrastructure workflows
✅ Orchestrator handles initialization sequences
✅ Orchestrator manages failover scenarios
✅ Orchestrator coordinates health checks

### Full Migration Complete When:
✅ All nodes follow proper ONEX architecture
✅ Reducer and orchestrator operational
✅ End-to-end integration tests pass
✅ Performance benchmarks met
✅ Documentation complete

---

## 🎯 Ready to Execute?

**I have enough information to:**
1. ✅ Extract all archived content to active src/
2. ✅ Update all imports from omnibase.* to omnibase_core.*
3. ✅ Organize infrastructure utilities
4. ✅ Migrate existing adapter nodes
5. ✅ Create infrastructure reducer from scratch
6. ✅ Create infrastructure orchestrator from scratch
7. ✅ Set up proper dependency injection
8. ✅ Write comprehensive tests

**Estimated Timeline:**
- Foundation + Adapters: 1-2 weeks
- Reducer + Orchestrator: 1-2 weeks
- Integration + Testing: 1 week
- **Total: 3-5 weeks for complete migration**

**Next Step:** Begin Stage 1 (Foundation Setup) by creating directory structure and migrating models.
