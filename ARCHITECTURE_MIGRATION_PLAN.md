# ONEX Infrastructure Architecture Migration Plan

## Executive Summary

This document provides a comprehensive migration plan for transitioning ONEX infrastructure from the current mixed architecture to proper ONEX 4-node patterns. The analysis identifies **6 files** in `src/omnibase_infra/infrastructure/` that need migration, with **postgres_connection_manager.py** and **kafka_producer_pool.py** being the primary targets as they should both become proper EFFECT nodes.

## 🎯 Primary Migration Targets

### **postgres_connection_manager.py** → **node_postgres_connection_manager_effect**

**Current Location:** `src/omnibase_infra/infrastructure/postgres_connection_manager.py`  
**Target Location:** `src/omnibase_infra/nodes/node_postgres_connection_manager_effect/v1_0_0/`  
**ONEX Pattern:** EFFECT node (external database service interactions)

#### Migration Rationale

| Criteria | Analysis | ONEX Pattern |
|----------|----------|--------------|
| **External Service Interaction** | ✅ Manages PostgreSQL database connections | **EFFECT** |
| **Connection Pooling** | ✅ Connection pool management and lifecycle | **EFFECT** |
| **Database Operations** | ✅ Query execution, transaction management | **EFFECT** |
| **Complex Business Logic** | ✅ Connection health monitoring, pool sizing | **Node Required** |
| **Service Boundaries** | ✅ Clear database access interface | **Node Required** |

#### Current Architecture Issues

1. **Utility Classification** - Currently treated as utility but has complex business logic
2. **Direct Service Implementation** - Not following ONEX contract-driven pattern
3. **Mixed Responsibilities** - Connection pooling + query execution + health monitoring
4. **Hardcoded Configuration** - Not using ONEX configuration contracts

#### Target Node Architecture

```
src/omnibase_infra/nodes/node_postgres_connection_manager_effect/v1_0_0/
├── contract.yaml                    # ONEX contract with shared model dependencies
├── node.py                         # NodeEffectService implementation
├── models/                         # Node-specific adapter models
│   ├── model_postgres_connection_manager_input.py
│   └── model_postgres_connection_manager_output.py
└── registry/
    └── registry_postgres_connection_manager.py
```

---

### **kafka_producer_pool.py** → **node_kafka_producer_pool_effect**

**Current Location:** `src/omnibase_infra/infrastructure/kafka_producer_pool.py`  
**Target Location:** `src/omnibase_infra/nodes/node_kafka_producer_pool_effect/v1_0_0/`  
**ONEX Pattern:** EFFECT node (external service interactions)

#### Migration Rationale

| Criteria | Analysis | ONEX Pattern |
|----------|----------|--------------|
| **External Service Interaction** | ✅ Manages Kafka/RedPanda connections | **EFFECT** |
| **Connection Pooling** | ✅ Producer pool management and lifecycle | **EFFECT** |
| **Message Publishing** | ✅ Publishes messages to external Kafka topics | **EFFECT** |
| **Complex Business Logic** | ✅ Circuit breaker integration, health monitoring | **Node Required** |
| **Service Boundaries** | ✅ Clear interface for event publishing | **Node Required** |

#### Current Architecture Issues

1. **Singleton Pattern** - Uses global `_producer_pool` instance
2. **Direct Service Implementation** - Not following ONEX contract-driven pattern
3. **Mixed Responsibilities** - Connection pooling + message publishing + health monitoring
4. **Hardcoded Configuration** - Not using ONEX configuration contracts

#### Target Node Architecture

```
src/omnibase_infra/nodes/node_kafka_producer_pool_effect/v1_0_0/
├── contract.yaml                    # ONEX contract with shared model dependencies
├── node.py                         # NodeEffectService implementation
├── models/                         # Node-specific adapter models
│   ├── model_kafka_producer_pool_input.py
│   └── model_kafka_producer_pool_output.py
└── registry/
    └── registry_kafka_producer_pool.py
```

#### Shared Models (DRY Pattern)

**Create in:** `src/omnibase_infra/models/kafka/`
- `model_kafka_producer_config.py` ✅ (Already exists)
- `model_kafka_producer_pool_stats.py` ✅ (Already exists)
- `model_kafka_message_payload.py` ✅ (Already exists)
- `model_kafka_security_config.py` ✅ (Already exists)

#### Migration Steps

1. **Phase 1: Contract Creation**
   - Create ONEX contract with proper model dependencies
   - Define io_operations for producer management, message publishing, health checks
   - Reference existing shared Kafka models as dependencies

2. **Phase 2: Node Implementation**
   - Convert `KafkaProducerPool` class to `NodeKafkaProducerPoolEffect`
   - Implement `NodeEffectService` base class
   - Add proper container injection: `def __init__(self, container: ONEXContainer)`
   - Replace singleton pattern with dependency injection

3. **Phase 3: Integration Updates**
   - Update `container.py` to register node instead of direct pool
   - Update all references to use container.get_service() pattern
   - Update imports throughout codebase

4. **Phase 4: Testing & Validation**
   - Unit tests for node functionality
   - Integration tests with event bus
   - Performance validation for producer pooling

---

## 🏗️ Secondary Migration Targets

### Infrastructure Files Analysis

| File | Current Purpose | Migration Strategy | Priority | Target Name |
|------|----------------|-------------------|----------|-------------|
| **postgres_connection_manager.py** | Database connection pooling | **MIGRATE TO EFFECT NODE** | **CRITICAL** | `node_postgres_connection_manager_effect` |
| **event_bus_circuit_breaker.py** | Circuit breaker for event reliability | **MIGRATE TO COMPUTE NODE** | High | `node_event_bus_circuit_breaker_compute` |
| **distributed_tracing.py** | OpenTelemetry tracing integration | **MIGRATE TO COMPUTE NODE** | Medium | `node_distributed_tracing_compute` |
| **container.py** | Dependency injection container | **KEEP AS INFRASTRUCTURE** | ✅ Correctly placed | N/A |
| **infrastructure_health_monitor.py** | Health monitoring aggregation | **MIGRATE TO ORCHESTRATOR NODE** | Medium | `node_infrastructure_health_monitor_orchestrator` |
| **infrastructure_observability.py** | Metrics collection and monitoring | **MIGRATE TO COMPUTE NODE** | Low | `node_infrastructure_observability_compute` |

### Detailed Migration Analysis

#### 1. **event_bus_circuit_breaker.py** → **node_event_bus_circuit_breaker_compute**

**ONEX Pattern:** COMPUTE (message processing and transformation)

```yaml
Migration Rationale:
- Processes event envelopes through circuit breaker logic
- Transforms events based on circuit state
- Complex business logic requiring contract-driven configuration
- Clear input/output model boundaries

Target: src/omnibase_infra/nodes/node_event_bus_circuit_breaker_compute/v1_0_0/
```

**Shared Models to Create:**
- `src/omnibase_infra/models/infrastructure/model_circuit_breaker_state.py`
- `src/omnibase_infra/models/infrastructure/model_circuit_breaker_metrics.py`

**Migration Steps:**
1. Create contract with circuit breaker configuration
2. Convert `EventBusCircuitBreaker` to `NodeEventBusCircuitBreakerCompute`
3. Implement proper event envelope processing
4. Update environment configuration integration

#### 2. **distributed_tracing.py** → **node_distributed_tracing_compute**

**ONEX Pattern:** COMPUTE (trace context processing and enrichment)

```yaml
Migration Rationale:
- Processes trace context in event envelopes
- Enriches events with tracing metadata
- Transforms events for trace propagation
- Integration point requiring proper contract boundaries

Target: src/omnibase_infra/nodes/node_distributed_tracing_compute/v1_0_0/
```

**Shared Models to Create:**
- `src/omnibase_infra/models/observability/model_trace_context.py`
- `src/omnibase_infra/models/observability/model_trace_configuration.py`

#### 3. **infrastructure_health_monitor.py** → **node_infrastructure_health_monitor_orchestrator**

**ONEX Pattern:** ORCHESTRATOR (workflow coordination across components)

```yaml
Migration Rationale:
- Orchestrates health checks across multiple services
- Coordinates monitoring workflows
- Aggregates results from multiple EFFECT and COMPUTE nodes
- Clear orchestration responsibilities

Target: src/omnibase_infra/nodes/node_infrastructure_health_monitor_orchestrator/v1_0_0/
```

#### 4. **infrastructure_observability.py** → **node_infrastructure_observability_compute**

**ONEX Pattern:** COMPUTE (metrics processing and aggregation)

```yaml
Migration Rationale:
- Processes metrics data and transforms for export
- Aggregates observability data from multiple sources
- Low priority - can remain utility for now

Target: Future consideration - src/omnibase_infra/nodes/node_infrastructure_observability_compute/v1_0_0/
```

---

## 🎯 Migration Priority Matrix

| Priority | Component | Impact | Effort | Dependencies |
|----------|-----------|--------|--------|--------------|
| **CRITICAL** | postgres_connection_manager.py | High | Medium | None |
| **CRITICAL** | kafka_producer_pool.py | High | Medium | None |
| **HIGH** | event_bus_circuit_breaker.py | Medium | Medium | postgres_connection_manager, kafka_producer_pool |
| **MEDIUM** | distributed_tracing.py | Low | High | OpenTelemetry integration |
| **MEDIUM** | infrastructure_health_monitor.py | Medium | Low | All other nodes |
| **LOW** | infrastructure_observability.py | Low | High | Prometheus integration |

---

## 📋 Implementation Phases

### Phase 1: Foundation (postgres_connection_manager.py & kafka_producer_pool.py)
- **Duration:** 2-3 days
- **Deliverables:**
  - `node_postgres_connection_manager_effect` with full ONEX compliance
  - `node_kafka_producer_pool_effect` with full ONEX compliance
  - Updated container registration for both nodes
  - Integration tests for database and event bus connectivity
- **Success Criteria:**
  - All existing functionality preserved for both components
  - Contract-driven configuration implemented
  - Dependency injection working correctly
  - Performance benchmarks maintained
  - No breaking changes to existing consumers

### Phase 2: Reliability (event_bus_circuit_breaker.py)
- **Duration:** 1 day
- **Dependencies:** Phase 1 complete
- **Deliverables:**
  - `node_event_bus_circuit_breaker_compute` with event envelope processing
  - Environment-specific configuration via contracts
- **Success Criteria:**
  - Circuit breaker functionality preserved
  - Integration with Phase 1 nodes working
  - All reliability patterns maintained

### Phase 3: Observability (distributed_tracing.py, infrastructure_health_monitor.py)
- **Duration:** 2-3 days
- **Dependencies:** Phase 1-2 complete
- **Deliverables:**
  - `node_distributed_tracing_compute` for trace context processing
  - `node_infrastructure_health_monitor_orchestrator` for monitoring orchestration
  - End-to-end observability chain
- **Success Criteria:**
  - Full trace propagation working
  - Health monitoring aggregation functional
  - Dashboard integration maintained

### Phase 4: Advanced Observability (infrastructure_observability.py)
- **Duration:** 1-2 days
- **Dependencies:** All phases complete
- **Deliverables:**
  - `node_infrastructure_observability_compute` for metrics processing
  - Prometheus integration via node architecture
- **Success Criteria:**
  - All metrics collection working
  - Export functionality preserved
  - Performance monitoring maintained

---

## 📁 Existing Nodes Directory Audit

### Current Node Structure Issues

The existing `nodes/` directory has inconsistent naming that needs correction:

| Current Directory | Issue | Correct Name |
|------------------|-------|--------------|
| `consul/v1_0_0/` | Missing `node_` prefix | `node_consul_effect/v1_0_0/` |
| `node_postgres_adapter_effect/v1_0_0/` | ✅ Already correct | No change needed |
| `consul_projector/v1_0_0/` | Missing `node_` prefix | `node_consul_projector_effect/v1_0_0/` |

### Node Naming Convention Standard

**MANDATORY**: All nodes must follow the `node_` prefix convention:
- Pattern: `node_{name}_{type}/v{major}_{minor}_{patch}/`
- Example: `node_kafka_producer_pool_effect/v1_0_0/`
- Types: `effect`, `compute`, `reducer`, `orchestrator`

### Directory Rename Actions Required

```bash
# Rename existing directories to follow convention
mv nodes/consul/v1_0_0/ nodes/node_consul_effect/v1_0_0/
mv nodes/consul_projector/v1_0_0/ nodes/node_consul_projector_effect/v1_0_0/

# Update contract.yaml files with correct node_name
# Update pyproject.toml entry points
# Update container registration references
```

---

## 🔄 Consistency Patterns

### Shared Model Architecture

All infrastructure nodes follow the **Shared Model Pattern**:

```
src/omnibase_infra/
├── models/                         # Shared models (DRY pattern)
│   ├── kafka/                      # Kafka-specific models
│   ├── postgres/                   # PostgreSQL-specific models
│   ├── consul/                     # Consul-specific models
│   ├── vault/                      # Vault-specific models
│   ├── infrastructure/             # Infrastructure-specific models
│   └── observability/              # Observability models
└── nodes/                          # Contract-driven nodes
    ├── node_postgres_connection_manager_effect/
    │   └── v1_0_0/
    │       ├── contract.yaml       # References shared postgres models as dependencies
    │       └── models/             # Node-specific adapter models only
    ├── node_kafka_producer_pool_effect/
    │   └── v1_0_0/
    │       ├── contract.yaml       # References shared kafka models as dependencies
    │       └── models/             # Node-specific adapter models only
    ├── node_event_bus_circuit_breaker_compute/
    │   └── v1_0_0/
    │       ├── contract.yaml       # References shared infrastructure models
    │       └── models/             # Node-specific adapter models only
    └── [...other infrastructure nodes with proper node_ naming]
```

### Contract Standards

All infrastructure contracts include:

```yaml
dependencies:
  # Protocol dependencies
  - name: "protocol_event_bus"
    type: "protocol"
    class_name: "ProtocolEventBus"
    module: "omnibase_core.protocol.protocol_event_bus"
    
  # Shared model dependencies (examples)
  - name: "model_postgres_connection_config"
    type: "model"
    class_name: "ModelPostgresConnectionConfig"
    module: "omnibase_infra.models.postgres.model_postgres_connection_config"
  - name: "model_kafka_producer_config"
    type: "model"
    class_name: "ModelKafkaProducerConfig"
    module: "omnibase_infra.models.kafka.model_kafka_producer_config"
  - name: "model_consul_service_config"
    type: "model"
    class_name: "ModelConsulServiceConfig"
    module: "omnibase_infra.models.consul.model_consul_service_config"
```

### Node Implementation Standards

All infrastructure nodes follow:

```python
class NodePostgresConnectionManagerEffect(NodeEffectService):
    def __init__(self, container: ONEXContainer):
        # Container injection pattern
        contract_path = Path(__file__).parent / "contract.yaml"
        super().__init__(container, contract_path)
        
        # Protocol-based dependency resolution (duck typing)
        self.event_bus = container.get_service("ProtocolEventBus")
        self.environment_config = container.get_service("EnvironmentConfig")

class NodeKafkaProducerPoolEffect(NodeEffectService):
    def __init__(self, container: ONEXContainer):
        # Container injection pattern
        contract_path = Path(__file__).parent / "contract.yaml"
        super().__init__(container, contract_path)
        
        # Protocol-based dependency resolution (duck typing)
        self.event_bus = container.get_service("ProtocolEventBus")
        self.postgres_manager = container.get_service("NodePostgresConnectionManagerEffect")
```

---

## ⚠️ Migration Risks & Mitigations

### High-Risk Areas

1. **Global State Dependencies**
   - **Risk:** Breaking existing singleton patterns
   - **Mitigation:** Phased migration with backward compatibility shims

2. **Performance Impact**
   - **Risk:** Additional abstraction layers affecting performance
   - **Mitigation:** Performance benchmarks and optimization

3. **Configuration Changes**
   - **Risk:** Breaking existing environment configuration
   - **Mitigation:** Maintain environment variable compatibility

### Validation Strategy

1. **Unit Testing:** All nodes have comprehensive unit tests
2. **Integration Testing:** End-to-end event publishing and processing
3. **Performance Testing:** Benchmarks for all critical paths
4. **Load Testing:** Producer pool performance under high load

---

## 🎯 Success Metrics

### Technical Metrics

- **ONEX Compliance:** 100% contract-driven infrastructure nodes
- **Performance:** No degradation in event publishing throughput
- **Reliability:** Circuit breaker functionality fully preserved
- **Observability:** All monitoring and metrics collection maintained

### Architecture Quality

- **Dependency Injection:** All services use container.get_service()
- **Duck Typing:** Zero isinstance() usage in infrastructure code
- **Strong Typing:** Zero Any type usage
- **Contract Coverage:** All infrastructure logic defined in contracts

### Migration Completeness

- **File Migration:** All identified files migrated to proper patterns
- **Legacy Cleanup:** `src/omnibase_infra/infrastructure/` directory cleaned up
- **Documentation:** Complete architectural documentation updated
- **Testing:** 100% test coverage for all migrated components

---

## 📝 Next Steps

1. **Immediate:** Begin Phase 1 migration of `postgres_connection_manager.py` and `kafka_producer_pool.py` (both CRITICAL)
2. **Short-term:** Complete high-priority migrations (`event_bus_circuit_breaker.py`)
3. **Medium-term:** Address observability and monitoring nodes (`distributed_tracing.py`, `infrastructure_health_monitor.py`)
4. **Long-term:** Complete advanced observability (`infrastructure_observability.py`)
5. **Cleanup:** Rename existing nodes to follow proper `node_` prefix convention

This migration plan ensures ONEX compliance while maintaining all existing functionality and performance characteristics. The corrected assessment recognizes `postgres_connection_manager.py` as a CRITICAL EFFECT node requiring immediate migration, not a utility. The phased approach minimizes risk while providing clear milestones and success criteria.