> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Declarative Effect Nodes Plan

# Effect Nodes Plan for omnibase_infra

**Created**: December 2, 2025
**Status**: Planning
**Dependencies**: MVP_EXECUTION_PLAN.md, EFFECT_NODES_SPEC.md (omniintelligence)
**Target**: omnibase-core ^0.4.0 (post contract-driven effect release)

---

## Executive Summary

This document outlines the integration of contract-driven effect nodes into `omnibase_infra`, building on the existing MVP execution plan. The contract-driven pattern will eliminate boilerplate code, standardize resilience patterns, and enable runtime configuration changes without code deployment.

**Naming Convention Change (December 2025)**: Contract-driven nodes are now the DEFAULT implementation and use standard names (no suffix). Previous imperative implementations are renamed with a `Legacy` suffix during the migration period.

**Key Benefits**:
- Reduce legacy imperative effect node code by ~60%
- Standardize retry, circuit breaker, and DLQ patterns
- Enable infrastructure-as-code for all effect nodes
- Simplify testing through protocol handler mocking

---

## Naming Convention

### Standard Names (Contract-Driven - DEFAULT)

Contract-driven effect nodes use standard names without any suffix. These are the default implementation:

| Class Name | Description |
|------------|-------------|
| `NodeEffect` | Base class for contract-driven effect nodes |
| `NodePostgresAdapter` | PostgreSQL adapter (contract-driven) |
| `NodeKafkaAdapter` | Kafka adapter (contract-driven) |
| `NodeConsulAdapter` | Consul adapter (contract-driven) |
| `NodeVaultAdapterEffect` | Vault adapter (contract-driven) |
| `NodeKeycloakAdapterEffect` | Keycloak adapter (contract-driven) |

### Legacy Names (Imperative - Deprecated)

Previous imperative implementations are renamed with a `Legacy` suffix during the migration period:

| Legacy Class Name | Original Name | Deprecation Timeline |
|-------------------|---------------|----------------------|
| `NodeEffectLegacy` | `NodeEffect` | Remove in v0.5.0 |
| `NodePostgresAdapterLegacy` | `NodePostgresAdapter` | Remove in v0.5.0 |
| `NodeKafkaAdapterLegacy` | `NodeKafkaAdapter` | Remove in v0.5.0 |
| `NodeConsulAdapterLegacy` | `NodeConsulAdapter` | Remove in v0.5.0 |
| `NodeVaultAdapterEffectLegacy` | `NodeVaultAdapterEffect` | Remove in v0.5.0 |
| `NodeKeycloakAdapterEffectLegacy` | `NodeKeycloakAdapterEffect` | Remove in v0.5.0 |

### Migration Path

1. **Phase 1 (v0.4.0)**: Legacy suffix added to imperative nodes, contract-driven nodes take standard names
2. **Phase 2 (v0.4.x)**: Deprecation warnings on Legacy imports
3. **Phase 3 (v0.5.0)**: Legacy classes removed entirely

### Import Examples

```python
# NEW (v0.4.0+): Contract-driven effect nodes (DEFAULT)
from omnibase_core.nodes import NodeEffect  # Base class
from omnibase_infra.nodes import NodePostgresAdapter  # Contract-driven

# DEPRECATED (v0.4.0-v0.4.x): Legacy imperative nodes
from omnibase_core.nodes import NodeEffectLegacy  # Will be removed in v0.5.0
from omnibase_infra.nodes import NodePostgresAdapterLegacy  # Will be removed in v0.5.0
```

---

## Core Design Invariants

These invariants apply to all contract-driven effect nodes:

1. **All behavior is contract-driven** - YAML contracts define everything
2. **NodeRuntime is the only executable event loop** - Effect nodes don't run their own loops
3. **Node logic is pure: no I/O, no mixins, no inheritance**
4. **Core never depends on SPI or infra** - Dependency: infra -> spi -> core
5. **SPI only defines protocols, never implementations**
6. **Infra owns all I/O and real system integrations** - Handlers live in infra

**Critical Invariant**:
```
No code in omnibase_core may initiate network I/O, database I/O,
file I/O, or external process execution.
```

This means:
- Handler implementations (HTTP, Postgres, Kafka, Bolt) MUST be in omnibase_infra
- Protocol interfaces are defined in omnibase_spi
- NodeRuntime and NodeInstance are in omnibase_core

---

## Handler Placement Rule

**Rule**: Any code that touches the network, filesystem, sockets, or external systems
MUST live in `omnibase_infra`. Core and SPI must remain pure and dependency-free.

**Correct Structure**:
```
omnibase_infra/
    handlers/
        http/
            http_rest_handler.py
            http_retry_policy.py
            http_circuit_breaker.py
        db/
            postgres_handler.py
            connection_pool.py
        graph/
            bolt_handler.py
        event/
            kafka_handler.py
    resilience/
        retry_policy.py
        circuit_breaker.py
        rate_limiter.py
    runtime_host/
        entrypoint.py
        wiring.py
```

---

## Integration with Runtime Host Model

Contract-driven effect nodes are designed to work with the Runtime Host model:

```
RuntimeHostProcess (omnibase_infra)
    +-- NodeRuntime (omnibase_core)
          |-- NodeInstance(contract-driven effect 1)
          |-- NodeInstance(contract-driven effect 2)
          +-- handlers:
                 postgres_handler (infra)
                 kafka_handler (infra)
                 http_rest_handler (infra)
```

**Key Points**:
- Contract-driven effects are loaded as NodeInstances
- Handlers are registered with the runtime at startup
- Effect contracts declare which handler types they need
- Runtime routes operations to appropriate handlers

---

## 1. Integration with Existing MVP Plan

### 1.1 Timeline Integration

The contract-driven effect work slots into the MVP as a **Phase 2.5** enhancement:

| Phase | Description | Status | Contract-Driven Impact |
|-------|-------------|--------|------------------------|
| Phase 0 | Pre-flight | Existing | No change |
| Phase 1 | Foundation | Existing | No change |
| Phase 2 | Effect Nodes (Legacy) | Existing | **Baseline, renamed to Legacy** |
| **Phase 2.5** | **Contract-Driven Effect Migration** | **NEW** | **New nodes take standard names** |
| Phase 3 | Stamping Service | Existing | Can use contract-driven Postgres |
| Phase 4 | Contract-Driven Nodes (Reducer/Orchestrator) | Existing | Foundation for effect contracts |
| Phase 5 | Compute Nodes | Existing | No change |
| Phase 6 | Testing & Validation | Existing | Extended for contracts |

### 1.2 Dependency Chain

```
omnibase-spi ^0.2.0 (protocol definitions)
       |
       +-- ProtocolHandler protocol (abstract, no I/O)
       +-- Protocol contracts and interfaces
       |
       v
omnibase-core ^0.4.0 (contract-driven effects release)
       |
       +-- NodeEffect base class (contract-driven, DEFAULT)
       +-- NodeEffectLegacy base class (imperative, DEPRECATED)
       +-- NodeRuntime (event loop, pure orchestration)
       +-- NodeInstance (pure node logic)
       +-- Effect contract JSON schema
       |
       v
omnibase_infra ^0.2.0 (post-migration)
       |
       +-- YAML effect contracts
       +-- Protocol handlers (HTTP, Bolt, Postgres, Kafka) - ALL I/O HERE
       +-- ProtocolHandlerRegistry (manages handler lifecycle)
       +-- Resilience utilities (RetryPolicy, CircuitBreaker)
       +-- RuntimeHostProcess (wiring and entrypoint)
       +-- NodePostgresAdapter, NodeKafkaAdapter, etc. (contract-driven)
       +-- NodePostgresAdapterLegacy, etc. (deprecated, for fallback)
       +-- Infrastructure-specific operations
```

### 1.3 Parallel vs Sequential Work

**Can Run in Parallel with MVP**:
- Contract YAML design (no code dependency)
- Protocol handler implementation in omnibase_infra
- Documentation and examples

**Must Wait for omnibase-core 0.4.0**:
- Actual node conversion
- Integration testing
- Runtime validation

---

## 2. Effect Nodes Inventory

### 2.1 Current Effect Nodes in omnibase_infra

| Node | Location | Protocol | Priority | Effort |
|------|----------|----------|----------|--------|
| `postgres_adapter` | `nodes/postgres_adapter/v1_0_0/` | postgres | P0 | Medium |
| `kafka_adapter` | `nodes/kafka_adapter/v1_0_0/` | kafka | P0 | Medium |
| `consul_adapter` | `nodes/consul_adapter/v1_0_0/` | http_rest | P1 | Low |
| `node_vault_adapter_effect` | `nodes/node_vault_adapter_effect/v1_0_0/` | http_rest | P1 | Low |
| `node_keycloak_adapter_effect` | `nodes/node_keycloak_adapter_effect/v1_0_0/` | http_rest | P2 | Medium |
| `hook_node` (webhook) | `nodes/hook_node/v1_0_0/` | http_rest | P2 | Low |
| `node_consul_projector_effect` | `nodes/node_consul_projector_effect/v1_0_0/` | http_rest | P3 | Low |

### 2.2 Priority Justification

**P0 - Foundation (Week 1)**:
- **Postgres**: Workflow state storage, FSM transitions - core to all operations
- **Kafka**: Event publishing - enables loose coupling between services

**P1 - Service Integration (Week 2)**:
- **Consul**: Service discovery, health checks - required for deployment
- **Vault**: Secret management - security critical but can use env vars initially

**P2 - Extended Features (Week 3)**:
- **Keycloak**: Authentication flows - can defer to later if using API keys
- **Webhook**: Notification delivery - enhancement, not critical path

**P3 - Projections (Week 4)**:
- **Consul Projector**: Read-model projection - optimization, not core

### 2.3 Effort Estimation

| Effort Level | Hours | Description |
|--------------|-------|-------------|
| Low | 2-4h | Simple HTTP REST, 3-5 operations |
| Medium | 4-8h | Complex protocol, many operations, custom validation |
| High | 8-16h | New protocol handler, extensive testing |

---

## 3. YAML Contracts to Create

### 3.1 Contract Directory Structure

```
omnibase_infra/
├── contracts/
│   └── effects/
│       ├── _schema/
│       │   └── effect_contract_schema.json    # JSON Schema for validation
│       ├── postgres_workflow.yaml              # Workflow state persistence
│       ├── postgres_fsm.yaml                   # FSM transition storage
│       ├── kafka_event.yaml                    # Event publishing
│       ├── consul_registry.yaml                # Service discovery
│       ├── consul_kv.yaml                      # KV store operations
│       ├── vault_secret.yaml                   # Secret management
│       ├── keycloak_auth.yaml                  # Authentication
│       ├── webhook_delivery.yaml               # Webhook notifications
│       └── valkey_cache.yaml                   # Caching (future)
└── src/omnibase_infra/
    ├── handlers/                               # At package root, NOT under nodes
    │   ├── __init__.py
    │   ├── http/
    │   │   ├── http_rest_handler.py            # HTTP REST protocol handler
    │   │   ├── http_retry_policy.py            # HTTP-specific retry logic
    │   │   └── http_circuit_breaker.py         # HTTP circuit breaker
    │   ├── db/
    │   │   ├── postgres_handler.py             # PostgreSQL protocol handler
    │   │   └── connection_pool.py              # Connection pool management
    │   ├── graph/
    │   │   └── bolt_handler.py                 # Neo4j Bolt protocol handler
    │   ├── event/
    │   │   └── kafka_handler.py                # Kafka protocol handler
    │   └── cache/
    │       └── valkey_handler.py               # Valkey (Redis-compatible) handler
    ├── resilience/                             # Shared resilience patterns
    │   ├── __init__.py
    │   ├── retry_policy.py                     # Generic retry policy
    │   ├── circuit_breaker.py                  # Generic circuit breaker
    │   └── rate_limiter.py                     # Rate limiting utilities
    ├── runtime_host/                           # Runtime host wiring
    │   ├── entrypoint.py                       # Process entrypoint
    │   └── wiring.py                           # Handler registration
    └── nodes/                                  # Nodes are separate from handlers
        ├── effect_nodes/                       # Contract-driven effect nodes (DEFAULT)
        │   ├── __init__.py
        │   └── loader.py                       # Contract loader utility
        └── effect_nodes_legacy/                # Legacy imperative nodes (DEPRECATED)
            ├── __init__.py
            └── ...                             # To be removed in v0.5.0
```

**Important**: Handlers live at the package root (`src/omnibase_infra/handlers/`), NOT under
`nodes/`. This follows the architectural invariant that handlers contain I/O code which must
remain separate from pure node logic.

### 3.2 postgres_workflow.yaml

```yaml
# Workflow state persistence for infrastructure orchestration
name: postgres_workflow_effect
version:
  major: 1
  minor: 0
  patch: 0

description: |
  PostgreSQL effect for workflow state persistence.
  Stores workflow execution states, checkpoints, and audit logs.

protocol:
  type: postgres
  version: "14"

connection:
  host: ${POSTGRES_HOST}
  port: ${POSTGRES_PORT}
  database: ${POSTGRES_DB}
  pool:
    min_size: 2
    max_size: 10
    max_idle_time_ms: 300000
  timeout_ms: 30000
  tls:
    enabled: ${POSTGRES_TLS_ENABLED}
    verify: true

authentication:
  type: basic
  basic:
    username: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}

operations:
  save_workflow_state:
    description: "Persist workflow execution state"
    request:
      sql: |
        INSERT INTO workflow_states (
          workflow_id, execution_id, state, payload, checkpoint, created_at
        ) VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (workflow_id, execution_id)
        DO UPDATE SET state = $3, payload = $4, checkpoint = $5, updated_at = NOW()
        RETURNING id, workflow_id, execution_id, state
      sql_params:
        - ${input.workflow_id}
        - ${input.execution_id}
        - ${input.state}
        - ${input.payload}
        - ${input.checkpoint}
    response:
      mapping:
        record_id: "$.rows[0].id"
        workflow_id: "$.rows[0].workflow_id"
    validation:
      required_fields:
        - workflow_id
        - execution_id
        - state

  get_workflow_state:
    description: "Retrieve workflow state by execution ID"
    request:
      sql: |
        SELECT id, workflow_id, execution_id, state, payload, checkpoint, created_at, updated_at
        FROM workflow_states
        WHERE workflow_id = $1 AND execution_id = $2
      sql_params:
        - ${input.workflow_id}
        - ${input.execution_id}
    response:
      mapping:
        state: "$.rows[0].state"
        payload: "$.rows[0].payload"
        checkpoint: "$.rows[0].checkpoint"

  list_workflow_history:
    description: "List workflow execution history"
    request:
      sql: |
        SELECT id, workflow_id, execution_id, state, created_at, updated_at
        FROM workflow_states
        WHERE workflow_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
      sql_params:
        - ${input.workflow_id}
        - ${input.limit}
        - ${input.offset}
    response:
      mapping:
        executions: "$.rows"
        count: "$.row_count"

  delete_workflow_state:
    description: "Delete workflow state (for cleanup)"
    request:
      sql: |
        DELETE FROM workflow_states
        WHERE workflow_id = $1 AND execution_id = $2
        RETURNING id
      sql_params:
        - ${input.workflow_id}
        - ${input.execution_id}
    response:
      mapping:
        deleted: "$.affected_rows"

resilience:
  retry:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 100
    max_delay_ms: 2000
    backoff_multiplier: 2.0
    jitter: true
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    success_threshold: 2
    timeout_ms: 30000
  timeout:
    request_ms: 5000
    operation_ms: 30000

events:
  consume:
    topic: dev.omnibase-infra.effect.postgres-workflow.request.v1
    group_id: postgres-workflow-effect-consumer
  produce:
    success_topic: dev.omnibase-infra.effect.postgres-workflow.response.v1
    failure_topic: dev.omnibase-infra.effect.postgres-workflow.failure.v1
    dlq_topic: dev.omnibase-infra.effect.postgres-workflow.request.v1.dlq

observability:
  metrics:
    enabled: true
    prefix: omnibase_infra_postgres_workflow
    labels:
      service: omnibase_infra
      component: postgres_workflow_effect
  logging:
    level: INFO
    sanitize_secrets: true
    secret_patterns:
      - password
      - secret
      - token

metadata:
  author: OmniInfra Team
  created_at: "2025-12-02"
  tags:
    - infrastructure
    - persistence
    - workflow
  documentation: https://docs.omninode.ai/infra/effects/postgres-workflow
```

### 3.3 postgres_fsm.yaml

```yaml
# FSM state transition storage
name: postgres_fsm_effect
version:
  major: 1
  minor: 0
  patch: 0

description: |
  PostgreSQL effect for FSM state transition persistence.
  Stores state machine states, transitions, and audit history.

protocol:
  type: postgres
  version: "14"

connection:
  host: ${POSTGRES_HOST}
  port: ${POSTGRES_PORT}
  database: ${POSTGRES_DB}
  pool:
    min_size: 2
    max_size: 10
  timeout_ms: 30000

authentication:
  type: basic
  basic:
    username: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}

operations:
  record_transition:
    description: "Record an FSM state transition"
    request:
      sql: |
        INSERT INTO fsm_transitions (
          entity_id, entity_type, fsm_type, from_state, to_state,
          event, context, correlation_id, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        RETURNING id, entity_id, from_state, to_state
      sql_params:
        - ${input.entity_id}
        - ${input.entity_type}
        - ${input.fsm_type}
        - ${input.from_state}
        - ${input.to_state}
        - ${input.event}
        - ${input.context}
        - ${context.correlation_id}
    response:
      mapping:
        transition_id: "$.rows[0].id"
        entity_id: "$.rows[0].entity_id"
    validation:
      required_fields:
        - entity_id
        - entity_type
        - fsm_type
        - from_state
        - to_state
        - event

  get_current_state:
    description: "Get current FSM state for an entity"
    request:
      sql: |
        SELECT to_state as current_state, event as last_event, created_at as last_transition
        FROM fsm_transitions
        WHERE entity_id = $1 AND entity_type = $2 AND fsm_type = $3
        ORDER BY created_at DESC
        LIMIT 1
      sql_params:
        - ${input.entity_id}
        - ${input.entity_type}
        - ${input.fsm_type}
    response:
      mapping:
        current_state: "$.rows[0].current_state"
        last_event: "$.rows[0].last_event"
        last_transition: "$.rows[0].last_transition"

  get_transition_history:
    description: "Get FSM transition history for an entity"
    request:
      sql: |
        SELECT id, from_state, to_state, event, context, correlation_id, created_at
        FROM fsm_transitions
        WHERE entity_id = $1 AND entity_type = $2 AND fsm_type = $3
        ORDER BY created_at DESC
        LIMIT $4 OFFSET $5
      sql_params:
        - ${input.entity_id}
        - ${input.entity_type}
        - ${input.fsm_type}
        - ${input.limit}
        - ${input.offset}
    response:
      mapping:
        transitions: "$.rows"
        count: "$.row_count"

resilience:
  retry:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 100
    max_delay_ms: 2000
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    timeout_ms: 30000

events:
  consume:
    topic: dev.omnibase-infra.effect.postgres-fsm.request.v1
    group_id: postgres-fsm-effect-consumer
  produce:
    success_topic: dev.omnibase-infra.effect.postgres-fsm.response.v1
    dlq_topic: dev.omnibase-infra.effect.postgres-fsm.request.v1.dlq

observability:
  metrics:
    enabled: true
    prefix: omnibase_infra_postgres_fsm
  logging:
    level: INFO
    sanitize_secrets: true

metadata:
  author: OmniInfra Team
  created_at: "2025-12-02"
  tags:
    - infrastructure
    - fsm
    - state-management
```

### 3.4 kafka_event.yaml

```yaml
# Kafka event publishing for infrastructure events
name: kafka_event_effect
version:
  major: 1
  minor: 0
  patch: 0

description: |
  Kafka effect for event publishing.
  Publishes infrastructure events with delivery confirmation.

protocol:
  type: kafka
  version: "3.0"

connection:
  url: ${KAFKA_BOOTSTRAP_SERVERS}
  timeout_ms: 30000
  pool:
    max_size: 5

authentication:
  type: none  # SASL config can be added if needed

operations:
  publish_event:
    description: "Publish an event to a Kafka topic"
    request:
      topic: ${input.topic}
      key: ${input.key}
      payload: ${input.payload}
    response:
      mapping:
        partition: "$.partition"
        offset: "$.offset"
        topic: "$.topic"
    validation:
      required_fields:
        - topic
        - payload

  publish_batch:
    description: "Publish multiple events to a topic"
    request:
      topic: ${input.topic}
      messages: ${input.messages}
    response:
      mapping:
        published_count: "$.count"
        partitions: "$.partitions"
    validation:
      required_fields:
        - topic
        - messages

  publish_workflow_event:
    description: "Publish a workflow lifecycle event"
    request:
      topic: dev.omnibase-infra.workflow.${input.event_type}.v1
      key: ${input.workflow_id}
      payload:
        event_id: ${context.event_id}
        event_type: ${input.event_type}
        workflow_id: ${input.workflow_id}
        execution_id: ${input.execution_id}
        state: ${input.state}
        timestamp: ${context.timestamp}
        correlation_id: ${context.correlation_id}
    response:
      mapping:
        partition: "$.partition"
        offset: "$.offset"

  publish_fsm_transition:
    description: "Publish an FSM transition event"
    request:
      topic: dev.omnibase-infra.fsm.transition.v1
      key: ${input.entity_id}
      payload:
        event_id: ${context.event_id}
        entity_id: ${input.entity_id}
        entity_type: ${input.entity_type}
        fsm_type: ${input.fsm_type}
        from_state: ${input.from_state}
        to_state: ${input.to_state}
        event: ${input.event}
        timestamp: ${context.timestamp}
        correlation_id: ${context.correlation_id}
    response:
      mapping:
        partition: "$.partition"
        offset: "$.offset"

resilience:
  retry:
    enabled: true
    max_attempts: 5
    initial_delay_ms: 500
    max_delay_ms: 10000
    backoff_multiplier: 2.0
  circuit_breaker:
    enabled: true
    failure_threshold: 10
    success_threshold: 3
    timeout_ms: 60000
  timeout:
    request_ms: 10000
    operation_ms: 60000

observability:
  metrics:
    enabled: true
    prefix: omnibase_infra_kafka_event
    labels:
      service: omnibase_infra
  logging:
    level: INFO
    include_request_body: false

metadata:
  author: OmniInfra Team
  created_at: "2025-12-02"
  tags:
    - infrastructure
    - events
    - kafka
```

### 3.5 consul_registry.yaml

```yaml
# Consul service registry operations
name: consul_registry_effect
version:
  major: 1
  minor: 0
  patch: 0

description: |
  Consul effect for service registration and discovery.
  Manages service lifecycle in the Consul catalog.

protocol:
  type: http_rest
  version: "HTTP/1.1"
  content_type: application/json

connection:
  url: http://${CONSUL_HOST}:${CONSUL_PORT}
  timeout_ms: 10000
  pool:
    max_size: 5

authentication:
  type: api_key
  api_key:
    header: X-Consul-Token
    prefix: ""
    value: ${CONSUL_ACL_TOKEN}

operations:
  register_service:
    description: "Register a service with Consul"
    request:
      method: PUT
      path: /v1/agent/service/register
      body:
        ID: ${input.service_id}
        Name: ${input.service_name}
        Tags: ${input.tags}
        Address: ${input.address}
        Port: ${input.port}
        Check:
          HTTP: http://${input.address}:${input.port}${input.health_path}
          Interval: ${input.check_interval}
          Timeout: ${input.check_timeout}
    response:
      success_codes: [200]
      mapping:
        registered: "true"
    validation:
      required_fields:
        - service_id
        - service_name
        - address
        - port

  deregister_service:
    description: "Deregister a service from Consul"
    request:
      method: PUT
      path: /v1/agent/service/deregister/${input.service_id}
    response:
      success_codes: [200]
      mapping:
        deregistered: "true"
    validation:
      required_fields:
        - service_id

  get_service:
    description: "Get service details by ID"
    request:
      method: GET
      path: /v1/catalog/service/${input.service_name}
    response:
      success_codes: [200]
      mapping:
        services: "$[*]"
        count: "$.length"

  list_services:
    description: "List all registered services"
    request:
      method: GET
      path: /v1/catalog/services
    response:
      success_codes: [200]
      mapping:
        services: "$"

  health_check:
    description: "Get health status of a service"
    request:
      method: GET
      path: /v1/health/service/${input.service_name}
      query:
        passing: ${input.passing_only}
    response:
      success_codes: [200]
      mapping:
        instances: "$[*]"
        healthy_count: "$.length"

resilience:
  retry:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 500
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    timeout_ms: 30000

events:
  consume:
    topic: dev.omnibase-infra.effect.consul-registry.request.v1
    group_id: consul-registry-effect-consumer
  produce:
    success_topic: dev.omnibase-infra.effect.consul-registry.response.v1
    dlq_topic: dev.omnibase-infra.effect.consul-registry.request.v1.dlq

observability:
  metrics:
    enabled: true
    prefix: omnibase_infra_consul_registry

metadata:
  author: OmniInfra Team
  created_at: "2025-12-02"
  tags:
    - infrastructure
    - service-discovery
    - consul
```

### 3.6 valkey_cache.yaml

```yaml
# Valkey (Redis-compatible) caching operations
name: valkey_cache_effect
version:
  major: 1
  minor: 0
  patch: 0

description: |
  Valkey effect for caching operations.
  Provides high-performance key-value caching with TTL support.

protocol:
  type: valkey  # Custom handler required
  version: "7.0"

connection:
  host: ${VALKEY_HOST}
  port: ${VALKEY_PORT}
  database: ${VALKEY_DB}
  pool:
    min_size: 2
    max_size: 20
  timeout_ms: 5000
  tls:
    enabled: ${VALKEY_TLS_ENABLED}

authentication:
  type: basic
  basic:
    password: ${VALKEY_PASSWORD}

operations:
  get:
    description: "Get a value by key"
    request:
      command: GET
      key: ${input.key}
    response:
      mapping:
        value: "$.value"
        exists: "$.exists"
    validation:
      required_fields:
        - key

  set:
    description: "Set a value with optional TTL"
    request:
      command: SET
      key: ${input.key}
      value: ${input.value}
      ttl_seconds: ${input.ttl}
    response:
      mapping:
        success: "$.ok"
    validation:
      required_fields:
        - key
        - value

  delete:
    description: "Delete a key"
    request:
      command: DEL
      key: ${input.key}
    response:
      mapping:
        deleted: "$.count"

  get_many:
    description: "Get multiple values by keys"
    request:
      command: MGET
      keys: ${input.keys}
    response:
      mapping:
        values: "$.values"

  set_many:
    description: "Set multiple key-value pairs"
    request:
      command: MSET
      pairs: ${input.pairs}
    response:
      mapping:
        success: "$.ok"

  increment:
    description: "Increment a numeric value"
    request:
      command: INCR
      key: ${input.key}
      amount: ${input.amount}
    response:
      mapping:
        new_value: "$.value"

  expire:
    description: "Set TTL on an existing key"
    request:
      command: EXPIRE
      key: ${input.key}
      ttl_seconds: ${input.ttl}
    response:
      mapping:
        success: "$.ok"

resilience:
  retry:
    enabled: true
    max_attempts: 2
    initial_delay_ms: 50
    max_delay_ms: 500
  circuit_breaker:
    enabled: true
    failure_threshold: 10
    timeout_ms: 10000
  timeout:
    request_ms: 1000
    operation_ms: 5000

events:
  consume:
    topic: dev.omnibase-infra.effect.valkey-cache.request.v1
    group_id: valkey-cache-effect-consumer
  produce:
    success_topic: dev.omnibase-infra.effect.valkey-cache.response.v1
    dlq_topic: dev.omnibase-infra.effect.valkey-cache.request.v1.dlq

observability:
  metrics:
    enabled: true
    prefix: omnibase_infra_valkey_cache
    histograms:
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
  logging:
    level: DEBUG
    include_request_body: false

metadata:
  author: OmniInfra Team
  created_at: "2025-12-02"
  tags:
    - infrastructure
    - caching
    - valkey
  dependencies:
    - valkey_handler  # Custom protocol handler
```

---

## 4. Integration with omnibase_core

### 4.1 Importing NodeEffect (Contract-Driven)

```python
"""Example of using contract-driven effect nodes in omnibase_infra."""

from pathlib import Path
from omnibase_core.nodes import NodeEffect  # Contract-driven base class (DEFAULT)
from omnibase_core.models import ModelEffectInput

# Path to contracts directory
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts" / "effects"

async def create_postgres_workflow_effect() -> NodeEffect:
    """Create and initialize Postgres workflow effect node."""
    node = NodeEffect(
        contract_path=CONTRACTS_DIR / "postgres_workflow.yaml",
        config_overrides={
            # Override env vars if needed for testing
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
        }
    )
    await node.initialize()
    return node

async def save_workflow_state(
    node: NodeEffect,
    workflow_id: str,
    execution_id: str,
    state: str,
    payload: dict
) -> dict:
    """Save workflow state using contract-driven effect."""
    result = await node.execute_effect(
        ModelEffectInput(
            operation="save_workflow_state",
            params={
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "state": state,
                "payload": payload,
                "checkpoint": None,
            }
        )
    )

    if not result.success:
        raise RuntimeError(f"Failed to save workflow state: {result.error}")

    return result.data
```

### 4.1.1 Legacy Fallback (Deprecated)

For migration purposes, you can still use the legacy imperative nodes:

```python
"""Legacy imperative nodes - DEPRECATED, will be removed in v0.5.0."""

import warnings
from omnibase_core.nodes import NodeEffectLegacy  # DEPRECATED

# This will emit a deprecation warning
warnings.warn(
    "NodeEffectLegacy is deprecated and will be removed in v0.5.0. "
    "Use NodeEffect with YAML contracts instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 4.2 Registering Custom Protocol Handlers

**Architectural Note**: All handler implementations live in `omnibase_infra`, never in
`omnibase_core`. Core only contains pure logic (NodeRuntime, NodeInstance). Protocol
interfaces (abstract base classes) are defined in `omnibase_spi`.

For custom protocols (e.g., Valkey):

```python
"""Custom Valkey protocol handler for omnibase_infra.

Location: src/omnibase_infra/handlers/cache/valkey_handler.py

This handler implements the ProtocolHandler protocol from omnibase_spi
and provides the actual I/O operations for Valkey/Redis.
"""

from typing import Any
# Protocol interface from SPI (abstract, no I/O)
from omnibase_spi.protocols import ProtocolHandler
# Models can be shared or defined in infra
from omnibase_infra.handlers.models import ModelProtocolRequest, ModelProtocolResponse
# Handler registry lives in infra (manages I/O components)
from omnibase_infra.handlers.registry import ProtocolHandlerRegistry
import valkey  # or redis-py with valkey compatibility

class ValkeyHandler(ProtocolHandler):
    """Protocol handler for Valkey (Redis-compatible) operations."""

    def __init__(self):
        self.client: valkey.Redis | None = None

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize Valkey connection pool."""
        self.client = valkey.Redis(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("database", 0),
            password=config.get("authentication", {}).get("basic", {}).get("password"),
            decode_responses=True,
            max_connections=config.get("pool", {}).get("max_size", 10),
        )

    async def shutdown(self) -> None:
        """Close Valkey connection."""
        if self.client:
            await self.client.close()

    async def execute(
        self,
        request: ModelProtocolRequest,
        operation_config: dict[str, Any]
    ) -> ModelProtocolResponse:
        """Execute Valkey command."""
        import time
        start_time = time.perf_counter()

        try:
            req_config = operation_config.get("request", {})
            command = req_config.get("command", "").upper()

            # Route to appropriate Valkey command
            if command == "GET":
                value = await self.client.get(request.params.get("input", {}).get("key"))
                data = {"value": value, "exists": value is not None}
            elif command == "SET":
                params = request.params.get("input", {})
                ttl = params.get("ttl")
                if ttl:
                    result = await self.client.setex(params["key"], ttl, params["value"])
                else:
                    result = await self.client.set(params["key"], params["value"])
                data = {"ok": result}
            elif command == "DEL":
                count = await self.client.delete(request.params.get("input", {}).get("key"))
                data = {"count": count}
            elif command == "MGET":
                keys = request.params.get("input", {}).get("keys", [])
                values = await self.client.mget(keys)
                data = {"values": dict(zip(keys, values))}
            elif command == "INCR":
                params = request.params.get("input", {})
                amount = params.get("amount", 1)
                if amount == 1:
                    value = await self.client.incr(params["key"])
                else:
                    value = await self.client.incrby(params["key"], amount)
                data = {"value": value}
            else:
                raise ValueError(f"Unsupported Valkey command: {command}")

            duration_ms = (time.perf_counter() - start_time) * 1000

            return ModelProtocolResponse(
                success=True,
                data=data,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelProtocolResponse(
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )

    async def health_check(self) -> bool:
        """Check Valkey connectivity."""
        try:
            return await self.client.ping()
        except Exception:
            return False

# Register custom handler at module load
def register_custom_handlers():
    """Register omnibase_infra custom protocol handlers."""
    ProtocolHandlerRegistry.register("valkey", ValkeyHandler)
```

### 4.3 Version Pinning Strategy

In `pyproject.toml`:

```toml
[project]
dependencies = [
    # Pin to specific minor version for stability
    "omnibase-core>=0.4.0,<0.5.0",
    # Allow patch updates within 0.2.x
    "omnibase-spi>=0.2.0,<0.3.0",
]
```

**Upgrade Policy**:
1. **Patch versions** (0.4.x): Auto-update, run tests
2. **Minor versions** (0.5.0): Review changelog, test in staging
3. **Major versions** (1.0.0): Full compatibility review, migration plan

---

## 5. Migration Path

### 5.1 Phased Migration Strategy

```
Phase A: Naming Transition (Week 1)
├── Rename current imperative nodes to *Legacy suffix
├── Create YAML contracts for new contract-driven nodes
├── New contract-driven nodes take standard names (no suffix)
├── Feature flag: USE_CONTRACT_DRIVEN_EFFECTS=false
└── No production impact, import aliases maintained

Phase B: Parallel Implementation (Week 2)
├── Implement contract-driven nodes alongside Legacy code
├── Implement wrapper that can use either
├── Deprecation warnings on Legacy imports
└── Feature flag: USE_CONTRACT_DRIVEN_EFFECTS=false

Phase C: Shadow Mode (Week 3)
├── Run both implementations in parallel
├── Compare results for consistency
├── Log discrepancies
└── Feature flag: USE_CONTRACT_DRIVEN_EFFECTS=shadow

Phase D: Gradual Rollout (Week 4)
├── Enable contract-driven for non-critical paths
├── Monitor metrics and error rates
├── Quick rollback to Legacy capability
└── Feature flag: USE_CONTRACT_DRIVEN_EFFECTS=true (per-node)

Phase E: Full Migration (Week 5-6)
├── All nodes using contract-driven implementation
├── Legacy nodes deprecated but available
├── Plan removal for v0.5.0
└── Feature flag: USE_CONTRACT_DRIVEN_EFFECTS=true (default)

Phase F: Legacy Removal (v0.5.0)
├── Remove all *Legacy classes
├── Remove feature flags
├── Contract-driven is the only implementation
└── Clean up deprecated imports
```

### 5.1.1 Naming Transition Details

| Current Name | v0.4.0 Name | v0.5.0 Name |
|--------------|-------------|-------------|
| `NodeEffect` (imperative) | `NodeEffectLegacy` | REMOVED |
| N/A | `NodeEffect` (contract-driven) | `NodeEffect` (contract-driven) |
| `NodePostgresAdapter` (imperative) | `NodePostgresAdapterLegacy` | REMOVED |
| N/A | `NodePostgresAdapter` (contract-driven) | `NodePostgresAdapter` (contract-driven) |
| `NodeKafkaAdapter` (imperative) | `NodeKafkaAdapterLegacy` | REMOVED |
| N/A | `NodeKafkaAdapter` (contract-driven) | `NodeKafkaAdapter` (contract-driven) |

### 5.2 Feature Flag Implementation

```python
"""Feature flags for contract-driven effect migration."""

import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.nodes import NodeEffect

class EffectNodeMode(str, Enum):
    """Migration mode for effect nodes."""
    LEGACY = "legacy"        # Use Legacy imperative only (deprecated)
    SHADOW = "shadow"        # Run both, compare, use Legacy
    CONTRACT = "contract"    # Use contract-driven only (DEFAULT in v0.4.0+)
    PER_NODE = "per_node"    # Check per-node flags

def get_effect_mode() -> EffectNodeMode:
    """Get current effect node mode from environment."""
    mode = os.getenv("USE_CONTRACT_DRIVEN_EFFECTS", "contract").lower()
    return EffectNodeMode(mode)

def should_use_contract_driven(node_name: str) -> bool:
    """Check if a specific node should use contract-driven implementation."""
    mode = get_effect_mode()

    if mode == EffectNodeMode.LEGACY:
        return False
    elif mode == EffectNodeMode.CONTRACT:
        return True
    elif mode == EffectNodeMode.PER_NODE:
        # Check node-specific flag
        flag = os.getenv(f"USE_CONTRACT_DRIVEN_{node_name.upper()}", "true")
        return flag.lower() == "true"
    elif mode == EffectNodeMode.SHADOW:
        # Shadow mode handled separately
        return False

    return True  # Default to contract-driven
```

### 5.3 Rollback Strategy

```python
"""Rollback utilities for contract-driven effect migration."""

import logging
from typing import TypeVar, Generic
from omnibase_core.nodes import NodeEffect, NodeEffectLegacy
from omnibase_core.models import ModelEffectInput

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=NodeEffectLegacy)

class EffectNodeFallback(Generic[T]):
    """
    Fallback wrapper that tries contract-driven first,
    falls back to Legacy imperative on failure.

    Note: This is a temporary migration utility. Once Legacy
    nodes are removed in v0.5.0, this class will be deprecated.
    """

    def __init__(
        self,
        contract_node: NodeEffect,
        legacy_fallback: T,
        fallback_threshold: int = 3,
    ):
        self.contract = contract_node
        self.legacy = legacy_fallback  # Deprecated, for fallback only
        self.threshold = fallback_threshold
        self.consecutive_failures = 0
        self._using_legacy = False

    async def execute(self, operation: str, params: dict) -> dict:
        """Execute with automatic fallback to Legacy."""
        if self._using_legacy:
            return await self._execute_legacy(operation, params)

        try:
            result = await self.contract.execute_effect(
                ModelEffectInput(operation=operation, params=params)
            )

            if result.success:
                self.consecutive_failures = 0
                return result.data
            else:
                raise RuntimeError(result.error)

        except Exception as e:
            self.consecutive_failures += 1
            logger.warning(
                f"Contract-driven effect failed | "
                f"operation={operation} | "
                f"failures={self.consecutive_failures} | "
                f"error={e}"
            )

            if self.consecutive_failures >= self.threshold:
                logger.error(
                    f"Switching to Legacy fallback | "
                    f"node={self.contract.contract.get('name')}"
                )
                self._using_legacy = True

            return await self._execute_legacy(operation, params)

    async def _execute_legacy(self, operation: str, params: dict) -> dict:
        """Execute using Legacy imperative implementation (deprecated)."""
        # Call the appropriate method on the Legacy node
        method = getattr(self.legacy, f"execute_{operation}", None)
        if method:
            return await method(**params)
        raise ValueError(f"Unknown operation: {operation}")

    def reset_fallback(self) -> None:
        """Reset to try contract-driven again."""
        self._using_legacy = False
        self.consecutive_failures = 0
```

---

## 6. Timeline Integration

### 6.1 Detailed Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| W1 | MVP Phase 2 | Complete Legacy effect nodes, rename with suffix | Working *Legacy adapters |
| W2 | MVP Phase 3-4 | Stamping service, contract-driven reducer | Stamping node, FSM contracts |
| W3 | **Phase 2.5a** | Design YAML contracts | 5 contract files, schema validation |
| W4 | **Phase 2.5b** | Wait for omnibase-core 0.4.0 | Dependency available (NodeEffect, NodeEffectLegacy) |
| W5 | **Phase 2.5c** | Implement contract-driven nodes | New nodes with standard names, feature flags |
| W6 | **Phase 2.5d** | Shadow mode testing | Comparison logs, discrepancy fixes |
| W7 | MVP Phase 6 | Full testing, validation | Test coverage, quality gates |
| W8 | Release | Production deployment | omnibase-infra 0.2.0 |
| Future | v0.5.0 | Remove Legacy nodes | Clean codebase, only contract-driven |

### 6.2 Dependencies and Blockers

**Hard Dependencies**:
1. omnibase-core 0.4.0 release with `NodeEffect` (contract-driven) and `NodeEffectLegacy`
2. Protocol handlers (HTTP, Bolt, Postgres, Kafka) implemented in omnibase_infra
3. `ProtocolHandler` protocol defined in omnibase_spi
4. `ProtocolHandlerRegistry` API stable (in omnibase_infra)

**Soft Dependencies**:
1. MVP Phase 2 Legacy effect nodes (for fallback baseline)
2. MVP Phase 4 FSM contracts (shared patterns)
3. Integration test infrastructure

**Potential Blockers**:
1. **omnibase-core 0.4.0 delay**: Proceed with contract design only
2. **Protocol handler gaps**: Implement custom handlers in infra
3. **Schema incompatibility**: Version contracts, maintain backward compat
4. **Legacy removal timing**: Ensure sufficient deprecation period before v0.5.0

### 6.3 Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Core release delay | Medium | High | Parallel contract design, fallback to Legacy |
| Contract schema changes | Medium | Medium | Version contracts, maintain v1 compatibility |
| Performance regression | Low | High | Shadow mode comparison, benchmarking |
| Protocol handler bugs | Medium | Medium | Custom handler overrides, rapid patching |
| Premature Legacy removal | Low | High | Clear deprecation timeline, warning logs |

---

## 7. Success Criteria

### 7.1 Technical Criteria

- [ ] All 5 core contracts (postgres x2, kafka, consul, valkey) validated
- [ ] Custom Valkey handler working
- [ ] Feature flag system operational
- [ ] Shadow mode comparison passing (>99% consistency)
- [ ] Performance within 5% of Legacy baseline
- [ ] All existing tests passing with contract-driven nodes
- [ ] Legacy nodes renamed and deprecation warnings active

### 7.2 Quality Criteria

- [ ] Contract schema validation passing
- [ ] mypy --strict on new code
- [ ] Test coverage >85% for contract-driven wrappers
- [ ] Documentation for contract authoring
- [ ] Runbook for rollback to Legacy procedures
- [ ] Clear deprecation timeline documented

### 7.3 Operational Criteria

- [ ] Metrics dashboards for contract-driven vs Legacy comparison
- [ ] Alerting for circuit breaker opens
- [ ] DLQ monitoring configured
- [ ] Health check endpoints including contract validation
- [ ] Deprecation warning logs for Legacy usage

---

## Appendix A: Contract Validation

```python
"""Utility for validating effect contracts."""

import json
import yaml
from pathlib import Path
from jsonschema import validate, ValidationError

SCHEMA_PATH = Path(__file__).parent / "_schema" / "effect_contract_schema.json"

def load_schema() -> dict:
    """Load the effect contract JSON schema."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)

def validate_contract(contract_path: Path) -> list[str]:
    """
    Validate an effect contract against the schema.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        with open(contract_path, encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        schema = load_schema()
        validate(instance=contract, schema=schema)

    except ValidationError as e:
        errors.append(f"Schema validation failed: {e.message}")
        errors.append(f"  Path: {'.'.join(str(p) for p in e.path)}")

    except yaml.YAMLError as e:
        errors.append(f"YAML parsing failed: {e}")

    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return errors

def validate_all_contracts(contracts_dir: Path) -> dict[str, list[str]]:
    """Validate all contracts in a directory."""
    results = {}

    for contract_file in contracts_dir.glob("*.yaml"):
        if contract_file.name.startswith("_"):
            continue  # Skip schema/internal files

        errors = validate_contract(contract_file)
        results[contract_file.name] = errors

    return results
```

---

## Appendix B: Quick Reference

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OMNIBASE_INFRA_DB_URL` | Full PostgreSQL DSN (primary) | (required) |
| `USE_CONTRACT_DRIVEN_EFFECTS` | Migration mode | `contract` |
| `USE_CONTRACT_DRIVEN_{NODE_NAME}` | Per-node override | `true` |
| `POSTGRES_HOST` | PostgreSQL host (fallback) | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port (fallback) | `5432` |
| `POSTGRES_DB` | Database name (fallback) | `omnibase_infra` |
| `POSTGRES_USER` | Database user (fallback) | `omni` |
| `POSTGRES_PASSWORD` | Database password (fallback for Docker) | (see `.env`) |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers | `localhost:9092` |
| `CONSUL_HOST` | Consul host | `localhost` |
| `CONSUL_PORT` | Consul port | `8500` |
| `CONSUL_ACL_TOKEN` | Consul ACL token | (optional) |
| `VALKEY_HOST` | Valkey host | `localhost` |
| `VALKEY_PORT` | Valkey port | `6379` |

### CLI Commands

```bash
# Validate all contracts
uv run python -m omnibase_infra.contracts.validate

# Run with contract-driven effects (DEFAULT in v0.4.0+)
USE_CONTRACT_DRIVEN_EFFECTS=contract uv run pytest

# Run with Legacy effects (deprecated)
USE_CONTRACT_DRIVEN_EFFECTS=legacy uv run pytest

# Run shadow mode comparison
USE_CONTRACT_DRIVEN_EFFECTS=shadow uv run pytest tests/integration/

# Enable specific node for contract-driven
USE_CONTRACT_DRIVEN_POSTGRES_WORKFLOW=true uv run pytest

# Force Legacy for specific node (deprecated)
USE_CONTRACT_DRIVEN_POSTGRES_WORKFLOW=false uv run pytest
```

### Class Names Quick Reference

| Standard Name (v0.4.0+) | Legacy Name (Deprecated) | Removal |
|-------------------------|--------------------------|---------|
| `NodeEffect` | `NodeEffectLegacy` | v0.5.0 |
| `NodePostgresAdapter` | `NodePostgresAdapterLegacy` | v0.5.0 |
| `NodeKafkaAdapter` | `NodeKafkaAdapterLegacy` | v0.5.0 |
| `NodeConsulAdapter` | `NodeConsulAdapterLegacy` | v0.5.0 |
| `NodeVaultAdapterEffect` | `NodeVaultAdapterEffectLegacy` | v0.5.0 |

---

*This plan was created on December 2, 2025 and should be reviewed when omnibase-core 0.4.0 is released. Naming convention updated to make contract-driven the default (December 2025).*
