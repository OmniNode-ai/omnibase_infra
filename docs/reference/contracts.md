# Contract.yaml Reference

This reference documents the complete structure of ONEX contract files.

## Overview

ONEX contracts are YAML files that declare node behavior. There are two types:

| File | Purpose | Location |
|------|---------|----------|
| `contract.yaml` | Node contract | `nodes/<node_name>/contract.yaml` |
| `handler_contract.yaml` | Handler contract | `contracts/handlers/<handler>/handler_contract.yaml` |

**Key Principle**: Contracts define WHAT a node does. Python code defines HOW (but should be minimal).

## Minimal Contract

Every node contract requires these fields:

```yaml
# Required fields
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_example"
node_type: "EFFECT_GENERIC"  # See node types below
description: "What this node does."

input_model:
  name: "ModelExampleInput"
  module: "omnibase_infra.nodes.node_example.models"

output_model:
  name: "ModelExampleOutput"
  module: "omnibase_infra.nodes.node_example.models"
```

## Node Types

Use `_GENERIC` suffix in contracts:

| Contract Value | EnumNodeKind | Purpose |
|----------------|--------------|---------|
| `EFFECT_GENERIC` | `EFFECT` | External I/O operations |
| `COMPUTE_GENERIC` | `COMPUTE` | Pure transformations |
| `REDUCER_GENERIC` | `REDUCER` | State + FSM + intents |
| `ORCHESTRATOR_GENERIC` | `ORCHESTRATOR` | Workflow coordination |

---

## Common Fields

These fields apply to all node types:

### Version Information

```yaml
contract_version:
  major: 1       # Breaking changes
  minor: 0       # New features (backward compatible)
  patch: 0       # Bug fixes

node_version: "1.0.0"  # Implementation version
```

### Basic Identity

```yaml
name: "node_example"            # Unique identifier (snake_case)
node_type: "EFFECT_GENERIC"     # One of the four archetypes
description: "Human-readable description of what this node does."
```

### Input/Output Models

```yaml
input_model:
  name: "ModelExampleInput"
  module: "omnibase_infra.nodes.node_example.models"
  description: "Optional description of input model."

output_model:
  name: "ModelExampleOutput"
  module: "omnibase_infra.nodes.node_example.models"
  description: "Optional description of output model."
```

### Capabilities

```yaml
capabilities:
  - name: "registration.storage"
    description: "Store and query registration records"
  - name: "registration.storage.query"
  - name: "registration.storage.upsert"
  - name: "registration.storage.delete"
```

**Naming Convention**: Use capability-oriented names (what it does), not technology names.
- Good: `registration.storage`, `service.discovery`
- Bad: `postgres`, `consul`

### Dependencies

```yaml
dependencies:
  protocols:
    - name: "ProtocolPostgresAdapter"
      module: "omnibase_spi.protocols"
      required: true
    - name: "ProtocolProjectionReader"
      module: "omnibase_spi.protocols"
      required: false

  nodes:
    - name: "node_registry_effect"
      required: true

  services:
    - name: "postgresql"
      version: ">=15.0"
      required: true
```

### Error Handling

```yaml
error_handling:
  retry_policy:
    max_retries: 3
    exponential_base: 2
    max_delay_seconds: 60
    retry_on:
      - "InfraConnectionError"
      - "InfraTimeoutError"
    do_not_retry_on:
      - "InfraAuthenticationError"

  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_seconds: 60
    half_open_max_calls: 3
    per_backend: true  # Separate circuit per backend

  error_codes:
    - code: "REGISTRATION_001"
      description: "Node ID not found"
      severity: "ERROR"
```

### Health Check

```yaml
health_check:
  enabled: true
  endpoint: "/health"
  interval_seconds: 30
  timeout_seconds: 5
  failure_threshold: 3
```

### Metadata

```yaml
metadata:
  author: "ONEX Team"
  created_at: "2025-01-15"
  updated_at: "2025-01-18"
  tags:
    - "infrastructure"
    - "registration"
  license: "MIT"
```

---

## EFFECT-Specific Fields

### IO Operations

```yaml
io_operations:
  - operation: "store_registration"
    description: "Persist a registration record"
    input_fields:
      - record: "ModelRegistrationRecord"
      - correlation_id: "UUID | None"
    output_fields:
      - result: "ModelUpsertResult"
    idempotent: true
    timeout_ms: 5000

  - operation: "query_registration"
    description: "Query registration by node_id"
    input_fields:
      - node_id: "str"
    output_fields:
      - record: "ModelRegistrationRecord | None"
    idempotent: true
    read_only: true
```

### Handler Routing (Effect)

```yaml
handler_routing:
  routing_strategy: "handler_type_match"
  default_handler: "postgresql"
  execution_mode: "parallel"          # parallel | sequential
  partial_failure_handling: true
  aggregation_strategy: "all_or_partial"

  handlers:
    - handler_type: "postgresql"
      handler:
        name: "HandlerRegistrationStoragePostgres"
        module: "omnibase_infra.handlers.registration_storage"
      backend: "postgres"

    - handler_type: "mock"
      handler:
        name: "HandlerRegistrationStorageMock"
        module: "omnibase_infra.handlers.registration_storage"
      backend: "mock"
```

### Database Schema (Optional)

```yaml
database_schema:
  tables:
    - name: "node_registrations"
      columns:
        - name: "node_id"
          type: "UUID"
          primary_key: true
        - name: "node_type"
          type: "VARCHAR(50)"
          not_null: true
        - name: "endpoints"
          type: "JSONB"
        - name: "created_at"
          type: "TIMESTAMP"
          default: "NOW()"
      indexes:
        - columns: ["node_type"]
          name: "idx_node_type"
```

---

## COMPUTE-Specific Fields

### Validation Rules

```yaml
validation_rules:
  - rule_id: "ARCH-001"
    name: "No Direct Handler Dispatch"
    severity: "WARNING"
    description: "Handlers must be routed through RuntimeHost"
    detection_strategy:
      type: "ast_pattern"
      patterns:
        - "direct_handler_call"
        - "handler.handle("
    suggested_fix: "Route through RuntimeHost.process_event()"

  - rule_id: "ARCH-002"
    name: "No Handler Publishing Events"
    severity: "ERROR"
    detection_strategy:
      type: "signature_analysis"
      checks:
        - no_bus_in_init_params
        - no_bus_instance_attributes
```

### Transformation Rules

```yaml
transformations:
  - name: "normalize_endpoint"
    input_type: "str"
    output_type: "str"
    description: "Normalize endpoint URL format"

  - name: "aggregate_capabilities"
    input_type: "list[str]"
    output_type: "set[str]"
    description: "Deduplicate capabilities"
```

---

## REDUCER-Specific Fields

### State Machine

```yaml
state_machine:
  state_machine_name: "registration_fsm"
  initial_state: "idle"

  states:
    - state_name: "idle"
      description: "Waiting for events"
      entry_actions: []
      exit_actions: []

    - state_name: "pending"
      description: "Processing registration"
      entry_actions:
        - "emit_consul_intent"
        - "emit_postgres_intent"
      timeout_seconds: 30

    - state_name: "partial"
      description: "Partial completion"

    - state_name: "complete"
      description: "Successfully completed"
      terminal: false

    - state_name: "failed"
      description: "Failed state"
      terminal: false

  transitions:
    - from_state: "idle"
      to_state: "pending"
      trigger: "introspection_received"
      conditions:
        - expression: "node_id is_present"
          required: true
        - expression: "node_type is_valid"
          required: true
      actions:
        - action_name: "build_consul_intent"
          action_type: "intent_emission"
        - action_name: "build_postgres_intent"
          action_type: "intent_emission"
      guard: "not is_duplicate_event"

    - from_state: "pending"
      to_state: "partial"
      trigger: "consul_confirmed"

    - from_state: "partial"
      to_state: "complete"
      trigger: "postgres_confirmed"
      conditions:
        - expression: "consul_confirmed is_true"

    - from_state: "*"           # Wildcard: any state
      to_state: "failed"
      trigger: "error_received"

    - from_state: "failed"
      to_state: "idle"
      trigger: "reset_requested"
```

### Intent Emission

```yaml
intent_emission:
  enabled: true
  intents:
    - intent_type: "consul.register"
      target_pattern: "consul://service/{service_name}"
      payload_model: "ModelPayloadConsulRegister"
      payload_module: "omnibase_infra.nodes.reducers.models"

    - intent_type: "postgres.upsert_registration"
      target_pattern: "postgres://node_registrations/{node_id}"
      payload_model: "ModelPayloadPostgresUpsertRegistration"
      payload_module: "omnibase_infra.nodes.reducers.models"
```

### Idempotency

```yaml
idempotency:
  enabled: true
  strategy: "event_id_tracking"
  description: "Track processed event_ids to prevent duplicates"
  storage: "in_state"           # in_state | external
```

### Validation

```yaml
validation:
  event_validation:
    - field: "node_id"
      required: true
      type: "str"
      min_length: 1
    - field: "node_type"
      required: true
      type: "EnumNodeKind"
      allowed_values: ["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"]
```

---

## ORCHESTRATOR-Specific Fields

### Workflow Coordination

```yaml
workflow_coordination:
  workflow_definition:
    workflow_metadata:
      workflow_name: "node_registration_workflow"
      workflow_version:
        major: 1
        minor: 0
        patch: 0
      description: "Two-way registration with Consul and PostgreSQL"

    execution_graph:
      nodes:
        - node_id: "receive_event"
          node_type: EFFECT_GENERIC
          description: "Receive incoming event"
          step_config:
            event_pattern: ["node-introspection.*"]

        - node_id: "read_projection"
          node_type: EFFECT_GENERIC
          depends_on: ["receive_event"]
          description: "Query current state"
          step_config:
            protocol: "ProtocolProjectionReader"

        - node_id: "compute_intents"
          node_type: REDUCER_GENERIC
          depends_on: ["read_projection"]
          description: "FSM transition and intent computation"

        - node_id: "execute_consul"
          node_type: EFFECT_GENERIC
          depends_on: ["compute_intents"]
          step_config:
            intent_filter: "consul.*"

        - node_id: "execute_postgres"
          node_type: EFFECT_GENERIC
          depends_on: ["compute_intents"]
          step_config:
            intent_filter: "postgres.*"

        - node_id: "aggregate_results"
          node_type: COMPUTE_GENERIC
          depends_on: ["execute_consul", "execute_postgres"]

    coordination_rules:
      execution_mode: parallel
      parallel_execution_allowed: true
      max_parallel_branches: 2
      failure_recovery_strategy: retry
      max_retries: 3
      timeout_ms: 30000
      checkpointing:
        enabled: true
        checkpoint_after_steps: ["compute_intents"]
```

### Handler Routing (Orchestrator)

```yaml
handler_routing:
  routing_strategy: "payload_type_match"

  handlers:
    - event_model:
        name: "ModelNodeIntrospectionEvent"
        module: "omnibase_infra.models.registration"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers"
      output_events:
        - "ModelNodeRegistrationInitiated"
      state_decision_matrix:
        - current_state: null
          action: "emit_registration_initiated"
          description: "New node - initiate registration"
        - current_state: "LIVENESS_EXPIRED"
          action: "emit_registration_initiated"
          description: "Expired node - re-register"
        - current_state: "ACTIVE"
          action: "no_op"
          description: "Already active - skip"

    - event_model:
        name: "ModelRuntimeTick"
        module: "omnibase_infra.runtime.models"
      handler:
        name: "HandlerRuntimeTick"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers"
      timeout_evaluation:
        uses_injected_now: true
        queries_pending_entities: true

  handler_dependencies:
    projection_reader:
      protocol: "ProtocolProjectionReader"
      implementation: "ProjectionReaderRegistration"
      shared: true
```

### Consumed Events

```yaml
consumed_events:
  - topic: "{env}.{namespace}.onex.evt.node-introspection.v1"
    event_type: "NodeIntrospectionEvent"
    description: "Node startup introspection"
    consumer_group: "registration-orchestrator"

  - topic: "{env}.{namespace}.onex.cmd.node-registration-acked.v1"
    event_type: "NodeRegistrationAcked"
    description: "Node acknowledges registration"

  - topic: "{env}.{namespace}.onex.internal.runtime-tick.v1"
    event_type: "RuntimeTick"
    description: "Periodic tick for timeout evaluation"
```

### Published Events

```yaml
published_events:
  - topic: "{env}.{namespace}.onex.evt.node-registration-initiated.v1"
    event_type: "NodeRegistrationInitiated"
    description: "Registration workflow started"

  - topic: "{env}.{namespace}.onex.evt.node-became-active.v1"
    event_type: "NodeBecameActive"
    description: "Node transitioned to ACTIVE"

  - topic: "{env}.{namespace}.onex.evt.node-liveness-expired.v1"
    event_type: "NodeLivenessExpired"
    description: "Heartbeat deadline passed"
```

### Intent Consumption

```yaml
intent_consumption:
  intent_routing_table:
    "consul.register": "node_registry_effect"
    "consul.deregister": "node_registry_effect"
    "postgres.upsert_registration": "node_registry_effect"
    "postgres.deactivate_registration": "node_registry_effect"
```

---

## Handler Contract

Handler contracts define standalone handlers:

```yaml
# contracts/handlers/consul/handler_contract.yaml

name: handler-consul
handler_class: omnibase_infra.handlers.handler_consul.HandlerConsul
handler_type: effect

description: "HashiCorp Consul integration for service discovery and KV store"

tags:
  - infrastructure
  - service-discovery
  - effect

operations:
  - name: "consul.kv_get"
    description: "Get value from Consul KV store"
  - name: "consul.kv_put"
    description: "Put value to Consul KV store"
  - name: "consul.register"
    description: "Register service with Consul"
  - name: "consul.deregister"
    description: "Deregister service from Consul"

dependencies:
  - protocol: "ProtocolConsulClient"
    required: true

security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: true
  allowed_operations:
    - "consul.kv_get"
    - "consul.kv_put"
    - "consul.register"
    - "consul.deregister"

error_handling:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
```

---

## Contract Validation

Contracts are validated at runtime. Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| Missing required field | `name`, `node_type`, or I/O models missing | Add required fields |
| Invalid node_type | Not one of the four archetypes | Use `*_GENERIC` suffix |
| Module not found | `module` path doesn't resolve | Check import path |
| Invalid transition | FSM transition references unknown state | Define all states first |
| Ambiguous contracts | Both `contract.yaml` and `handler_contract.yaml` exist | Use only one per directory |

## Best Practices

1. **Use capabilities, not technologies**: Name by what it does, not what it uses
2. **Keep contracts focused**: One node = one responsibility
3. **Define error handling**: Always include retry and circuit breaker config
4. **Version intentionally**: Bump major for breaking changes
5. **Document thoroughly**: Use description fields liberally
6. **Validate early**: Run `onex validate` in CI

## Related Documentation

| Topic | Document |
|-------|----------|
| Quick start | [Quick Start Guide](../getting-started/quickstart.md) |
| Node archetypes | [Node Archetypes Reference](node-archetypes.md) |
| Architecture | [Architecture Overview](../architecture/overview.md) |
| Registration example | [2-Way Registration](../guides/registration-example.md) |
| All patterns | [Pattern Documentation](../patterns/README.md) |
| Handler patterns | [Handler Plugin Loader](../patterns/handler_plugin_loader.md) |
