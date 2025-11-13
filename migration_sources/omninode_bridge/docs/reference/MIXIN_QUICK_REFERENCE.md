# OmniBase Core Mixin Quick Reference

**Version**: 1.0 | **Last Updated**: 2025-11-04

Quick reference card for omnibase_core mixins - for developers creating or regenerating nodes.

---

## 🏥 Health & Monitoring

### MixinHealthCheck
**Purpose**: Add health check endpoints and component monitoring
**When to use**: Node interacts with external services (DB, API, Kafka)
**Config**:
```yaml
mixins:
  - name: MixinHealthCheck
    config:
      check_interval_ms: 30000
      timeout_seconds: 5.0
      components:
        - name: "database"
          critical: true
          timeout_seconds: 5.0
```
**Methods to implement**: `async def _check_<component>_health() -> bool`

### MixinMetrics
**Purpose**: Performance metrics collection (latency, throughput, errors)
**When to use**: Production nodes requiring observability
**Config**:
```yaml
mixins:
  - name: MixinMetrics
    config:
      collect_latency: true
      collect_throughput: true
      percentiles: [50, 95, 99]
```
**Auto-collected**: Request latency, error rates, throughput

### MixinLogData
**Purpose**: Structured logging with correlation tracking
**When to use**: All nodes (recommended)
**Config**:
```yaml
mixins:
  - name: MixinLogData
    config:
      log_level: "INFO"
      structured: true
      include_correlation_id: true
```
**Built-in**: Correlation ID tracking, structured JSON logs

---

## 🔄 Event-Driven Patterns

### MixinEventDrivenNode
**Purpose**: Event consumption from Kafka topics
**When to use**: Node consumes events (database adapter, metrics collector)
**Config**:
```yaml
mixins:
  - name: MixinEventDrivenNode
    config:
      kafka_bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      consumer_group: "my_service_consumers"
      topics:
        - "workflow-started"
        - "workflow-completed"
```
**Methods to implement**: `async def handle_<event_type>(event: dict)`

### MixinEventBus
**Purpose**: Event publishing to Kafka topics
**When to use**: Node publishes events
**Config**: Minimal (uses Kafka connection from EventDrivenNode)
**Usage**: `await self.publish_event(topic="my-topic", event=event_data)`

### MixinEventHandler
**Purpose**: Event handler registration and routing
**When to use**: Complex event handling logic
**Usage**: Register handlers for specific event types

### MixinEventListener
**Purpose**: Event listening with filtering
**When to use**: Selective event consumption
**Usage**: Filter events before processing

---

## 🌐 Service Integration

### MixinServiceRegistry
**Purpose**: Service discovery and registration with Consul
**When to use**: Microservices architecture, service mesh
**Config**:
```yaml
mixins:
  - name: MixinServiceRegistry
    config:
      consul_host: "${CONSUL_HOST}"
      consul_port: 8500
      service_name: "my_service"
      health_check_interval_seconds: 10
```
**Auto-registered**: Service name, host, port, health check endpoint

### MixinDiscoveryResponder
**Purpose**: Respond to service discovery requests
**When to use**: Node participates in discovery protocol
**Usage**: Auto-responds to discovery queries

---

## ⚙️ Execution Patterns

### MixinNodeExecutor
**Purpose**: Node execution orchestration
**When to use**: Complex execution logic
**Built-in**: Execution lifecycle management

### MixinNodeLifecycle
**Purpose**: Lifecycle management (init → running → shutdown)
**When to use**: All nodes (automatically included in NodeEffect)
**Hooks**: `initialize()`, `shutdown()`, `health_check()`

### MixinHybridExecution
**Purpose**: Hybrid sync/async execution modes
**When to use**: Node supports both sync and async calls
**Usage**: Automatic mode detection and execution

### MixinToolExecution
**Purpose**: Tool/command execution framework
**When to use**: Node executes external tools or commands
**Usage**: Structured tool invocation with timeout and retry

### MixinDagSupport
**Purpose**: DAG workflow support (LlamaIndex-style)
**When to use**: Workflow orchestration nodes
**Built-in**: DAG execution, step dependencies, parallel execution

---

## 💾 Data Handling

### MixinHashComputation
**Purpose**: Hash computation utilities (BLAKE3, SHA256)
**When to use**: Content addressing, deduplication
**Usage**: `hash = self.compute_hash(data, algorithm="blake3")`

### MixinCaching
**Purpose**: In-memory and distributed caching
**When to use**: Expensive operations, API calls
**Config**:
```yaml
mixins:
  - name: MixinCaching
    config:
      ttl_seconds: 300
      max_entries: 1000
```
**Usage**: `@cached(ttl=300)` decorator

### MixinLazyEvaluation
**Purpose**: Lazy loading and evaluation
**When to use**: Expensive initializations, optional features
**Usage**: Properties loaded on first access

### MixinCompletionData
**Purpose**: Completion tracking and metadata
**When to use**: Long-running operations
**Built-in**: Progress tracking, ETA calculation

---

## 📝 Serialization

### MixinCanonicalYAMLSerializer
**Purpose**: Canonical YAML serialization (stable, deterministic)
**When to use**: Config files, contract generation
**Usage**: `yaml_str = self.to_canonical_yaml()`

### MixinYAMLSerialization
**Purpose**: Standard YAML operations
**When to use**: General YAML I/O
**Usage**: `self.to_yaml()`, `self.from_yaml()`

### SerializableMixin
**Purpose**: Generic serialization (JSON, msgpack, pickle)
**When to use**: Data persistence, API responses
**Usage**: `self.to_dict()`, `self.to_json()`

### MixinSensitiveFieldRedaction
**Purpose**: Redact sensitive fields in logs and outputs
**When to use**: Handling secrets, PII, credentials
**Config**:
```yaml
mixins:
  - name: MixinSensitiveFieldRedaction
    config:
      redact_fields:
        - "password"
        - "api_key"
        - "secret"
```
**Auto-redacted**: Configured fields in logs and serialization

---

## 📋 Contract & Metadata

### MixinContractMetadata
**Purpose**: Contract metadata handling
**When to use**: Contract introspection
**Usage**: Access contract fields programmatically

### MixinContractStateReducer
**Purpose**: Contract state reduction (aggregate multiple contracts)
**When to use**: Orchestrator nodes
**Usage**: Merge and reduce contract states

### MixinIntrospectFromContract
**Purpose**: Introspection driven by contract
**When to use**: Dynamic node configuration
**Usage**: Auto-configure from contract.yaml

### MixinNodeIdFromContract
**Purpose**: Node ID derived from contract
**When to use**: Deterministic node IDs
**Usage**: Stable node identification

### MixinNodeIntrospection
**Purpose**: Node introspection (schema, capabilities)
**When to use**: Dynamic discovery, documentation generation
**Usage**: `self.introspect()` returns node schema

---

## 🖥️ CLI & Debugging

### MixinCLIHandler
**Purpose**: CLI command handling
**When to use**: Node has CLI interface
**Usage**: Register commands and handlers

### MixinDebugDiscoveryLogging
**Purpose**: Debug-level discovery logging
**When to use**: Troubleshooting service discovery
**Usage**: Automatically logs discovery events

### MixinFailFast
**Purpose**: Fail-fast validation patterns
**When to use**: Strict validation requirements
**Usage**: Validates inputs and fails early on errors

---

## 🎯 Recommended Combinations

### Effect Node (Database Adapter)
```yaml
mixins:
  - MixinHealthCheck      # Monitor DB connection
  - MixinMetrics          # Track operation performance
  - MixinEventDrivenNode  # Consume Kafka events
  - MixinLogData          # Structured logging
  - MixinSensitiveFieldRedaction  # Redact secrets
```

### Orchestrator Node
```yaml
mixins:
  - MixinHealthCheck      # Monitor component health
  - MixinMetrics          # Track workflow metrics
  - MixinDagSupport       # Workflow DAGs
  - MixinLogData          # Structured logging
  - MixinServiceRegistry  # Service registration
```

### Reducer Node (Aggregation)
```yaml
mixins:
  - MixinMetrics          # Track aggregation metrics
  - MixinEventDrivenNode  # Consume events
  - MixinCaching          # Cache aggregation results
  - MixinLogData          # Structured logging
```

### API Gateway
```yaml
mixins:
  - MixinHealthCheck      # Health endpoint
  - MixinMetrics          # API metrics
  - MixinServiceRegistry  # Service discovery
  - MixinLogData          # Request logging
  - MixinRequestResponseIntrospection  # Request tracking
```

---

## ⚠️ NodeEffect Built-Ins (No Mixin Needed)

These features are **already included** in NodeEffect base class:

✅ **Circuit Breakers** - Automatic failure handling
✅ **Retry Policies** - Exponential backoff retry
✅ **Transaction Support** - Rollback-capable transactions
✅ **Timeout Management** - Per-operation timeouts
✅ **Concurrent Execution** - Semaphore-based concurrency control
✅ **Performance Metrics** - Basic execution metrics
✅ **Event Emission** - State change event publishing
✅ **File Operations** - Atomic file I/O with rollback

**Configure via `advanced_features` in contract, not mixins.**

---

## 🚫 Mixin Conflicts & Overlaps

**Avoid combining**:
- `MixinMetrics` + manual metrics (use mixin only)
- `MixinHealthCheck` + custom health checks (use mixin, register components)
- `MixinEventBus` + manual Kafka producer (use mixin)

**Compatible combinations**:
- `MixinEventDrivenNode` (consumer) + `MixinEventBus` (producer) ✅
- `MixinHealthCheck` + `MixinMetrics` ✅
- `MixinServiceRegistry` + `MixinDiscoveryResponder` ✅

---

## 📖 Further Reading

- **Full Mixin Catalog**: `docs/reference/OMNIBASE_CORE_MIXIN_CATALOG.md`
- **Contract Schema**: `docs/reference/CONTRACT_SCHEMA.md`
- **Code Generation Guide**: `docs/guides/CODE_GENERATION_GUIDE.md`
- **Master Plan**: `docs/planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md`

---

**Questions?** Check troubleshooting guide or ask in #omninode-dev Slack channel.
