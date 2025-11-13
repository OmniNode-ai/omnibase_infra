# Contract Schema v2.0 - Complete Reference

**Version:** v2.0.0
**Last Updated:** 2025-11-04
**Status:** Production-Ready

## Overview

Contract Schema v2.0 introduces **mixin support** and **advanced features configuration** to ONEX v2.0 node contracts. This major enhancement enables:

- **Declarative mixin integration** - Compose nodes from reusable capability mixins
- **Advanced features** - Circuit breakers, retry policies, DLQ, transactions, security, observability
- **Backward compatibility** - Existing contracts work without modification
- **Schema versioning** - Future-proof contract evolution

## Table of Contents

1. [Schema Structure](#schema-structure)
2. [New Features in v2.0](#new-features-in-v20)
3. [Mixin Declarations](#mixin-declarations)
4. [Advanced Features](#advanced-features)
5. [Validation Examples](#validation-examples)
6. [Migration Guide](#migration-guide)
7. [Schema Files](#schema-files)
8. [Best Practices](#best-practices)

---

## Schema Structure

### Complete Contract Schema

```yaml
# Contract Schema v2.0 - Complete Structure
schema_version: "v2.0.0"  # NEW: Schema version for validation

# EXISTING: Core contract fields (unchanged)
name: "NodeMyEffect"
version:
  major: 1
  minor: 0
  patch: 0
node_type: "effect"
description: "Node description"

capabilities:
  - name: "capability_name"
    description: "Capability description"

endpoints: {}
dependencies: {}
configuration: {}

# NEW: Mixin declarations
mixins:
  - name: "MixinHealthCheck"
    enabled: true
    config: {}
  - name: "MixinMetrics"
    enabled: true
    config: {}

# NEW: Advanced features configuration
advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
  retry_policy:
    enabled: true
    max_attempts: 3
  dead_letter_queue:
    enabled: true
  transactions:
    enabled: true
  security_validation:
    enabled: true
  observability:
    tracing:
      enabled: true

# EXISTING: Subcontracts (unchanged)
subcontracts:
  effect:
    operations: ["initialize", "shutdown"]

# EXISTING: Other contract sections (unchanged)
performance_targets: {}
health_checks: {}
input_state: {}
output_state: {}
io_operations: []
definitions: {}
```

### Schema Version Evolution

| Version | Released | Key Features | Breaking Changes |
|---------|----------|--------------|------------------|
| **v1.0.0** | 2024-01 | Initial ONEX v2.0 contract schema | N/A |
| **v2.0.0** | 2025-11 | Mixins, advanced features, schema versioning | None (backward compatible) |

---

## New Features in v2.0

### 1. Schema Versioning

All contracts now include a `schema_version` field for validation:

```yaml
schema_version: "v2.0.0"
```

**Benefits:**
- Explicit schema version tracking
- Forward compatibility validation
- Migration path clarity

### 2. Mixin Declarations

Declare mixins directly in contracts for automatic code generation:

```yaml
mixins:
  - name: "MixinHealthCheck"
    enabled: true
    config:
      check_interval_ms: 30000
      components:
        - name: "database"
          critical: true
```

**Benefits:**
- Declarative composition over manual inheritance
- Validation of mixin configuration
- Conditional mixin inclusion
- Auto-generated integration code

### 3. Advanced Features

Configure advanced capabilities through standardized schema:

```yaml
advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
  retry_policy:
    enabled: true
    max_attempts: 3
```

**Benefits:**
- Consistent configuration across nodes
- Validated configuration values
- Built-in NodeEffect features + custom mixins
- Single source of truth for features

---

## Mixin Declarations

### Mixin Schema Structure

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "pattern": "^Mixin[A-Z][a-zA-Z0-9]*$"
      },
      "enabled": {
        "type": "boolean",
        "default": true
      },
      "config": {
        "type": "object"
      }
    },
    "required": ["name"]
  }
}
```

### Available Mixins

#### MixinHealthCheck

**Purpose:** Health monitoring with component-level checks

**Configuration:**

```yaml
mixins:
  - name: "MixinHealthCheck"
    enabled: true
    config:
      check_interval_ms: 30000       # Health check interval
      timeout_seconds: 5.0            # Overall timeout
      components:
        - name: "database"
          critical: true              # Failure fails overall health
          timeout_seconds: 5.0
        - name: "kafka_consumer"
          critical: false             # Failure is non-critical
          timeout_seconds: 3.0
```

**Generated Methods:**
- `async def health_check() -> HealthCheckResponse`
- `async def check_component(name: str) -> bool`

#### MixinMetrics

**Purpose:** Performance metrics collection

**Configuration:**

```yaml
mixins:
  - name: "MixinMetrics"
    enabled: true
    config:
      collect_latency: true
      collect_throughput: true
      collect_error_rates: true
      percentiles: [50, 95, 99]
      histogram_buckets: [10, 50, 100, 500, 1000]
```

**Generated Methods:**
- `async def record_metric(name: str, value: float)`
- `async def get_metrics() -> MetricsSnapshot`

#### MixinEventDrivenNode

**Purpose:** Kafka event consumption

**Configuration:**

```yaml
mixins:
  - name: "MixinEventDrivenNode"
    enabled: true
    config:
      kafka_bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      consumer_group: "${SERVICE_NAME}_consumers"
      topics:
        - "workflow-started"
        - "workflow-completed"
      auto_offset_reset: "latest"
      enable_auto_commit: true
```

**Generated Methods:**
- `async def consume_events()`
- `async def handle_event(event: OnexEnvelopeV1)`

#### MixinServiceRegistry

**Purpose:** Consul service discovery integration

**Configuration:**

```yaml
mixins:
  - name: "MixinServiceRegistry"
    enabled: false  # Optional
    config:
      consul_host: "${CONSUL_HOST}"
      consul_port: 8500
      service_name: "${SERVICE_NAME}"
      health_check_interval_seconds: 10
      deregister_critical_service_after: "30m"
```

**Generated Methods:**
- `async def register_service()`
- `async def deregister_service()`

#### MixinLogData

**Purpose:** Structured logging with redaction

**Configuration:**

```yaml
mixins:
  - name: "MixinLogData"
    enabled: true
    config:
      log_level: "INFO"
      structured: true
      include_correlation_id: true
      redact_sensitive_fields: true
      sensitive_field_patterns:
        - "password"
        - "token"
        - "api_key"
```

**Generated Methods:**
- `def log_info(msg: str, **kwargs)`
- `def log_error(msg: str, **kwargs)`

### Mixin Configuration Validation

Each mixin has a specific configuration schema. Invalid configurations are rejected during validation:

```python
# Example: Invalid MixinHealthCheck configuration
{
  "name": "MixinHealthCheck",
  "config": {
    "check_interval_ms": 500  # INVALID: Below minimum 1000ms
  }
}

# Validation Error:
# MixinHealthCheck.config.check_interval_ms: 500 is less than minimum 1000
```

---

## Advanced Features

### Circuit Breaker

**Purpose:** Fault tolerance with circuit breaker pattern

**Configuration:**

```yaml
advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5              # Failures before opening
    recovery_timeout_ms: 60000        # Time before retry
    half_open_max_calls: 3            # Max calls in half-open state
    services:
      - name: "postgres"
        failure_threshold: 3          # Service-specific threshold
        recovery_timeout_ms: 30000
      - name: "kafka"
        failure_threshold: 10
        recovery_timeout_ms: 120000
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "enabled": {"type": "boolean", "default": true},
    "failure_threshold": {"type": "integer", "minimum": 1, "maximum": 100},
    "recovery_timeout_ms": {"type": "integer", "minimum": 1000, "maximum": 600000},
    "half_open_max_calls": {"type": "integer", "minimum": 1, "maximum": 10},
    "services": {"type": "array"}
  }
}
```

### Retry Policy

**Purpose:** Exponential backoff retry with configurable exceptions

**Configuration:**

```yaml
advanced_features:
  retry_policy:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 1000
    max_delay_ms: 30000
    backoff_multiplier: 2.0
    retryable_exceptions:
      - "TimeoutError"
      - "ConnectionError"
      - "TemporaryFailure"
    retryable_status_codes: [429, 500, 502, 503, 504]
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "enabled": {"type": "boolean", "default": true},
    "max_attempts": {"type": "integer", "minimum": 1, "maximum": 10},
    "initial_delay_ms": {"type": "integer", "minimum": 100, "maximum": 10000},
    "max_delay_ms": {"type": "integer", "minimum": 1000, "maximum": 300000},
    "backoff_multiplier": {"type": "number", "minimum": 1.0, "maximum": 5.0},
    "retryable_exceptions": {"type": "array"},
    "retryable_status_codes": {"type": "array"}
  }
}
```

### Dead Letter Queue

**Purpose:** Failed event handling with monitoring

**Configuration:**

```yaml
advanced_features:
  dead_letter_queue:
    enabled: true
    max_retries: 3
    topic_suffix: ".dlq"
    retry_delay_ms: 5000
    alert_threshold: 100              # Alert if >100 in DLQ
    monitoring:
      enabled: true
      check_interval_seconds: 60
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "enabled": {"type": "boolean", "default": true},
    "max_retries": {"type": "integer", "minimum": 1, "maximum": 10},
    "topic_suffix": {"type": "string", "pattern": "^\\.[a-z0-9_-]+$"},
    "retry_delay_ms": {"type": "integer", "minimum": 1000, "maximum": 300000},
    "alert_threshold": {"type": "integer", "minimum": 1, "maximum": 10000},
    "monitoring": {"type": "object"}
  }
}
```

### Transactions

**Purpose:** Database transaction management

**Configuration:**

```yaml
advanced_features:
  transactions:
    enabled: true
    isolation_level: "READ_COMMITTED"
    timeout_seconds: 30
    rollback_on_error: true
    savepoints: true
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "enabled": {"type": "boolean", "default": true},
    "isolation_level": {
      "type": "string",
      "enum": ["READ_UNCOMMITTED", "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"]
    },
    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
    "rollback_on_error": {"type": "boolean", "default": true},
    "savepoints": {"type": "boolean", "default": true}
  }
}
```

### Security Validation

**Purpose:** Input sanitization and SQL injection prevention

**Configuration:**

```yaml
advanced_features:
  security_validation:
    enabled: true
    sanitize_inputs: true
    sanitize_logs: true
    validate_sql: true
    max_input_length: 10000
    forbidden_patterns:
      - "(?i)(DROP|DELETE|TRUNCATE)\\s+TABLE"
      - "(?i)EXEC(UTE)?\\s+"
    redact_fields:
      - "password"
      - "api_key"
      - "secret"
      - "token"
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "enabled": {"type": "boolean", "default": true},
    "sanitize_inputs": {"type": "boolean", "default": true},
    "sanitize_logs": {"type": "boolean", "default": true},
    "validate_sql": {"type": "boolean", "default": true},
    "max_input_length": {"type": "integer", "minimum": 100, "maximum": 1000000},
    "forbidden_patterns": {"type": "array"},
    "redact_fields": {"type": "array"}
  }
}
```

### Observability

**Purpose:** Tracing, metrics, and logging configuration

**Configuration:**

```yaml
advanced_features:
  observability:
    tracing:
      enabled: true
      exporter: "otlp"
      endpoint: "${OTEL_EXPORTER_OTLP_ENDPOINT}"
      sample_rate: 1.0
    metrics:
      enabled: true
      prometheus_port: 9090
      export_interval_seconds: 15
    logging:
      structured: true
      json_format: true
      correlation_tracking: true
```

**Schema:**

```json
{
  "type": "object",
  "properties": {
    "tracing": {
      "type": "object",
      "properties": {
        "enabled": {"type": "boolean", "default": true},
        "exporter": {"type": "string", "enum": ["otlp", "jaeger", "zipkin", "console"]},
        "endpoint": {"type": "string"},
        "sample_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      }
    },
    "metrics": {
      "type": "object",
      "properties": {
        "enabled": {"type": "boolean", "default": true},
        "prometheus_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
        "export_interval_seconds": {"type": "integer", "minimum": 5, "maximum": 300}
      }
    },
    "logging": {
      "type": "object",
      "properties": {
        "structured": {"type": "boolean", "default": true},
        "json_format": {"type": "boolean", "default": true},
        "correlation_tracking": {"type": "boolean", "default": true}
      }
    }
  }
}
```

---

## Validation Examples

### Valid Contract Examples

#### Example 1: Minimal Contract (Backward Compatible)

```yaml
# Valid v1.0.0 contract (works with v2.0.0 validator)
name: "NodeSimpleEffect"
version:
  major: 1
  minor: 0
  patch: 0
node_type: "effect"
description: "Simple effect node without mixins"

subcontracts:
  effect:
    operations: ["initialize", "shutdown"]
```

**Validation Result:** ✅ Valid (backward compatible)

#### Example 2: Contract with Mixins

```yaml
# Valid v2.0.0 contract with mixins
schema_version: "v2.0.0"
name: "NodeDatabaseAdapterEffect"
version:
  major: 1
  minor: 0
  patch: 0
node_type: "effect"
description: "Database adapter with health checks and metrics"

mixins:
  - name: "MixinHealthCheck"
    enabled: true
    config:
      check_interval_ms: 30000
      components:
        - name: "database"
          critical: true
          timeout_seconds: 5.0
  - name: "MixinMetrics"
    enabled: true
    config:
      collect_latency: true
      percentiles: [50, 95, 99]

subcontracts:
  effect:
    operations: ["initialize", "shutdown", "process"]
```

**Validation Result:** ✅ Valid

#### Example 3: Complete Contract with Advanced Features

```yaml
# Valid v2.0.0 contract with all features
schema_version: "v2.0.0"
name: "NodeAdvancedEffect"
version:
  major: 2
  minor: 0
  patch: 0
node_type: "effect"
description: "Advanced effect node with all features enabled"

mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true
  - name: "MixinEventDrivenNode"
    enabled: true
    config:
      kafka_bootstrap_servers: "localhost:9092"
      consumer_group: "advanced_consumers"
      topics: ["events"]

advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
  retry_policy:
    enabled: true
    max_attempts: 3
  dead_letter_queue:
    enabled: true
    max_retries: 3
  transactions:
    enabled: true
    isolation_level: "READ_COMMITTED"
  security_validation:
    enabled: true
    sanitize_inputs: true
  observability:
    tracing:
      enabled: true
      exporter: "otlp"
    metrics:
      enabled: true

subcontracts:
  effect:
    operations: ["initialize", "shutdown", "process"]
```

**Validation Result:** ✅ Valid

### Invalid Contract Examples

#### Example 1: Invalid Mixin Name

```yaml
mixins:
  - name: "InvalidMixin"  # Doesn't match Mixin* pattern
```

**Validation Error:**
```
ValidationError: mixins[0].name: 'InvalidMixin' does not match pattern '^Mixin[A-Z][a-zA-Z0-9]*$'
```

#### Example 2: Invalid Health Check Config

```yaml
mixins:
  - name: "MixinHealthCheck"
    config:
      check_interval_ms: 500  # Below minimum 1000
```

**Validation Error:**
```
ValidationError: mixins[0].config.check_interval_ms: 500 is less than minimum 1000
```

#### Example 3: Invalid Circuit Breaker Config

```yaml
advanced_features:
  circuit_breaker:
    failure_threshold: 150  # Above maximum 100
```

**Validation Error:**
```
ValidationError: advanced_features.circuit_breaker.failure_threshold: 150 is greater than maximum 100
```

#### Example 4: Missing Required Field

```yaml
mixins:
  - enabled: true  # Missing required 'name' field
```

**Validation Error:**
```
ValidationError: mixins[0]: 'name' is a required property
```

---

## Migration Guide

### Migrating from v1.0.0 to v2.0.0

**Good News:** v2.0.0 is **100% backward compatible**. Existing v1.0.0 contracts require no changes.

### Migration Strategies

#### Strategy 1: No Changes (Keep v1.0.0)

Your existing contracts work as-is:

```yaml
# Existing v1.0.0 contract - no changes required
name: "NodeMyEffect"
version: {major: 1, minor: 0, patch: 0}
node_type: "effect"
description: "My effect node"
```

✅ **Works with v2.0.0 validator**
✅ **No code changes needed**
✅ **Generated code unchanged**

#### Strategy 2: Add Schema Version Only

Add version tracking without new features:

```yaml
# Add schema_version field
schema_version: "v2.0.0"  # NEW

name: "NodeMyEffect"
version: {major: 1, minor: 0, patch: 0}
# ... rest unchanged
```

✅ **Explicit version tracking**
✅ **Future-proof**
✅ **No functional changes**

#### Strategy 3: Gradual Mixin Adoption

Add mixins one at a time:

```yaml
schema_version: "v2.0.0"
name: "NodeMyEffect"
# ... existing fields ...

# Add first mixin
mixins:
  - name: "MixinHealthCheck"
    enabled: true
```

✅ **Incremental enhancement**
✅ **Test each mixin separately**
✅ **Low risk**

#### Strategy 4: Full v2.0.0 Migration

Adopt all new features at once:

```yaml
schema_version: "v2.0.0"
name: "NodeMyEffect"
# ... existing fields ...

mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true

advanced_features:
  circuit_breaker:
    enabled: true
  retry_policy:
    enabled: true
```

✅ **Full feature set**
⚠️ **Requires thorough testing**
⚠️ **Larger code changes**

### Step-by-Step Migration Process

#### Step 1: Validate Existing Contract

Ensure your v1.0.0 contract is valid:

```bash
# Validate with v2.0.0 schema
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract path/to/contract.yaml

# Or with custom schema directory
poetry run python -m omninode_bridge.codegen.contract_validator \
  --schema src/omninode_bridge/codegen/schemas/ \
  --contract path/to/contract.yaml
```

Expected output:
```
Validating contract: path/to/contract.yaml

✅ Contract validation successful (contract.yaml)

Warnings:
⚠️  Contract is using schema version v1.0.0 - consider upgrading to v2.0.0 for mixin support
```

#### Step 2: Add Schema Version (Optional)

```yaml
# Add to top of contract
schema_version: "v2.0.0"
```

#### Step 3: Identify Needed Mixins

Review your node's manual implementations:

| Manual Implementation | Replace With Mixin |
|----------------------|-------------------|
| Custom health checks | MixinHealthCheck |
| Manual metrics collection | MixinMetrics |
| Kafka consumer setup | MixinEventDrivenNode |
| Consul registration | MixinServiceRegistry |
| Structured logging | MixinLogData |

#### Step 4: Add Mixins One by One

Start with non-critical mixins:

```yaml
# Week 1: Add metrics
mixins:
  - name: "MixinMetrics"
    enabled: true

# Week 2: Add health checks
mixins:
  - name: "MixinMetrics"
    enabled: true
  - name: "MixinHealthCheck"
    enabled: true
    config:
      components:
        - name: "database"
          critical: true
```

#### Step 5: Configure Advanced Features

Enable built-in features:

```yaml
advanced_features:
  # Start with circuit breaker
  circuit_breaker:
    enabled: true
    failure_threshold: 5

  # Add retry policy
  retry_policy:
    enabled: true
    max_attempts: 3
```

#### Step 6: Remove Deprecated Sections

If you have legacy `error_handling` section:

```yaml
# OLD (v1.0.0)
error_handling:
  retry_policy:
    max_attempts: 3
  circuit_breaker:
    enabled: true

# NEW (v2.0.0) - Move to advanced_features
advanced_features:
  retry_policy:
    enabled: true
    max_attempts: 3
  circuit_breaker:
    enabled: true
```

#### Step 7: Regenerate Node Code

```bash
# Regenerate with new contract
python -m omninode_bridge.codegen.generator \
  --contract path/to/contract.yaml \
  --output-dir src/nodes/my_effect/v1_0_0/
```

#### Step 8: Update Tests

Adjust tests for new mixin methods:

```python
# Test health check from mixin
async def test_health_check():
    node = NodeMyEffect()
    result = await node.health_check()
    assert result.healthy is True

# Test metrics from mixin
async def test_metrics():
    node = NodeMyEffect()
    await node.record_metric("latency", 10.5)
    metrics = await node.get_metrics()
    assert metrics is not None
```

#### Step 9: Deploy and Monitor

- Deploy to staging first
- Verify mixin functionality
- Monitor for errors
- Gradually roll to production

### Deprecation Timeline

| Feature | v1.0.0 | v2.0.0 | v3.0.0 (Future) |
|---------|--------|--------|-----------------|
| **error_handling** | ✅ Primary | ⚠️ Deprecated | ❌ Removed |
| **advanced_features** | ❌ N/A | ✅ Primary | ✅ Primary |
| **mixins** | ❌ N/A | ✅ Supported | ✅ Required |

**Recommendation:** Migrate to `advanced_features` before v3.0.0 release.

---

## Schema Files

### File Locations

```
src/omninode_bridge/codegen/schemas/
├── contract_schema_v2.json          # Complete contract schema
├── mixins_schema.json               # Mixin declarations schema
└── advanced_features_schema.json    # Advanced features schema
```

### Schema References

The main contract schema references sub-schemas:

```json
{
  "mixins": {
    "$ref": "mixins_schema.json"
  },
  "advanced_features": {
    "$ref": "advanced_features_schema.json"
  }
}
```

### Validation Tools

#### Python Validation

```python
import jsonschema
import yaml

# Load schemas
with open("schemas/contract_schema_v2.json") as f:
    schema = json.load(f)

# Load contract
with open("contract.yaml") as f:
    contract = yaml.safe_load(f)

# Validate
try:
    jsonschema.validate(contract, schema)
    print("✅ Valid contract")
except jsonschema.ValidationError as e:
    print(f"❌ Validation failed: {e.message}")
```

#### CLI Validation

```bash
# Validate single contract
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract nodes/my_effect/v1_0_0/contract.yaml

# Or with custom schema directory
poetry run python -m omninode_bridge.codegen.contract_validator \
  --schema src/omninode_bridge/codegen/schemas/ \
  --contract nodes/my_effect/v1_0_0/contract.yaml

# Validate all contracts
find nodes -name "contract.yaml" | xargs -I {} \
  poetry run python -m omninode_bridge.codegen.contract_validator \
    --contract {}
```

---

## Best Practices

### Mixin Selection

✅ **DO:**
- Use `MixinHealthCheck` for all production nodes
- Use `MixinMetrics` for performance-critical nodes
- Use `MixinEventDrivenNode` for event consumers
- Enable mixins conditionally based on environment

❌ **DON'T:**
- Add mixins you don't need (bloats generated code)
- Override mixin methods without understanding implications
- Disable critical mixins in production

### Configuration

✅ **DO:**
- Use environment variables for sensitive config (`${VAR_NAME}`)
- Set reasonable timeouts based on SLAs
- Configure service-specific circuit breakers
- Use structured logging in production

❌ **DON'T:**
- Hard-code credentials or secrets
- Set extremely low timeouts (causes false failures)
- Disable security validation in production
- Use default configs without reviewing

### Schema Evolution

✅ **DO:**
- Add `schema_version` to all new contracts
- Test contracts with validator before deployment
- Document custom mixin configurations
- Keep contracts in version control

❌ **DON'T:**
- Skip schema validation
- Mix v1.0.0 and v2.0.0 patterns inconsistently
- Modify generated code directly (use mixins/contracts)

### Performance

✅ **DO:**
- Monitor mixin overhead in performance tests
- Disable non-critical mixins for latency-sensitive nodes
- Use sampling for high-volume tracing
- Cache mixin configuration at startup

❌ **DON'T:**
- Enable all mixins without profiling
- Set check intervals too aggressively
- Collect metrics synchronously in hot paths

---

## Summary

Contract Schema v2.0 provides:

✅ **Backward Compatibility** - Existing contracts work unchanged
✅ **Mixin Support** - Declarative composition of capabilities
✅ **Advanced Features** - Standardized configuration for circuit breakers, retry, DLQ, etc.
✅ **Schema Versioning** - Future-proof evolution path
✅ **Validation** - Comprehensive JSON Schema validation

**Next Steps:**
1. Review your existing contracts
2. Decide on migration strategy (none, gradual, or full)
3. Add mixins as needed
4. Configure advanced features
5. Validate with new schema
6. Regenerate and test

**Resources:**
- [Mixin Enhancement Master Plan](../planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)
- [Code Generation Guide](../guides/MIXIN_ENHANCED_GENERATION_QUICKSTART.md)
- [Schema Files](../../src/omninode_bridge/codegen/schemas/)
