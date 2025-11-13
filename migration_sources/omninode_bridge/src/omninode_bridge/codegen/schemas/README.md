# ONEX v2.0 Contract Schemas

This directory contains JSON Schema definitions for ONEX v2.0 node contracts, including support for mixin declarations and advanced features configuration.

## Schema Files

### Core Schemas

**`contract_schema_v2.json`** - Complete Contract Schema
- Main schema for ONEX v2.0 contracts
- Includes all core fields, mixins, and advanced features
- References sub-schemas for modularity
- Backward compatible with v1.0.0 contracts

**`mixins_schema.json`** - Mixin Declarations Schema
- Schema for mixin array in contracts
- Validates mixin names, configuration, and structure
- Includes configuration schemas for built-in mixins
- Referenced by main contract schema

**`advanced_features_schema.json`** - Advanced Features Schema
- Schema for advanced features configuration
- Validates circuit breaker, retry policy, DLQ, transactions, security, observability
- Service-specific configuration support
- Referenced by main contract schema

### Schema Structure

```
schemas/
├── contract_schema_v2.json          # Main contract schema (references others)
├── mixins_schema.json               # Mixin declarations
├── advanced_features_schema.json    # Advanced features
├── examples/                        # Example contracts
│   ├── valid_minimal_v1.yaml
│   ├── valid_with_mixins_v2.yaml
│   ├── valid_complete_v2.yaml
│   ├── invalid_mixin_name.yaml
│   ├── invalid_health_check_interval.yaml
│   ├── invalid_circuit_breaker_threshold.yaml
│   ├── invalid_missing_required_field.yaml
│   └── README.md
└── README.md                        # This file
```

## Quick Start

### Validate a Contract

```bash
# Using Python jsonschema library
python << 'EOF'
import json
import yaml
from jsonschema import validate, ValidationError

# Load schema
with open("contract_schema_v2.json") as f:
    schema = json.load(f)

# Load contract
with open("path/to/contract.yaml") as f:
    contract = yaml.safe_load(f)

# Validate
try:
    validate(contract, schema)
    print("✅ Contract is valid")
except ValidationError as e:
    print(f"❌ Validation failed: {e.message}")
EOF
```

### Using Contract Validator Tool

```bash
# Validate single contract
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract nodes/my_effect/v1_0_0/contract.yaml

# Or with custom schema directory
poetry run python -m omninode_bridge.codegen.contract_validator \
  --schema src/omninode_bridge/codegen/schemas/ \
  --contract nodes/my_effect/v1_0_0/contract.yaml

# Validate all contracts in directory
find nodes -name "contract.yaml" -exec \
  poetry run python -m omninode_bridge.codegen.contract_validator \
    --contract {} \;
```

## Schema Features

### 1. Schema Versioning

Contracts can specify their schema version:

```yaml
schema_version: "v2.0.0"
```

**Benefits:**
- Explicit version tracking
- Future-proof contract evolution
- Enables automated migrations

**Note:** Optional in v2.0.0 for backward compatibility, required in v3.0.0

### 2. Mixin Declarations

Declare mixins for automatic code generation:

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

**Available Mixins:**
- `MixinHealthCheck` - Health monitoring
- `MixinMetrics` - Performance metrics
- `MixinEventDrivenNode` - Kafka event consumption
- `MixinServiceRegistry` - Consul service discovery
- `MixinLogData` - Structured logging

**Validation:**
- Mixin name must match pattern: `^Mixin[A-Z][a-zA-Z0-9]*$`
- Configuration validated against mixin-specific schema
- Required dependencies checked

### 3. Advanced Features

Configure advanced capabilities:

```yaml
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
```

**Features:**
- **Circuit Breaker** - Fault tolerance with service-specific configs
- **Retry Policy** - Exponential backoff with configurable exceptions
- **Dead Letter Queue** - Failed event handling with monitoring
- **Transactions** - Database transaction management
- **Security Validation** - Input sanitization and SQL injection prevention
- **Observability** - Tracing, metrics, logging configuration

### 4. Backward Compatibility

v1.0.0 contracts work without modification:

```yaml
# Valid v1.0.0 contract (no changes needed)
name: "NodeMyEffect"
version: {major: 1, minor: 0, patch: 0}
node_type: "effect"
description: "My effect node"
```

**Compatibility:**
- ✅ All v1.0.0 contracts valid in v2.0.0
- ✅ Optional fields for new features
- ✅ No breaking changes

## Schema Validation Rules

### Mixin Validation

1. **Name Pattern**: Must match `^Mixin[A-Z][a-zA-Z0-9]*$`
2. **Required Fields**: `name` (required), `enabled` (default: true), `config` (optional)
3. **Configuration**: Must conform to mixin-specific schema
4. **Dependencies**: Required dependencies must be satisfied

**Example Validation Error:**

```
ValidationError: mixins[0].name: 'InvalidName' does not match pattern '^Mixin[A-Z][a-zA-Z0-9]*$'
```

### Advanced Features Validation

1. **Type Checking**: All fields must match expected types
2. **Range Constraints**: Numeric values within valid ranges
3. **Enum Values**: String fields must match allowed values
4. **Nested Validation**: Service-specific configs validated recursively

**Example Validation Error:**

```
ValidationError: advanced_features.circuit_breaker.failure_threshold: 150 is greater than maximum 100
```

### Core Contract Validation

1. **Required Fields**: `name`, `version`, `node_type`, `description`
2. **Node Naming**: Must follow `Node<Name><Type>` pattern
3. **Version Format**: Semantic versioning with major, minor, patch
4. **Node Type**: Must be one of: effect, compute, reducer, orchestrator

## Examples

See `examples/` directory for comprehensive examples:

- **Valid Examples**:
  - `valid_minimal_v1.yaml` - Minimal backward-compatible contract
  - `valid_with_mixins_v2.yaml` - Contract with mixins
  - `valid_complete_v2.yaml` - Complete contract with all features

- **Invalid Examples**:
  - `invalid_mixin_name.yaml` - Wrong mixin name pattern
  - `invalid_health_check_interval.yaml` - Value below minimum
  - `invalid_circuit_breaker_threshold.yaml` - Value above maximum
  - `invalid_missing_required_field.yaml` - Missing required field

## Integration with Code Generator

### Validation in Generation Pipeline

```python
from omninode_bridge.codegen.generator import NodeGenerator

# Generator validates contracts automatically
generator = NodeGenerator()

# Load and validate contract
try:
    contract = generator.load_contract("path/to/contract.yaml")
    # Generation proceeds...
except ValidationError as e:
    print(f"Contract validation failed: {e}")
```

### Pre-Commit Validation

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
for contract in $(git diff --cached --name-only | grep "contract.yaml"); do
  poetry run python -m omninode_bridge.codegen.contract_validator \
    --contract "$contract" || exit 1
done
```

## Schema Reference URIs

Each schema has a unique URI for external reference:

```json
{
  "$id": "https://omninode.dev/schemas/contract/v2.0.0",
  "$schema": "http://json-schema.org/draft-07/schema#"
}
```

**Usage in external tools:**

```bash
# Download schema
curl https://schemas.omninode.dev/contract/v2.0.0 > contract_schema_v2.json

# Validate with downloaded schema
jsonschema -i contract.yaml contract_schema_v2.json
```

## Mixin Configuration Reference

### MixinHealthCheck

```yaml
config:
  check_interval_ms: 30000       # Integer, min=1000, max=300000
  timeout_seconds: 5.0            # Number, min=1.0, max=60.0
  components:                     # Array
    - name: "database"            # String, required
      critical: true              # Boolean, default=true
      timeout_seconds: 5.0        # Number, min=0.5, max=30.0
```

### MixinMetrics

```yaml
config:
  collect_latency: true           # Boolean, default=true
  collect_throughput: true        # Boolean, default=true
  collect_error_rates: true       # Boolean, default=true
  percentiles: [50, 95, 99]       # Array<Integer>, min=1, max=100
  histogram_buckets: [10, 50]     # Array<Integer>, min=1
```

### MixinEventDrivenNode

```yaml
config:
  kafka_bootstrap_servers: "..."  # String, required
  consumer_group: "..."           # String, required
  topics: ["topic1"]              # Array<String>, minItems=1, required
  auto_offset_reset: "latest"     # Enum: earliest, latest, none
  enable_auto_commit: true        # Boolean, default=true
```

### MixinServiceRegistry

```yaml
config:
  consul_host: "..."              # String, required
  consul_port: 8500               # Integer, min=1, max=65535, default=8500
  service_name: "..."             # String, required
  health_check_interval_seconds: 10  # Integer, min=5, max=300, default=10
  deregister_critical_service_after: "30m"  # String, pattern=^[0-9]+[smh]$
```

### MixinLogData

```yaml
config:
  log_level: "INFO"               # Enum: DEBUG, INFO, WARNING, ERROR, CRITICAL
  structured: true                # Boolean, default=true
  include_correlation_id: true    # Boolean, default=true
  redact_sensitive_fields: true   # Boolean, default=true
  sensitive_field_patterns:       # Array<String>
    - "password"
```

## Advanced Features Reference

### Circuit Breaker

```yaml
circuit_breaker:
  enabled: true                   # Boolean, default=true
  failure_threshold: 5            # Integer, min=1, max=100, default=5
  recovery_timeout_ms: 60000      # Integer, min=1000, max=600000
  half_open_max_calls: 3          # Integer, min=1, max=10, default=3
  services:                       # Array (optional)
    - name: "postgres"            # String, required
      failure_threshold: 3        # Integer, min=1, max=100
      recovery_timeout_ms: 30000  # Integer, min=1000, max=600000
```

### Retry Policy

```yaml
retry_policy:
  enabled: true                   # Boolean, default=true
  max_attempts: 3                 # Integer, min=1, max=10, default=3
  initial_delay_ms: 1000          # Integer, min=100, max=10000, default=1000
  max_delay_ms: 30000             # Integer, min=1000, max=300000
  backoff_multiplier: 2.0         # Number, min=1.0, max=5.0, default=2.0
  retryable_exceptions:           # Array<String>
    - "TimeoutError"
  retryable_status_codes:         # Array<Integer>, min=100, max=599
    - 429
```

### Dead Letter Queue

```yaml
dead_letter_queue:
  enabled: true                   # Boolean, default=true
  max_retries: 3                  # Integer, min=1, max=10, default=3
  topic_suffix: ".dlq"            # String, pattern=^\.[a-z0-9_-]+$
  retry_delay_ms: 5000            # Integer, min=1000, max=300000
  alert_threshold: 100            # Integer, min=1, max=10000
  monitoring:
    enabled: true                 # Boolean, default=true
    check_interval_seconds: 60    # Integer, min=10, max=3600
```

### Transactions

```yaml
transactions:
  enabled: true                   # Boolean, default=true
  isolation_level: "READ_COMMITTED"  # Enum: READ_UNCOMMITTED, READ_COMMITTED, REPEATABLE_READ, SERIALIZABLE
  timeout_seconds: 30             # Integer, min=1, max=300, default=30
  rollback_on_error: true         # Boolean, default=true
  savepoints: true                # Boolean, default=true
```

### Security Validation

```yaml
security_validation:
  enabled: true                   # Boolean, default=true
  sanitize_inputs: true           # Boolean, default=true
  sanitize_logs: true             # Boolean, default=true
  validate_sql: true              # Boolean, default=true
  max_input_length: 10000         # Integer, min=100, max=1000000
  forbidden_patterns:             # Array<String> (regex patterns)
    - "(?i)(DROP|DELETE)\\s+TABLE"
  redact_fields:                  # Array<String>
    - "password"
```

### Observability

```yaml
observability:
  tracing:
    enabled: true                 # Boolean, default=true
    exporter: "otlp"              # Enum: otlp, jaeger, zipkin, console
    endpoint: "..."               # String
    sample_rate: 1.0              # Number, min=0.0, max=1.0, default=1.0
  metrics:
    enabled: true                 # Boolean, default=true
    prometheus_port: 9090         # Integer, min=1024, max=65535
    export_interval_seconds: 15   # Integer, min=5, max=300
  logging:
    structured: true              # Boolean, default=true
    json_format: true             # Boolean, default=true
    correlation_tracking: true    # Boolean, default=true
```

## Documentation

- **[Contract Schema v2.0 Reference](../../../docs/reference/CONTRACT_SCHEMA_V2.md)** - Complete documentation with examples and migration guide
- **[Schema Versioning Strategy](../../../docs/reference/SCHEMA_VERSIONING_STRATEGY.md)** - Schema evolution and versioning approach
- **[Mixin Enhancement Master Plan](../../../docs/planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)** - Original enhancement plan

## Testing Schemas

### Unit Tests

```python
import pytest
from jsonschema import validate, ValidationError

def test_valid_minimal_contract():
    """Test minimal v1.0.0 contract validates."""
    contract = {
        "name": "NodeMinimalEffect",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Minimal node"
    }
    validate(contract, schema)  # Should not raise

def test_invalid_mixin_name():
    """Test invalid mixin name raises ValidationError."""
    contract = {
        # ... valid fields ...
        "mixins": [{"name": "InvalidName"}]
    }
    with pytest.raises(ValidationError):
        validate(contract, schema)
```

### Integration Tests

```bash
# Test all example contracts
pytest tests/codegen/test_schema_validation.py -v

# Test schema evolution
pytest tests/codegen/test_schema_versioning.py -v
```

## Contributing

When modifying schemas:

1. **Maintain backward compatibility** in MINOR versions
2. **Update examples** to reflect schema changes
3. **Add validation tests** for new constraints
4. **Document changes** in CONTRACT_SCHEMA_V2.md
5. **Update version** in schema $id field

## Summary

The ONEX v2.0 contract schemas provide:

✅ **Comprehensive validation** - JSON Schema for all contract sections
✅ **Mixin support** - Declarative composition with validated configuration
✅ **Advanced features** - Circuit breakers, retry, DLQ, transactions, security, observability
✅ **Backward compatibility** - All v1.0.0 contracts remain valid
✅ **Schema versioning** - Clear evolution path with migration support

**Resources:**
- Main Documentation: `/docs/reference/CONTRACT_SCHEMA_V2.md`
- Versioning Strategy: `/docs/reference/SCHEMA_VERSIONING_STRATEGY.md`
- Examples: `examples/`
- Schema Files: `contract_schema_v2.json`, `mixins_schema.json`, `advanced_features_schema.json`
