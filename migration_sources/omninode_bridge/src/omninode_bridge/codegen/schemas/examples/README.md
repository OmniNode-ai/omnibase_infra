# Contract Schema Examples

This directory contains example contracts demonstrating valid and invalid contract patterns for ONEX v2.0.

## Valid Examples

### Backward Compatibility

**`valid_minimal_v1.yaml`** - Minimal v1.0.0 Contract
- Demonstrates backward compatibility
- No schema_version, mixins, or advanced_features
- Validates successfully with v2.0.0 schema
- Use case: Legacy contracts that don't need v2.0 features

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract valid_minimal_v1.yaml
```

**Expected:** ✅ Valid (v1.0.0 compatible)

---

### Mixin Usage

**`valid_with_mixins_v2.yaml`** - Contract with Mixins
- Demonstrates mixin declarations
- Includes MixinHealthCheck and MixinMetrics
- Proper mixin configuration
- Use case: Production nodes with health monitoring and metrics

**Key Features:**
- Health check with multiple components
- Metrics collection with custom percentiles
- Component-level timeout configuration

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract valid_with_mixins_v2.yaml
```

**Expected:** ✅ Valid (v2.0.0 with mixins)

---

### Complete Feature Set

**`valid_complete_v2.yaml`** - Complete v2.0.0 Contract
- Demonstrates all v2.0.0 features
- 5 mixins with full configuration
- All advanced features enabled
- Complete subcontracts
- Use case: Production-grade event processor

**Key Features:**
- MixinHealthCheck, MixinMetrics, MixinEventDrivenNode, MixinServiceRegistry, MixinLogData
- Circuit breaker with service-specific configs
- Retry policy with exponential backoff
- Dead letter queue with monitoring
- Database transactions
- Security validation
- Full observability (tracing, metrics, logging)

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract valid_complete_v2.yaml
```

**Expected:** ✅ Valid (v2.0.0 complete)

---

## Invalid Examples

These examples demonstrate common validation errors and their messages.

### Invalid Mixin Name

**`invalid_mixin_name.yaml`**
- **Error:** Mixin name doesn't match required pattern
- **Pattern:** `^Mixin[A-Z][a-zA-Z0-9]*$`
- **Issue:** Name doesn't start with "Mixin"

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract invalid_mixin_name.yaml
```

**Expected Error:**
```
❌ ValidationError: mixins[0].name: 'InvalidMixinName' does not match pattern '^Mixin[A-Z][a-zA-Z0-9]*$'
```

---

### Invalid Health Check Configuration

**`invalid_health_check_interval.yaml`**
- **Error:** check_interval_ms below minimum threshold
- **Minimum:** 1000ms
- **Actual:** 500ms

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract invalid_health_check_interval.yaml
```

**Expected Error:**
```
❌ ValidationError: mixins[0].config.check_interval_ms: 500 is less than minimum 1000
```

---

### Invalid Circuit Breaker Configuration

**`invalid_circuit_breaker_threshold.yaml`**
- **Error:** failure_threshold exceeds maximum
- **Maximum:** 100
- **Actual:** 150

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract invalid_circuit_breaker_threshold.yaml
```

**Expected Error:**
```
❌ ValidationError: advanced_features.circuit_breaker.failure_threshold: 150 is greater than maximum 100
```

---

### Missing Required Field

**`invalid_missing_required_field.yaml`**
- **Error:** Missing required 'name' field in mixin declaration
- **Required:** name
- **Provided:** enabled, config

**Validation:**
```bash
poetry run python -m omninode_bridge.codegen.contract_validator \
  --contract invalid_missing_required_field.yaml
```

**Expected Error:**
```
❌ ValidationError: mixins[0]: 'name' is a required property
```

---

## Validation Testing

### Validate All Examples

```bash
# Validate all valid examples (should pass)
for contract in valid_*.yaml; do
  echo "Validating: $contract"
  poetry run python -m omninode_bridge.codegen.contract_validator \
    --contract "$contract"
done

# Validate all invalid examples (should fail with expected errors)
for contract in invalid_*.yaml; do
  echo "Validating: $contract (expecting error)"
  poetry run python -m omninode_bridge.codegen.contract_validator \
    --contract "$contract" || echo "Expected validation error"
done
```

### Python Validation

```python
import json
import yaml
from jsonschema import validate, ValidationError

# Load schema
with open("../contract_schema_v2.json") as f:
    schema = json.load(f)

# Load contract
with open("valid_complete_v2.yaml") as f:
    contract = yaml.safe_load(f)

# Validate
try:
    validate(contract, schema)
    print("✅ Contract is valid")
except ValidationError as e:
    print(f"❌ Validation failed: {e.message}")
```

---

## Example Usage Patterns

### Pattern 1: Minimal Node (No Mixins)

Best for: Simple utilities, testing, prototypes

```yaml
name: "NodeSimpleEffect"
version: {major: 1, minor: 0, patch: 0}
node_type: "effect"
description: "Simple node"
subcontracts:
  effect:
    operations: ["initialize", "shutdown"]
```

### Pattern 2: Production Node (Health + Metrics)

Best for: Production services, critical paths

```yaml
schema_version: "v2.0.0"
name: "NodeProductionEffect"
# ... core fields ...
mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true
```

### Pattern 3: Event-Driven Node

Best for: Kafka consumers, stream processors

```yaml
schema_version: "v2.0.0"
name: "NodeEventProcessorEffect"
# ... core fields ...
mixins:
  - name: "MixinEventDrivenNode"
    enabled: true
    config:
      kafka_bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      topics: ["events"]
```

### Pattern 4: Full-Featured Node

Best for: Complex services, mission-critical systems

```yaml
schema_version: "v2.0.0"
name: "NodeAdvancedEffect"
# ... core fields ...
mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true
  - name: "MixinEventDrivenNode"
    enabled: true
advanced_features:
  circuit_breaker: {enabled: true}
  retry_policy: {enabled: true}
  observability:
    tracing: {enabled: true}
```

---

## Common Validation Errors

### Error: Invalid Node Name Pattern

```
mixins[0].name: 'MyMixin' does not match pattern '^Mixin[A-Z][a-zA-Z0-9]*$'
```

**Fix:** Ensure mixin name starts with "Mixin" followed by PascalCase name

### Error: Value Below/Above Threshold

```
mixins[0].config.check_interval_ms: 500 is less than minimum 1000
```

**Fix:** Review schema constraints and adjust value to valid range

### Error: Missing Required Property

```
mixins[0]: 'name' is a required property
```

**Fix:** Add required field to mixin declaration

### Error: Additional Properties Not Allowed

```
mixins[0]: Additional properties are not allowed ('extra_field' was unexpected)
```

**Fix:** Remove fields not defined in schema or check for typos

---

## Resources

- **Schema Files**: `../contract_schema_v2.json`, `../mixins_schema.json`, `../advanced_features_schema.json`
- **Documentation**: `/docs/reference/CONTRACT_SCHEMA_V2.md`
- **Versioning**: `/docs/reference/SCHEMA_VERSIONING_STRATEGY.md`
- **Master Plan**: `/docs/planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md`

---

## Contributing Examples

To add a new example:

1. Create YAML file in this directory
2. Prefix with `valid_` or `invalid_`
3. Add descriptive comment header explaining the example
4. Update this README with example description
5. Test validation before committing

**Example template:**

```yaml
# Example: [Brief Description]
#
# [Detailed explanation of what this demonstrates]
# [Expected validation result]

schema_version: "v2.0.0"
name: "NodeExampleEffect"
# ... contract fields ...
```
