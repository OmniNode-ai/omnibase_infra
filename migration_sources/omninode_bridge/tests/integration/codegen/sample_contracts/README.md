# Sample Contracts for Mixin E2E Testing

This directory contains 12 sample YAML contracts for comprehensive mixin-enhanced code generation testing.

## Contract Categories

### Backward Compatibility

1. **minimal_effect.yaml** - Minimal Effect node without mixins (v1.0 style)
   - Purpose: Validate backward compatibility
   - Mixins: None
   - Use Case: Legacy contract support

### Single Mixin

2. **health_check_only.yaml** - Effect node with single mixin
   - Purpose: Test single mixin integration
   - Mixins: MixinHealthCheck
   - Use Case: Basic health monitoring

### Dual Mixins

3. **health_metrics.yaml** - Effect node with two mixins
   - Purpose: Test common dual-mixin pattern
   - Mixins: MixinHealthCheck, MixinMetrics
   - Use Case: Monitored services

### Event-Driven Patterns

4. **event_driven_service.yaml** - Complete event-driven service
   - Purpose: Test event-driven architecture
   - Mixins: MixinEventDrivenNode, MixinServiceRegistry, MixinHealthCheck
   - Use Case: Event processing services

### Production Patterns

5. **database_adapter.yaml** - Full database adapter pattern
   - Purpose: Test production database adapter
   - Mixins: MixinHealthCheck, MixinMetrics, MixinCaching, MixinServiceRegistry, MixinLogData
   - Use Case: Database access layer

6. **api_client.yaml** - API client pattern
   - Purpose: Test external API integration
   - Mixins: MixinHealthCheck, MixinMetrics, MixinRequestResponseIntrospection, MixinLogData
   - Use Case: External service clients

### Node Type Variants

7. **compute_cached.yaml** - Compute node with caching
   - Purpose: Test Compute node type with mixins
   - Mixins: MixinCaching, MixinMetrics, MixinHashComputation
   - Use Case: Expensive computation caching

8. **reducer_persistent.yaml** - Reducer with persistence
   - Purpose: Test Reducer node type with mixins
   - Mixins: MixinMetrics, MixinEventBus, MixinHealthCheck, MixinCanonicalYAMLSerializer
   - Use Case: Stream aggregation

9. **orchestrator_workflow.yaml** - Workflow orchestrator
   - Purpose: Test Orchestrator node type with mixins
   - Mixins: MixinEventDrivenNode, MixinMetrics, MixinHealthCheck, MixinServiceRegistry
   - Use Case: Workflow coordination

### Error Cases

10. **invalid_mixin_name.yaml** - Error case with unknown mixin
    - Purpose: Test error handling for invalid mixin names
    - Mixins: MixinDoesNotExist (invalid), MixinHealthCheck
    - Use Case: Validation testing

11. **missing_dependency.yaml** - Error case with missing dependencies
    - Purpose: Test mixin dependency validation
    - Mixins: MixinEventDrivenNode (missing required dependencies)
    - Use Case: Dependency checking

### Stress Testing

12. **maximum_mixins.yaml** - Maximum mixin complexity
    - Purpose: Stress test with many mixins
    - Mixins: 9 mixins (all compatible mixins together)
    - Use Case: Complexity testing

## Mixin Coverage

### Mixins Tested

- ✅ MixinHealthCheck (8+ contracts)
- ✅ MixinMetrics (8+ contracts)
- ✅ MixinLogData (2+ contracts)
- ✅ MixinRequestResponseIntrospection (2+ contracts)
- ✅ MixinEventDrivenNode (3+ contracts)
- ✅ MixinEventBus (2+ contracts)
- ✅ MixinServiceRegistry (4+ contracts)
- ✅ MixinCaching (3+ contracts)
- ✅ MixinHashComputation (2+ contracts)
- ✅ MixinCanonicalYAMLSerializer (1+ contracts)

**Coverage**: 10/10 mixins (100%)

## Node Type Coverage

- ✅ Effect (8 contracts)
- ✅ Compute (1 contract)
- ✅ Reducer (1 contract)
- ✅ Orchestrator (1 contract)

**Coverage**: 4/4 node types (100%)

## Usage in Tests

These contracts are used by test fixtures in `test_mixin_generation_e2e.py`:

```python
# Load contract by name
contract = load_contract("health_metrics")

# Generate node from contract
node_file = await generate_node(contract)

# Read and validate generated code
code = node_file.read_text()
assert_valid_python(code)
assert_has_mixin(code, "MixinHealthCheck")
```

## Adding New Contracts

When adding new sample contracts:

1. Create YAML file in this directory
2. Follow naming convention: `{pattern}_{type}.yaml`
3. Include required fields: `node_id`, `node_type`, `version`, `metadata`
4. Add mixin configuration if testing mixins
5. Document purpose in this README
6. Add corresponding test in `test_mixin_generation_e2e.py`

## Contract Structure

All contracts follow this structure:

```yaml
---
node_id: "unique_node_id"
node_type: "effect|compute|reducer|orchestrator"
version: "v1_0_0"
namespace: "omninode.services.{domain}"

metadata:
  name: "Human-readable name"
  description: "Description of purpose"
  author: "Test Generator"
  tags: ["tag1", "tag2"]

mixin_configuration:  # Optional
  mixins:
    - mixin_name: "MixinName"
      config:
        key: value

event_patterns:  # Optional
  subscribes:
    - topic: "topic.name"
      event_type: "EventType"
  publishes:
    - topic: "topic.name"
      event_type: "EventType"

input_schema:  # Optional
  type: "object"
  properties: {}

output_schema:  # Optional
  type: "object"
  properties: {}
```

## Test Coverage

Each contract is used in one or more test methods to validate:

- ✅ Syntax validity
- ✅ Mixin imports
- ✅ Mixin inheritance
- ✅ Method generation
- ✅ Configuration handling
- ✅ Error cases (for error contracts)

## References

- **Test Suite**: `../test_mixin_generation_e2e.py`
- **Test Guide**: `../MIXIN_E2E_TEST_GUIDE.md`
- **Test Summary**: `../MIXIN_E2E_TEST_SUMMARY.md`
- **MixinInjector**: `src/omninode_bridge/codegen/mixin_injector.py`
