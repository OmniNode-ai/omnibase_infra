# PluginCompute Unit Tests

Comprehensive unit tests for the PluginCompute system (OMN-813 Phase 2b).

## Current Status

**✅ TESTS ACTIVE** - Implementation complete:
- `PluginComputeBase` abstract base class
- `PluginJsonNormalizer` example plugin
- ONEX 4-node architecture compliance verification

## Test Coverage

The test suite includes 35+ comprehensive tests covering:

### Protocol Conformance (2 tests)
- ✅ `test_protocol_conformance_with_base_class` - Verify base class conforms to protocol
- ✅ `test_protocol_conformance_with_example` - Verify example plugin conforms to protocol

### Abstract Base Class Behavior (3 tests)
- ✅ `test_base_class_is_abstract` - Cannot instantiate base class directly
- ✅ `test_execute_method_is_abstract` - Must override execute() method
- ✅ `test_validation_hooks_are_optional` - Can use base class without validation hooks

### Validation Hook Execution (3 tests)
- ✅ `test_validate_input_called_before_execute` - Hook execution order
- ✅ `test_validate_output_called_after_execute` - Hook execution order
- ✅ `test_validation_errors_propagate` - Validation exceptions bubble up

### Determinism Guarantees (3 tests)
- ✅ `test_same_input_produces_same_output` - Core determinism guarantee
- ✅ `test_determinism_across_multiple_calls` - Repeated calls consistency
- ✅ `test_different_input_produces_different_output` - Input sensitivity

### PluginJsonNormalizer Functionality (6 tests)
- ✅ `test_normalizes_simple_dict` - Sorts keys in simple dict
- ✅ `test_normalizes_nested_dict` - Recursive key sorting
- ✅ `test_normalizes_dict_with_lists` - Handles lists in dict
- ✅ `test_normalizes_empty_dict` - Edge case: empty input
- ✅ `test_preserves_values` - Values unchanged, only keys sorted
- ✅ `test_deterministic_normalization` - Same JSON → same output

### Validation Hook Integration (3 tests)
- ✅ `test_validation_hooks_execute_in_order` - Hooks with example plugin
- ✅ `test_input_validation_failure` - Input validation errors
- ✅ `test_output_validation_failure` - Output validation errors

### Context Propagation (2 tests)
- ✅ `test_context_available_in_execute` - Context parameter in execute()
- ✅ `test_context_available_in_validation_hooks` - Context in validation hooks

### ONEX Architecture Compliance (13+ tests)
- ✅ `test_plugin_must_not_perform_network_io` - Network I/O belongs in EFFECT layer
- ✅ `test_plugin_must_not_perform_file_io` - File I/O belongs in EFFECT layer
- ✅ `test_plugin_must_not_access_database` - Database access belongs in EFFECT layer
- ✅ `test_plugin_must_not_use_global_state` - Global state breaks determinism
- ✅ `test_plugin_must_be_deterministic` - Same inputs → same outputs
- ✅ `test_plugin_must_not_use_non_deterministic_randomness` - No random without seed
- ✅ `test_plugin_can_use_deterministic_randomness_with_seed` - Seeded random allowed
- ✅ `test_plugin_must_not_access_current_time_non_deterministically` - Time must be provided
- ✅ `test_plugin_can_use_time_if_provided_in_context` - Time as input allowed
- ✅ `test_plugin_must_not_modify_input_data` - No side effects on input
- ✅ `test_plugin_must_not_modify_context` - No side effects on context
- ✅ `test_plugin_separation_from_effect_layer` - COMPUTE vs EFFECT separation
- ✅ `test_plugin_must_not_perform_multi_source_aggregation` - REDUCER layer responsibility
- ✅ `test_plugin_must_not_coordinate_workflows` - ORCHESTRATOR layer responsibility

### Architectural Benefits (3 tests)
- ✅ `test_compute_plugins_are_easily_testable` - No mocking required
- ✅ `test_compute_plugins_are_composable` - Plugins compose without coordination
- ✅ `test_compute_plugins_enable_horizontal_scaling` - Stateless scalability

## Running Tests

Tests are now active and can be run directly:

1. **Run all plugin tests**:
   ```bash
   uv run pytest tests/unit/plugins/ -v
   ```

2. **Run specific test suites**:
   ```bash
   # Protocol conformance and base functionality
   uv run pytest tests/unit/plugins/test_plugin_compute_base.py -v

   # ONEX architecture compliance verification
   uv run pytest tests/unit/plugins/test_onex_architecture_compliance.py -v
   ```

3. **Expected result**: All 35+ tests should pass with >90% coverage

## Test Design Principles

Following ONEX testing standards:
- ✅ Clear test names describing what is tested
- ✅ Arrange-Act-Assert pattern
- ✅ Both happy path and edge cases
- ✅ Comprehensive coverage (22 tests)
- ✅ Deterministic tests (no flakiness)
- ✅ Clear test documentation

## Dependencies

Tests require:
- `pytest` framework
- `omnibase_infra.protocols.protocol_plugin_compute` (✅ exists)
- `omnibase_infra.plugins.plugin_compute_base` (✅ exists)
- `omnibase_infra.plugins.examples.plugin_json_normalizer` (✅ exists)

## ONEX 4-Node Architecture Documentation

Compute plugins integrate with the COMPUTE layer of ONEX's 4-node architecture:

### Architecture Overview

- **EFFECT (NodeEffectService)**:
  * External I/O operations (database, network, filesystem, message bus)
  * Service integrations (Kafka, Consul, Vault, Redis, PostgreSQL)
  * Connection pooling and circuit breaker patterns
  * Examples: postgres_adapter, consul_adapter, kafka_adapter

- **COMPUTE (NodeComputeService)** - WHERE PLUGINS LIVE:
  * Pure data transformations without side effects
  * Deterministic algorithms and business logic
  * Stateless computation (THIS PROTOCOL)
  * Examples: data validation, JSON normalization, aggregation

- **REDUCER (NodeReducerService)**:
  * State aggregation from multiple sources
  * Event sourcing and state reconstruction
  * Multi-source data consolidation
  * Examples: infrastructure_reducer, state_aggregator

- **ORCHESTRATOR (NodeOrchestratorService)**:
  * Workflow coordination across multiple nodes
  * Multi-step process management
  * Service orchestration patterns
  * Examples: infrastructure_orchestrator, workflow_coordinator

### Why Plugins Belong in COMPUTE Layer

1. **Determinism Guarantee**: COMPUTE nodes require reproducible outputs
2. **No Side Effects**: EFFECT nodes handle all I/O, COMPUTE stays pure
3. **Testability**: Pure functions are trivially testable (no mocking required)
4. **Composability**: Plugins can be combined without coordination complexity
5. **Scalability**: Stateless computation enables horizontal scaling

### Architectural Benefits

- Clear separation of concerns (I/O vs computation)
- Simplified testing (mock I/O, test computation independently)
- Enhanced reusability (plugins can be shared across nodes)
- Improved scalability (stateless plugins enable easy scaling)
- Better maintainability (changes to computation don't affect I/O layer)

## Related Work

- **OMN-813 Phase 1**: Protocol definition (completed)
- **OMN-813 Phase 2a**: Base class and example plugin (completed)
- **OMN-813 Phase 2b**: Unit tests (completed and active)
- **PR #45**: ONEX architecture integration documentation (this work)
