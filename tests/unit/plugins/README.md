# ComputePlugin Unit Tests

Comprehensive unit tests for the ComputePlugin system (OMN-813 Phase 2b).

## Current Status

**⏸ TESTS PAUSED** - Waiting for implementation dependencies:
- `ComputePluginBase` abstract base class
- `JsonNormalizerPlugin` example plugin

## Test Coverage

The test suite includes 22 comprehensive tests covering:

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

### JsonNormalizerPlugin Functionality (6 tests)
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

## Activating Tests

Once `ComputePluginBase` and `JsonNormalizerPlugin` are implemented:

1. **Remove skip marker** from `test_compute_plugin_base.py`:
   ```python
   # Delete or comment out this line:
   # pytestmark = pytest.mark.skip(reason="Waiting for implementation")
   ```

2. **Run tests**:
   ```bash
   poetry run pytest tests/unit/plugins/test_compute_plugin_base.py -v
   ```

3. **Expected result**: All 22 tests should pass with >90% coverage

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
- `omnibase_infra.plugins.protocol_compute_plugin` (✅ exists)
- `omnibase_infra.plugins.compute_plugin_base` (⏸ pending)
- `omnibase_infra.plugins.examples.json_normalizer_plugin` (⏸ pending)

## Related Work

- **OMN-813 Phase 1**: Protocol definition (completed)
- **OMN-813 Phase 2a**: Base class and example plugin (pending)
- **OMN-813 Phase 2b**: Unit tests (this file - completed, waiting for activation)
