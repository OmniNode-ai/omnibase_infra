# Mixin-Enhanced Code Generation End-to-End Test Suite

**Comprehensive integration tests for the complete mixin-enhanced code generation pipeline.**

## Overview

This test suite provides comprehensive validation of the entire code generation workflow with mixin support, covering:

- ✅ **50+ integration test cases**
- ✅ **All mixin combinations** (single, dual, complex, maximum)
- ✅ **All 4 node types** (Effect, Compute, Reducer, Orchestrator)
- ✅ **Full validation pipeline** (syntax, imports, mixins, security)
- ✅ **Performance benchmarks** (generation speed, validation speed)
- ✅ **Backward compatibility** (v1.0 contracts without mixins)
- ✅ **Error handling** (invalid contracts, unknown mixins, malformed YAML)
- ✅ **Real-world patterns** (database adapters, API clients, event processors)

## Test Structure

### Test Files

```
tests/integration/codegen/
├── test_mixin_generation_e2e.py    # Main test suite (50+ tests)
├── test_helpers.py                  # Utilities and assertion helpers
├── sample_contracts/                # 11+ sample YAML contracts
│   ├── minimal_effect.yaml         # No mixins (backward compat)
│   ├── health_check_only.yaml      # Single mixin
│   ├── health_metrics.yaml         # Two mixins
│   ├── event_driven_service.yaml   # Event-driven pattern
│   ├── database_adapter.yaml       # Full database adapter
│   ├── api_client.yaml             # API client pattern
│   ├── compute_cached.yaml         # Compute with caching
│   ├── reducer_persistent.yaml     # Reducer with persistence
│   ├── orchestrator_workflow.yaml  # Workflow orchestrator
│   ├── invalid_mixin_name.yaml     # Error case
│   ├── missing_dependency.yaml     # Error case
│   └── maximum_mixins.yaml         # Stress test (8+ mixins)
└── MIXIN_E2E_TEST_GUIDE.md         # This file
```

### Test Classes

1. **TestBasicE2E** - Basic end-to-end workflows (5 tests)
2. **TestMixinCombinations** - Common mixin combinations (10 tests)
3. **TestAllNodeTypes** - All 4 node types (4 tests)
4. **TestValidationPipeline** - Validation integration (6 tests)
5. **TestPerformance** - Performance benchmarks (4 tests)
6. **TestBackwardCompatibility** - v1.0 contract support (4 tests)
7. **TestErrorHandling** - Error scenarios (5 tests)
8. **TestRealWorldScenarios** - Production patterns (4 tests)

**Total: 42+ test methods** (expandable to 50+ with parametrization)

## Running Tests

### Run All Mixin E2E Tests

```bash
# Run all tests
pytest tests/integration/codegen/test_mixin_generation_e2e.py -v

# Run with coverage
pytest tests/integration/codegen/test_mixin_generation_e2e.py --cov=src/omninode_bridge/codegen -v

# Run specific test class
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestBasicE2E -v

# Run specific test
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestBasicE2E::test_generate_effect_node_with_health_check -v
```

### Run Performance Tests

```bash
# Performance tests only
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestPerformance -v

# With performance markers
pytest tests/integration/codegen/test_mixin_generation_e2e.py -m performance -v

# With timing
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestPerformance -vv --durations=10
```

### Run Error Handling Tests

```bash
# Error handling tests
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestErrorHandling -v

# Show full error output
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestErrorHandling -vv
```

## Test Coverage Areas

### 1. Basic End-to-End Tests (TestBasicE2E)

**Purpose**: Validate basic generation workflows

- ✅ Generate Effect node with single mixin (MixinHealthCheck)
- ✅ Generate Effect node with multiple mixins (Health + Metrics)
- ✅ Generate minimal Effect node without mixins (backward compat)
- ✅ Generate node with event patterns configured
- ✅ Generate node with service registry configuration

**Success Criteria**:
- Valid Python syntax
- Correct mixin imports
- Correct mixin inheritance
- Mixin methods present
- No syntax errors

### 2. Mixin Combinations (TestMixinCombinations)

**Purpose**: Test common mixin patterns

**Tested Combinations**:
1. ✅ MixinHealthCheck + MixinMetrics
2. ✅ MixinEventDrivenNode + MixinServiceRegistry + MixinHealthCheck
3. ✅ MixinCaching + MixinMetrics (Compute node)
4. ✅ MixinEventDrivenNode + MixinMetrics (Orchestrator)
5. ✅ MixinHealthCheck + MixinMetrics + MixinCaching + MixinServiceRegistry
6. ✅ MixinHealthCheck + MixinMetrics + MixinRequestResponseIntrospection
7. ✅ Maximum mixins (8+ mixins for stress testing)
8. ✅ Single mixin only
9. ✅ MixinEventBus + MixinMetrics (Reducer)
10. ✅ MixinHashComputation + MixinCaching (Compute)

**Success Criteria**:
- All declared mixins in inheritance
- All mixins imported correctly
- Valid Python syntax
- No duplicate mixins

### 3. All Node Types (TestAllNodeTypes)

**Purpose**: Validate generation for all 4 node types

- ✅ Effect node generation
- ✅ Compute node generation
- ✅ Reducer node generation
- ✅ Orchestrator node generation

**Success Criteria**:
- Correct base class (NodeEffect, NodeCompute, etc.)
- Correct execute method (execute_effect, execute_compute, etc.)
- Valid syntax for each type
- Type-specific patterns

### 4. Validation Pipeline (TestValidationPipeline)

**Purpose**: Test NodeValidator integration

**Validation Stages Tested**:
1. ✅ Syntax validation (AST parsing)
2. ✅ Import validation (all mixins imported)
3. ✅ Mixin validation (declared vs inherited)
4. ✅ Security validation (no eval/exec)
5. ✅ ONEX compliance (naming conventions)
6. ✅ Type checking (optional, disabled for speed)

**Performance Targets**:
- Validation < 200ms without type checking
- Validation < 1s with type checking (if enabled)

### 5. Performance Tests (TestPerformance)

**Purpose**: Benchmark generation and validation speed

**Benchmarks**:
- ✅ Simple node generation < 5 seconds
- ✅ Complex node generation (5+ mixins) < 10 seconds
- ✅ Validation < 200ms (fast mode)
- ✅ Batch generation (10 nodes) < 30 seconds

**Metrics Tracked**:
- Generation time (seconds)
- Validation time (milliseconds)
- Lines of code generated
- Memory usage (future)

### 6. Backward Compatibility (TestBackwardCompatibility)

**Purpose**: Ensure v1.0 contracts still work

- ✅ Parse v1.0 contract without errors
- ✅ Generate from v1.0 contract (no mixins)
- ✅ Handle deprecated fields gracefully
- ✅ Regenerate existing nodes

**Success Criteria**:
- No breaking changes for v1.0 contracts
- Graceful degradation (warnings, not errors)
- Consistent output format

### 7. Error Handling (TestErrorHandling)

**Purpose**: Test error cases and recovery

**Error Scenarios**:
- ✅ Invalid contract schema (missing required fields)
- ✅ Unknown mixin name (MixinDoesNotExist)
- ✅ Missing mixin dependencies
- ✅ Invalid node_type field
- ✅ Malformed YAML syntax
- ✅ Missing required fields (node_id, node_type)

**Success Criteria**:
- Clear error messages
- Graceful failure (no crashes)
- Helpful debugging information

### 8. Real-World Scenarios (TestRealWorldScenarios)

**Purpose**: Test production-ready patterns

**Patterns Tested**:
1. ✅ Database adapter (Health + Metrics + Caching + ServiceRegistry)
2. ✅ API client (Health + Metrics + Introspection)
3. ✅ Event processor (EventDriven + Health + Metrics)
4. ✅ Workflow coordinator (EventDriven + Metrics + ServiceRegistry)

**Success Criteria**:
- Production-ready code quality
- All required mixins present
- Proper configuration
- Valid syntax and structure

## Sample Contracts Reference

### Minimal Effect (No Mixins)

```yaml
# tests/integration/codegen/sample_contracts/minimal_effect.yaml
node_id: "minimal_effect"
node_type: "effect"
version: "v1_0_0"
# No mixin_configuration - backward compatibility
```

### Health Check Only (Single Mixin)

```yaml
# tests/integration/codegen/sample_contracts/health_check_only.yaml
node_id: "health_check_effect"
node_type: "effect"
version: "v1_0_0"

mixin_configuration:
  mixins:
    - mixin_name: "MixinHealthCheck"
      config:
        health_check_interval_seconds: 30
```

### Database Adapter (Full Production Pattern)

```yaml
# tests/integration/codegen/sample_contracts/database_adapter.yaml
node_id: "database_adapter_effect"
node_type: "effect"
version: "v1_0_0"

mixin_configuration:
  mixins:
    - mixin_name: "MixinHealthCheck"
    - mixin_name: "MixinMetrics"
    - mixin_name: "MixinCaching"
    - mixin_name: "MixinServiceRegistry"
    - mixin_name: "MixinLogData"
```

### Maximum Mixins (Stress Test)

```yaml
# tests/integration/codegen/sample_contracts/maximum_mixins.yaml
node_id: "maximum_mixins_effect"
node_type: "effect"
version: "v1_0_0"

mixin_configuration:
  mixins:
    - mixin_name: "MixinHealthCheck"
    - mixin_name: "MixinMetrics"
    - mixin_name: "MixinLogData"
    - mixin_name: "MixinRequestResponseIntrospection"
    - mixin_name: "MixinEventBus"
    - mixin_name: "MixinServiceRegistry"
    - mixin_name: "MixinCaching"
    - mixin_name: "MixinHashComputation"
    - mixin_name: "MixinCanonicalYAMLSerializer"
```

## Test Utilities

### Code Analysis Helpers

```python
from test_helpers import CodeAnalyzer

# Analyze generated code
result = CodeAnalyzer.analyze_code(code)

print(f"Valid Python: {result.is_valid_python}")
print(f"Class: {result.class_name}")
print(f"Mixins: {result.mixins}")
print(f"Methods: {result.methods}")
```

### Mixin Analysis Helpers

```python
from test_helpers import MixinAnalyzer

# Analyze mixin usage
result = MixinAnalyzer.analyze_mixins(code, contract)

print(f"Declared: {result.declared_mixins}")
print(f"Inherited: {result.inherited_mixins}")
print(f"Missing imports: {result.missing_imports}")
```

### Assertion Helpers

```python
from test_helpers import AssertionHelpers

# Custom assertions
AssertionHelpers.assert_valid_python(code)
AssertionHelpers.assert_has_mixin(code, "MixinHealthCheck")
AssertionHelpers.assert_has_method(code, "execute_effect")
AssertionHelpers.assert_imports_mixin(code, "MixinMetrics")
```

## Success Metrics

### Test Coverage

- **Total Tests**: 42+ test methods (expandable to 50+)
- **Coverage Target**: 95%+ of codegen module
- **Pass Rate Target**: 100% (all tests passing)
- **Execution Time**: < 2 minutes for full suite

### Code Quality

- **Valid Python**: 100% of generated code
- **Mixin Accuracy**: 100% (declared == inherited == imported)
- **Performance**: All benchmarks met
- **Error Handling**: All error cases covered

### Real-World Validation

- ✅ Database adapter pattern validated
- ✅ API client pattern validated
- ✅ Event processor pattern validated
- ✅ Workflow orchestrator pattern validated

## Troubleshooting

### Common Issues

**Issue**: Tests failing with "Contract not found"
**Solution**: Ensure sample_contracts directory exists with all YAML files

**Issue**: Import errors for omnibase_core mixins
**Solution**: Ensure omnibase_core is installed and up to date

**Issue**: Performance tests timing out
**Solution**: Increase timeout or run performance tests separately

**Issue**: Validation tests failing
**Solution**: Check NodeValidator is initialized with correct settings

### Debug Mode

```bash
# Run with verbose output
pytest tests/integration/codegen/test_mixin_generation_e2e.py -vv

# Run with print statements
pytest tests/integration/codegen/test_mixin_generation_e2e.py -s

# Run single test with debugging
pytest tests/integration/codegen/test_mixin_generation_e2e.py::TestBasicE2E::test_generate_effect_node_with_health_check -vv -s
```

## Future Enhancements

### Planned Additions

1. **Parametrized Tests** - Generate test combinations programmatically
2. **Contract Fuzzing** - Random contract generation for stress testing
3. **Performance Profiling** - Memory usage and CPU profiling
4. **Integration with CI/CD** - Automated test runs on PR
5. **Test Report Generation** - HTML test reports with coverage
6. **Mixin Dependency Validation** - Auto-detect missing dependencies
7. **Contract Schema Validation** - JSON Schema validation for contracts
8. **Generated Code Linting** - Run flake8/black on generated code

### Expansion Areas

- **More Node Types**: Support for future node types
- **Custom Mixins**: Test with user-defined mixins
- **Complex Workflows**: Multi-node workflow testing
- **Edge Cases**: More error scenarios and boundary conditions

## Contributing

### Adding New Tests

1. Create sample contract in `sample_contracts/`
2. Add test method to appropriate test class
3. Use existing fixtures and helpers
4. Document expected behavior
5. Run tests to verify

### Adding New Fixtures

1. Add fixture to `test_mixin_generation_e2e.py`
2. Document fixture purpose and usage
3. Use type hints for clarity
4. Consider scope (function, class, module)

### Adding New Helpers

1. Add helper to `test_helpers.py`
2. Add to appropriate class (CodeAnalyzer, MixinAnalyzer, etc.)
3. Include docstring with examples
4. Add type hints

## References

- **Main Test File**: `test_mixin_generation_e2e.py`
- **Helpers**: `test_helpers.py`
- **Sample Contracts**: `sample_contracts/*.yaml`
- **YAMLContractParser**: `src/omninode_bridge/codegen/yaml_contract_parser.py`
- **MixinInjector**: `src/omninode_bridge/codegen/mixin_injector.py`
- **NodeValidator**: `src/omninode_bridge/codegen/validation/validator.py`
- **TemplateEngine**: `src/omninode_bridge/codegen/template_engine.py`

---

**Last Updated**: 2024-11-04
**Test Suite Version**: 1.0
**Author**: Test Generator
