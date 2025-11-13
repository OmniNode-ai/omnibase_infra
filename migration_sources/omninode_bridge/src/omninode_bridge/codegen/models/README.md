# Test Contract Models - ONEX Standards Compliant

**Version**: 1.0.0
**Status**: ✅ Validated and Ready for Use
**Pattern**: Follows ModelContractEffect architecture

## Overview

This package provides Pydantic models for test contract generation following ONEX v2.0 standards. These models enable codified test specifications that can be used to automatically generate comprehensive test suites.

**Note**: This is a prototype implementation in omninode_bridge that will be migrated to omnibase_core once the pattern is validated in production.

## Architecture

The test contract system follows the same architectural patterns as `ModelContractEffect` from omnibase_core:

- **INTERFACE_VERSION**: Locked at 1.0.0 for code generation stability
- **ZERO TOLERANCE**: No `Any` types allowed in implementation
- **Field Validators**: Comprehensive validation using Pydantic v2
- **YAML Support**: `from_yaml()` and `to_yaml()` methods
- **Type Safety**: Full type hints throughout

## Models

### Core Models

#### `ModelContractTest`
Main contract model for test generation specifications.

**Key Fields**:
- Test identification (name, version, description)
- Target node information (target_node, target_version, target_node_type)
- Coverage requirements (coverage_minimum, coverage_target)
- Test types (unit, integration, contract, performance, e2e, etc.)
- Test targets (methods/scenarios to test)
- Mock requirements and configuration
- Pytest configuration

**Validators**:
- `validate_coverage_target`: Ensures target >= minimum
- `validate_target_node_type`: Validates against ONEX node types
- `validate_test_types_consistency`: Ensures at least one test type
- `_validate_test_contract_requirements`: Post-init validation

### Supporting Models

#### `EnumTestType`
Test type enumeration with values:
- UNIT, INTEGRATION, CONTRACT, PERFORMANCE, E2E
- LOAD, STRESS, SECURITY, REGRESSION, SMOKE

#### `ModelTestTarget`
Specification for individual test cases including:
- Target method/function/component
- Test scenarios and expected behaviors
- Input parameters and expected outputs
- Edge cases and error conditions
- Assertions and test priority

#### `ModelMockRequirements`
Mock configuration including:
- Dependencies to mock (classes, modules)
- External services (APIs, databases)
- Database, HTTP, Kafka mocking
- Filesystem and datetime mocking
- Custom mock specifications
- Fixture configuration

#### `ModelTestConfiguration`
Pytest and execution configuration:
- Pytest markers, plugins, options
- Parallel execution settings
- Coverage configuration
- Required fixtures
- Test data and environment setup
- Quality gates (type checking, linting)

## Usage

### Loading from YAML

```python
from omninode_bridge.codegen.models.model_contract_test import ModelContractTest

# Load contract from YAML file
with open("test_contract.yaml") as f:
    yaml_content = f.read()

contract = ModelContractTest.from_yaml(yaml_content)

print(f"Contract: {contract.name} v{contract.version}")
print(f"Target: {contract.target_node} v{contract.target_version}")
print(f"Coverage: {contract.coverage_target}%")
print(f"Test Types: {[t.value for t in contract.test_types]}")
```

### Creating Programmatically

```python
from omninode_bridge.codegen.models.model_contract_test import ModelContractTest
from omninode_bridge.codegen.models.enum_test_type import EnumTestType
from omnibase_core.primitives.model_semver import ModelSemVer

contract = ModelContractTest(
    name="test_my_node",
    version=ModelSemVer(major=1, minor=0, patch=0),
    description="Test contract for MyNode",
    target_node="MyNode",
    target_version="1.0.0",
    target_node_type="effect",
    test_suite_name="test_my_node",
    test_types=[EnumTestType.UNIT, EnumTestType.INTEGRATION],
    coverage_target=95,
)

# Export to YAML
yaml_output = contract.to_yaml()
```

### Validation

Run the included validation script:

```bash
poetry run python src/omninode_bridge/codegen/validate_test_contract.py
```

This validates:
- YAML parsing and model validation
- Field validators
- Coverage requirements
- Test targets and mock requirements
- YAML serialization

## Example Contract

A comprehensive example contract is provided at:
```
src/omninode_bridge/codegen/templates/test_contract_example.yaml
```

This example demonstrates:
- Complete test contract for NodeBridgeOrchestrator
- 4 test targets with scenarios and assertions
- Mock requirements for PostgreSQL, Kafka, and HTTP
- Pytest configuration with parallel execution
- Coverage targets and quality gates

## Design Decisions

1. **Standalone Model**: Does not inherit from `ModelContractBase` since tests aren't nodes in the 4-node architecture. Uses similar patterns but as a standalone contract.

2. **Flexible Test Types**: Supports both standard test types (unit, integration) and specialized types (load, stress, security) via enum.

3. **Comprehensive Mocking**: Extensive mock configuration options covering databases, HTTP, Kafka, filesystem, datetime, and custom mocks.

4. **Quality Gates**: Built-in support for coverage thresholds, type checking, linting, and test isolation enforcement.

5. **YAML-First**: Designed to be primarily defined in YAML with programmatic fallback.

## Validation Results

✅ All validations passed successfully:
- Model: ModelContractTest v1.0.0
- Zero Any types: COMPLIANT
- Field validators: WORKING
- YAML from_yaml: WORKING
- YAML to_yaml: WORKING
- Pattern matches: ModelContractEffect

## Migration Path

Once this pattern is validated in production:

1. Move models to `omnibase_core.models.contracts.test/`
2. Create `ModelContractTest` as official contract type
3. Update code generation system to use test contracts
4. Add test contract validation to CI/CD pipeline
5. Generate test documentation from contracts

## Files

```
src/omninode_bridge/codegen/models/
├── __init__.py
├── README.md (this file)
├── enum_test_type.py
├── model_contract_test.py
├── model_mock_requirements.py
├── model_test_configuration.py
└── model_test_target.py

src/omninode_bridge/codegen/templates/
└── test_contract_example.yaml

src/omninode_bridge/codegen/
└── validate_test_contract.py
```

## See Also

- `ModelContractEffect` in omnibase_core for architectural pattern reference
- ONEX v2.0 architecture documentation
- Bridge Nodes Guide for integration examples
