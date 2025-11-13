# Test Templates - Comprehensive Test Generation for ONEX Nodes

**Version**: 1.0.0
**Status**: Production Ready
**ONEX v2.0 Compliant**

## Overview

This directory contains Jinja2 templates for generating comprehensive pytest-compatible test files for ONEX nodes. These templates support all test types (unit, integration, contract, performance) and follow pytest best practices.

## Quick Start

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from omninode_bridge.codegen.models.model_contract_test import ModelContractTest

# Load template environment
template_dir = Path("src/omninode_bridge/codegen/templates/test_templates")
env = Environment(loader=FileSystemLoader(template_dir))

# Create test contract
test_contract = ModelContractTest(
    name="test_postgres_crud",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Test contract for PostgresCRUD node",
    target_node="NodePostgresCrudEffect",
    target_version="1.0.0",
    target_node_type="effect",
    test_suite_name="test_postgres_crud",
    test_types=["unit", "integration", "contract", "performance"],
    coverage_target=95,
)

# Render unit test template
template = env.get_template("test_unit.py.j2")
unit_test_code = template.render(
    node_name="NodePostgresCrudEffect",
    module_path="omninode_bridge.nodes.postgres_crud.v1_0_0.node",
    node_type="effect",
    test_contract=test_contract,
    fixtures=[],
)

# Write to file
output_path = Path("tests/test_unit.py")
output_path.write_text(unit_test_code)
```

## Available Templates

### 1. test_unit.py.j2

**Purpose**: Generate unit tests with mocked dependencies

**Features**:
- Individual method testing in isolation
- Comprehensive mocking support (database, Kafka, HTTP, filesystem, datetime)
- Parametrized test generation
- Error handling and edge case testing
- Mock verification tests
- Quality gate enforcement (isolation, determinism)

**Template Variables**:
```python
{
    "node_name": "NodeExampleEffect",
    "module_path": "omninode_bridge.nodes.example.v1_0_0.node",
    "node_type": "effect",
    "test_contract": ModelContractTest(...),
    "fixtures": [
        {
            "name": "sample_data",
            "description": "Sample test data",
            "code": "return {'id': '123', 'name': 'test'}",
            "scope": "function"  # optional
        }
    ]
}
```

**Example Output**:
```python
# test_unit.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def node(container):
    return NodeExampleEffect(container)

@pytest.mark.asyncio
async def test_execute_effect_success(node, sample_contract):
    result = await node.execute_effect(sample_contract)
    assert result is not None
```

### 2. test_integration.py.j2

**Purpose**: Generate integration tests with real dependencies

**Features**:
- End-to-end workflow testing
- Real database/service integration
- Concurrent execution testing
- Data persistence validation
- External service integration
- Error recovery testing

**Template Variables**:
```python
{
    "node_name": "NodeExampleEffect",
    "module_path": "omninode_bridge.nodes.example.v1_0_0.node",
    "node_type": "effect",
    "test_contract": ModelContractTest(...),
}
```

**Example Output**:
```python
# test_integration.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_workflow_success(integration_node, integration_contract):
    result = await integration_node.execute_effect(integration_contract)
    assert result is not None
    # Verify database state, Kafka events, etc.
```

### 3. test_contract.py.j2

**Purpose**: Validate ONEX v2.0 contract compliance

**Features**:
- Contract YAML validation
- Field presence and type checking
- Method signature validation
- ONEX naming convention validation
- I/O operation specification (for Effect nodes)
- Metadata and performance requirement validation

**Template Variables**:
```python
{
    "node_name": "NodeExampleEffect",
    "module_path": "omninode_bridge.nodes.example.v1_0_0.node",
    "node_type": "effect",
    "test_contract": ModelContractTest(...),
    "contract_file": "contracts/example_effect_contract.yaml",
}
```

**Example Output**:
```python
# test_contract.py
@pytest.mark.contract
def test_contract_required_fields(contract_data):
    required_fields = ["name", "version", "description", "node_type", ...]
    for field in required_fields:
        assert field in contract_data
```

### 4. test_performance.py.j2

**Purpose**: Benchmark performance and validate thresholds

**Features**:
- Execution time measurement
- Throughput testing (sequential and concurrent)
- Latency percentile analysis (P50, P95, P99)
- Load and stress testing
- Spike load handling
- Memory usage monitoring

**Template Variables**:
```python
{
    "node_name": "NodeExampleEffect",
    "module_path": "omninode_bridge.nodes.example.v1_0_0.node",
    "node_type": "effect",
    "test_contract": ModelContractTest(...),
    "performance_thresholds": {
        "execution_time_target_ms": 100,
        "execution_time_max_ms": 1000,
        "throughput_min_rps": 100,
        "throughput_target_rps": 500,
        "p50_latency_ms": 50,
        "p95_latency_ms": 200,
        "p99_latency_ms": 500,
    }
}
```

**Example Output**:
```python
# test_performance.py
@pytest.mark.performance
@pytest.mark.asyncio
async def test_latency_percentiles(performance_node, performance_contract):
    metrics = await measure_performance(performance_node, performance_contract, 1000)
    assert metrics.p95_latency_ms < 200
```

### 5. conftest.py.j2

**Purpose**: Shared pytest fixtures and configuration

**Features**:
- Custom pytest marker registration
- Shared test fixtures (containers, nodes, contracts)
- Mock fixtures (database, Kafka, HTTP, datetime)
- Data fixtures (test data directory, sample data)
- Cleanup fixtures (automatic resource cleanup)
- Utility fixtures (logging, timers, correlation IDs)

**Template Variables**:
```python
{
    "node_name": "NodeExampleEffect",
    "module_path": "omninode_bridge.nodes.example.v1_0_0.node",
    "node_type": "effect",
    "test_contract": ModelContractTest(...),
    "fixtures": [...]  # Custom fixtures
}
```

**Example Output**:
```python
# conftest.py
@pytest.fixture
def test_container() -> ModelContainer:
    config_dict = {"environment": "test"}
    container = ModelContainer(value=config_dict, container_type="config")
    return container
```

## Template Variable Reference

### Common Variables (All Templates)

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `node_name` | str | Yes | Node class name (e.g., "NodePostgresCrudEffect") |
| `module_path` | str | Yes | Import path (e.g., "omninode_bridge.nodes.postgres_crud.v1_0_0.node") |
| `node_type` | str | Yes | Node type: "effect", "compute", "reducer", "orchestrator" |
| `test_contract` | ModelContractTest | Yes | Test contract model with configuration |

### Test Contract Model Fields

The `ModelContractTest` provides comprehensive test configuration:

```python
class ModelContractTest(BaseModel):
    # Target identification
    target_node: str                    # "NodePostgresCrudEffect"
    target_version: str                 # "1.0.0"
    target_node_type: str               # "effect"

    # Coverage requirements
    coverage_minimum: int = 85          # Minimum coverage %
    coverage_target: int = 95           # Target coverage %

    # Test types to generate
    test_types: List[EnumTestType]      # [UNIT, INTEGRATION, CONTRACT, PERFORMANCE]

    # Test targets (methods to test)
    test_targets: List[ModelTestTarget]

    # Mock requirements
    mock_requirements: ModelMockRequirements

    # Test configuration
    test_configuration: ModelTestConfiguration

    # Options
    include_docstrings: bool = True
    include_type_hints: bool = True
    use_async_tests: bool = False
    parametrize_tests: bool = True

    # Quality gates
    enforce_test_isolation: bool = True
    enforce_deterministic_tests: bool = True
```

### Mock Requirements Configuration

```python
class ModelMockRequirements(BaseModel):
    # Dependencies to mock
    mock_dependencies: List[str] = []           # ["omninode_bridge.db.connection"]
    mock_external_services: List[str] = []      # ["api.service", "cache.redis"]

    # Specific mocks
    mock_database: bool = False
    mock_kafka_producer: bool = False
    mock_kafka_consumer: bool = False
    mock_http_clients: bool = False
    mock_filesystem: bool = False
    mock_datetime: bool = False
    mock_environment: bool = False

    # Mock behavior
    mock_return_values: Dict[str, Any] = {}
    mock_side_effects: List[str] = []
    mock_exceptions: List[str] = []
```

### Test Configuration

```python
class ModelTestConfiguration(BaseModel):
    # Pytest configuration
    pytest_markers: List[str] = []              # ["slow", "integration"]
    pytest_plugins: List[str] = []

    # Execution settings
    parallel_execution: bool = False
    parallel_workers: int = 1
    timeout_seconds: int = 300

    # Coverage configuration
    coverage_enabled: bool = True
    coverage_threshold: int = 85
    coverage_fail_under: bool = True

    # Test database
    use_test_database: bool = False
    test_database_config: Dict[str, Any] = {}

    # Output configuration
    verbose_output: bool = True
    generate_html_report: bool = True
```

### Performance Thresholds

```python
performance_thresholds = {
    # Execution time (milliseconds)
    "execution_time_target_ms": 100,
    "execution_time_max_ms": 1000,

    # Throughput (requests per second)
    "throughput_min_rps": 100,
    "throughput_target_rps": 500,

    # Memory (MB)
    "memory_target_mb": 128,
    "memory_max_mb": 512,

    # Latency percentiles (milliseconds)
    "p50_latency_ms": 50,
    "p95_latency_ms": 200,
    "p99_latency_ms": 500,

    # Concurrency
    "max_concurrent_requests": 100,
}
```

## Best Practices

### 1. Template Selection

Choose templates based on test requirements:

- **Unit Tests**: Always generate for all nodes (fast, isolated, high coverage)
- **Integration Tests**: Generate for nodes with external dependencies (database, Kafka, APIs)
- **Contract Tests**: Always generate (validates ONEX compliance)
- **Performance Tests**: Generate for critical path nodes (latency-sensitive, high throughput)

### 2. Mock Configuration

Configure mocks appropriately:

```python
# Good: Specific mocks for dependencies
mock_requirements = ModelMockRequirements(
    mock_database=True,
    mock_kafka_producer=True,
    mock_dependencies=["omninode_bridge.db.postgres_client"],
)

# Bad: Over-mocking (makes tests less valuable)
mock_requirements = ModelMockRequirements(
    mock_everything=True  # Don't do this
)
```

### 3. Test Targets

Specify explicit test targets for complex nodes:

```python
test_targets = [
    ModelTestTarget(
        target_name="execute_effect",
        test_scenarios=["success", "database_error", "timeout"],
        expected_behaviors=["returns result", "handles errors", "respects timeout"],
        edge_cases=["empty input", "large payload"],
        error_conditions=["connection lost", "invalid contract"],
    )
]
```

### 4. Coverage Targets

Set realistic coverage targets:

- **85%**: Minimum acceptable coverage
- **95%**: Target for production-critical nodes
- **100%**: Only for simple, critical nodes (hard to maintain)

### 5. Performance Thresholds

Base thresholds on actual requirements:

```python
# Good: Based on SLA requirements
performance_thresholds = {
    "execution_time_max_ms": 100,  # SLA: 99% requests < 100ms
    "p95_latency_ms": 50,          # SLA: 95% requests < 50ms
}

# Bad: Arbitrary numbers
performance_thresholds = {
    "execution_time_max_ms": 1,  # Unrealistic
}
```

## Template Customization

### Adding Custom Fixtures

Add custom fixtures to the `fixtures` parameter:

```python
fixtures = [
    {
        "name": "sample_database_record",
        "description": "Sample database record for testing",
        "code": """
            return {
                'id': str(uuid4()),
                'created_at': datetime.now(UTC),
                'status': 'active',
            }
        """,
        "scope": "function",
    },
    {
        "name": "mock_api_client",
        "description": "Mock API client with predefined responses",
        "code": """
            mock = AsyncMock()
            mock.get.return_value = {'status': 'success'}
            return mock
        """,
        "scope": "module",
    }
]
```

### Adding Custom Assertions

Specify custom assertion functions:

```python
test_contract = ModelContractTest(
    # ... other fields ...
    custom_assertions=["assert_valid_uuid", "assert_sorted_list"],
)
```

The template will generate stub functions:

```python
def assert_valid_uuid(value):
    """Custom assertion for UUID validation."""
    from uuid import UUID
    assert isinstance(UUID(value), UUID)
```

## Integration with NodeTestGeneratorEffect

These templates are designed to work with `NodeTestGeneratorEffect`:

```python
from omninode_bridge.codegen.nodes.test_generator_effect import NodeTestGeneratorEffect
from omninode_bridge.codegen.models.model_contract_test import ModelContractTest

# Create test contract
test_contract = ModelContractTest(...)

# Generate tests
generator = NodeTestGeneratorEffect(container)
result = await generator.execute_effect(test_contract)

# result contains generated test files:
# - tests/test_unit.py
# - tests/test_integration.py
# - tests/test_contract.py
# - tests/test_performance.py
# - tests/conftest.py
```

## Template Validation

Validate generated tests before writing:

```python
import ast
import pytest

# Validate Python syntax
def validate_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Run pytest collection to verify
def validate_pytest_collection(test_file: Path) -> bool:
    result = pytest.main(["--collect-only", str(test_file)])
    return result == 0
```

## Troubleshooting

### Common Issues

1. **Template variable missing**
   - Error: `jinja2.exceptions.UndefinedError: 'fixtures' is undefined`
   - Solution: Provide all required template variables

2. **Generated tests have syntax errors**
   - Error: `SyntaxError: invalid syntax`
   - Solution: Validate Jinja2 escaping and indentation in templates

3. **Pytest collection fails**
   - Error: `ERROR collecting tests/test_unit.py`
   - Solution: Check imports and fixture names in generated code

4. **Performance tests timeout**
   - Error: `asyncio.TimeoutError`
   - Solution: Increase `timeout_seconds` in test configuration

### Debug Mode

Enable verbose template rendering:

```python
from jinja2 import Environment, DebugUndefined

env = Environment(
    loader=FileSystemLoader(template_dir),
    undefined=DebugUndefined,  # Shows undefined variables
)
```

## Version History

### 1.0.0 (2024-10-30)

- Initial release
- Full ONEX v2.0 compliance
- Support for all 4 node types (Effect, Compute, Reducer, Orchestrator)
- 5 comprehensive templates (unit, integration, contract, performance, conftest)
- ModelContractTest integration
- Pytest best practices

## Future Enhancements

Planned features for future versions:

- **Security Testing**: Template for security vulnerability tests
- **Regression Testing**: Baseline comparison templates
- **Mutation Testing**: Test quality verification
- **Property-Based Testing**: Hypothesis integration
- **Visual Regression**: Snapshot testing for UI nodes

## Contributing

To modify templates:

1. Edit template files in `test_templates/`
2. Update tests in `tests/codegen/test_templates/`
3. Update this README with changes
4. Bump version in `__init__.py`

## License

Part of OmniNode Bridge - ONEX v2.0 Compliant Code Generation System
