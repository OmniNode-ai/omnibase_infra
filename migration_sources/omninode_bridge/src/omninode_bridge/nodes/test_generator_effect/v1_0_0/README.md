# NodeTestGeneratorEffect - ONEX v2.0 Compliant

## Overview

**Status**: ✅ Implementation Complete
**Type**: Effect Node
**Purpose**: Generate comprehensive test files from ModelContractTest contracts using Jinja2 templates

## Features

- **Multiple Test Types**: Unit, Integration, Contract, Performance tests
- **Template-Based Generation**: Jinja2 templates for flexible code generation
- **Syntax Validation**: AST-based validation of generated Python code
- **Comprehensive Metrics**: Tracks generation time, files created, lines of code
- **Configurable**: Template directory, fixtures, overwrite policy
- **ONEX v2.0 Compliant**: Full compliance with ONEX v2.0 standards

## Architecture

### Node Class: `NodeTestGeneratorEffect`

Extends `NodeEffect` from omnibase_core with:
- Template rendering engine (Jinja2)
- Python syntax validation (AST)
- File writing with overwrite protection
- Metrics tracking

### Data Models

1. **ModelTestGeneratorRequest**: Input data model
   - `test_contract_yaml`: YAML string of ModelContractTest
   - `output_directory`: Where to write test files
   - `node_name`: Name of node being tested
   - `template_directory`: Optional custom template directory
   - `enable_fixtures`: Generate conftest.py
   - `overwrite_existing`: Overwrite existing files

2. **ModelTestGeneratorResponse**: Output data model
   - `generated_files`: List of generated test files
   - `file_count`: Total files generated
   - `total_lines_of_code`: Total LOC across all files
   - `duration_ms`: Total generation time
   - `template_render_ms`: Template rendering time
   - `file_write_ms`: File writing time
   - `success`: Generation status
   - `warnings`: Non-critical warnings

3. **ModelTestGeneratorConfig**: Configuration model
   - Template settings (directory, autoescape)
   - File writing settings (overwrite, create_directories)
   - Code generation options (docstrings, type hints, async tests)
   - Performance limits (max render time, max write time)
   - Validation settings (syntax validation, naming enforcement)

### Contract

**ONEX v2.0 Format** (`contract.yaml`):
- Name: `test_generator`
- Version: `1.0.0`
- Node Type: `EFFECT`
- IO Operations: `file_write`, `template_render`
- Performance: 2000ms target, 3000ms max

## Usage

### Basic Usage

```python
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.enums.enum_node_type import EnumNodeType
from omninode_bridge.nodes.test_generator_effect.v1_0_0 import NodeTestGeneratorEffect

# Initialize node
container = ModelContainer(
    value={"environment": "production"},
    container_type="config"
)
node = NodeTestGeneratorEffect(container)

# Create contract
contract = ModelContractEffect(
    name="generate_tests",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Generate tests for NodePostgresCrudEffect",
    node_type=EnumNodeType.EFFECT,
    input_model="ModelTestGeneratorRequest",
    output_model="ModelTestGeneratorResponse",
    tool_specification={
        "tool_name": "test_generator",
        "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect"
    },
    io_operations=[
        {
            "operation_type": "file_write",
            "atomic": True,
            "timeout_seconds": 30,
            "validation_enabled": True,
        }
    ],
    input_state={
        "test_contract_yaml": yaml_string,
        "output_directory": "/path/to/tests",
        "node_name": "postgres_crud_effect",
        "enable_fixtures": True,
        "overwrite_existing": False,
    }
)

# Execute
response = await node.execute_effect(contract)

# Check results
print(f"Generated {response.file_count} test files")
print(f"Total LOC: {response.total_lines_of_code}")
print(f"Duration: {response.duration_ms:.2f}ms")

for generated_file in response.generated_files:
    print(f"  - {generated_file.file_path.name} ({generated_file.lines_of_code} LOC)")
```

### Sample Test Contract YAML

```yaml
name: "test_postgres_crud"
version:
  major: 1
  minor: 0
  patch: 0
description: "Test contract for PostgreSQL CRUD operations"

target_node: "postgres_crud_effect"
target_version: "1.0.0"
target_node_type: "effect"
test_suite_name: "test_postgres_crud_effect"

coverage_minimum: 85
coverage_target: 95

test_types:
  - unit
  - integration

test_targets:
  - target_name: "execute_effect"
    target_type: "method"
    test_scenarios:
      - "successful execution"
      - "error handling"
    expected_behaviors:
      - "returns success response"
      - "handles errors gracefully"

mock_requirements:
  mock_dependencies:
    - "postgres_client"
  mock_external_services:
    - "kafka_client"

test_configuration:
  pytest_markers:
    - "asyncio"
    - "unit"
  timeout_seconds: 30
```

## Template Support

Templates located in: `src/omninode_bridge/codegen/templates/test_templates/`

### Available Templates

1. **test_unit.py.j2**: Unit tests for node methods
2. **test_integration.py.j2**: Integration tests for end-to-end workflows
3. **test_contract.py.j2**: Contract validation tests
4. **test_performance.py.j2**: Performance/load tests
5. **conftest.py.j2**: Pytest fixtures and configuration

### Template Context Variables

All templates have access to:
- `test_contract`: ModelContractTest instance
- `node_name`: Name of node being tested
- `target_node`: Full node class name
- `node_type`: Node type (effect, compute, reducer, orchestrator)
- `test_types`: List of test types to generate
- `test_targets`: List of test targets with scenarios
- `mock_requirements`: Mock configuration
- `test_configuration`: Pytest configuration
- `coverage_minimum`: Minimum coverage percentage
- `coverage_target`: Target coverage percentage
- `generated_at`: ISO timestamp
- `correlation_id`: UUID for tracking

## Performance

### Targets

- Template rendering: < 2000ms (configurable)
- File writing: < 1000ms (configurable)
- Total generation: < 3000ms
- Success rate: > 95%

### Metrics

Track via `node.get_metrics()`:
- `total_generations`: Total number of generations
- `total_files_generated`: Total files created
- `avg_files_per_generation`: Average files per generation
- `avg_generation_time_ms`: Average generation time

## Testing

### Unit Tests

Located in: `tests/test_node.py`

Run with:
```bash
pytest src/omninode_bridge/nodes/test_generator_effect/v1_0_0/tests/test_node.py -v
```

Tests:
- Node initialization
- Test file generation
- Fixture generation
- Syntax validation
- Metrics tracking
- Template context building

### Integration Tests

Located in: `tests/test_integration.py`

Run with:
```bash
pytest src/omninode_bridge/nodes/test_generator_effect/v1_0_0/tests/test_integration.py -v -m integration
```

Tests:
- End-to-end test generation
- Overwrite protection
- Metrics tracking across generations
- Generated code validation

## Configuration

### Environment Variables

None required - all configuration via ModelTestGeneratorConfig.

### Default Configuration

```python
ModelTestGeneratorConfig(
    template_directory=Path("src/omninode_bridge/codegen/templates/test_templates"),
    template_autoescape=False,
    enable_fixtures=True,
    overwrite_existing=False,
    create_directories=True,
    include_docstrings=True,
    include_type_hints=True,
    use_async_tests=True,
    parametrize_tests=True,
    max_template_render_time_ms=2000,
    max_file_write_time_ms=1000,
    validate_generated_code=True,
    enforce_test_naming=True,
)
```

## Error Handling

### Common Errors

1. **Template Not Found**:
   - Error Code: `CONFIGURATION_ERROR`
   - Solution: Check template directory exists and contains required templates

2. **Syntax Errors in Generated Code**:
   - Error Code: `VALIDATION_ERROR`
   - Solution: Fix template or disable syntax validation

3. **File Already Exists**:
   - Error Code: `VALIDATION_ERROR`
   - Solution: Set `overwrite_existing=True` or delete existing files

4. **Invalid Test Contract YAML**:
   - Error Code: `VALIDATION_ERROR`
   - Solution: Fix YAML structure to match ModelContractTest schema

## Integration Points

### Called By

- NodeCodegenOrchestrator (Stage 9: Test Generation)
- Manual invocation for standalone test generation

### Dependencies

- **Jinja2**: Template rendering
- **omnibase_core**: Base node classes, error handling
- **omninode_bridge.codegen.models**: Test contract models

### Events Published

Future enhancement: Kafka events for observability
- `test_generation_started`
- `test_generation_completed`
- `test_generation_failed`

## Compliance

### ONEX v2.0

✅ Suffix-based naming: `NodeTestGeneratorEffect`
✅ Contract-driven architecture
✅ Event-driven patterns (future)
✅ Comprehensive error handling (ModelOnexError)
✅ Strong typing (Pydantic models)
✅ Performance monitoring

### Code Quality

✅ Type hints on all methods
✅ Comprehensive docstrings
✅ Zero tolerance for `Any` types
✅ Pydantic validation
✅ AST-based syntax validation

## Future Enhancements

1. **Kafka Event Publishing**: Publish generation events for observability
2. **Template Caching**: Cache compiled templates for performance
3. **Parallel Generation**: Generate multiple test files in parallel
4. **AI-Assisted Test Generation**: Use LLMs to generate test scenarios
5. **Coverage Analysis**: Analyze existing code coverage and generate missing tests

## Known Issues

1. **Template Bugs**: Some templates may have minor syntax issues requiring fixes
2. **Limited Template Support**: Not all test types have corresponding templates yet

## Maintenance

### Adding New Templates

1. Create new `.j2` template in `src/omninode_bridge/codegen/templates/test_templates/`
2. Add template mapping in `template_map` (line 443 of `node.py`)
3. Test template rendering with sample contracts
4. Update documentation

### Updating Models

Follow ONEX stability guarantees:
- ✅ Add optional fields (minor version bump)
- ❌ Remove fields (major version bump)
- ❌ Change field types (major version bump)

## Support

For issues or questions:
- Check logs for detailed error messages
- Verify template directory exists and contains templates
- Validate test contract YAML structure
- Enable debug logging for detailed tracing

---

**Generated**: 2025-10-30
**Version**: 1.0.0
**Author**: OmniNode Code Generation
