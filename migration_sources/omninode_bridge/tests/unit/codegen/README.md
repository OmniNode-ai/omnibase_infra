# Code Generation Test Suite

Comprehensive test suite for the unified code generation infrastructure.

## Test Structure

```
tests/
├── fixtures/
│   └── codegen/
│       ├── sample_requirements.py  # Test PRD requirements
│       └── mock_responses.py       # Mock LLM responses and code samples
├── unit/
│   └── codegen/
│       ├── conftest.py                    # Test fixtures
│       ├── test_service.py                # CodeGenerationService tests
│       ├── test_quality_gates.py          # QualityGatePipeline tests
│       └── strategies/
│           ├── test_jinja2_strategy.py    # Jinja2Strategy tests
│           ├── test_selector.py           # StrategySelector tests
│           ├── test_template_load_strategy.py  # TemplateLoadStrategy tests (placeholder)
│           └── test_hybrid_strategy.py    # HybridStrategy tests (placeholder)
└── integration/
    └── codegen/
        └── test_end_to_end_generation.py  # End-to-end workflow tests
```

## Test Coverage

### Completed Tests (✅)

1. **CodeGenerationService** (`test_service.py`)
   - Service initialization
   - Strategy registry management
   - Strategy registration and retrieval
   - Strategy selection logic
   - Requirements validation
   - Node generation workflow
   - Error handling

2. **Jinja2Strategy** (`test_jinja2_strategy.py`)
   - Strategy initialization
   - Node type support
   - Requirements validation (basic/standard/strict)
   - Code generation with template engine
   - Error handling
   - Performance tracking
   - Validation bypass

3. **StrategySelector** (`test_selector.py`)
   - Strategy selection for simple/moderate/complex requirements
   - Complexity calculation
   - Scoring algorithms (Jinja2/TemplateLoad/Hybrid)
   - Override strategy support
   - Fallback strategy ordering
   - Custom logic detection
   - Performance requirements impact

4. **QualityGatePipeline** (`test_quality_gates.py`)
   - Pipeline initialization
   - Syntax validation
   - Security validation
   - Code injection detection
   - ONEX compliance validation
   - Validation level handling (strict/permissive/development)
   - Quality score calculation
   - Execution time tracking

5. **End-to-End Integration** (`test_end_to_end_generation.py`)
   - Full generation pipeline
   - Auto-strategy selection
   - Multiple node types
   - Validation integration
   - Error handling
   - Correlation ID tracking

### Placeholder Tests (⏳)

1. **TemplateLoadStrategy** (`test_template_load_strategy.py`)
   - Requires mocking: TemplateEngine, BusinessLogicGenerator, ArtifactConverter
   - Tests skipped pending complex mock setup

2. **HybridStrategy** (`test_hybrid_strategy.py`)
   - Requires mocking: Jinja2Strategy, CodeInjector, BusinessLogicGenerator, QualityGatePipeline
   - Tests skipped pending complex mock setup

## Running Tests

### Run all code generation tests:
```bash
pytest tests/unit/codegen/ tests/integration/codegen/ -v
```

### Run specific test file:
```bash
pytest tests/unit/codegen/test_service.py -v
```

### Run with coverage:
```bash
pytest tests/unit/codegen/ --cov=src/omninode_bridge/codegen --cov-report=html
```

### Run only non-skipped tests:
```bash
pytest tests/unit/codegen/ -v --ignore-skipped
```

## Test Fixtures

### Sample Requirements (from `sample_requirements.py`):
- `get_simple_crud_requirements()` - Low complexity (< 5)
- `get_moderate_complexity_requirements()` - Moderate complexity (5-10)
- `get_complex_orchestration_requirements()` - High complexity (> 10)
- `get_reducer_requirements()` - Reducer node specific
- `get_invalid_requirements()` - Invalid for validation testing

### Mock Responses (from `mock_responses.py`):
- `MOCK_LLM_RESPONSE_SIMPLE` - Simple business logic
- `MOCK_LLM_RESPONSE_MODERATE` - Moderate complexity logic
- `MOCK_LLM_RESPONSE_COMPLEX` - Complex orchestration logic
- `SAMPLE_VALID_CODE` - Valid generated code
- `SAMPLE_CODE_WITH_STUBS` - Code with stubs for injection testing
- `SAMPLE_CODE_WITH_SYNTAX_ERROR` - Code with syntax errors
- `SAMPLE_CODE_WITH_SECURITY_ISSUES` - Code with security vulnerabilities

### Pytest Fixtures (from `conftest.py`):
- `simple_crud_requirements` - Simple CRUD requirements
- `moderate_complexity_requirements` - Moderate complexity requirements
- `complex_orchestration_requirements` - Complex requirements
- `effect_classification` - Effect node classification
- `compute_classification` - Compute node classification
- `mock_llm_node` - Mocked LLM node
- `sample_valid_code` - Valid code sample
- `temp_output_dir` - Temporary output directory

## Expected Coverage

**Target**: >80% for new code generation components

**Coverage by Component**:
- ✅ CodeGenerationService: ~85%
- ✅ StrategyRegistry: ~90%
- ✅ Jinja2Strategy: ~85%
- ✅ StrategySelector: ~90%
- ✅ QualityGatePipeline: ~80%
- ⏳ TemplateLoadStrategy: ~0% (placeholder tests)
- ⏳ HybridStrategy: ~0% (placeholder tests)

## Key Test Patterns

### Mocking LLM Calls
```python
mock_llm_response = "# Generated implementation"
mock_node = AsyncMock()
mock_node.execute_effect.return_value = ModelLLMResponse(
    generated_text=mock_llm_response,
    ...
)
```

### Mocking Template Engine
```python
mock_artifacts = ModelGeneratedArtifacts(...)
with patch.object(
    strategy.template_engine,
    'generate',
    new_callable=AsyncMock,
    return_value=mock_artifacts
):
    result = await strategy.generate(request)
```

### Testing Validation
```python
is_valid, errors = strategy.validate_requirements(
    requirements,
    EnumValidationLevel.STRICT
)
assert is_valid is True
assert len(errors) == 0
```

## Notes

- **Async Tests**: Use `@pytest.mark.asyncio` for async functions
- **Mocking**: Heavy use of `unittest.mock` to avoid external dependencies
- **Fixtures**: Shared fixtures in `conftest.py` for consistency
- **Performance**: Tests should complete in < 5 seconds total
- **Integration Marker**: Use `@pytest.mark.integration` for integration tests

## TODO

1. Implement full tests for TemplateLoadStrategy
2. Implement full tests for HybridStrategy
3. Add performance benchmarks
4. Add stress tests (1000+ concurrent generations)
5. Add LLM cost tracking tests
6. Add backward compatibility tests (ensure existing TemplateEngine still works)
