# Quality Gates Pipeline

**Status**: ✅ Complete
**Version**: 1.0.0
**Integration**: CodeValidator + Multi-Stage Validation

## Overview

The Quality Gates Pipeline provides comprehensive, multi-stage validation for LLM-generated code with configurable strictness levels. It integrates with the existing `CodeValidator` and adds additional validation stages including type checking, code injection detection, and more.

## Features

### 5-Stage Validation Pipeline

1. **Syntax Validation** (Critical)
   - AST parsing to verify Python syntax
   - Always runs, blocks pipeline if failed
   - Execution time: <10ms typical

2. **Type Checking** (Optional)
   - mypy integration for type hint validation
   - Strict mode type checking
   - Skipped in development mode
   - Requires mypy installation: `pip install mypy`

3. **ONEX v2.0 Compliance** (Framework-Specific)
   - Validates ONEX patterns (ModelOnexError, emit_log_event)
   - Checks for omnibase_core imports
   - Type hints and docstring validation
   - Skipped in development mode

4. **Code Injection Validation** (Quality)
   - Detects unimplemented stubs (TODO, pass, NotImplementedError)
   - Ensures generated code replaced template stubs
   - Skipped in development mode
   - Critical for production deployments

5. **Security Scanning** (Critical)
   - Hardcoded secrets detection
   - SQL injection vulnerability checks
   - Dangerous patterns (eval, exec, pickle)
   - Always runs

### Validation Levels

The pipeline supports three validation levels with different strictness:

| Level | Description | Use Case |
|-------|-------------|----------|
| **STRICT** | All checks must pass, no warnings | Production deployments |
| **PERMISSIVE** | Critical checks must pass, warnings OK | Testing/staging |
| **DEVELOPMENT** | Only syntax + critical security | Rapid iteration |

#### Stage Execution by Level

| Stage | STRICT | PERMISSIVE | DEVELOPMENT |
|-------|--------|------------|-------------|
| Syntax | ✅ Required | ✅ Required | ✅ Required |
| Type Checking | ✅ Required | ✅ Required | ⏭️ Skipped |
| ONEX Compliance | ✅ Required | ⚠️ Warnings | ⏭️ Skipped |
| Code Injection | ✅ Required | ⚠️ Warnings | ⏭️ Skipped |
| Security | ✅ Required | ✅ Required | ✅ Required |

### Quality Scoring

Quality score (0.0-1.0) calculated using weighted stages:

- **Syntax**: 25% (critical)
- **Type Checking**: 20%
- **ONEX Compliance**: 25%
- **Code Injection**: 15%
- **Security**: 15% (critical)

Partial credit given for stages with few issues:
- 0-2 issues: 50% credit
- 3-5 issues: 25% credit
- 6+ issues: 0% credit

## Usage

### Basic Usage

```python
from omninode_bridge.codegen import QualityGatePipeline, ValidationLevel
from omninode_bridge.codegen.business_logic.models import GenerationContext

# Initialize pipeline
pipeline = QualityGatePipeline(
    validation_level="strict",
    enable_mypy=True
)

# Create context
context = GenerationContext(
    node_type="effect",
    service_name="postgres_crud",
    method_name="execute_effect"
)

# Validate code
result = await pipeline.validate(
    generated_code=llm_output,
    context=context
)

# Check results
if result.passed:
    print(f"✅ Validation passed! Quality score: {result.quality_score:.2f}")
else:
    print(f"❌ Validation failed: {result.failed_stages}")
    for issue in result.all_issues:
        print(f"  - {issue}")
```

### Integration with BusinessLogicGenerator

```python
from omninode_bridge.codegen import (
    BusinessLogicGenerator,
    QualityGatePipeline,
)

# Initialize components
generator = BusinessLogicGenerator(enable_llm=True)
quality_gates = QualityGatePipeline(validation_level="strict")

# Generate code
enhanced = await generator.enhance_artifacts(
    artifacts=template_artifacts,
    requirements=prd_requirements
)

# Validate each generated method
for method in enhanced.methods_generated:
    context = GenerationContext(
        node_type=prd_requirements.node_type,
        service_name=prd_requirements.service_name,
        method_name=method.method_name
    )

    result = await quality_gates.validate(
        method.generated_code,
        context
    )

    if not result.passed:
        print(f"⚠️ Method {method.method_name} failed validation")
```

### Custom mypy Configuration

```python
from pathlib import Path

pipeline = QualityGatePipeline(
    validation_level="strict",
    enable_mypy=True,
    mypy_config_path=Path("mypy.ini")
)
```

### Validation Level Selection

Choose validation level based on environment:

```python
import os

# Environment-based validation level
env = os.getenv("ENVIRONMENT", "development")
validation_levels = {
    "production": "strict",
    "staging": "permissive",
    "development": "development"
}

pipeline = QualityGatePipeline(
    validation_level=validation_levels[env]
)
```

## Validation Result Structure

```python
class ValidationResult(BaseModel):
    # Overall result
    passed: bool
    validation_level: ValidationLevel
    quality_score: float  # 0.0-1.0

    # Stage results
    stage_results: list[StageResult]
    failed_stages: list[str]
    passed_stages: list[str]
    skipped_stages: list[str]

    # Aggregated issues
    all_issues: list[str]
    all_warnings: list[str]
    critical_issues: list[str]

    # Detailed validator result
    validator_result: Optional[ValidatorResult]

    # Metrics
    total_execution_time_ms: float
    total_issues_count: int
    total_warnings_count: int
    timestamp: float
```

### Stage Result Structure

```python
class StageResult(BaseModel):
    stage: ValidationStage  # Enum: SYNTAX, TYPE_CHECKING, etc.
    passed: bool
    issues: list[str]
    warnings: list[str]
    execution_time_ms: float
    skipped: bool
```

## Example Output

### Successful Validation (Strict)

```
✅ Overall Result: PASSED
✓ Quality Score: 1.00
✓ Execution Time: 45.2ms

✓ Stage Results:
  ✅ syntax                PASS     (8.1ms)
  ✅ type_checking         PASS     (15.3ms)
  ✅ onex_compliance       PASS     (12.4ms)
  ✅ code_injection        PASS     (5.7ms)
  ✅ security              PASS     (3.7ms)

✅ No issues or warnings found!
```

### Failed Validation (Strict)

```
❌ Overall Result: FAILED
✓ Quality Score: 0.35
✓ Execution Time: 38.6ms

✓ Stage Results:
  ✅ syntax                PASS     (7.8ms)
  ⏭️ type_checking         SKIPPED  (0.0ms)
  ❌ onex_compliance       FAIL     (11.2ms)
  ❌ code_injection        FAIL     (4.3ms)
  ❌ security              FAIL     (2.1ms)

⚠️ Issues Found (5):
  - Should use ModelOnexError instead of generic exceptions
  - Should use emit_log_event for structured logging
  - Line 15: TODO comment found (stub not replaced)
  - Line 23: Hardcoded api_key (use environment variables)
  - Security: Potential SQL injection (f-string with SQL execute)
```

### Permissive Mode (Warnings Allowed)

```
✅ Overall Result: PASSED
✓ Quality Score: 0.75
✓ Execution Time: 42.3ms

✓ Stage Results:
  ✅ syntax                PASS     (8.4ms)
  ✅ type_checking         PASS     (16.1ms)
  ✅ onex_compliance       PASS     (10.5ms)
  ✅ code_injection        PASS     (4.8ms)
  ✅ security              PASS     (2.5ms)

⚠️ Warnings (3):
  - Function 'helper_method' missing return type hint
  - Function 'helper_method' has short docstring (15 chars, minimum: 20)
  - High cyclomatic complexity (12), consider simplifying
```

## Integration Points

### With CodeValidator

Quality gates pipeline integrates existing `CodeValidator`:

```python
from omninode_bridge.codegen.business_logic.validator import CodeValidator

# CodeValidator handles:
# - Syntax validation (AST parsing)
# - ONEX compliance patterns
# - Type hint checking
# - Security scanning
# - Code quality metrics

# QualityGatePipeline orchestrates:
# - Multi-stage validation
# - mypy integration
# - Code injection detection
# - Configurable strictness levels
# - Comprehensive reporting
```

### With CodeGenerationPipeline

```python
from omninode_bridge.codegen import CodeGenerationPipeline, QualityGatePipeline

# Generate code
pipeline = CodeGenerationPipeline(enable_llm=True)
result = await pipeline.generate_node(
    node_type="effect",
    version="v1_0_0",
    requirements={...}
)

# Validate generated code
quality_gates = QualityGatePipeline(validation_level="strict")
validation = await quality_gates.validate(
    result.enhanced_node_file,
    context=GenerationContext(...)
)

# Deploy only if validation passes
if validation.passed:
    deploy(result.enhanced_node_file)
```

## Performance

### Execution Times (Typical)

| Stage | Time | Notes |
|-------|------|-------|
| Syntax | 5-10ms | AST parsing |
| Type Checking | 50-200ms | mypy subprocess |
| ONEX Compliance | 10-20ms | Pattern matching |
| Code Injection | 5-10ms | Stub detection |
| Security | 5-10ms | Pattern scanning |
| **Total** | **75-250ms** | Depends on mypy |

### Optimization Tips

1. **Disable mypy in development**: Set `enable_mypy=False` for faster iteration
2. **Use development mode**: Skip non-critical stages during active development
3. **Cache mypy results**: Use persistent mypy daemon for repeated validations
4. **Parallel validation**: Validate multiple methods in parallel

## Best Practices

### Production Deployments

```python
# Always use strict mode for production
pipeline = QualityGatePipeline(
    validation_level="strict",
    enable_mypy=True
)

# Block deployment if validation fails
result = await pipeline.validate(code, context)
if not result.passed:
    raise DeploymentError(
        f"Code validation failed: {result.failed_stages}"
    )

# Log quality metrics
logger.info(
    f"Code quality score: {result.quality_score:.2f}, "
    f"execution_time: {result.total_execution_time_ms:.1f}ms"
)
```

### Development Workflow

```python
# Use development mode for rapid iteration
pipeline = QualityGatePipeline(
    validation_level="development",
    enable_mypy=False
)

# Only critical checks run (syntax + security)
result = await pipeline.validate(code, context)
if not result.passed:
    print(f"Critical issues: {result.critical_issues}")
```

### CI/CD Integration

```yaml
# .github/workflows/codegen-validation.yml
- name: Validate Generated Code
  run: |
    python -m scripts.validate_codegen \
      --validation-level strict \
      --enable-mypy \
      --fail-on-warnings
```

## Troubleshooting

### mypy Not Found

If mypy type checking fails with "mypy not found":

```bash
# Install mypy
pip install mypy

# Or disable in pipeline
pipeline = QualityGatePipeline(enable_mypy=False)
```

### False Positives

If validation flags valid code:

1. Check validation level - use `permissive` or `development` mode
2. Review specific stage results to identify which check failed
3. Add exceptions to validation rules if needed (extend CodeValidator)

### Performance Issues

If validation is slow:

1. Disable mypy: `enable_mypy=False`
2. Use development mode: `validation_level="development"`
3. Run mypy in daemon mode: `mypy --daemon`

## Future Enhancements

Planned improvements:

- [ ] Caching layer for validation results
- [ ] Integration with additional linters (ruff, pylint)
- [ ] Custom validation rule plugins
- [ ] Parallel stage execution
- [ ] Incremental validation (only changed code)
- [ ] Integration with CI/CD systems
- [ ] Validation result persistence to database
- [ ] Quality trend tracking over time

## Related Documentation

- [BusinessLogicGenerator](./BUSINESS_LOGIC_GENERATOR.md) - LLM-powered code generation
- [CodeValidator](./CODE_VALIDATOR.md) - Core validation logic
- [CodeGenerationPipeline](./CODE_GENERATION_PIPELINE.md) - Complete generation workflow
- [ONEX v2.0 Compliance](../../architecture/ONEX_COMPLIANCE.md) - Framework standards

## API Reference

### QualityGatePipeline

```python
class QualityGatePipeline:
    def __init__(
        self,
        validation_level: str = "strict",
        enable_mypy: bool = True,
        mypy_config_path: Optional[Path] = None,
    ) -> None: ...

    async def validate(
        self,
        generated_code: str,
        context: Optional[GenerationContext] = None,
    ) -> ValidationResult: ...
```

### ValidationResult

```python
class ValidationResult(BaseModel):
    passed: bool
    validation_level: ValidationLevel
    quality_score: float
    stage_results: list[StageResult]
    failed_stages: list[str]
    passed_stages: list[str]
    skipped_stages: list[str]
    all_issues: list[str]
    all_warnings: list[str]
    critical_issues: list[str]
    validator_result: Optional[ValidatorResult]
    total_execution_time_ms: float
    total_issues_count: int
    total_warnings_count: int
    timestamp: float
```

### ValidationLevel

```python
class ValidationLevel(str, Enum):
    STRICT = "strict"
    PERMISSIVE = "permissive"
    DEVELOPMENT = "development"
```

### ValidationStage

```python
class ValidationStage(str, Enum):
    SYNTAX = "syntax"
    TYPE_CHECKING = "type_checking"
    ONEX_COMPLIANCE = "onex_compliance"
    CODE_INJECTION = "code_injection"
    SECURITY = "security"
```

## Demonstration

Run the demonstration script to see quality gates in action:

```bash
# Run demo (if dependencies available)
python scripts/demo_quality_gates.py

# Expected output: Multiple validation scenarios showcasing:
# - ONEX-compliant code (all levels)
# - Syntax errors
# - Security issues
# - Unimplemented stubs
# - Comparison of validation levels
```

## Summary

The Quality Gates Pipeline provides:

✅ **Multi-Stage Validation** - 5 comprehensive validation stages
✅ **Configurable Strictness** - 3 validation levels for different environments
✅ **Integration Ready** - Works with CodeValidator and BusinessLogicGenerator
✅ **Detailed Reporting** - Comprehensive issue tracking and quality scoring
✅ **Performance Optimized** - <250ms typical execution time
✅ **Production Ready** - Used in OmniNode code generation pipeline

**Status**: Production-ready, fully integrated with codegen pipeline.
