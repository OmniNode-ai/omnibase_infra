# Framework Integration Guide

How omnibase_infra validation integrates with the omnibase_core validation framework.

## Architecture Overview

```
omnibase_infra/validation/
├── infra_validators.py          # Infrastructure-specific wrappers
└── __init__.py                  # Public API exports

omnibase_core/validation/        # Core validation framework
├── validate_architecture()
├── validate_contracts()
├── validate_patterns()
├── validate_union_usage()
└── circular_import_validator.py
```

## Integration Pattern

omnibase_infra provides thin wrappers around omnibase_core validators with infrastructure-appropriate defaults:

```python
# omnibase_infra wrapper
def validate_infra_architecture(
    directory: str | Path = INFRA_SRC_PATH,
    max_violations: int = INFRA_MAX_VIOLATIONS,
) -> ValidationResult:
    """Infrastructure architecture validation with strict defaults."""
    return validate_architecture(str(directory), max_violations=max_violations)
```

**Benefits**:
- Infrastructure-specific defaults (strict enforcement for infrastructure)
- Consistent API across ONEX ecosystem
- Centralized validation logic in omnibase_core
- Type safety with `ValidationResult` alias

## Validation Models

### ModelValidationResult[None]

Most validators return `ModelValidationResult[None]`:

```python
from omnibase_core.validation import ModelValidationResult

class ModelValidationResult(Generic[T], BaseModel):
    is_valid: bool
    errors: list[str]
    metadata: ModelValidationMetadata | None = None
    data: T | None = None
```

**Type Alias** (omnibase_infra convenience):
```python
ValidationResult = ModelValidationResult[None]
```

### CircularImportValidationResult

Import validator uses specialized result model:

```python
from omnibase_core.validation import CircularImportValidationResult

class CircularImportValidationResult(BaseModel):
    has_circular_imports: bool
    circular_imports: list[str]
    import_errors: list[ModelImportError]
    unexpected_errors: list[str]
    total_files: int
    success_count: int
    failure_count: int
    success_rate: float
```

**Key Difference**: Uses `has_circular_imports` instead of `is_valid`.

## Validator Integration Details

### 1. Architecture Validator

**Core Function**: `omnibase_core.validation.validate_architecture`

**Integration**:
```python
from omnibase_core.validation import validate_architecture

result = validate_architecture(
    directory=str(directory),
    max_violations=max_violations
)
```

**Metadata**:
- `files_processed`: Number of Python files analyzed
- `violations_found`: Count of multi-model files
- `max_violations`: Configured threshold

### 2. Contract Validator

**Core Function**: `omnibase_core.validation.validate_contracts`

**Integration**:
```python
from omnibase_core.validation import validate_contracts

result = validate_contracts(str(directory))
```

**Metadata**:
- `yaml_files_found`: Number of contract YAML files
- `violations_found`: Count of invalid contracts

**Deep Validation**: Use `validate_yaml_file()` for comprehensive checking:
```python
from omnibase_core.validation import validate_yaml_file

result = validate_yaml_file(Path(contract_path))
```

### 3. Pattern Validator

**Core Function**: `omnibase_core.validation.validate_patterns`

**Integration**:
```python
from omnibase_core.validation import validate_patterns

result = validate_patterns(
    directory=str(directory),
    strict=strict
)
```

**Metadata**:
- `files_processed`: Python files analyzed
- `strict_mode`: Boolean strict mode flag
- `violations_found`: Pattern violation count

**Strict Mode Effects**:
- `True`: Enforce all naming conventions and anti-patterns
- `False`: Relaxed mode (not recommended for infrastructure)

### 4. Union Usage Validator

**Core Function**: `omnibase_core.validation.validate_union_usage`

**Integration**:
```python
from omnibase_core.validation import validate_union_usage

result = validate_union_usage(
    directory=str(directory),
    max_unions=max_unions,
    strict=strict
)
```

**Metadata**:
- `total_unions`: Count of complex union types found
- `max_unions`: Configured threshold

**Infrastructure Justification**: Infrastructure code needs typed unions for:
- Protocol implementations (Consul, Kafka, Vault adapters)
- Message routing and handler dispatch
- Service integration type safety

### 5. Circular Import Validator

**Core Class**: `omnibase_core.validation.circular_import_validator.CircularImportValidator`

**Integration**:
```python
from omnibase_core.validation.circular_import_validator import CircularImportValidator

validator = CircularImportValidator(source_path=Path(directory))
result = validator.validate()
```

**Result Attributes**:
- `has_circular_imports`: Primary pass/fail indicator
- `circular_imports`: List of modules in cycles
- `import_errors`: Detailed import error information
- `success_rate`: Percentage of successful imports

**Error Handling**: Distinguishes between:
1. Circular imports (dependency cycles)
2. Missing dependencies (import errors)
3. Unexpected errors (validator bugs)

## Type Safety

### Type Alias Usage

Infrastructure wrappers use `ValidationResult` type alias for cleaner signatures:

```python
# Before (verbose)
def validate_infra_architecture(...) -> ModelValidationResult[None]:
    ...

# After (clean)
def validate_infra_architecture(...) -> ValidationResult:
    ...
```

### Union Type Handling

Result aggregation uses union types:

```python
def validate_infra_all(...) -> dict[str, ValidationResult | CircularImportValidationResult]:
    results: dict[str, ValidationResult | CircularImportValidationResult] = {}
    # ...
```

**Justification**: Different validators return different result types. isinstance check is justified for result type discrimination.

## Configuration Constants

Infrastructure-specific constants override core defaults:

```python
# Infrastructure paths
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"

# Infrastructure thresholds (stricter than core defaults)
INFRA_MAX_VIOLATIONS = 0        # Zero tolerance for architecture violations
INFRA_PATTERNS_STRICT = True    # Strict pattern enforcement with documented exemptions (OMN-983)

# Union validation thresholds
INFRA_MAX_UNIONS = 491          # Buffer above ~485 baseline (target: <200)
INFRA_UNIONS_STRICT = True      # Strict union validation (flags actual violations per OMN-983)
```

## Error Handling

All validators follow consistent error handling:

1. **Import Errors**: Gracefully handle missing omnibase_core
2. **Path Errors**: Validate directory existence
3. **Validation Errors**: Return structured error messages
4. **Unexpected Errors**: Log and propagate with context

Example from `scripts/validate.py`:

```python
try:
    from omnibase_core.validation import validate_architecture
    result = validate_architecture(...)
    return bool(result.is_valid)
except ImportError as e:
    print(f"Skipping validation: {e}")
    return True  # Don't fail CI on missing dependencies
```

## Extension Points

### Custom Validators

Add infrastructure-specific validators:

```python
def validate_infra_custom(
    directory: str | Path = INFRA_SRC_PATH,
) -> ValidationResult:
    """Custom infrastructure validation logic."""
    # Custom validation implementation
    return ModelValidationResult(
        is_valid=True,
        errors=[],
        metadata=None,
    )
```

### Validator Aggregation

Combine validators in `validate_infra_all()`:

```python
def validate_infra_all(...) -> dict[str, ValidationResult | CircularImportValidationResult]:
    results = {}

    # HIGH priority validators
    results["architecture"] = validate_infra_architecture(directory)
    results["contracts"] = validate_infra_contracts(nodes_directory)
    results["patterns"] = validate_infra_patterns(directory)

    # MEDIUM priority validators
    results["union_usage"] = validate_infra_union_usage(directory)
    results["circular_imports"] = validate_infra_circular_imports(directory)

    # Custom validators
    results["custom"] = validate_infra_custom(directory)

    return results
```

## Version Compatibility

**Current Versions**:
- omnibase_core: 0.3.5
- omnibase_infra: 0.1.0

**Compatibility Requirements**:
- omnibase_core >= 0.3.0 (validation framework introduced)
- Python >= 3.12 (required for type annotations)

**Breaking Changes**: When omnibase_core validator API changes, update wrappers in omnibase_infra accordingly.

## Performance Considerations

See [Performance Notes](performance_notes.md) for detailed performance analysis.

**Quick Summary**:
- Architecture validator: O(n) file scans
- Contract validator: O(n) YAML parsing
- Pattern validator: O(n) AST analysis
- Union validator: O(n) type annotation parsing
- Import validator: O(n²) dependency graph analysis

For large codebases, consider incremental validation (validate changed files only).

## Testing Integration

Validation framework is tested in omnibase_infra:

```python
# tests/unit/test_smoke.py
def test_validation_module_import() -> None:
    """Verify validators are importable and callable."""
    from omnibase_infra.validation import validate_infra_all

    assert validate_infra_all is not None
    assert callable(validate_infra_all)
```

Integration tests validate actual validation logic (not just imports).

## Next Steps

- [Validator Reference](validator_reference.md) - Detailed validator documentation
- [Performance Notes](performance_notes.md) - Performance optimization strategies
- [Troubleshooting](troubleshooting.md) - Common integration issues
