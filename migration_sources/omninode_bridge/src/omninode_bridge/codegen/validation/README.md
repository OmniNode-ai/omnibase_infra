# Node Validation Pipeline

Comprehensive validation pipeline for ONEX v2.0 mixin-enhanced nodes.

## Overview

The `NodeValidator` provides 6-stage validation of generated node code:

1. **Syntax Validation** (<10ms) - Python compile check
2. **AST Validation** (<20ms) - Structure and method signatures
3. **Import Resolution** (<50ms) - Verify imports can be resolved
4. **ONEX Compliance** (<100ms) - Mixin verification and patterns
5. **Security Scanning** (<100ms) - Dangerous pattern detection
6. **Type Checking** (1-3s, optional) - mypy integration

**Performance Target**: <200ms without type checking, <3s with type checking

## Quick Start

```python
from omninode_bridge.codegen.validation import NodeValidator
from omninode_bridge.codegen.models_contract import ModelEnhancedContract, ModelVersionInfo

# Create validator
validator = NodeValidator(
    enable_type_checking=False,  # Optional mypy checking
    enable_security_scan=True,   # Recommended
)

# Validate generated node
results = await validator.validate_generated_node(
    node_file_content=generated_code,
    contract=enhanced_contract
)

# Check results
for result in results:
    if not result.passed:
        print(f"❌ {result.stage.value} failed:")
        for error in result.errors:
            print(f"  - {error}")
```

## Validation Stages

### 1. Syntax Validation

**What it checks:**
- Python syntax correctness
- IndentationError, TabError, SyntaxError

**Performance**: ~1-10ms

**Example Error:**
```
Line 42: expected ':'
    def broken_function(self
                           ^
```

**Fix Suggestions:**
- Fix syntax error before proceeding with other validation stages

### 2. AST Validation

**What it checks:**
- Valid class definition exists
- Required methods present: `__init__`, `initialize`, `shutdown`
- Async method markers correct
- Proper method signatures
- Execute method matches node type

**Performance**: ~10-20ms

**Example Errors:**
```
Missing required method: async def initialize(self)
Method 'initialize' must be async
```

**Fix Suggestions:**
```python
# Add required method:
async def initialize(self) -> None:
    await super().initialize()
    # Your initialization code here
```

### 3. Import Resolution

**What it checks:**
- All imported modules exist
- Imported names exist in modules
- Special handling for `omnibase_core` (allowed even if not installed)

**Performance**: ~20-50ms

**Example Warnings:**
```
Module 'some_package' not found (may be available at runtime)
```

**Fix Suggestions:**
- Ensure all dependencies are listed in `pyproject.toml` or `requirements.txt`

### 4. ONEX Compliance

**What it checks:**
- Inherits from correct base class (e.g., `NodeEffect`)
- All declared mixins present in inheritance chain
- No duplicate mixins
- Proper `super().__init__(container)` call
- Proper `await super().initialize()` call
- No NodeEffect built-in duplication

**Performance**: ~50-100ms

**Example Errors:**
```
Declared mixin 'MixinHealthCheck' not found in inheritance chain
Missing super().__init__(container) call in __init__
```

**Fix Suggestions:**
```python
# Add mixin to inheritance:
class NodeMyEffect(NodeEffect, MixinHealthCheck, ...):
    ...

# Add super().__init__ call:
def __init__(self, container: ModelContainer):
    super().__init__(container)  # Add this as first line
    ...
```

### 5. Security Scanning

**What it checks:**

**Dangerous Patterns (Errors):**
- `eval()` calls
- `exec()` calls
- `__import__()` dynamic imports
- `os.system()` calls
- `subprocess` with `shell=True`
- Hardcoded secrets (passwords, API keys, tokens)

**Suspicious Patterns (Warnings):**
- `pickle` usage
- Unsafe `yaml.load()` (without Loader)

**Performance**: ~50-100ms

**Example Errors:**
```
Line 45: eval() call detected - security risk
Line 12: Hardcoded API key detected
Line 67: subprocess with shell=True - potential command injection
```

**Fix Suggestions:**
- Remove dangerous patterns and hardcoded secrets
- Use environment variables or secret management for credentials
- When using subprocess, avoid shell=True and validate inputs

### 6. Type Checking (Optional)

**What it checks:**
- Type hint correctness via mypy
- Strict mode compliance
- Type consistency

**Performance**: ~1-3 seconds

**Requires**: `pip install mypy`

**Example Errors:**
```
Type error: Function is missing a return type annotation
Type error: Incompatible types in assignment
```

**Fix Suggestions:**
- Add type hints to all function signatures
- Use `-> None` for functions that don't return values

## Configuration

### Basic Configuration

```python
validator = NodeValidator(
    enable_type_checking=False,   # Disable for faster validation
    enable_security_scan=True,    # Always recommended
)
```

### With Type Checking

```python
from pathlib import Path

validator = NodeValidator(
    enable_type_checking=True,
    enable_security_scan=True,
    mypy_config_path=Path("mypy.ini"),  # Optional
)
```

## Result Interpretation

### ModelValidationResult

```python
@dataclass
class ModelValidationResult:
    stage: EnumValidationStage       # Which validation stage
    passed: bool                      # Did stage pass?
    errors: list[str]                 # Critical errors
    warnings: list[str]               # Non-critical warnings
    execution_time_ms: float          # Time taken
    suggestions: list[str]            # Fix suggestions
```

### Checking Results

```python
results = await validator.validate_generated_node(code, contract)

# Check if all passed
all_passed = all(r.passed for r in results)

# Get failed stages
failed_stages = [r for r in results if not r.passed]

# Get all errors
all_errors = []
for result in results:
    all_errors.extend(result.errors)

# Check performance
total_time = sum(r.execution_time_ms for r in results)
print(f"Validation took {total_time:.1f}ms")
```

### Pretty Printing

```python
for result in results:
    print(result)  # Uses __str__ for formatted output
```

Output:
```
Stage: syntax - ✅ PASSED (2.3ms)

Stage: onex_compliance - ❌ FAILED (45.7ms)
  Errors (2):
    • Declared mixin 'MixinHealthCheck' not found in inheritance chain
    • Missing super().__init__(container) call in __init__
  Suggestions:
    💡 Add MixinHealthCheck to class inheritance:
       class NodeMyEffect(NodeEffect, MixinHealthCheck, ...):
```

## Integration with Code Generation

### In Generation Pipeline

```python
from omninode_bridge.codegen.validation import NodeValidator

class CodeGenerationPipeline:
    def __init__(self):
        self.validator = NodeValidator(
            enable_type_checking=False,  # Too slow for real-time
            enable_security_scan=True,
        )

    async def generate_node(self, contract):
        # ... generate code ...

        # Validate generated code
        results = await self.validator.validate_generated_node(
            node_file_content=generated_code,
            contract=contract
        )

        # Check for failures
        if not all(r.passed for r in results):
            # Collect errors for regeneration
            errors = []
            for result in results:
                errors.extend(result.errors)

            # Retry with error feedback
            return await self.regenerate_with_fixes(errors)

        return generated_code
```

### Post-Generation Validation

```python
# After writing files to disk
from pathlib import Path

node_file = Path("generated_nodes/my_node/node.py")
code = node_file.read_text()

validator = NodeValidator(enable_security_scan=True)
results = await validator.validate_generated_node(code, contract)

if all(r.passed for r in results):
    print("✅ Generated node is valid!")
else:
    print("❌ Validation failed:")
    for result in results:
        if not result.passed:
            print(f"\n{result.stage.value}:")
            for error in result.errors:
                print(f"  - {error}")
```

## Performance Guidelines

### Fast Validation (Development)

```python
# Disable slow stages for rapid iteration
validator = NodeValidator(
    enable_type_checking=False,  # Skip mypy (1-3s)
    enable_security_scan=False,  # Skip security (if trusted)
)
# Expected: ~50-100ms total
```

### Standard Validation (CI/CD)

```python
# Balance speed and thoroughness
validator = NodeValidator(
    enable_type_checking=False,  # Skip mypy
    enable_security_scan=True,   # Always scan security
)
# Expected: ~100-200ms total
```

### Strict Validation (Production)

```python
# Full validation including type checking
validator = NodeValidator(
    enable_type_checking=True,   # Enable mypy
    enable_security_scan=True,   # Always scan security
)
# Expected: ~1-3s total
```

## Common Issues

### 1. Import Warnings

**Issue**: Warnings about missing modules

**Solution**: Normal if modules aren't installed in dev environment. They'll be available at runtime.

```python
# These warnings are safe to ignore:
"Module 'asyncpg' not found (may be available at runtime)"
"omnibase_core imports are always allowed"
```

### 2. Mixin Not Found

**Issue**:
```
Declared mixin 'MixinHealthCheck' not found in inheritance chain
```

**Solution**: Add mixin to class inheritance
```python
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

class NodeMyEffect(NodeEffect, MixinHealthCheck):  # Add mixin here
    ...
```

### 3. Missing super() Calls

**Issue**:
```
Missing super().__init__(container) call in __init__
```

**Solution**: Add super() calls
```python
def __init__(self, container: ModelContainer):
    super().__init__(container)  # Add this first
    # Your initialization

async def initialize(self) -> None:
    await super().initialize()  # Add this first
    # Your initialization
```

### 4. Security Issues

**Issue**:
```
Line 45: eval() call detected - security risk
```

**Solution**: Remove dangerous patterns
```python
# ❌ Bad
result = eval(user_input)

# ✅ Good - use safe alternatives
import ast
result = ast.literal_eval(user_input)  # Only for literals
```

## Testing

### Unit Tests

See `tests/unit/codegen/validation/test_validator.py` for comprehensive tests:

- Syntax validation tests
- AST validation tests
- Import resolution tests
- ONEX compliance tests
- Security scanning tests
- Performance tests

### Integration Tests

See `tests/integration/codegen/validation/test_node_validation.py` for real-world scenarios:

- Complex nodes with multiple mixins
- All node types (effect, compute, reducer, orchestrator)
- Performance under load
- Error message quality

### Running Tests

```bash
# Run all validation tests
pytest tests/unit/codegen/validation/ -v

# Run specific test
pytest tests/unit/codegen/validation/test_validator.py::test_validate_syntax_valid_code -v

# Run with coverage
pytest tests/unit/codegen/validation/ --cov=omninode_bridge.codegen.validation --cov-report=html
```

## Architecture

### Class Hierarchy

```
NodeValidator
├── _validate_syntax()        # Compile check
├── _validate_ast()            # Structure check
├── _validate_imports()        # Import resolution
├── _validate_onex_compliance()  # ONEX patterns
├── _validate_security()       # Security scan
├── _validate_types()          # Optional mypy
└── Helper methods:
    ├── _extract_class_from_ast()
    ├── _extract_methods_from_class()
    ├── _check_init_signature()
    ├── _check_super_call()
    └── _get_name_from_node()
```

### Data Flow

```
node_file_content + contract
         ↓
validate_generated_node()
         ↓
    [Stage 1: Syntax] ─→ ModelValidationResult
         ↓ (if passed)
    [Stage 2: AST] ─→ ModelValidationResult
         ↓
    [Stage 3: Imports] ─→ ModelValidationResult
         ↓
    [Stage 4: ONEX] ─→ ModelValidationResult
         ↓
    [Stage 5: Security] ─→ ModelValidationResult
         ↓ (if enabled)
    [Stage 6: Types] ─→ ModelValidationResult
         ↓
list[ModelValidationResult]
```

## Future Enhancements

### Planned Features

1. **Caching**: Cache validation results for unchanged code
2. **Incremental Validation**: Only re-validate changed stages
3. **Custom Rules**: User-defined validation rules
4. **Auto-Fix**: Automatic fix suggestions for common issues
5. **Metrics Collection**: Track validation metrics over time
6. **Parallel Validation**: Run independent stages in parallel

### Performance Optimizations

1. **AST Caching**: Cache parsed ASTs
2. **Import Memoization**: Cache import resolution results
3. **Regex Compilation**: Pre-compile security patterns
4. **Lazy mypy**: Only run mypy if other stages pass

## License

Part of OmniNode Bridge - See main project LICENSE

## Support

For issues or questions:
- GitHub Issues: [omninode_bridge/issues](https://github.com/yourorg/omninode_bridge/issues)
- Documentation: `docs/guides/VALIDATION_GUIDE.md`
- Examples: `examples/codegen/validation_examples.py`
