# Business Logic Code Validator

**Status**: ✅ Complete (Phase 2)
**Test Coverage**: 95.8% (validator.py), 98.1% (validation_rules.py)
**Tests**: 31 tests, all passing

## Overview

The CodeValidator provides comprehensive validation of LLM-generated code for the business logic generation pipeline. It ensures generated code meets quality, security, and ONEX compliance standards before injection into node implementations.

## Components

### 1. CodeValidator (`validator.py`)

Main validation class that orchestrates all validation checks.

**Key Features:**
- AST-based syntax validation
- ONEX compliance checking
- Type hint validation
- Security vulnerability detection
- Code quality assessment
- Quality score calculation (0.0-1.0)

**Usage:**
```python
from omninode_bridge.codegen.business_logic import CodeValidator, GenerationContext

validator = CodeValidator(strict=True)
context = GenerationContext(
    node_type="effect",
    method_name="execute_effect",
    service_name="postgres_crud"
)

result = await validator.validate(generated_code, context)

if result.passed:
    print(f"Code validation passed! Quality score: {result.quality_score}")
else:
    print(f"Validation failed: {result.issues}")
```

### 2. Validation Rules (`validation_rules.py`)

Defines validation rules and provides helper functions for validation checks.

**Security Checks:**
- Hardcoded secrets detection (passwords, API keys, tokens)
- SQL injection vulnerabilities (f-strings with SQL)
- Dangerous patterns (eval, exec, pickle, dynamic imports)

**ONEX Compliance:**
- ModelOnexError usage (instead of generic exceptions)
- emit_log_event usage (instead of logger/print)
- omnibase_core imports

**Code Quality:**
- Cyclomatic complexity estimation
- Function length checking
- Docstring presence and quality

## Validation Modes

### Strict Mode (default)
Fails validation on any issue (syntax, security, ONEX, type hints, quality).

```python
validator = CodeValidator(strict=True)
```

### Lenient Mode
Only fails on critical issues (syntax errors, security vulnerabilities).

```python
validator = CodeValidator(strict=False)
```

## Validation Result

```python
class ValidationResult(BaseModel):
    # Overall status
    passed: bool
    quality_score: float  # 0.0-1.0

    # Component results
    syntax_valid: bool
    onex_compliant: bool
    has_type_hints: bool
    security_clean: bool

    # Issues by category
    issues: List[str]           # All issues
    syntax_errors: List[str]
    onex_issues: List[str]
    type_hint_issues: List[str]
    security_issues: List[str]
    quality_issues: List[str]

    # Metrics
    complexity_score: int
    line_count: int
```

## Quality Scoring

Quality score is calculated using weighted components:

| Component | Weight | Critical? |
|-----------|--------|-----------|
| Syntax Validation | 30% | ✅ Yes |
| Security Checks | 25% | ✅ Yes |
| ONEX Compliance | 20% | No |
| Type Hints | 15% | No |
| Code Quality | 10% | No |

**Additional penalty**: -0.05 per issue (max -0.2)

## Validation Checks

### 1. Syntax Validation
- AST parsing to verify Python syntax
- Error messages include line numbers
- Catches syntax errors before execution

### 2. ONEX Compliance
- ✅ ModelOnexError for exceptions
- ✅ emit_log_event for logging
- ✅ omnibase_core imports
- ❌ Generic Exception usage
- ❌ logger.info/print() usage

### 3. Type Hints
- Function return type annotations
- Argument type annotations
- Exceptions: `__init__`, `__str__`, `__repr__`, private methods

### 4. Security Checks
- **Hardcoded Secrets**: Detects passwords, API keys, tokens
- **SQL Injection**: F-strings with SQL keywords + execute()
- **Dangerous Functions**: eval(), exec(), pickle.loads(), __import__()

### 5. Code Quality
- **Complexity**: Cyclomatic complexity ≤ 10
- **Length**: Function length ≤ 50 lines
- **Documentation**: Docstrings ≥ 20 characters

## Test Coverage

**31 tests covering all aspects:**

**Syntax Tests (3):**
- Valid syntax
- Invalid syntax
- Syntax error line numbers

**ONEX Compliance Tests (3):**
- Compliant code
- Missing ModelOnexError
- Missing emit_log_event

**Type Hint Tests (4):**
- Complete type hints
- Missing return type
- Missing argument types
- Magic method exceptions

**Security Tests (5):**
- Secure code
- Hardcoded passwords
- SQL injection (f-strings)
- Dangerous eval()
- Dangerous exec()

**Code Quality Tests (5):**
- Good quality code
- High complexity
- Missing docstring
- Short docstring
- Long function

**Quality Score Tests (2):**
- Perfect code (score ≥ 0.9)
- Poor code (score < 0.5)

**Mode Tests (3):**
- Strict mode behavior
- Lenient mode (passes on warnings)
- Lenient mode (fails on critical)

**Integration Tests (2):**
- ValidationResult structure
- Issue aggregation

**Validation Rules Tests (4):**
- Hardcoded secret detection
- Dangerous pattern detection
- ONEX compliance checking
- Complexity estimation

## Integration with BusinessLogicGenerator

The CodeValidator is used by BusinessLogicGenerator to validate each generated method:

```python
# In BusinessLogicGenerator
validator = CodeValidator(strict=True)

for method in methods_to_generate:
    # Generate code via LLM
    generated_code = await llm.generate(prompt)

    # Validate generated code
    validation = await validator.validate(generated_code, context)

    if not validation.passed:
        # Retry or log issues
        logger.warning(f"Validation failed: {validation.issues}")
    else:
        # Inject into node file
        inject_implementation(generated_code)
```

## Configuration

```python
# validation_rules.py constants
MAX_COMPLEXITY = 10        # Maximum cyclomatic complexity
MAX_FUNCTION_LENGTH = 50   # Maximum lines per function
MIN_DOCSTRING_LENGTH = 20  # Minimum docstring length

# Security keywords to check
SECURITY_KEYWORDS = [
    "password", "api_key", "secret", "token",
    "credentials", "auth", "private_key"
]

# Dangerous patterns
DANGEROUS_PATTERNS = [
    (r"eval\(", "Use of eval() is dangerous"),
    (r"exec\(", "Use of exec() is dangerous"),
    (r"__import__\(", "Dynamic imports should be avoided"),
    (r"pickle\.loads?\(", "Pickle can execute arbitrary code"),
]
```

## Future Enhancements

1. **AST Manipulation**: Direct AST modification instead of string replacement
2. **Custom Rules**: User-defined validation rules via configuration
3. **Performance Optimization**: Caching parsed ASTs
4. **Advanced Security**: Integration with bandit/semgrep
5. **Context-Aware Validation**: Validate based on node type requirements

## References

- **Implementation Plan**: `docs/planning/LLM_BUSINESS_LOGIC_GENERATION_PLAN.md` (lines 1181-1340)
- **Tests**: `tests/unit/codegen/business_logic/test_validator.py`
- **Parent Component**: BusinessLogicGenerator (Phase 2)

## Changelog

**2025-10-31**: Initial implementation
- Core CodeValidator class
- Comprehensive validation rules
- 31 unit tests with 95%+ coverage
- Security, ONEX, and quality checks
