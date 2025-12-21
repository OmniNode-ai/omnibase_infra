# Validator Reference

Comprehensive reference documentation for all ONEX infrastructure validators.

## Table of Contents

1. [validate_infra_architecture](#validate_infra_architecture)
2. [validate_infra_contracts](#validate_infra_contracts)
3. [validate_infra_patterns](#validate_infra_patterns)
4. [validate_infra_contract_deep](#validate_infra_contract_deep)
5. [validate_infra_union_usage](#validate_infra_union_usage)
6. [validate_infra_circular_imports](#validate_infra_circular_imports)
7. [validate_infra_all](#validate_infra_all)
8. [get_validation_summary](#get_validation_summary)

---

## validate_infra_architecture

Validate infrastructure architecture with strict one-model-per-file enforcement.

### Signature

```python
def validate_infra_architecture(
    directory: str | Path = INFRA_SRC_PATH,
    max_violations: int = INFRA_MAX_VIOLATIONS,
) -> ValidationResult:
```

### Parameters

- **directory** (`str | Path`, optional): Directory to validate. Defaults to `"src/omnibase_infra/"`.
- **max_violations** (`int`, optional): Maximum allowed violations. Defaults to `0` (strict enforcement).

### Returns

`ValidationResult` with:
- `is_valid`: `True` if violations ≤ max_violations
- `errors`: List of violation messages
- `metadata`: `ModelValidationMetadata` with:
  - `files_processed`: Number of Python files analyzed
  - `violations_found`: Count of multi-model files
  - `max_violations`: Configured threshold

### What It Validates

- **One model per file**: Each Python file should contain at most one Pydantic model class
- **Model identification**: Classes inheriting from `BaseModel` or with `Model*` prefix
- **File organization**: Proper separation of model definitions

### Common Violations

```python
# VIOLATION: Multiple models in one file
# File: src/omnibase_infra/models/bad_example.py
class ModelUserData(BaseModel):
    name: str

class ModelUserSettings(BaseModel):  # VIOLATION: Second model
    theme: str
```

**Fix**: Split into separate files:
```
src/omnibase_infra/models/model_user_data.py
src/omnibase_infra/models/model_user_settings.py
```

### Example Usage

```python
from omnibase_infra.validation import validate_infra_architecture

# Default strict validation
result = validate_infra_architecture()
if not result.is_valid:
    for error in result.errors:
        print(f"Architecture violation: {error}")

# Custom directory and threshold
result = validate_infra_architecture(
    directory="src/custom_module/",
    max_violations=5
)
```

### CI/CD Integration

```bash
poetry run python scripts/validate.py architecture
```

---

## validate_infra_contracts

Validate YAML contract files for infrastructure nodes.

### Signature

```python
def validate_infra_contracts(
    directory: str | Path = INFRA_NODES_PATH,
) -> ValidationResult:
```

### Parameters

- **directory** (`str | Path`, optional): Directory containing node contracts. Defaults to `"src/omnibase_infra/nodes/"`.

### Returns

`ValidationResult` with:
- `is_valid`: `True` if all contracts valid
- `errors`: List of contract validation errors
- `metadata`: `ModelValidationMetadata` with:
  - `yaml_files_found`: Number of contract YAML files
  - `violations_found`: Count of invalid contracts

### What It Validates

- **Contract schema compliance**: All required fields present
- **YAML syntax**: Valid YAML formatting
- **Contract completeness**: Input/output models defined
- **Dependency declarations**: Valid dependency references

### Common Violations

```yaml
# VIOLATION: Missing required fields
contract_version: "1.0.0"
# node_type: MISSING (required)
# node_name: MISSING (required)
```

**Fix**: Add required fields:
```yaml
contract_version: "1.0.0"
node_type: "EFFECT"
node_name: "infrastructure_consul_adapter"
input_model: "ModelConsulAdapterInput"
output_model: "ModelConsulAdapterOutput"
```

### Example Usage

```python
from omnibase_infra.validation import validate_infra_contracts

# Validate all node contracts
result = validate_infra_contracts()
if not result.is_valid:
    print(f"Found {result.metadata.violations_found} invalid contracts")
    for error in result.errors:
        print(f"Contract error: {error}")

# Custom nodes directory
result = validate_infra_contracts(directory="custom/nodes/")
```

### CI/CD Integration

```bash
poetry run python scripts/validate.py contracts
```

---

## validate_infra_patterns

Validate infrastructure code patterns and naming conventions.

### Signature

```python
def validate_infra_patterns(
    directory: str | Path = INFRA_SRC_PATH,
    strict: bool = INFRA_PATTERNS_STRICT,
) -> ValidationResult:
```

### Parameters

- **directory** (`str | Path`, optional): Directory to validate. Defaults to `"src/omnibase_infra/"`.
- **strict** (`bool`, optional): Enable strict mode. Defaults to `True` (INFRA_PATTERNS_STRICT per OMN-983).

### Returns

`ValidationResult` with:
- `is_valid`: `True` if all patterns compliant
- `errors`: List of pattern violations
- `metadata`: `ModelValidationMetadata` with:
  - `files_processed`: Python files analyzed
  - `strict_mode`: Boolean strict mode flag
  - `violations_found`: Pattern violation count

### What It Validates

**Strict Mode** (default per OMN-983):
- Model prefix naming (`Model*`)
- snake_case file naming
- Anti-pattern detection (no `*Manager`, `*Handler`, `*Helper`)
- Documented exemptions in `exempted_patterns` list (KafkaEventBus, RuntimeHostProcess, etc.)

**Relaxed Mode** (strict=False):
- Model prefix naming (warnings only)
- File naming (warnings only)

### Common Violations

```python
# VIOLATION: Missing Model prefix
class ConsulAdapterInput(BaseModel):  # Should be ModelConsulAdapterInput
    service_name: str

# VIOLATION: Anti-pattern class name
class ConfigManager:  # *Manager is an anti-pattern
    pass

# VIOLATION: CamelCase file name
# File: ConsulAdapter.py  # Should be consul_adapter.py
```

**Fix**: Follow ONEX conventions:
```python
# CORRECT: Model prefix
class ModelConsulAdapterInput(BaseModel):
    service_name: str

# CORRECT: Use specific pattern
class ConsulConfigRegistry:  # Instead of ConfigManager
    pass

# CORRECT: snake_case file name
# File: consul_adapter.py
```

### Example Usage

```python
from omnibase_infra.validation import validate_infra_patterns

# Strict validation (recommended)
result = validate_infra_patterns(strict=True)
if not result.is_valid:
    for error in result.errors:
        print(f"Pattern violation: {error}")

# Relaxed validation
result = validate_infra_patterns(strict=False)
```

### CI/CD Integration

```bash
poetry run python scripts/validate.py patterns
```

---

## validate_infra_contract_deep

Perform deep contract validation for ONEX compliance.

### Signature

```python
def validate_infra_contract_deep(
    contract_path: str | Path,
    contract_type: Literal["effect", "compute", "reducer", "orchestrator"] = "effect",
) -> ModelContractValidationResult:
```

### Parameters

- **contract_path** (`str | Path`): Path to the contract YAML file.
- **contract_type** (`Literal`, optional): Type of contract to validate. Defaults to `"effect"`.

### Returns

`ModelContractValidationResult` with:
- `is_valid`: `True` if contract passes comprehensive validation
- `validation_score`: Quality score (0.0-1.0)
- `errors`: List of validation errors
- `warnings`: List of validation warnings
- `suggestions`: List of improvement suggestions

### What It Validates

Comprehensive validation suitable for autonomous code generation:
- **Schema compliance**: Complete contract structure
- **Type definitions**: All models and schemas defined
- **Dependency resolution**: Valid dependency references
- **Contract completeness**: All required metadata present
- **Generation readiness**: Contract suitable for code generation

### Example Usage

```python
from omnibase_infra.validation import validate_infra_contract_deep

# Deep validation for code generation
result = validate_infra_contract_deep(
    contract_path="src/omnibase_infra/nodes/consul_adapter/v1_0_0/contract.yaml",
    contract_type="effect"
)

print(f"Validation score: {result.validation_score:.2%}")
if not result.is_valid:
    print("Errors:", result.errors)
if result.warnings:
    print("Warnings:", result.warnings)
if result.suggestions:
    print("Suggestions:", result.suggestions)
```

### Use Cases

- **Pre-generation validation**: Ensure contract is complete before code generation
- **Contract quality scoring**: Assess contract completeness and quality
- **Migration validation**: Verify migrated contracts meet ONEX standards

---

## validate_infra_union_usage

Validate Union type usage to prevent overly complex types.

Counts actual **VIOLATIONS** (problematic patterns), not total unions.
Valid `X | None` patterns are not counted as violations.

### Signature

```python
def validate_infra_union_usage(
    directory: str | Path = INFRA_SRC_PATH,
    max_violations: int = INFRA_MAX_UNION_VIOLATIONS,
    strict: bool = INFRA_UNIONS_STRICT,
) -> ValidationResult:
```

### Parameters

- **directory** (`str | Path`, optional): Directory to validate. Defaults to `"src/omnibase_infra/"`.
- **max_violations** (`int`, optional): Maximum allowed union violations. Defaults to `10` (INFRA_MAX_UNION_VIOLATIONS).
- **strict** (`bool`, optional): Enable strict mode. Defaults to `True` (INFRA_UNIONS_STRICT per OMN-983).

### Returns

`ValidationResult` with:
- `is_valid`: `True` if violations within threshold
- `errors`: List of union violations (problematic patterns only)
- `summary`: Human-readable summary with violation count and total unions
- `metadata`: `ModelValidationMetadata` with:
  - `total_unions`: Total count of all unions (informational)

### What Counts as a Violation

- **Primitive soup**: `str | int | float | bool` (4+ primitive types)
- **Mixed type unions**: `str | int | MyModel` (primitives mixed with models)
- **Legacy syntax**: `Union[X, None]` instead of `X | None`

### What is NOT Counted (Valid Patterns)

- **Nullable patterns**: `X | None` (ONEX-preferred, PEP 604 compliant)
- **Discriminated unions**: `ModelA | ModelB`
- **Simple unions**: 2-3 type unions for legitimate use cases

### Infrastructure Justification

Infrastructure code needs typed unions for:
- Protocol implementations (Consul, Kafka, Vault adapters)
- Message routing and handler dispatch
- Service integration type safety

**Why max_unions=400**: Infrastructure has many service adapters with typed handlers, protocol implementations, message routing, and registration event models. The threshold is set as a tight buffer above the current baseline (~379 unions as of 2025-12-20). Most unions are legitimate `X | None` nullable patterns (ONEX-preferred PEP 604 syntax) which are counted but NOT flagged as violations. The target is to reduce to <200 through ongoing `dict[str, object]` to `JsonValue` migration.

### Example Usage

```python
from omnibase_infra.validation import validate_infra_union_usage

# Default validation
result = validate_infra_union_usage()
if not result.is_valid:
    print(f"Found {result.metadata.total_unions} unions (max: {result.metadata.max_unions})")

# Strict validation (fail on any union)
result = validate_infra_union_usage(strict=True, max_unions=0)
```

### CI/CD Integration

```bash
poetry run python scripts/validate.py unions
```

---

## validate_infra_circular_imports

Check for circular import dependencies.

### Signature

```python
def validate_infra_circular_imports(
    directory: str | Path = INFRA_SRC_PATH,
) -> CircularImportValidationResult:
```

### Parameters

- **directory** (`str | Path`, optional): Directory to check. Defaults to `"src/omnibase_infra/"`.

### Returns

`CircularImportValidationResult` with:
- `has_circular_imports`: `True` if circular imports detected
- `circular_imports`: List of modules in cycles
- `import_errors`: Detailed import error information
- `unexpected_errors`: List of unexpected errors
- `total_files`: Number of files analyzed
- `success_count`: Successful imports
- `failure_count`: Failed imports
- `success_rate`: Percentage of successful imports

### What It Validates

- **Circular dependencies**: Modules importing each other in cycles
- **Import errors**: Missing dependencies or syntax errors
- **Import health**: Overall import success rate

### Common Violations

```python
# VIOLATION: Circular import
# File: module_a.py
from module_b import FunctionB

# File: module_b.py
from module_a import FunctionA  # Creates cycle: A -> B -> A
```

**Fix**: Break circular dependency:
```python
# Solution 1: Move shared code to common module
# File: module_common.py
class SharedClass:
    pass

# File: module_a.py
from module_common import SharedClass

# File: module_b.py
from module_common import SharedClass

# Solution 2: Use TYPE_CHECKING imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_b import FunctionB  # Only for type hints
```

### Example Usage

```python
from omnibase_infra.validation import validate_infra_circular_imports

# Check for circular imports
result = validate_infra_circular_imports()

if result.has_circular_imports:
    print("Circular import cycles detected:")
    for module in result.circular_imports:
        print(f"  - {module}")

if result.import_errors:
    print("\nImport errors:")
    for error in result.import_errors[:5]:
        print(f"  - {error.module_name}: {error.error_message}")

print(f"\nSuccess rate: {result.success_rate:.1%}")
```

### CI/CD Integration

```bash
poetry run python scripts/validate.py imports
```

---

## validate_infra_all

Run all validators on infrastructure code.

### Signature

```python
def validate_infra_all(
    directory: str | Path = INFRA_SRC_PATH,
    nodes_directory: str | Path = INFRA_NODES_PATH,
) -> dict[str, ValidationResult | CircularImportValidationResult]:
```

### Parameters

- **directory** (`str | Path`, optional): Main source directory. Defaults to `"src/omnibase_infra/"`.
- **nodes_directory** (`str | Path`, optional): Nodes directory for contract validation. Defaults to `"src/omnibase_infra/nodes/"`.

### Returns

Dictionary mapping validator name to result:
- `"architecture"`: `ValidationResult`
- `"contracts"`: `ValidationResult`
- `"patterns"`: `ValidationResult`
- `"union_usage"`: `ValidationResult`
- `"circular_imports"`: `CircularImportValidationResult`

### Validation Priority

**HIGH Priority** (critical for infrastructure):
1. Architecture
2. Contracts
3. Patterns

**MEDIUM Priority** (important but not blocking):
4. Union usage
5. Circular imports

### Example Usage

```python
from omnibase_infra.validation import validate_infra_all, get_validation_summary

# Run all validators
results = validate_infra_all()

# Check individual results
if not results["architecture"].is_valid:
    print("Architecture validation failed")

# Get summary
summary = get_validation_summary(results)
print(f"Passed: {summary['passed']}/{summary['total_validators']}")

if summary['failed'] > 0:
    print(f"Failed validators: {', '.join(summary['failed_validators'])}")
```

### CI/CD Integration

```bash
# Run all validators
poetry run python scripts/validate.py all --verbose

# Quick mode (skip medium priority)
poetry run python scripts/validate.py all --quick
```

---

## get_validation_summary

Generate a summary of validation results.

### Signature

```python
def get_validation_summary(
    results: dict[str, ValidationResult | CircularImportValidationResult],
) -> dict[str, int | list[str]]:
```

### Parameters

- **results** (`dict`): Dictionary of validation results from `validate_infra_all()`.

### Returns

Dictionary with summary statistics:
- `"total_validators"` (`int`): Total number of validators run
- `"passed"` (`int`): Number of validators that passed
- `"failed"` (`int`): Number of validators that failed
- `"failed_validators"` (`list[str]`): Names of failed validators

### Example Usage

```python
from omnibase_infra.validation import validate_infra_all, get_validation_summary

results = validate_infra_all()
summary = get_validation_summary(results)

print(f"""
Validation Summary
==================
Total Validators: {summary['total_validators']}
Passed: {summary['passed']}
Failed: {summary['failed']}
""")

if summary['failed'] > 0:
    print("Failed Validators:")
    for validator in summary['failed_validators']:
        print(f"  - {validator}")
        print(f"    Errors: {results[validator].errors}")
```

---

## Constants Reference

```python
# Type aliases
ValidationResult = ModelValidationResult[None]

# Infrastructure paths
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"

# Validation thresholds
INFRA_MAX_UNION_VIOLATIONS = 10 # Max union violations (not total unions)
INFRA_MAX_VIOLATIONS = 0        # Zero tolerance for architecture violations

# Strict mode flags (OMN-983)
INFRA_PATTERNS_STRICT = True    # Strict pattern enforcement with documented exemptions
INFRA_UNIONS_STRICT = True      # Strict union validation (flags actual violations)
```

---

## Next Steps

- [Framework Integration](framework_integration.md) - Integration with omnibase_core
- [Performance Notes](performance_notes.md) - Performance optimization
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
