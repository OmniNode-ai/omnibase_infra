> **Navigation**: [Home](../index.md) > Validation

# ONEX Infrastructure Validation Framework

Comprehensive validation system for omnibase_infra, ensuring code quality, architecture compliance, and ONEX standards adherence.

> **Note**: For authoritative coding rules and standards that validators enforce, see [CLAUDE.md](../../CLAUDE.md).

## Quick Start

```bash
# Run all validators with verbose output
poetry run python scripts/validate.py all --verbose

# Run specific validators
poetry run python scripts/validate.py architecture
poetry run python scripts/validate.py contracts
poetry run python scripts/validate.py patterns
poetry run python scripts/validate.py unions
poetry run python scripts/validate.py imports

# Quick mode (skip medium priority validators)
poetry run python scripts/validate.py all --quick
```

## Available Validators

### 1. Architecture Validator (HIGH Priority)
**Purpose**: Enforce ONEX one-model-per-file principle

**Configuration**:
- Max violations: `0` (strict enforcement)
- Directory: `src/omnibase_infra/`

**What it checks**:
- One model class per file
- Proper file naming conventions
- Model class organization

**Why it matters**: Critical for infrastructure nodes that depend on contract-driven code generation. Multiple models per file break the generator's assumptions.

### 2. Contract Validator (HIGH Priority)
**Purpose**: Validate YAML contract files for infrastructure nodes

**Configuration**:
- Directory: `src/omnibase_infra/nodes/`
- Contract types: effect, compute, reducer, orchestrator

**What it checks**:
- Contract schema compliance
- Required fields presence
- Type definitions completeness
- Dependency declarations

**Why it matters**: Infrastructure nodes (Consul, Kafka, Vault, PostgreSQL adapters) are contract-driven. Invalid contracts prevent node initialization.

### 3. Pattern Validator (HIGH Priority)
**Purpose**: Enforce ONEX naming conventions and anti-patterns

**Configuration**:
- Strict mode: `True`
- Directory: `src/omnibase_infra/`

**What it checks**:
- Model prefix naming (`Model*`)
- snake_case file naming
- Anti-pattern detection (no `*Manager`, `*Handler`, `*Helper`)

**Why it matters**: Consistency across infrastructure service adapters. Anti-patterns indicate poor architecture.

### 4. Union Usage Validator (MEDIUM Priority)
**Purpose**: Prevent overly complex union types

**Configuration**:
- Max unions: `491` (buffer above ~485 baseline, target: <200)
- Strict mode: `True` (flags actual violations per OMN-983)
- Directory: `src/omnibase_infra/`

**What it checks**:
- Total union count (including `X | None` patterns which are counted but NOT flagged)
- Actual violations (primitive soup like `str | int | bool | float`, `Union[X, None]` syntax)

**Why it matters**: Infrastructure code has typed unions for protocol implementations and message routing. The threshold is set above the current baseline to prevent regression while ongoing `dict[str, object]` to `JsonValue` migration reduces the count toward <200.

### 5. Circular Import Validator (MEDIUM Priority)
**Purpose**: Detect circular import dependencies

**Configuration**:
- Directory: `src/omnibase_infra/`

**What it checks**:
- Import cycles between modules
- Circular dependency chains
- Import errors

**Why it matters**: Infrastructure packages have complex dependencies (Consul, Kafka, Vault). Circular imports cause runtime issues that are hard to debug.

## Validation Results

Each validator returns a `ModelValidationResult[None]` (or `CircularImportValidationResult` for imports) with:

- `is_valid`: Boolean indicating pass/fail
- `errors`: List of error messages
- `metadata`: Additional validation statistics
- `has_circular_imports`: (Import validator only) Boolean for circular import detection

## CI/CD Integration

Validators run automatically in GitHub Actions:

```yaml
# .github/workflows/test.yml
- name: Run ONEX validators
  run: poetry run python scripts/validate.py all --verbose
```

**CI Validation Jobs**:
1. Smoke Tests (fail fast on basic issues)
2. Full Tests (comprehensive test suite)
3. Code Quality (black, isort, ruff, mypy)
4. **ONEX Validators** (infrastructure-specific validation)

All jobs must pass for PR merge approval.

## Exit Codes

- `0`: All validators passed
- `1`: One or more validators failed

Use exit codes in CI/CD pipelines:

```bash
poetry run python scripts/validate.py all || exit 1
```

## Configuration

### Infrastructure-Specific Defaults

Validators are pre-configured with infrastructure-appropriate defaults:

```python
# src/omnibase_infra/validation/infra_validators.py
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"
INFRA_MAX_UNIONS = 491          # Buffer above ~485 baseline (target: <200)
INFRA_MAX_VIOLATIONS = 0
INFRA_PATTERNS_STRICT = True    # Strict mode with documented exemptions (OMN-983)
INFRA_UNIONS_STRICT = True      # Strict union validation (OMN-983)
```

### Customization

Override defaults when calling validators:

```python
from omnibase_infra.validation import validate_infra_architecture

# Custom max violations
result = validate_infra_architecture(
    directory="custom/path",
    max_violations=5
)
```

## Integration with omnibase_core

All validators leverage omnibase_core validation framework:

- Architecture: `omnibase_core.validation.validate_architecture`
- Contracts: `omnibase_core.validation.validate_contracts`
- Patterns: `omnibase_core.validation.validate_patterns`
- Unions: `omnibase_core.validation.validate_union_usage`
- Imports: `omnibase_core.validation.circular_import_validator.CircularImportValidator`

See [Framework Integration](framework_integration.md) for details.

## Documentation

- [Framework Integration](framework_integration.md) - Integration with omnibase_core
- [Validator Reference](validator_reference.md) - Detailed validator documentation
- [Performance Notes](performance_notes.md) - Performance considerations
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Examples

### Programmatic Usage

```python
from omnibase_infra.validation import (
    validate_infra_all,
    get_validation_summary,
)

# Run all validators
results = validate_infra_all()

# Get summary
summary = get_validation_summary(results)
print(f"Passed: {summary['passed']}/{summary['total_validators']}")

if summary['failed'] > 0:
    print(f"Failed validators: {summary['failed_validators']}")
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: onex-validators
        name: ONEX Infrastructure Validators
        entry: poetry run python scripts/validate.py all --quick
        language: system
        pass_filenames: false
```

## Support

For issues or questions:
1. Check [Troubleshooting Guide](troubleshooting.md)
2. Review [Validator Reference](validator_reference.md)
3. File GitHub issue with validation output
