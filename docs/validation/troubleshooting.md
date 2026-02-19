> **Navigation**: [Home](../index.md) > [Validation](README.md) > Troubleshooting

# Validation Troubleshooting Guide

Common issues and solutions for ONEX infrastructure validation.

## Table of Contents

1. [Import Errors](#import-errors)
2. [Validation Failures](#validation-failures)
3. [Performance Issues](#performance-issues)
4. [CI/CD Issues](#cicd-issues)
5. [Integration Issues](#integration-issues)

---

## Import Errors

### Issue: "Cannot import omnibase_core"

**Symptom**:
```bash
uv run python scripts/validate.py all
ImportError: No module named 'omnibase_core'
```

**Root Cause**: omnibase_core not installed or wrong Python environment

**Solution**:
```bash
# Install dependencies
uv sync

# Verify omnibase_core version
uv pip show omnibase-core

# Expected: omnibase-core 0.3.5 or later
```

**Prevention**: Always use `uv run` to execute scripts in the project environment.

---

### Issue: "CircularImportValidator not available"

**Symptom**:
```
Imports: SKIP (CircularImportValidator not available: ...)
```

**Root Cause**: omnibase_core version too old (< 0.3.0)

**Solution**:
```bash
# Update omnibase_core
uv add --upgrade omnibase-core

# Verify version
uv pip show omnibase-core
# Should be >= 0.3.0
```

**Prevention**: Lock omnibase_core version in pyproject.toml:
```toml
[project]
dependencies = [
    "omnibase-core>=0.3.5",
]
```

---

### Issue: "Validator API incompatible"

**Symptom**:
```
Imports: ERROR (Validator API incompatible: 'ModelValidationResult' object has no attribute 'is_valid')
```

**Root Cause**: omnibase_core API changed, omnibase_infra out of sync

**Solution**:
```bash
# Update omnibase_core to compatible version
uv add --upgrade omnibase-core

# If issue persists, check omnibase_core changelog
# Update infra_validators.py to match new API
```

**Prevention**: Pin omnibase_core version to known compatible version.

---

## Validation Failures

### Issue: Architecture Violations - Multiple Models Per File

**Symptom**:
```
Architecture: FAIL
  - src/omnibase_infra/models/database_models.py contains 3 models (max: 1)
```

**Root Cause**: Multiple Pydantic model classes in single file

**Solution**:

1. **Identify models in file**:
   ```bash
   grep -n "class Model" src/omnibase_infra/models/database_models.py
   ```

2. **Split into separate files**:
   ```
   Before:
   src/omnibase_infra/models/database_models.py
     - ModelUserData
     - ModelPostgresConfig
     - ModelConnectionPool

   After:
   src/omnibase_infra/models/model_user_data.py
   src/omnibase_infra/models/model_postgres_config.py
   src/omnibase_infra/models/model_connection_pool.py
   ```

3. **Update imports**:
   ```python
   # Before
   from omnibase_infra.models.database_models import ModelUserData

   # After
   from omnibase_infra.models.model_user_data import ModelUserData
   ```

**Prevention**: Follow ONEX one-model-per-file principle from the start.

---

### Issue: Contract Validation Failures

**Symptom**:
```
Contracts: FAIL
  - src/omnibase_infra/nodes/consul_adapter/v1_0_0/contract.yaml: Missing required field 'node_type'
```

**Root Cause**: Contract YAML missing required fields

**Solution**:

1. **Check contract schema requirements**:
   ```yaml
   # Required fields
   contract_version: "1.0.0"
   node_version: "1.0.0"
   node_type: "EFFECT_GENERIC"  # Use _GENERIC variants
   node_name: "infrastructure_consul_adapter"
   input_model: "ModelConsulAdapterInput"
   output_model: "ModelConsulAdapterOutput"
   ```

2. **Add missing fields**:
   ```yaml
   node_type: "EFFECT_GENERIC"  # For service adapters
   # Other valid types: COMPUTE_GENERIC, REDUCER_GENERIC, ORCHESTRATOR_GENERIC
   ```

3. **Re-validate**:
   ```bash
   uv run python scripts/validate.py contracts
   ```

**Prevention**: Use contract templates from omnibase_core when creating new nodes.

---

### Issue: Pattern Violations - Anti-patterns Detected

**Symptom**:
```
Patterns: FAIL
  - src/omnibase_infra/utils/config_manager.py: Class 'ConfigManager' uses anti-pattern suffix '*Manager'
```

**Root Cause**: Class names using anti-pattern suffixes

**Solution**:

1. **Identify anti-pattern classes**:
   ```bash
   grep -rn "class.*Manager\|class.*Handler\|class.*Helper" src/
   ```

2. **Rename to specific patterns**:
   ```python
   # Anti-pattern
   class ConfigManager:
       pass

   # ONEX pattern
   class ConfigRegistry:  # or ConfigService, ConfigAdapter
       pass
   ```

3. **Update all references**:
   ```bash
   # Find all usages
   grep -rn "ConfigManager" src/
   # Update imports and usages
   ```

**Prevention**: Follow ONEX naming conventions. Avoid generic suffixes.

**Allowed Patterns**:
- `*Registry` (for dependency injection)
- `*Service` (for business logic)
- `*Adapter` (for external service integration)
- `*Factory` (for object creation)
- `*Builder` (for complex object construction)

---

### Issue: Union Usage Threshold Exceeded

**Symptom**:
```
Unions: FAIL
  - Found 495 unions (max: 491)
```

**Root Cause**: Too many Union types in codebase

**Context**: The current INFRA_MAX_UNIONS threshold is 491, set as a buffer above
the baseline (~485 unions as of 2025-12-21). Most unions are legitimate `X | None`
nullable patterns (ONEX-preferred PEP 604 syntax) which are counted but NOT flagged
as violations. The target is to reduce to <200 through `dict[str, object]` to `JsonValue` migration.

**Solution**:

1. **Identify union-heavy files**:
   ```bash
   # Search for Union usage (both old and new syntax)
   grep -rn "Union\[" src/ | wc -l
   grep -rn " | None" src/ | wc -l
   ```

2. **Refactor excessive unions**:
   ```python
   # Before: Excessive union
   HandlerType = Union[
       ConsulAdapter,
       KafkaAdapter,
       HandlerInfisical,
       PostgresAdapter,
       RedisAdapter,
       # ... 15 more
   ]

   # After: Use protocol or base class
   from typing import Protocol

   class ServiceHandler(Protocol):
       def handle(self, request: Request) -> Response:
           ...

   # Now use ServiceHandler instead of Union
   ```

3. **Migrate dict[str, object] to JsonValue** (preferred approach):
   ```python
   # Before: Contributes to union count
   metadata: dict[str, object]

   # After: Strongly typed, reduces union pressure
   from omnibase_infra.models.types import JsonValue
   metadata: JsonValue
   ```

4. **Adjust threshold only if absolutely necessary**:
   ```python
   # Current threshold is 491 (buffer above ~485 baseline as of 2025-12-21)
   # Only increase with documented justification and ticket reference
   INFRA_MAX_UNIONS = 491  # Document justification per OMN-983
   ```

**Prevention**: Prefer protocols/base classes over large unions. Migrate
`dict[str, object]` patterns to `JsonValue` for stronger typing.

---

### Issue: Circular Imports Detected

**Symptom**:
```
Imports: FAIL
  Circular import cycles detected:
    - omnibase_infra.models.consul_models
    - omnibase_infra.nodes.consul_adapter.models.model_consul_adapter_input
```

**Root Cause**: Modules importing each other in cycles

**Solution**:

**Method 1: Move shared code to common module**
```python
# Create common module
# src/omnibase_infra/models/common/model_consul_types.py
class ConsulServiceType(Enum):
    KV = "kv"
    AGENT = "agent"

# Import from common in both modules
from omnibase_infra.models.common.model_consul_types import ConsulServiceType
```

**Method 2: Use TYPE_CHECKING imports**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type checking, not at runtime
    from omnibase_infra.nodes.consul_adapter.models import ModelConsulAdapterInput

def process_consul_data(data: "ModelConsulAdapterInput") -> None:
    # Use string annotation to avoid runtime import
    ...
```

**Method 3: Restructure dependencies**
```python
# Before: A imports B, B imports A (cycle)

# After: Extract interface/protocol
# protocol_consul_handler.py
class ProtocolConsulHandler(Protocol):
    def handle(self, request: Request) -> Response:
        ...

# Both A and B import protocol (no cycle)
```

**Prevention**: Design module hierarchy to avoid bidirectional dependencies.

---

## Performance Issues

### Issue: Validation Takes Too Long (>5s)

**Symptom**:
```bash
uv run python scripts/validate.py all
# Takes 8+ seconds
```

**Root Cause**: Large codebase or inefficient validation

**Solution**:

1. **Use quick mode for local development**:
   ```bash
   uv run python scripts/validate.py all --quick
   # Skips medium priority validators (union, imports)
   ```

2. **Profile validators to identify bottleneck**:
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   from omnibase_infra.validation import validate_infra_all
   validate_infra_all()

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   ```

3. **Implement incremental validation**:
   ```bash
   # Only validate changed files
   git diff --name-only HEAD | grep '\.py$' > changed.txt
   # Validate only these files
   ```

4. **Enable caching** (if available):
   ```bash
   export ONEX_VALIDATION_CACHE=.onex_cache/validation
   uv run python scripts/validate.py all
   ```

**Prevention**: See [Performance Notes](performance_notes.md) for optimization strategies.

---

### Issue: CI/CD Validation Timeout

**Symptom**:
```yaml
# GitHub Actions
onex-validation:
  timeout-minutes: 10
  # Job times out after 10 minutes
```

**Root Cause**: Validation too slow for CI/CD environment

**Solution**:

1. **Enable dependency caching**:
   ```yaml
   - name: Load cached venv
     uses: actions/cache@v4
     with:
       path: .venv
       key: venv-${{ hashFiles('uv.lock') }}
   ```

2. **Use quick mode in CI**:
   ```yaml
   - name: Run ONEX validators
     run: uv run python scripts/validate.py all --quick
   ```

3. **Parallelize validation jobs**:
   ```yaml
   jobs:
     validate-architecture:
       run: uv run python scripts/validate.py architecture
     validate-contracts:
       run: uv run python scripts/validate.py contracts
   ```

4. **Increase timeout if necessary**:
   ```yaml
   onex-validation:
     timeout-minutes: 15  # Increase from 10
   ```

**Prevention**: Monitor CI/CD performance and optimize proactively.

---

## CI/CD Issues

### Issue: Validators Pass Locally But Fail in CI

**Symptom**:
```
Local: All validators PASS
CI:    Architecture FAIL
```

**Root Cause**: Environment differences (Python version, dependencies, file paths)

**Solution**:

1. **Verify Python version consistency**:
   ```yaml
   # .github/workflows/test.yml
   python-version: "3.12"  # Must match local
   ```

   ```bash
   # Local check
   python --version
   # Should be 3.12.x
   ```

2. **Check dependency versions**:
   ```bash
   # Local
   uv pip show omnibase-core
   # omnibase-core 0.3.5

   # CI logs should show same version
   ```

3. **Reproduce CI environment locally**:
   ```bash
   # Use same uv version
   uv --version

   # Fresh install
   rm -rf .venv
   uv sync
   uv run python scripts/validate.py all
   ```

4. **Check file paths**:
   ```bash
   # CI runs from repo root
   cd /path/to/repo
   uv run python scripts/validate.py all
   ```

**Prevention**: Use same Python/uv versions locally and in CI.

---

### Issue: ONEX Validation Job Not Running

**Symptom**: CI passes without running ONEX validators

**Root Cause**: Job not defined or dependency missing

**Solution**:

1. **Verify job exists in workflow**:
   ```yaml
   # .github/workflows/test.yml
   jobs:
     onex-validation:
       name: ONEX Validators
       # ...
   ```

2. **Check job dependencies**:
   ```yaml
   test-summary:
     needs: [smoke-test, test, lint, onex-validation]
     # Must include onex-validation
   ```

3. **Verify trigger conditions**:
   ```yaml
   on:
     pull_request:
       branches: [main, develop]
     push:
       branches: [main, develop]
   ```

**Prevention**: Review CI workflow after adding new jobs.

---

## Integration Issues

### Issue: Validators Not Available in Python Code

**Symptom**:
```python
from omnibase_infra.validation import validate_infra_all
ImportError: cannot import name 'validate_infra_all'
```

**Root Cause**: Function not exported in `__init__.py`

**Solution**:

1. **Check `__init__.py` exports**:
   ```python
   # src/omnibase_infra/validation/__init__.py
   from .infra_validators import (
       validate_infra_all,  # Must be listed
       validate_infra_architecture,
       # ...
   )

   __all__ = [
       "validate_infra_all",  # Must be in __all__
       # ...
   ]
   ```

2. **Verify function exists**:
   ```bash
   grep -n "def validate_infra_all" src/omnibase_infra/validation/infra_validators.py
   ```

3. **Reimport after changes**:
   ```python
   # In Python REPL
   import importlib
   import omnibase_infra.validation
   importlib.reload(omnibase_infra.validation)
   ```

**Prevention**: Always add new validators to `__all__`.

---

### Issue: Type Checking Failures with ValidationResult

**Symptom**:
```bash
uv run mypy src/
error: Name 'ValidationResult' is not defined
```

**Root Cause**: Type alias not exported or not recognized by mypy

**Solution**:

1. **Verify type alias export**:
   ```python
   # src/omnibase_infra/validation/infra_validators.py
   ValidationResult = ModelValidationResult[None]

   __all__ = [
       "ValidationResult",  # Must be exported
       # ...
   ]
   ```

2. **Use TYPE_CHECKING import if needed**:
   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from omnibase_infra.validation import ValidationResult

   def process_result(result: "ValidationResult") -> bool:
       return result.is_valid
   ```

3. **Check mypy configuration**:
   ```toml
   # pyproject.toml
   [tool.mypy]
   plugins = ["pydantic.mypy"]
   ignore_missing_imports = true  # May hide import issues
   ```

**Prevention**: Test type checking after adding new type aliases.

---

## Getting Help

If issues persist after trying these solutions:

1. **Enable verbose output**:
   ```bash
   uv run python scripts/validate.py all --verbose
   ```

2. **Check validator versions**:
   ```bash
   uv pip show omnibase-core omnibase-infra
   ```

3. **Review recent changes**:
   ```bash
   git log --oneline --all --graph -10
   ```

4. **File GitHub issue** with:
   - Full error output
   - Validator versions
   - Python version
   - Minimal reproduction steps

---

## Next Steps

- [Validator Reference](validator_reference.md) - Detailed validator documentation
- [Performance Notes](performance_notes.md) - Performance optimization
- [Framework Integration](framework_integration.md) - Integration patterns
