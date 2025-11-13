# Backwards Compatibility Validation Guide

## Overview

The backwards compatibility validation script ensures that code changes maintain API compatibility with previous versions, preventing accidental breaking changes in your public APIs.

## Features

### Detection Capabilities

The validator detects the following breaking changes:

1. **API Removals**
   - Removed public functions
   - Removed public classes
   - Removed public methods
   - Removed public constants

2. **Signature Changes**
   - Changed function parameters
   - Changed return types
   - Changed type annotations
   - Added required parameters

3. **Pydantic Model Changes**
   - Removed model fields
   - Changed field types
   - Changed field from optional to required
   - Changed field validators

4. **Class Changes**
   - Removed classes
   - Changed class inheritance
   - Removed public methods

## Installation

The script is already integrated into the project. No additional installation required.

## Usage

### Command Line

```bash
# Validate specific files
python scripts/validate_backwards_compatibility.py src/module.py

# Validate all staged files (for pre-commit)
python scripts/validate_backwards_compatibility.py --staged

# Validate against specific baseline
python scripts/validate_backwards_compatibility.py --baseline v1.0.0 src/module.py

# Strict mode (fail even on exempted issues)
python scripts/validate_backwards_compatibility.py --strict --staged

# Generate default configuration
python scripts/validate_backwards_compatibility.py --generate-config
```

### Pre-commit Hook

The validator is integrated into pre-commit hooks and runs automatically on commit:

```bash
# Normal commit - runs validation automatically
git commit -m "Your commit message"

# Skip validation if needed (use sparingly!)
git commit --no-verify -m "Skip validation"
```

## Configuration

Configuration is stored in `config/backwards_compatibility_config.yaml`.

### Key Configuration Options

```yaml
# Git reference for baseline comparison
baseline_version: "main"

# Strict mode: fail on exempted issues
strict_mode: false

# Feature flags
check_functions: true
check_classes: true
check_models: true
check_type_annotations: true

# Ignore patterns
ignore_patterns:
  - "tests/"
  - "examples/"
  - "scripts/"

# Exemptions for intentional breaking changes
exemptions:
  "src/my_module.py:MyClass.my_method": "Intentional breaking change for v2.0"
```

### Adding Exemptions

When you intentionally make a breaking change:

1. Run the validator to see the detected issues
2. Add exemptions to the config file:

```yaml
exemptions:
  "src/models.py:UserModel.email": "Changed email field to EmailStr for validation"
  "src/api.py:create_user": "Added required 'tenant_id' parameter for multi-tenancy"
```

3. Document the breaking change in CHANGELOG.md
4. Consider if this requires a major version bump (semantic versioning)

## Examples

### Example 1: Detected Function Removal

**Before (baseline):**
```python
def process_data(data: dict) -> dict:
    """Process data."""
    return data
```

**After (current):**
```python
# Function removed
```

**Validation Output:**
```
❌ [api_removed] src/utils.py
   Function 'process_data' was removed
   Line: 10
   Signature: process_data(data: dict) -> dict
```

### Example 2: Detected Signature Change

**Before (baseline):**
```python
def create_user(name: str, email: str) -> User:
    """Create a user."""
    pass
```

**After (current):**
```python
def create_user(name: str, email: str, tenant_id: str) -> User:
    """Create a user."""
    pass
```

**Validation Output:**
```
❌ [api_signature_changed] src/users.py
   Function 'create_user' signature changed
   Line: 15
   Old: create_user(name: str, email: str) -> User
   New: create_user(name: str, email: str, tenant_id: str) -> User
```

### Example 3: Detected Pydantic Model Change

**Before (baseline):**
```python
class UserModel(BaseModel):
    name: str
    email: Optional[str] = None
    age: int
```

**After (current):**
```python
class UserModel(BaseModel):
    name: str
    email: EmailStr  # Changed from Optional[str]
    # age field removed
```

**Validation Output:**
```
❌ [model_field_removed] src/models.py
   Field 'age' removed from model 'UserModel'
   Line: 20

❌ [model_field_type_changed] src/models.py
   Field 'email' type changed in model 'UserModel'
   Line: 20
   Old: Optional[str]
   New: EmailStr

❌ [model_field_required_changed] src/models.py
   Field 'email' changed from optional to required in model 'UserModel'
   Line: 20
```

## Best Practices

### 1. Use Semantic Versioning

- **Major version bump (X.0.0)**: Breaking changes allowed
- **Minor version bump (0.X.0)**: No breaking changes, only additions
- **Patch version bump (0.0.X)**: No breaking changes, only fixes

### 2. Deprecation Strategy

Instead of removing APIs immediately:

```python
import warnings

def old_function(data: dict) -> dict:
    """
    Old function - DEPRECATED.

    .. deprecated:: 2.0.0
        Use :func:`new_function` instead.
    """
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(data)

def new_function(data: dict) -> dict:
    """New function with improved implementation."""
    pass
```

### 3. Version-Based Configuration

For projects with multiple active versions:

```yaml
version_baselines:
  v1.0:
    baseline_ref: "v1.0.0"
    strict_mode: true
  v2.0:
    baseline_ref: "v2.0.0"
    strict_mode: false
```

### 4. Document Breaking Changes

Always document breaking changes in:
- CHANGELOG.md
- Migration guides
- API documentation
- Commit messages

Example commit message:
```
feat!: Add required tenant_id parameter to create_user

BREAKING CHANGE: create_user() now requires tenant_id parameter
for multi-tenancy support. Update all calls to include tenant_id.

Before: create_user(name="John", email="john@example.com")
After: create_user(name="John", email="john@example.com", tenant_id="tenant1")
```

### 5. Safe Refactoring Patterns

#### Adding Optional Parameters
```python
# Safe - adds optional parameter
def create_user(name: str, email: str, tenant_id: Optional[str] = None) -> User:
    pass
```

#### Using *args and **kwargs
```python
# Safe - allows additional parameters
def process_data(data: dict, **kwargs) -> dict:
    pass
```

#### Using Protocols for Type Changes
```python
from typing import Protocol

class DataLike(Protocol):
    """Protocol for data-like objects."""
    def to_dict(self) -> dict: ...

# Safe - accepts any object implementing the protocol
def process_data(data: DataLike) -> dict:
    return data.to_dict()
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Backwards Compatibility Check

on:
  pull_request:
    branches: [main, develop]

jobs:
  compatibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for comparison

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run backwards compatibility check
        run: |
          poetry run python scripts/validate_backwards_compatibility.py \
            --baseline origin/main \
            --config config/backwards_compatibility_config.yaml

      - name: Upload compatibility report
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: compatibility-report
          path: compatibility_report.html
```

### Pre-commit Hook Example

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: backwards-compatibility
        name: Backwards compatibility validation
        entry: poetry run python scripts/validate_backwards_compatibility.py --staged
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]
```

## Troubleshooting

### Issue: "No baseline content found"

**Cause:** File doesn't exist in baseline branch

**Solution:** This is expected for new files. No action needed.

### Issue: "Syntax error in file"

**Cause:** File has Python syntax errors

**Solution:** Fix syntax errors before committing.

### Issue: "Too many false positives"

**Cause:** Validation is too strict for your use case

**Solution:**
1. Add exemptions for specific cases
2. Adjust `severity_threshold` in config
3. Use ignore patterns for test/example code

### Issue: "Validation is too slow"

**Cause:** Large number of files or complex AST parsing

**Solution:**
1. Use `--staged` to validate only changed files
2. Add more patterns to `ignore_patterns`
3. Run full validation only in CI, not pre-commit

## Advanced Usage

### Custom Baseline Comparisons

```bash
# Compare against specific commit
python scripts/validate_backwards_compatibility.py --baseline abc123 src/

# Compare against tag
python scripts/validate_backwards_compatibility.py --baseline v1.0.0 src/

# Compare against branch
python scripts/validate_backwards_compatibility.py --baseline develop src/
```

### Programmatic Usage

```python
from pathlib import Path
from scripts.validate_backwards_compatibility import CompatibilityValidator

# Initialize validator
validator = CompatibilityValidator(
    config_path=Path("config/backwards_compatibility_config.yaml"),
    baseline_version="main",
    strict_mode=False
)

# Validate files
files = ["src/models.py", "src/api.py"]
is_valid = validator.validate_files(files)

# Access detected issues
for issue in validator.issues:
    print(f"{issue.severity}: {issue.description}")
```

## Contributing

To improve the backwards compatibility validator:

1. Add new detection patterns to `ASTAnalyzer`
2. Add new change types to `ChangeType` enum
3. Update configuration schema for new features
4. Add tests for new detection capabilities

## References

- [Semantic Versioning](https://semver.org/)
- [Python AST Documentation](https://docs.python.org/3/library/ast.html)
- [Pydantic Models](https://docs.pydantic.dev/)
- [Python Deprecation](https://docs.python.org/3/library/warnings.html)
