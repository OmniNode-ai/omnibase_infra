# Backwards Compatibility Validator

**Quick reference for using the backwards compatibility validation script.**

## Quick Start

```bash
# Validate staged files (pre-commit)
python scripts/validate_backwards_compatibility.py --staged

# Validate specific files
python scripts/validate_backwards_compatibility.py src/my_module.py

# Validate against specific baseline
python scripts/validate_backwards_compatibility.py --baseline v1.0.0 src/

# Generate configuration file
python scripts/validate_backwards_compatibility.py --generate-config

# Run test suite
python scripts/test_backwards_compatibility.py
```

## What It Detects

- ❌ Removed public functions, classes, methods
- ❌ Changed function signatures (parameters, return types)
- ❌ Removed Pydantic model fields
- ❌ Changed model field types or requirements
- ❌ Changed type annotations
- ✅ Safe: Adding optional parameters
- ✅ Safe: Adding optional model fields
- ✅ Safe: Changes to private APIs (starting with `_`)

## Configuration

Edit `config/backwards_compatibility_config.yaml`:

```yaml
baseline_version: "main"
strict_mode: false

ignore_patterns:
  - "tests/"
  - "examples/"

exemptions:
  "src/module.py:function_name": "Reason for breaking change"
```

## Pre-commit Integration

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

## Exit Codes

- `0`: No breaking changes (validation passed)
- `1`: Breaking changes detected (validation failed)
- `2`: Script error

## Common Workflows

### Handle Breaking Change

**Option 1: Add Exemption**
```yaml
# config/backwards_compatibility_config.yaml
exemptions:
  "src/api.py:create_user": "Added tenant_id for v2.0 multi-tenancy"
```

**Option 2: Use Deprecation**
```python
import warnings

def old_function():
    warnings.warn("Use new_function instead", DeprecationWarning)
    return new_function()
```

**Option 3: Skip Validation (Not Recommended)**
```bash
git commit --no-verify -m "Breaking change"
```

### Test Validator

```bash
# Run test suite
python scripts/test_backwards_compatibility.py

# Should output:
# ✅ Function Removal (Breaking) - PASS
# ✅ Function Signature Change (Breaking) - PASS
# ✅ Optional Parameter Addition (Safe) - PASS
# ...
```

## Files

- `scripts/validate_backwards_compatibility.py` - Main validator script
- `scripts/test_backwards_compatibility.py` - Test suite
- `config/backwards_compatibility_config.yaml` - Configuration
- `docs/BACKWARDS_COMPATIBILITY_GUIDE.md` - Full documentation
- `docs/BACKWARDS_COMPATIBILITY_INTEGRATION.md` - Integration guide

## Support

- 📚 Full docs: `docs/BACKWARDS_COMPATIBILITY_GUIDE.md`
- 🔧 Integration: `docs/BACKWARDS_COMPATIBILITY_INTEGRATION.md`
- 🧪 Tests: `python scripts/test_backwards_compatibility.py`

## Example Output

```
🔍 Validating backwards compatibility against baseline: main
   Checking 3 file(s)...

================================================================================
BACKWARDS COMPATIBILITY VALIDATION RESULTS
================================================================================

❌ ERRORS (2):

❌ [api_removed] src/utils.py
   Function 'process_data' was removed
   Line: 10
   Signature: process_data(data: dict) -> dict

❌ [api_signature_changed] src/users.py
   Function 'create_user' signature changed
   Line: 15
   Old: create_user(name: str, email: str) -> User
   New: create_user(name: str, email: str, tenant_id: str) -> User

================================================================================
Total: 2 errors, 0 warnings, 0 exempted

❌ VALIDATION FAILED: Breaking changes detected
```

## Best Practices

1. **Semantic Versioning**
   - Major (X.0.0): Breaking changes allowed
   - Minor (0.X.0): Only additions, no breaks
   - Patch (0.0.X): Bug fixes only

2. **Deprecation Strategy**
   - Don't remove immediately
   - Deprecate in version N
   - Remove in version N+1

3. **Document Changes**
   - CHANGELOG.md
   - Migration guides
   - API documentation

4. **Communication**
   - Announce breaking changes
   - Provide migration paths
   - Give users time to adapt

## For omnibase_core Integration

To use this in omnibase_core:

1. **Copy files:**
   ```bash
   cp scripts/validate_backwards_compatibility.py /path/to/omnibase_core/scripts/
   cp config/backwards_compatibility_config.yaml /path/to/omnibase_core/config/
   ```

2. **Update config:**
   ```yaml
   # Adjust for omnibase_core
   baseline_version: "main"
   ignore_patterns:
     - "tests/"
     - "examples/"
   ```

3. **Add to pre-commit:**
   ```yaml
   - id: backwards-compatibility
     name: Backwards compatibility validation
     entry: poetry run python scripts/validate_backwards_compatibility.py --staged
     language: system
     types: [python]
   ```

4. **Test:**
   ```bash
   pre-commit run backwards-compatibility --all-files
   ```

## Dependencies

- Python 3.12+
- pyyaml (for config parsing)
- git (for baseline comparison)
- Standard library: ast, subprocess, argparse

No additional Python dependencies required!
