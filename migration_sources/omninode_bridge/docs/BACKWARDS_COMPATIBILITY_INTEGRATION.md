# Backwards Compatibility Validator - Integration Guide

## Quick Start

This guide provides step-by-step instructions for integrating the backwards compatibility validator into the omnibase_core project.

## Prerequisites

- Python 3.12+
- Git repository
- Poetry for dependency management
- Pre-commit hooks installed

## Installation Steps

### Step 1: Copy Files to omnibase_core

Copy the following files from this repository to omnibase_core:

```bash
# From omninode_bridge repository root
cd /path/to/omnibase_core

# Copy the validator script
cp /path/to/omninode_bridge/scripts/validate_backwards_compatibility.py scripts/

# Copy the configuration
mkdir -p config
cp /path/to/omninode_bridge/config/backwards_compatibility_config.yaml config/

# Copy the documentation
cp /path/to/omninode_bridge/docs/BACKWARDS_COMPATIBILITY_GUIDE.md docs/
cp /path/to/omninode_bridge/docs/BACKWARDS_COMPATIBILITY_INTEGRATION.md docs/
```

### Step 2: Update Configuration

Edit `config/backwards_compatibility_config.yaml` for your project:

```yaml
# Set your baseline branch (usually main or master)
baseline_version: "main"

# Configure ignored patterns specific to your project
ignore_patterns:
  - "tests/"
  - "test_*.py"
  - "examples/"
  - "scripts/"
  - "docs/"

# Add project-specific exemptions as needed
exemptions: {}
```

### Step 3: Add to Pre-commit Hooks

Add the following to your `.pre-commit-config.yaml`:

```yaml
repos:
  # ... existing hooks ...

  # Backwards Compatibility Validation
  - repo: local
    hooks:
      - id: backwards-compatibility
        name: Backwards compatibility validation
        entry: poetry run python scripts/validate_backwards_compatibility.py --staged
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]
        description: "Validate backwards compatibility of public APIs"
```

**Full Integration Example:**

```yaml
repos:
  # Fast file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  # Local hooks
  - repo: local
    hooks:
      # Code formatting
      - id: black-format
        name: black (auto-format)
        entry: poetry run black
        language: system
        types: [python]
        stages: [commit]

      - id: isort-format
        name: isort (auto-sort imports)
        entry: poetry run isort
        language: system
        types: [python]
        stages: [commit]

      # Linting
      - id: ruff-fix
        name: ruff (auto-fix linting issues)
        entry: poetry run ruff check --fix
        language: system
        types: [python]
        stages: [commit]

      # Backwards Compatibility (NEW)
      - id: backwards-compatibility
        name: Backwards compatibility validation
        entry: poetry run python scripts/validate_backwards_compatibility.py --staged
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]
        description: "Validate backwards compatibility of public APIs"

  # Secrets detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

### Step 4: Install Pre-commit Hooks

```bash
# Install/update pre-commit hooks
pre-commit install

# Test the hooks
pre-commit run --all-files
```

### Step 5: Generate Initial Configuration (Optional)

```bash
# Generate default configuration if needed
poetry run python scripts/validate_backwards_compatibility.py --generate-config
```

## Verification

### Test the Validator

Create a test file to verify the validator works:

```bash
# Create a test file
cat > test_compatibility.py << 'EOF'
def public_function(param: str) -> str:
    """A public function."""
    return param

class PublicClass:
    """A public class."""
    def public_method(self, value: int) -> int:
        """A public method."""
        return value
EOF

# Commit the test file
git add test_compatibility.py
git commit -m "test: Add compatibility test file"

# Now modify it to introduce breaking changes
cat > test_compatibility.py << 'EOF'
# Remove public_function - this should be detected

class PublicClass:
    """A public class."""
    # Change signature - this should be detected
    def public_method(self, value: str) -> str:
        """A public method."""
        return value
EOF

# Try to commit - should fail with detected breaking changes
git add test_compatibility.py
git commit -m "test: Breaking changes"

# Clean up
git reset HEAD test_compatibility.py
rm test_compatibility.py
```

## Configuration Options

### Basic Configuration

```yaml
# config/backwards_compatibility_config.yaml

# Baseline version to compare against
baseline_version: "main"

# Fail validation even on exempted issues
strict_mode: false

# Feature toggles
check_functions: true
check_classes: true
check_models: true
check_type_annotations: true
```

### Advanced Configuration

```yaml
# Severity overrides
severity_overrides:
  API_REMOVED: "error"
  API_SIGNATURE_CHANGED: "error"
  MODEL_FIELD_TYPE_CHANGED: "error"
  TYPE_ANNOTATION_CHANGED: "warning"

# Version-specific baselines
version_baselines:
  v1.0:
    baseline_ref: "v1.0.0"
    strict_mode: true
  v2.0:
    baseline_ref: "v2.0.0"
    strict_mode: false
```

## Integration with CI/CD

### GitHub Actions

Create `.github/workflows/compatibility.yml`:

```yaml
name: Backwards Compatibility

on:
  pull_request:
    branches: [main, develop]
    paths:
      - 'src/**/*.py'
      - 'omnibase_core/**/*.py'

jobs:
  compatibility:
    name: Check Backwards Compatibility
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for git comparisons

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run compatibility check
        run: |
          poetry run python scripts/validate_backwards_compatibility.py \
            --baseline origin/${{ github.base_ref }} \
            --config config/backwards_compatibility_config.yaml

      - name: Upload report on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: compatibility-report
          path: compatibility_report.html
          retention-days: 30
```

### GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
backwards-compatibility:
  stage: test
  image: python:3.12
  before_script:
    - pip install poetry
    - poetry install
  script:
    - |
      poetry run python scripts/validate_backwards_compatibility.py \
        --baseline origin/main \
        --config config/backwards_compatibility_config.yaml
  only:
    - merge_requests
  allow_failure: false
  artifacts:
    when: on_failure
    paths:
      - compatibility_report.html
    expire_in: 30 days
```

## Usage Patterns

### Development Workflow

```bash
# 1. Make your changes
vim src/my_module.py

# 2. Stage changes
git add src/my_module.py

# 3. Commit (validator runs automatically)
git commit -m "feat: Add new feature"

# 4. If breaking changes detected:
#    - Add exemptions if intentional
#    - Or fix the breaking changes
#    - Or skip validation (not recommended)
git commit --no-verify -m "feat: Intentional breaking change"
```

### Handling Breaking Changes

#### Option 1: Add Exemption (Recommended)

Edit `config/backwards_compatibility_config.yaml`:

```yaml
exemptions:
  "src/models.py:UserModel.email": "Changed to EmailStr for validation (v2.0)"
  "src/api.py:create_user": "Added tenant_id for multi-tenancy (v2.0)"
```

Then commit normally.

#### Option 2: Use Deprecation First

```python
# Instead of removing immediately, deprecate first
import warnings

@deprecated("Use new_function instead", version="2.0")
def old_function(data: dict) -> dict:
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(data)
```

#### Option 3: Skip Validation (Use Sparingly)

```bash
# Skip pre-commit hooks entirely
git commit --no-verify -m "Breaking change"
```

## Maintenance

### Updating Exemptions

Periodically review and clean up exemptions:

```bash
# Review current exemptions
cat config/backwards_compatibility_config.yaml | grep -A 20 "exemptions:"

# Remove obsolete exemptions after major version release
# Edit config/backwards_compatibility_config.yaml
```

### Updating Baseline

After major releases, update the baseline:

```yaml
# For v2.0 release, update to use v1.0 as baseline
baseline_version: "v1.0.0"

# Or use branch-specific baselines
version_baselines:
  v2.0:
    baseline_ref: "v1.0.0"
    strict_mode: false
```

## Troubleshooting

### Problem: Validator Not Running

**Solution:**
```bash
# Reinstall pre-commit hooks
pre-commit uninstall
pre-commit install

# Verify hook is registered
cat .git/hooks/pre-commit
```

### Problem: False Positives

**Solution:**
```yaml
# Add to ignore_patterns in config
ignore_patterns:
  - "tests/"
  - "internal/"
  - "_private_module.py"
```

### Problem: Git Baseline Not Found

**Solution:**
```bash
# Ensure you have the baseline ref
git fetch origin main

# Or specify a different baseline
python scripts/validate_backwards_compatibility.py --baseline origin/develop
```

### Problem: Too Slow

**Solution:**
```yaml
# Limit to source files only
ignore_patterns:
  - "tests/"
  - "examples/"
  - "docs/"
  - "scripts/"
  - "migrations/"
```

## Best Practices

### 1. Run Locally Before Pushing

```bash
# Run validator on all changes before pushing
poetry run python scripts/validate_backwards_compatibility.py src/
```

### 2. Document Breaking Changes

Always document in:
- CHANGELOG.md
- Migration guides
- API documentation

### 3. Use Semantic Versioning

- **Major (X.0.0)**: Breaking changes allowed
- **Minor (0.X.0)**: Only additions, no breaks
- **Patch (0.0.X)**: Bug fixes only

### 4. Gradual Migration

For large refactoring:
1. Add new API
2. Deprecate old API
3. Give users time to migrate
4. Remove old API in next major version

### 5. Communication

Communicate breaking changes:
- Release notes
- Migration guides
- Deprecation warnings
- Team announcements

## Support

### Getting Help

1. Check documentation: `docs/BACKWARDS_COMPATIBILITY_GUIDE.md`
2. Review examples in this guide
3. Check GitHub issues
4. Contact team lead

### Reporting Issues

When reporting validator issues, include:
- Command run
- Full error output
- Sample code that triggered issue
- Configuration file

### Contributing

To improve the validator:
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## Next Steps

After integration:

1. ✅ Run initial validation on entire codebase
2. ✅ Add any necessary exemptions
3. ✅ Document in team wiki/docs
4. ✅ Add to CI/CD pipeline
5. ✅ Train team on usage
6. ✅ Set up monitoring/alerts

## Additional Resources

- [Backwards Compatibility Guide](./BACKWARDS_COMPATIBILITY_GUIDE.md)
- [Python AST Documentation](https://docs.python.org/3/library/ast.html)
- [Semantic Versioning](https://semver.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
