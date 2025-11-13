# Post-Generation File Validation

**Purpose**: Validate actual generated files on disk to ensure truth-based reporting

**Status**: ✅ Complete (November 2025)

## Problem Statement

The code generation pipeline previously validated in-memory code before files were written to disk. This caused a critical issue where:

1. In-memory validation passed (reporting "✅ Syntax valid: True")
2. Files were written to disk
3. Actual files on disk had `IndentationError` and stub implementations
4. Success reports claimed completion, but files were broken

**Root Cause**: Validation happened on intermediate data structures, not actual file contents.

## Solution: Post-Generation File Validation

The `FileValidator` validates actual files **after they're written to disk**, ensuring reports reflect true file state.

### Key Features

1. **Syntax Validation** - AST parsing on actual file contents
2. **Stub Detection** - Detects placeholder code that should be implemented:
   - `# IMPLEMENTATION REQUIRED` markers
   - `# TODO` comments
   - Methods with only `pass` statement
   - `raise NotImplementedError`
3. **File Existence** - Verifies all expected files exist
4. **Multiple Files** - Validates entire directory trees recursively

### Architecture

```
Code Generation Pipeline
    ↓
Generate in-memory artifacts
    ↓
Write files to disk
    ↓
[NEW] FileValidator validates actual disk files
    ↓
Report success only if validation passes
```

## Usage

### Standalone Script

```bash
# Validate specific files
poetry run python scripts/validate_generated_files.py generated_vault_secrets_effect.py

# Validate directory (recursive)
poetry run python scripts/validate_generated_files.py generated_nodes/vault_secrets_effect/

# Strict mode (fail on warnings)
poetry run python scripts/validate_generated_files.py --strict generated_nodes/

# Verbose output
poetry run python scripts/validate_generated_files.py --verbose generated_nodes/
```

### Integration in Generation Scripts

```python
from omninode_bridge.codegen.file_validator import FileValidator

# After generating and writing files
all_files = result.artifacts.get_all_files()
for filename, content in all_files.items():
    file_path = output_dir / filename
    file_path.write_text(content)

# Validate actual files on disk
validator = FileValidator()
written_files = [output_dir / filename for filename in all_files.keys()]
validation_result = await validator.validate_generated_files(
    file_paths=written_files,
    strict_mode=True,
)

# Check validation result
if not validation_result.passed:
    print(f"⚠️ Validation failed: {len(validation_result.issues)} issues found")
    print(validator.format_validation_report(validation_result))
    sys.exit(1)
```

### Programmatic Usage

```python
from pathlib import Path
from omninode_bridge.codegen.file_validator import FileValidator

# Initialize validator
validator = FileValidator()

# Validate files
result = await validator.validate_generated_files(
    file_paths=[
        Path("generated_nodes/vault_secrets_effect/node.py"),
        Path("generated_nodes/vault_secrets_effect/__init__.py"),
    ],
    strict_mode=True,
)

# Check results
if result.passed:
    print(f"✅ All {result.files_validated} files valid")
else:
    print(f"❌ {result.files_failed}/{result.files_validated} files failed")
    for issue in result.issues:
        print(f"  {issue.file_path}:{issue.line_number} - {issue.message}")
```

## Validation Checks

### 1. Syntax Validation (AST Parsing)

Catches Python syntax errors:
- `SyntaxError`
- `IndentationError`
- `TabError`

**Example Detection**:
```
🔴 generated_vault_secrets_effect.py:28
   IndentationError: expected an indented block after 'try' statement on line 26
```

### 2. Stub Pattern Detection

Detects placeholder code:

#### IMPLEMENTATION REQUIRED Markers
```python
def execute_effect(self, contract):
    # IMPLEMENTATION REQUIRED
    pass
```
**Detection**: `🔴 Line 96: IMPLEMENTATION REQUIRED marker found (stub not replaced)`

#### TODO Comments
```python
def incomplete_method(self):
    # TODO: Implement this
    pass
```
**Detection**: `⚠️ Line 42: TODO comment found (stub not replaced)`

#### Bare Pass Methods
```python
def stub_method(self):
    """Method with only pass."""
    pass
```
**Detection**: `🔴 Method 'stub_method' contains only 'pass' statement (likely stub)`

#### NotImplementedError
```python
def not_ready(self):
    raise NotImplementedError("Not implemented yet")
```
**Detection**: `🔴 Line 15: NotImplementedError found (stub not replaced)`

### 3. File Existence Checks

Verifies expected files exist:
```
🔴 generated_nodes/vault_secrets_effect/node.py
   File does not exist: generated_nodes/vault_secrets_effect/node.py
```

## Validation Report Format

```
================================================================================
POST-GENERATION FILE VALIDATION REPORT
================================================================================

❌ Overall Status: FAILED

Summary:
  Files validated: 6
  Files passed: 4
  Files failed: 2
  Total issues: 3
  Execution time: 2ms

Issues by Type:
  Syntax errors: 0
  Stub issues: 3
  Missing files: 0

Detailed Issues:
  🔴 generated_nodes/vault_secrets_effect_final/node.py:96
     IMPLEMENTATION REQUIRED marker found (stub not replaced)

  🔴 generated_nodes/vault_secrets_effect_final/node.py:83
     Method 'execute_effect' contains only 'pass' statement (likely stub)

  🔴 generated_nodes/vault_secrets_effect_final/tests/test_integration.py:12
     Method 'test_end_to_end_workflow' contains only 'pass' statement (likely stub)

================================================================================
```

## Integration Points

### 1. Generation Scripts

Updated scripts:
- `scripts/regenerate_vault_secrets_effect.py` - Validates files after generation
- `scripts/regenerate_vault_secrets.py` - Same pattern
- All future generation scripts should follow this pattern

### 2. CodeGenerationService

Future enhancement: Add optional file validation to `CodeGenerationService.generate_node()`:

```python
result = await service.generate_node(
    requirements=requirements,
    classification=classification,
    output_directory=output_dir,
    validate_files_on_disk=True,  # Enable post-write validation
)
```

### 3. Quality Gates Pipeline

The `FileValidator` complements the existing `QualityGatePipeline`:
- **QualityGatePipeline**: Validates in-memory code during generation
- **FileValidator**: Validates actual files after writing

Both are important and serve different purposes.

## Test Coverage

Comprehensive test suite: `tests/unit/codegen/test_file_validator.py`

Tests cover:
- ✅ Valid files pass validation
- ✅ Syntax errors detected (SyntaxError, IndentationError, TabError)
- ✅ IMPLEMENTATION REQUIRED markers detected
- ✅ TODO comments detected
- ✅ Bare pass methods detected
- ✅ NotImplementedError detected
- ✅ Missing files detected
- ✅ Multiple file validation
- ✅ Validation report formatting
- ✅ Strict mode behavior

**Result**: 10/10 tests passing

## Performance

- **Single file**: ~0-2ms per file
- **Directory (6 files)**: ~2ms total
- **Large directory (50+ files)**: ~10-20ms total

Very fast validation with minimal overhead.

## Best Practices

### 1. Always Validate After Writing

```python
# ❌ BAD: Report success before validating files
for filename, content in all_files.items():
    (output_dir / filename).write_text(content)
print("✅ Generation complete!")

# ✅ GOOD: Validate files before reporting success
for filename, content in all_files.items():
    (output_dir / filename).write_text(content)

validator = FileValidator()
result = await validator.validate_generated_files([...])
if not result.passed:
    print("❌ Validation failed!")
    sys.exit(1)
print("✅ Generation complete!")
```

### 2. Use Strict Mode in CI/CD

```bash
# In CI/CD pipelines, always use strict mode
poetry run python scripts/validate_generated_files.py --strict generated_nodes/
```

### 3. Include Validation in Success Reports

```python
# Generate comprehensive report
validation_report = validator.format_validation_report(result)
print(validation_report)

# Or include in SUCCESS_REPORT.md
with open("SUCCESS_REPORT.md", "w") as f:
    f.write("## File Validation\n\n")
    f.write(validation_report)
```

### 4. Fail Fast on Validation Errors

```python
if not validation_result.passed:
    print(f"⚠️ Validation failed: {len(validation_result.issues)} issues")
    return 1  # Exit with error code
```

## Future Enhancements

### 1. Import Validation

Detect unused or missing imports:
```python
# Detect
import unused_module  # Never used
from missing_module import Foo  # Module doesn't exist
```

### 2. Type Hint Validation

Verify type hints are valid:
```python
# Detect
def method(x: NonExistentType) -> InvalidReturn:
    pass
```

### 3. ONEX Compliance Validation

Check ONEX patterns in generated files:
```python
# Verify
- ModelContract* usage
- Error handling patterns (ModelOnexError)
- Logging patterns (emit_log_event)
- Metrics patterns
```

### 4. Integration with QualityGatePipeline

Add file path support to `QualityGatePipeline`:
```python
pipeline = QualityGatePipeline(validation_level="strict")
result = await pipeline.validate_file(file_path=Path("node.py"))
```

## Related Documentation

- **[Quality Gates](./QUALITY_GATES.md)** - In-memory code validation
- **[Code Generation Service](./SERVICE.md)** - Generation pipeline architecture
- **[Test Suite Summary](../../tests/unit/codegen/TEST_SUITE_SUMMARY.md)** - Test coverage details

## Implementation Details

**Files**:
- `src/omninode_bridge/codegen/file_validator.py` - Core validator
- `scripts/validate_generated_files.py` - Standalone script
- `tests/unit/codegen/test_file_validator.py` - Test suite

**Classes**:
- `FileValidator` - Main validator class
- `FileValidationResult` - Result model with detailed reporting
- `FileValidationIssue` - Individual issue model

**Key Methods**:
- `validate_generated_files(file_paths, strict_mode)` - Main validation entry point
- `format_validation_report(result, include_file_paths)` - Report formatting
- `_validate_single_file(file_path, strict_mode)` - Single file validation
- `_validate_syntax(file_path, code)` - AST syntax validation
- `_detect_stubs(file_path, code)` - Stub pattern detection

---

**Last Updated**: November 2, 2025
**Status**: Production Ready ✅
