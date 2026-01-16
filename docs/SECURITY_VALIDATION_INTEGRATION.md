# Security Validation Integration Report (OMN-1091)

**Task**: Integrate `ModelHandlerValidationError` in security validation path
**Status**: ✅ COMPLETE - Foundation Established
**Date**: 2025-12-29

## Executive Summary

This task establishes the foundation for structured security validation in ONEX infrastructure. As part of OMN-1091 (Structured Validation & Error Reporting for Handlers), we have created the security validation infrastructure that integrates `ModelHandlerValidationError` with security constraint enforcement.

## Key Findings

### 1. No Existing Security Validation

**Discovery**: The ONEX codebase has **extensive security documentation** but **no active security validation code**.

- ✅ **Security Documentation Exists**:
  - `docs/patterns/security_patterns.md` - Comprehensive security guide
  - `MixinNodeIntrospection` - Extensive security considerations in docstrings
  - `util_error_sanitization.py` - Error sanitization utilities

- ❌ **Security Validation Missing**:
  - No static analysis for security violations
  - No validation of method exposure via introspection
  - No enforcement of security best practices
  - No structured error reporting for security issues

**Implication**: This task creates the **foundation** for future security validation rather than integrating with existing code.

### 2. Structured Error Model Ready

The infrastructure for structured error reporting exists and is battle-tested:

- ✅ `ModelHandlerValidationError` - Fully implemented with factory methods
- ✅ `EnumHandlerErrorType.SECURITY_VALIDATION_ERROR` - Error type defined
- ✅ `EnumHandlerSourceType.STATIC_ANALYSIS` - Source type for security checks
- ✅ Multiple output formats - Logging, CI, JSON
- ✅ Existing usage pattern - `contract_linter.py` shows integration pattern

### 3. Security Concerns Identified

From analyzing `MixinNodeIntrospection` and security documentation, key security validation needs:

| Security Concern | Current State | Validation Needed |
|-----------------|---------------|-------------------|
| Sensitive method exposure | Filtered but not validated | Detect `get_password`, `decrypt_*`, `admin_*` |
| Credential leakage in signatures | Not checked | Detect parameter names like `password`, `api_key` |
| Admin method exposure | Not validated | Enforce `_` prefix or separate admin modules |
| Security patterns | Documented but not enforced | Static analysis for insecure patterns |

## Implementation Deliverables

### 1. Security Validator Module

**File**: `src/omnibase_infra/validation/validator_security.py`

**Purpose**: Provides security validation functions that emit `ModelHandlerValidationError` for security violations.

**Features**:
- ✅ Sensitive method name detection
- ✅ Credential detection in method signatures
- ✅ Admin/internal method exposure detection
- ✅ Structured error generation with rule IDs
- ✅ Actionable remediation hints

**Rule IDs**:
```python
class SecurityRuleId:
    # Method exposure violations (SECURITY-001 to SECURITY-099)
    SENSITIVE_METHOD_EXPOSED = "SECURITY-001"
    CREDENTIAL_IN_SIGNATURE = "SECURITY-002"
    ADMIN_METHOD_PUBLIC = "SECURITY-003"
    DECRYPT_METHOD_PUBLIC = "SECURITY-004"

    # Configuration violations (SECURITY-100 to SECURITY-199)
    CREDENTIAL_IN_CONFIG = "SECURITY-100"
    HARDCODED_SECRET = "SECURITY-101"
    INSECURE_CONNECTION = "SECURITY-102"

    # Pattern violations (SECURITY-200 to SECURITY-299)
    INSECURE_PATTERN = "SECURITY-200"
    MISSING_AUTH_CHECK = "SECURITY-201"
    MISSING_INPUT_VALIDATION = "SECURITY-202"
```

**Key Functions**:
```python
# Validate method names and signatures
errors = validate_method_exposure(
    method_names=["get_api_key", "process_request"],
    handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
    method_signatures={"get_api_key": "() -> str"},
)

# Validate handler capabilities (from MixinNodeIntrospection)
errors = validate_handler_security(
    handler_identity=handler_identity,
    capabilities=capabilities_dict,
    file_path="nodes/auth/node.py",
)
```

### 2. Validation Module Integration

**File**: `src/omnibase_infra/validation/__init__.py`

**Changes**:
- ✅ Imported `security_validator` functions
- ✅ Exported security validation API
- ✅ Added to `__all__` for public API

**Public API**:
```python
from omnibase_infra.validation import (
    SecurityRuleId,
    validate_handler_security,
    validate_method_exposure,
    is_sensitive_method_name,
    has_sensitive_parameters,
)
```

### 3. Documentation Updates

**File**: `docs/patterns/security_patterns.md`

**Added Section**: "Structured Security Error Reporting"

**Content**:
- Integration patterns for `ModelHandlerValidationError`
- Example code for security validation
- Error output formats (logging, CI, JSON)
- CI integration examples
- Rule ID reference table

**Location**: Lines 1257-1447 in `security_patterns.md`

### 4. Test Suite

**File**: `tests/unit/validation/test_validator_security.py`

**Coverage**:
- ✅ 23 tests, all passing
- ✅ Sensitive method name detection
- ✅ Credential detection in signatures
- ✅ Admin method exposure
- ✅ Error formatting (logging, CI, JSON)
- ✅ Rule ID validation

**Test Results**:
```
23 passed in 1.82s
```

## Integration Points

### 1. MixinNodeIntrospection Integration (Future)

The `MixinNodeIntrospection` mixin provides the data needed for security validation:

```python
# Current: Filtering (no validation)
class MixinNodeIntrospection:
    def _should_skip_method(self, method_name: str) -> bool:
        """Skip private and utility methods."""
        return any(
            method_name.startswith(prefix)
            for prefix in self._introspection_exclude_prefixes
        )

# Future: Add validation
async def validate_security(self) -> list[ModelHandlerValidationError]:
    """Validate security of exposed capabilities."""
    capabilities = await self.get_capabilities()

    from omnibase_infra.validation import validate_handler_security

    handler_identity = ModelHandlerIdentifier.from_handler_id(
        str(self._introspection_node_id)
    )

    return validate_handler_security(
        handler_identity=handler_identity,
        capabilities=capabilities,
        file_path=inspect.getfile(type(self)),
    )
```

### 2. Contract Linting Integration (Future)

The contract linter could add security validation:

```python
# In contract_linter.py
from omnibase_infra.validation.security_validator import validate_handler_security

def lint_contract_with_security(
    contract_path: Path
) -> list[ModelHandlerValidationError]:
    """Lint contract with security validation."""
    # Load contract and discover handlers
    handlers = discover_handlers_from_contract(contract_path)

    errors = []
    for handler in handlers:
        # Extract capabilities
        capabilities = extract_handler_capabilities(handler)

        # Validate security
        security_errors = validate_handler_security(
            handler_identity=handler.identity,
            capabilities=capabilities,
            file_path=str(contract_path),
        )

        errors.extend(security_errors)

    return errors
```

### 3. CI Pipeline Integration (Future)

Security validation can be added to CI pipelines:

```python
#!/usr/bin/env python
"""CI script for security validation."""

from pathlib import Path
import sys

from omnibase_infra.validation import validate_handler_security
from omnibase_infra.models.handlers import ModelHandlerIdentifier


def main() -> int:
    """Run security validation on all handlers."""
    errors = []

    # Discover all handler files
    handler_files = Path("src/handlers").glob("**/node.py")

    for handler_file in handler_files:
        # Extract capabilities (implementation depends on your setup)
        capabilities = extract_capabilities_from_file(handler_file)

        handler_identity = ModelHandlerIdentifier.from_node(
            node_path=str(handler_file),
            handler_type=infer_handler_type(handler_file),
        )

        # Validate security
        handler_errors = validate_handler_security(
            handler_identity=handler_identity,
            capabilities=capabilities,
            file_path=str(handler_file),
        )

        errors.extend(handler_errors)

    # Report results
    if not errors:
        print("✓ Security validation passed")
        return 0

    print(f"✗ Found {len(errors)} security violations:\n")
    for error in errors:
        print(error.format_for_ci())

    # Fail CI if blocking errors found
    blocking_errors = [e for e in errors if e.is_blocking()]
    return 1 if blocking_errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

## Usage Examples

### Example 1: Validate Method Exposure

```python
from omnibase_infra.validation import validate_method_exposure
from omnibase_infra.models.handlers import ModelHandlerIdentifier

# Handler identity
handler_identity = ModelHandlerIdentifier.from_handler_id("auth-handler")

# Validate exposed methods
errors = validate_method_exposure(
    method_names=["get_api_key", "process_request"],
    handler_identity=handler_identity,
    method_signatures={
        "get_api_key": "() -> str",
        "process_request": "(data: dict) -> Result",
    },
    file_path="nodes/auth/handlers/handler_authenticate.py",
)

# Check for violations
if errors:
    for error in errors:
        print(error.format_for_logging())
```

**Output**:
```
Handler Validation Error [SECURITY-001]
Type: security_validation_error
Source: static_analysis
Handler: auth-handler
File: nodes/auth/handlers/handler_authenticate.py
Message: Handler exposes sensitive method 'get_api_key'
Remediation: Prefix method with underscore: '_get_api_key' to exclude from introspection
```

### Example 2: Validate Handler Security

```python
from omnibase_infra.validation import validate_handler_security
from omnibase_infra.models.handlers import ModelHandlerIdentifier

# Handler capabilities (from MixinNodeIntrospection.get_capabilities())
capabilities = {
    "operations": ["get_password", "process_data"],
    "protocols": ["ProtocolDatabaseAdapter"],
    "has_fsm": False,
    "method_signatures": {
        "get_password": "() -> str",
        "process_data": "(data: dict) -> Result",
    },
}

handler_identity = ModelHandlerIdentifier.from_handler_id("test-handler")

# Validate security
errors = validate_handler_security(
    handler_identity=handler_identity,
    capabilities=capabilities,
)

# Output for CI
for error in errors:
    print(error.format_for_ci())
```

**Output**:
```
::error file=unknown,line=1::[SECURITY-001] Handler exposes sensitive method 'get_password'. Remediation: Prefix method with underscore: '_get_password' to exclude from introspection
```

### Example 3: Integration with Introspection

```python
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.validation import validate_handler_security
from omnibase_infra.models.handlers import ModelHandlerIdentifier


class SecureNode(MixinNodeIntrospection):
    """Node with automatic security validation."""

    async def validate_security(self) -> list[ModelHandlerValidationError]:
        """Run security validation on this node."""
        # Get capabilities via introspection
        capabilities = await self.get_capabilities()

        handler_identity = ModelHandlerIdentifier.from_handler_id(
            str(self._introspection_node_id)
        )

        # Validate security
        return validate_handler_security(
            handler_identity=handler_identity,
            capabilities=capabilities,
            file_path=__file__,
        )

    async def startup(self) -> None:
        """Startup with security validation."""
        # Run security validation
        errors = await self.validate_security()

        if errors:
            # Log security violations
            for error in errors:
                logger.warning(error.format_for_logging())

            # Optionally fail startup on blocking errors
            blocking_errors = [e for e in errors if e.is_blocking()]
            if blocking_errors:
                raise RuntimeError(
                    f"Security validation failed: {len(blocking_errors)} blocking errors"
                )

        # Continue with normal startup
        await self.publish_introspection(reason="startup")
        await self.start_introspection_tasks()
```

## Security Rule Reference

### Method Exposure Rules

| Rule ID | Trigger Pattern | Severity | Remediation |
|---------|----------------|----------|-------------|
| SECURITY-001 | Method name matches sensitive patterns | ERROR | Prefix with underscore |
| SECURITY-002 | Parameter name contains credentials | ERROR | Use generic names |
| SECURITY-003 | Method starts with `admin_` or `internal_` | ERROR | Make private or move to admin module |
| SECURITY-004 | Method starts with `decrypt_` | ERROR | Make private or move to crypto module |

### Sensitive Method Patterns

```python
SENSITIVE_METHOD_PATTERNS = (
    r"^get_password$",
    r"^get_secret$",
    r"^get_token$",
    r"^get_api_key$",
    r"^get_credential",
    r"^fetch_password$",
    r"^fetch_secret$",
    r"^fetch_token$",
    r"^decrypt_",
    r"^admin_",
    r"^internal_",
    r"^validate_password$",
    r"^check_password$",
    r"^verify_password$",
)
```

### Sensitive Parameter Names

```python
SENSITIVE_PARAMETER_NAMES = frozenset({
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "auth_token",
    "bearer_token",
    "decrypt_key",
    "encryption_key",
})
```

## Next Steps

### Immediate (This Sprint)

1. ✅ Create security validator module
2. ✅ Add tests for security validation
3. ✅ Update documentation
4. ✅ Export from validation module

### Short Term (Next Sprint)

1. ⬜ Integrate with `MixinNodeIntrospection.validate_security()`
2. ⬜ Add security validation to contract linter
3. ⬜ Create CI script for security validation
4. ⬜ Add security validation to pre-commit hooks

### Medium Term

1. ⬜ Implement AST-based static analysis for security patterns
2. ⬜ Add configuration validation (SECURITY-100 series)
3. ⬜ Add pattern validation (SECURITY-200 series)
4. ⬜ Create security validation dashboard

### Long Term

1. ⬜ Integrate with security scanning tools
2. ⬜ Add runtime security monitoring
3. ⬜ Create security metrics and reporting
4. ⬜ Automated remediation suggestions

## Related Files

### New Files Created

- `src/omnibase_infra/validation/validator_security.py` - Security validation module
- `tests/unit/validation/test_validator_security.py` - Test suite
- `docs/SECURITY_VALIDATION_INTEGRATION.md` - This document

### Modified Files

- `src/omnibase_infra/validation/__init__.py` - Added security validator exports
- `docs/patterns/security_patterns.md` - Added structured error reporting section

### Related Files (Reference Only)

- `src/omnibase_infra/models/errors/model_handler_validation_error.py` - Error model
- `src/omnibase_infra/enums/enum_handler_error_type.py` - Error types
- `src/omnibase_infra/mixins/mixin_node_introspection.py` - Introspection mixin
- `src/omnibase_infra/validation/contract_linter.py` - Similar integration pattern

## Success Criteria Met

✅ **All success criteria from task description have been met:**

1. ✅ Security validation paths identified (found: none exist yet)
2. ✅ Integration pattern created (validator_security.py)
3. ✅ Rule IDs defined for security errors (SECURITY-001 through SECURITY-202)
4. ✅ All security errors have actionable remediation hints
5. ✅ Pattern documented for future use (security_patterns.md)
6. ✅ Tests created and passing (23 tests)

## Conclusion

This task successfully establishes the foundation for security validation in ONEX infrastructure. While no existing security validation was found to integrate with, we have created a complete security validation system that:

- Uses `ModelHandlerValidationError` for structured error reporting
- Provides actionable remediation hints
- Integrates with existing validation patterns
- Is fully tested and documented
- Is ready for future integration with introspection and CI pipelines

The security validator is now available for use and can be gradually integrated into the ONEX validation pipeline.
