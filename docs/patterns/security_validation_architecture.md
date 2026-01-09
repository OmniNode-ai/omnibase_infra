# Security Validation Architecture

## Overview

The ONEX infrastructure implements a **three-layer security validation system** that provides defense in depth for handler security. Each layer validates at a different point in the handler lifecycle, ensuring that security is enforced from handler definition through runtime execution.

**Core Principle**: Defense in depth - even if one layer fails or is bypassed, the other layers still provide protection.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Validation Flow](#validation-flow)
- [SecurityMetadataValidator](#securitymetadatavalidator)
- [RegistrationSecurityValidator](#registrationsecurityvalidator)
- [InvocationSecurityEnforcer](#invocationsecurityenforcer)
- [Security Rule Reference](#security-rule-reference)
- [Integration Example](#integration-example)
- [Related Patterns](#related-patterns)

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                    HANDLER LIFECYCLE                              |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+     +---------------------+     +-------+  |
|  |   Handler Code    | --> |   Handler Loading   | --> |       |  |
|  |   Definition      |     |   (Parse & Load)    |     |       |  |
|  +-------------------+     +---------------------+     |       |  |
|                                    |                   |       |  |
|                                    v                   |       |  |
|                        +------------------------+      |       |  |
|                        | SecurityMetadata-      |      |       |  |
|                        | Validator              |      | H     |  |
|                        | (SECURITY-305 to 308)  |      | A     |  |
|                        +------------------------+      | N     |  |
|                                    |                   | D     |  |
|                                    | PASS              | L     |  |
|                                    v                   | E     |  |
|                        +------------------------+      | R     |  |
|                        | RegistrationSecurity-  |      |       |  |
|                        | Validator              |      | R     |  |
|                        | (SECURITY-300 to 304)  |      | E     |  |
|                        +------------------------+      | G     |  |
|                                    |                   | I     |  |
|                                    | PASS              | S     |  |
|                                    v                   | T     |  |
|                        +------------------------+      | R     |  |
|                        | Handler Registered     |      | Y     |  |
|                        | (Ready for Use)        |      |       |  |
|                        +------------------------+      +-------+  |
|                                    |                              |
|                                    | INVOCATION                   |
|                                    v                              |
|                        +------------------------+                 |
|                        | InvocationSecurity-    |                 |
|                        | Enforcer               |                 |
|                        | (SECURITY-310 to 312)  |                 |
|                        +------------------------+                 |
|                                    |                              |
|                                    | PASS                         |
|                                    v                              |
|                        +------------------------+                 |
|                        | Handler Executes       |                 |
|                        | (Runtime Operation)    |                 |
|                        +------------------------+                 |
|                                                                   |
+------------------------------------------------------------------+
```

## Validation Flow

### Sequential Validation Diagram

```
   Handler Definition
         |
         v
   +-----------------------+
   | 1. LOADING TIME       |
   |    SecurityMetadata-  |
   |    Validator          |
   +-----------------------+
         |
         | Validates: Handler type vs security metadata consistency
         | Rules: SECURITY-305, 306, 307, 308
         |
         v
   +-----------------------+
   | 2. REGISTRATION TIME  |
   |    RegistrationSec-   |
   |    urityValidator     |
   +-----------------------+
         |
         | Validates: Handler policy vs environment constraints
         | Rules: SECURITY-300, 301, 302, 303, 304
         |
         v
   +-----------------------+
   | 3. INVOCATION TIME    |
   |    InvocationSecurity-|
   |    Enforcer           |
   +-----------------------+
         |
         | Validates: Runtime operations vs declared policy
         | Rules: SECURITY-310, 311, 312
         |
         v
     Handler Executes
```

### Timing Summary

| Validator | When | What it Validates |
|-----------|------|-------------------|
| `SecurityMetadataValidator` | Handler loading (parse time) | Handler type matches security metadata |
| `RegistrationSecurityValidator` | Handler registration | Policy fits environment constraints |
| `InvocationSecurityEnforcer` | Each handler invocation | Operations comply with declared policy |

---

## SecurityMetadataValidator

**Location**: `omnibase_infra.runtime.security_metadata_validator`

**Purpose**: Validates that handler security metadata is appropriate for the handler's declared behavioral category. This ensures handlers are correctly classified before they can even attempt registration.

### When It Runs

The `SecurityMetadataValidator` runs at **handler loading time**, before handlers are registered with the system. This is the earliest point in the handler lifecycle where security validation occurs.

### Rules Enforced

| Rule ID | Rule Name | Description |
|---------|-----------|-------------|
| SECURITY-305 | `EFFECT_MISSING_SECURITY_METADATA` | EFFECT handlers MUST have security metadata (secret_scopes, allowed_domains, or non-default data_classification) |
| SECURITY-306 | `COMPUTE_HAS_SECURITY_METADATA` | COMPUTE handlers MUST NOT have security metadata (they must be pure) |
| SECURITY-307 | `INVALID_SECRET_SCOPE` | Secret scopes must be valid (non-empty strings without whitespace) |
| SECURITY-308 | `INVALID_DOMAIN_PATTERN` | Domain patterns must be valid URL hostname patterns |

### Handler Type Security Requirements

| Handler Type | Security Metadata Required? | Rationale |
|--------------|----------------------------|-----------|
| EFFECT | Yes | Performs I/O, needs security constraints |
| COMPUTE | No | Pure transformation, no I/O |
| NONDETERMINISTIC_COMPUTE | Yes | Treated like EFFECT for security |

### Usage Example

```python
from omnibase_infra.runtime import SecurityMetadataValidator, validate_handler_security
from omnibase_infra.enums import EnumHandlerTypeCategory
from omnibase_infra.models.security import ModelHandlerSecurityPolicy

# Create validator instance
validator = SecurityMetadataValidator()

# Validate EFFECT handler (must have security metadata)
effect_policy = ModelHandlerSecurityPolicy(
    secret_scopes=frozenset({"database/readonly"}),
    allowed_domains=["api.example.com"],
)
result = validator.validate(
    handler_name="db_effect_handler",
    handler_type=EnumHandlerTypeCategory.EFFECT,
    security_policy=effect_policy,
)
assert result.valid  # Passes - has security metadata

# Validate COMPUTE handler (must NOT have security metadata)
compute_policy = ModelHandlerSecurityPolicy()  # No security metadata
result = validator.validate(
    handler_name="transform_compute_handler",
    handler_type=EnumHandlerTypeCategory.COMPUTE,
    security_policy=compute_policy,
)
assert result.valid  # Passes - no security metadata
```

---

## RegistrationSecurityValidator

**Location**: `omnibase_infra.validation.registration_security_validator`

**Purpose**: Validates handler security policies against environment-level constraints. This ensures that handlers are only registered in environments that permit their security requirements.

### When It Runs

The `RegistrationSecurityValidator` runs at **registration time**, after loading but before the handler can be used. If registration validation rejects a handler, invocation-time enforcement will never be needed for that handler.

### Rules Enforced

| Rule ID | Rule Name | Description |
|---------|-----------|-------------|
| SECURITY-300 | `SECRET_SCOPE_NOT_PERMITTED` | Handler requests secret scope not permitted in environment |
| SECURITY-301 | `CLASSIFICATION_EXCEEDS_MAX` | Handler's data classification exceeds environment maximum |
| SECURITY-302 | `ADAPTER_REQUESTING_SECRETS` | Adapter handler requesting direct secret access |
| SECURITY-303 | `ADAPTER_NON_EFFECT_CATEGORY` | Adapter handler has non-EFFECT category |
| SECURITY-304 | `ADAPTER_MISSING_DOMAIN_ALLOWLIST` | Adapter missing required explicit domain allowlist |

### Environment Policy Checks

The validator compares handler-declared policy against environment constraints:

```
Handler Policy                Environment Policy
--------------                ------------------
secret_scopes     <------->   permitted_secret_scopes
data_classification <----->   max_data_classification
is_adapter        <------->   adapter_secrets_override_allowed
allowed_domains   <------->   require_explicit_domain_allowlist
```

### Usage Example

```python
from omnibase_infra.validation import RegistrationSecurityValidator, validate_handler_registration
from omnibase_infra.models.security import ModelHandlerSecurityPolicy, ModelEnvironmentPolicy
from omnibase_infra.enums import EnumEnvironment
from omnibase_core.enums import EnumDataClassification

# Define environment constraints
env_policy = ModelEnvironmentPolicy(
    environment=EnumEnvironment.PRODUCTION,
    permitted_secret_scopes=frozenset({"api-keys", "database-creds"}),
    max_data_classification=EnumDataClassification.CONFIDENTIAL,
    require_explicit_domain_allowlist=True,
)

# Define handler policy
handler_policy = ModelHandlerSecurityPolicy(
    secret_scopes=frozenset({"api-keys"}),
    allowed_domains=["api.example.com"],
    data_classification=EnumDataClassification.INTERNAL,
)

# Stateful pattern - bind environment at construction
validator = RegistrationSecurityValidator(env_policy)
if validator.is_valid(handler_policy):
    print("Handler can register in this environment")
else:
    errors = validator.validate(handler_policy)
    for error in errors:
        print(f"Validation error: {error.message}")

# Stateless pattern - convenience function
errors = validate_handler_registration(handler_policy, env_policy)
```

---

## InvocationSecurityEnforcer

**Location**: `omnibase_infra.runtime.invocation_security_enforcer`

**Purpose**: Enforces handler security policies at invocation time. This validates that runtime operations comply with the handler's declared security policy.

### When It Runs

The `InvocationSecurityEnforcer` runs at **invocation time**, for each handler invocation. It is created when a handler is instantiated and performs security checks during handler execution.

### Rules Enforced

| Rule ID | Rule Name | Description |
|---------|-----------|-------------|
| SECURITY-310 | `DOMAIN_ACCESS_DENIED` | Handler tried to access domain not in allowlist |
| SECURITY-311 | `SECRET_SCOPE_ACCESS_DENIED` | Handler tried to access undeclared secret scope |
| SECURITY-312 | `CLASSIFICATION_CONSTRAINT_VIOLATION` | Data exceeds handler's classification level |

### Security Checks

| Check Method | What It Validates |
|--------------|-------------------|
| `check_domain_access(domain)` | Domain matches allowed patterns |
| `check_secret_scope_access(scope)` | Scope is declared in policy |
| `check_classification_constraint(level)` | Data level <= handler level |

### Usage Example

```python
from omnibase_infra.runtime import InvocationSecurityEnforcer, SecurityViolationError
from omnibase_infra.models.security import ModelHandlerSecurityPolicy
from omnibase_core.enums import EnumDataClassification
from uuid import uuid4

# Create enforcer with handler policy
policy = ModelHandlerSecurityPolicy(
    secret_scopes=frozenset({"api-keys"}),
    allowed_domains=["api.example.com", "*.internal.com"],
    data_classification=EnumDataClassification.CONFIDENTIAL,
)
enforcer = InvocationSecurityEnforcer(policy, correlation_id=uuid4())

# Check domain access
enforcer.check_domain_access("api.example.com")  # OK
enforcer.check_domain_access("service.internal.com")  # OK (wildcard match)
try:
    enforcer.check_domain_access("malicious.com")  # Raises SecurityViolationError
except SecurityViolationError as e:
    print(f"Blocked: {e.rule_id} - {e.message}")

# Check secret scope access
enforcer.check_secret_scope_access("api-keys")  # OK
try:
    enforcer.check_secret_scope_access("database-creds")  # Raises
except SecurityViolationError as e:
    print(f"Blocked: {e.rule_id} - {e.message}")

# Check data classification
enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)  # OK
enforcer.check_classification_constraint(EnumDataClassification.CONFIDENTIAL)  # OK
try:
    enforcer.check_classification_constraint(EnumDataClassification.RESTRICTED)  # Raises
except SecurityViolationError as e:
    print(f"Blocked: {e.rule_id} - {e.message}")
```

---

## Security Rule Reference

### Complete Rule ID Table

| Rule ID | Validator | Category | Description |
|---------|-----------|----------|-------------|
| SECURITY-300 | Registration | Secret | Secret scope not permitted by environment |
| SECURITY-301 | Registration | Classification | Data classification exceeds environment max |
| SECURITY-302 | Registration | Adapter | Adapter requesting secrets |
| SECURITY-303 | Registration | Adapter | Adapter with non-EFFECT category |
| SECURITY-304 | Registration | Adapter | Adapter missing domain allowlist |
| SECURITY-305 | Metadata | Handler Type | EFFECT handler missing security metadata |
| SECURITY-306 | Metadata | Handler Type | COMPUTE handler has security metadata |
| SECURITY-307 | Metadata | Validation | Invalid secret scope format |
| SECURITY-308 | Metadata | Validation | Invalid domain pattern format |
| SECURITY-310 | Invocation | Domain | Domain access denied |
| SECURITY-311 | Invocation | Secret | Secret scope access denied |
| SECURITY-312 | Invocation | Classification | Classification constraint violation |

### Rule Ranges

- **300-309**: Registration-time violations (policy declaration errors)
- **305-308**: Handler type validation (metadata consistency)
- **310-319**: Invocation-time violations (runtime enforcement errors)

---

## Integration Example

### Complete Two-Layer Validation Flow

```python
from uuid import uuid4
from omnibase_core.enums import EnumDataClassification
from omnibase_infra.enums import EnumEnvironment, EnumHandlerTypeCategory
from omnibase_infra.models.security import (
    ModelHandlerSecurityPolicy,
    ModelEnvironmentPolicy,
)
from omnibase_infra.runtime import (
    SecurityMetadataValidator,
    InvocationSecurityEnforcer,
)
from omnibase_infra.validation import validate_handler_registration


def register_and_use_handler():
    """Demonstrate complete security validation flow."""

    # Step 1: Define handler security policy
    handler_policy = ModelHandlerSecurityPolicy(
        secret_scopes=frozenset({"api-keys"}),
        allowed_domains=["api.example.com", "storage.example.com"],
        data_classification=EnumDataClassification.CONFIDENTIAL,
    )

    # Step 2: Handler loading - Validate metadata consistency
    metadata_validator = SecurityMetadataValidator()
    result = metadata_validator.validate(
        handler_name="my_effect_handler",
        handler_type=EnumHandlerTypeCategory.EFFECT,
        security_policy=handler_policy,
    )
    if not result.valid:
        raise ValueError(f"Handler metadata invalid: {result.errors}")
    print("1. Handler metadata validation PASSED")

    # Step 3: Registration - Validate against environment
    env_policy = ModelEnvironmentPolicy(
        environment=EnumEnvironment.PRODUCTION,
        permitted_secret_scopes=frozenset({"api-keys", "database-creds"}),
        max_data_classification=EnumDataClassification.CONFIDENTIAL,
    )
    errors = validate_handler_registration(handler_policy, env_policy)
    if errors:
        raise ValueError(f"Handler registration rejected: {errors}")
    print("2. Handler registration validation PASSED")

    # Step 4: Handler is now registered and can be invoked
    # Create enforcer for runtime checks
    enforcer = InvocationSecurityEnforcer(
        handler_policy,
        correlation_id=uuid4(),
    )

    # Step 5: Runtime checks during handler execution
    # These would be called by the handler when accessing resources
    enforcer.check_domain_access("api.example.com")  # OK
    enforcer.check_secret_scope_access("api-keys")  # OK
    enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)  # OK
    print("3. Runtime security checks PASSED")

    print("\nHandler successfully validated at all three layers!")


if __name__ == "__main__":
    register_and_use_handler()
```

---

## Related Patterns

- [Security Patterns](./security_patterns.md) - Comprehensive security patterns including error sanitization, input validation, and authentication
- [Error Handling Patterns](./error_handling_patterns.md) - Error context and sanitization for security errors
- [Policy Registry Trust Model](./policy_registry_trust_model.md) - Trust model for policy registration

## See Also

- [EnumSecurityRuleId](../../src/omnibase_infra/enums/enum_security_rule_id.py) - Security rule definitions
- [SecurityMetadataValidator](../../src/omnibase_infra/runtime/security_metadata_validator.py) - Handler loading validation
- [RegistrationSecurityValidator](../../src/omnibase_infra/validation/registration_security_validator.py) - Registration validation
- [InvocationSecurityEnforcer](../../src/omnibase_infra/runtime/invocation_security_enforcer.py) - Runtime enforcement
- [Two-Layer Security Tests](../../tests/integration/security/test_two_layer_security_validation.py) - Integration tests demonstrating the full flow
