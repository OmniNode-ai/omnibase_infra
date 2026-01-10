# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security validation rule IDs for OMN-1098.

This module defines the security rule identifiers used in validation errors
for handler security policy enforcement. These IDs provide structured error
codes for both registration-time and invocation-time security violations.

Rule ID Structure:
    - SECURITY-3xx: Security-related violations
    - 300-309: Registration-time violations (policy declaration errors)
    - 310-319: Invocation-time violations (runtime enforcement errors)

Usage:
    These rule IDs are used in validation error models to provide
    machine-readable error codes for security violations.

See Also:
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - ModelEnvironmentPolicy: Environment-level security constraints
"""

from enum import StrEnum


class EnumSecurityRuleId(StrEnum):
    """Security rule identifiers for validation errors.

    Provides structured error codes for handler security policy violations.
    Used to categorize and identify specific security constraint failures
    during handler registration and invocation.

    Two-Layer Security Validation Architecture:
        The security system validates handlers at two distinct points:

        1. Registration-Time Violations (300-309):
            Validated by: RegistrationSecurityValidator
            Location: omnibase_infra.validation.registration_security_validator
            When: Handler attempts to register with the system
            Purpose: Prevents misconfigured handlers from registering

        2. Invocation-Time Violations (310-319):
            Enforced by: InvocationSecurityEnforcer
            Location: omnibase_infra.runtime.invocation_security_enforcer
            When: Handler attempts to access resources at runtime
            Purpose: Enforces declared policy during handler execution

    Attributes:
        SECRET_SCOPE_NOT_PERMITTED: Handler requests secret scope not permitted
            in the current environment. (Registration-time, SECURITY-300)
        CLASSIFICATION_EXCEEDS_MAX: Handler's data classification exceeds the
            maximum allowed for the environment. (Registration-time, SECURITY-301)
        ADAPTER_REQUESTING_SECRETS: Adapter handler attempting to request secrets
            (adapters should not access secrets directly). (Registration-time, SECURITY-302)
        ADAPTER_NON_EFFECT_CATEGORY: Adapter handler has non-EFFECT category
            (adapters must be EFFECT handlers). (Registration-time, SECURITY-303)
        ADAPTER_MISSING_DOMAIN_ALLOWLIST: Adapter handler missing required
            domain allowlist in environment requiring explicit allowlists.
            (Registration-time, SECURITY-304)
        EFFECT_MISSING_SECURITY_METADATA: EFFECT handler missing required
            security metadata (must have secret_scopes, allowed_domains, or
            data_classification). (Handler-type validation, SECURITY-305)
        COMPUTE_HAS_SECURITY_METADATA: COMPUTE handler has security metadata
            (COMPUTE handlers must not have security metadata).
            (Handler-type validation, SECURITY-306)
        INVALID_SECRET_SCOPE: Secret scope is invalid (empty or whitespace-only).
            (Handler-type validation, SECURITY-307)
        INVALID_DOMAIN_PATTERN: Domain pattern in allowlist is invalid.
            (Handler-type validation, SECURITY-308)
        DOMAIN_ACCESS_DENIED: Handler attempted to access domain not in its
            declared allowlist. (Invocation-time, SECURITY-310)
        SECRET_SCOPE_ACCESS_DENIED: Handler attempted to access secret scope
            not in its declared scopes. (Invocation-time, SECURITY-311)
        CLASSIFICATION_CONSTRAINT_VIOLATION: Data processed exceeds handler's
            declared classification level. (Invocation-time, SECURITY-312)

    See Also:
        - RegistrationSecurityValidator: Registration-time validation
        - InvocationSecurityEnforcer: Invocation-time enforcement
        - ModelHandlerSecurityPolicy: Handler-declared security requirements
        - ModelEnvironmentPolicy: Environment-level security constraints
    """

    # Registration-time violations (300-309)
    SECRET_SCOPE_NOT_PERMITTED = "SECURITY-300"
    CLASSIFICATION_EXCEEDS_MAX = "SECURITY-301"
    ADAPTER_REQUESTING_SECRETS = "SECURITY-302"
    ADAPTER_NON_EFFECT_CATEGORY = "SECURITY-303"
    ADAPTER_MISSING_DOMAIN_ALLOWLIST = "SECURITY-304"

    # Handler type security violations (305-309) - OMN-1137
    EFFECT_MISSING_SECURITY_METADATA = "SECURITY-305"
    COMPUTE_HAS_SECURITY_METADATA = "SECURITY-306"
    INVALID_SECRET_SCOPE = "SECURITY-307"
    INVALID_DOMAIN_PATTERN = "SECURITY-308"

    # Invocation-time violations (310-319)
    DOMAIN_ACCESS_DENIED = "SECURITY-310"
    SECRET_SCOPE_ACCESS_DENIED = "SECURITY-311"
    CLASSIFICATION_CONSTRAINT_VIOLATION = "SECURITY-312"


__all__ = ["EnumSecurityRuleId"]
