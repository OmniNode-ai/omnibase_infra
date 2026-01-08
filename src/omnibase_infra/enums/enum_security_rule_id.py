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

    Registration-Time Violations (300-309):
        These violations are detected when a handler attempts to register
        with security requirements that exceed environment permissions.

    Invocation-Time Violations (310-319):
        These violations are detected at runtime when a handler attempts
        to access resources beyond its declared policy.

    Attributes:
        SECRET_SCOPE_NOT_PERMITTED: Handler requests secret scope not permitted
            in the current environment.
        CLASSIFICATION_EXCEEDS_MAX: Handler's data classification exceeds the
            maximum allowed for the environment.
        ADAPTER_REQUESTING_SECRETS: Adapter handler attempting to request secrets
            (adapters should not access secrets directly).
        ADAPTER_NON_EFFECT_CATEGORY: Adapter handler has non-EFFECT category
            (adapters must be EFFECT handlers).
        ADAPTER_MISSING_DOMAIN_ALLOWLIST: Adapter handler missing required
            domain allowlist in environment requiring explicit allowlists.
        DOMAIN_ACCESS_DENIED: Handler attempted to access domain not in its
            declared allowlist.
        SECRET_SCOPE_ACCESS_DENIED: Handler attempted to access secret scope
            not in its declared scopes.
        CLASSIFICATION_CONSTRAINT_VIOLATION: Data processed exceeds handler's
            declared classification level.
    """

    # Registration-time violations (300-309)
    SECRET_SCOPE_NOT_PERMITTED = "SECURITY-300"
    CLASSIFICATION_EXCEEDS_MAX = "SECURITY-301"
    ADAPTER_REQUESTING_SECRETS = "SECURITY-302"
    ADAPTER_NON_EFFECT_CATEGORY = "SECURITY-303"
    ADAPTER_MISSING_DOMAIN_ALLOWLIST = "SECURITY-304"

    # Invocation-time violations (310-319)
    DOMAIN_ACCESS_DENIED = "SECURITY-310"
    SECRET_SCOPE_ACCESS_DENIED = "SECURITY-311"
    CLASSIFICATION_CONSTRAINT_VIOLATION = "SECURITY-312"


__all__ = ["EnumSecurityRuleId"]
