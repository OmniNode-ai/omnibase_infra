# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security Validation for ONEX Infrastructure.

This module provides security validation utilities for handler introspection,
method exposure, and security constraint enforcement. Part of OMN-1091:
Structured Validation & Error Reporting for Handlers.

Security Validation Scope:
    - Method exposure via introspection (sensitive method names)
    - Credential leakage in public method signatures
    - Insecure patterns in handler code
    - Missing authentication/authorization decorators

Integration Status:
    This module is a PLACEHOLDER for future security validation integration.
    As of implementation, the security validation logic needs to be integrated
    with the following components:

    1. MixinNodeIntrospection._should_skip_method() - Add validation
    2. Contract linting - Add security rule checks
    3. Static analysis - Add AST-based security scanning
    4. Runtime validation - Add security constraint enforcement

Usage:
    >>> from omnibase_infra.validation.security_validator import (
    ...     validate_method_exposure,
    ...     validate_handler_security,
    ... )
    >>>
    >>> # Validate method names for security concerns
    >>> errors = validate_method_exposure(
    ...     method_names=["get_credentials", "process_request"],
    ...     handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
    ... )
    >>>
    >>> # Check for security violations
    >>> for error in errors:
    ...     print(error.format_for_logging())

See Also:
    - docs/patterns/security_patterns.md - Comprehensive security guide
    - MixinNodeIntrospection - Node capability discovery with security filtering
    - ModelHandlerValidationError - Structured error reporting
    - EnumHandlerErrorType.SECURITY_VALIDATION_ERROR - Error classification
"""

from __future__ import annotations

import re
from typing import Final

from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier

# Security rule IDs for structured error reporting
# These IDs enable tracking and remediation of specific security violations


class SecurityRuleId:
    """Rule IDs for security validation errors.

    These IDs provide unique identifiers for each type of security validation
    failure, enabling structured error tracking and remediation guidance.
    """

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


# Sensitive method name patterns that should be prefixed with underscore
# These patterns indicate methods that expose sensitive operations
# Pre-compiled for performance
_SENSITIVE_METHOD_PATTERN_STRINGS: Final[tuple[str, ...]] = (
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

# Pre-compiled regex patterns for efficient matching
SENSITIVE_METHOD_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern) for pattern in _SENSITIVE_METHOD_PATTERN_STRINGS
)

# Patterns that should map to ADMIN_METHOD_PUBLIC rule (SECURITY-003)
_ADMIN_INTERNAL_PATTERN_STRINGS: Final[tuple[str, ...]] = (
    r"^admin_",
    r"^internal_",
)

# Pre-compiled admin/internal patterns
_ADMIN_INTERNAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern) for pattern in _ADMIN_INTERNAL_PATTERN_STRINGS
)

# Parameter names that indicate sensitive data in method signatures
SENSITIVE_PARAMETER_NAMES: Final[frozenset[str]] = frozenset(
    {
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
    }
)


def is_sensitive_method_name(method_name: str) -> bool:
    """Check if method name matches sensitive operation patterns.

    Args:
        method_name: Name of the method to check

    Returns:
        True if method name indicates a sensitive operation

    Example:
        >>> is_sensitive_method_name("get_password")
        True
        >>> is_sensitive_method_name("process_request")
        False
    """
    method_lower = method_name.lower()

    for pattern in SENSITIVE_METHOD_PATTERNS:
        if pattern.match(method_lower):
            return True

    return False


def _get_sensitivity_rule_id(method_name: str) -> str | None:
    """Determine the appropriate security rule ID for a sensitive method.

    Args:
        method_name: Name of the method to check

    Returns:
        Security rule ID if method is sensitive, None otherwise

    Note:
        - admin_/internal_ methods map to SECURITY-003 (ADMIN_METHOD_PUBLIC)
        - Other sensitive patterns map to SECURITY-001 (SENSITIVE_METHOD_EXPOSED)
    """
    method_lower = method_name.lower()

    # Check admin/internal patterns first (SECURITY-003)
    for pattern in _ADMIN_INTERNAL_PATTERNS:
        if pattern.match(method_lower):
            return SecurityRuleId.ADMIN_METHOD_PUBLIC

    # Check other sensitive patterns (SECURITY-001)
    for pattern in SENSITIVE_METHOD_PATTERNS:
        if pattern.match(method_lower):
            return SecurityRuleId.SENSITIVE_METHOD_EXPOSED

    return None


def has_sensitive_parameters(signature: str) -> list[str]:
    """Extract sensitive parameter names from method signature.

    Args:
        signature: Method signature string (e.g., "(username: str, password: str)")

    Returns:
        List of sensitive parameter names found in signature

    Example:
        >>> has_sensitive_parameters("(username: str, password: str)")
        ['password']
        >>> has_sensitive_parameters("(user_id: UUID, data: dict)")
        []
    """
    signature_lower = signature.lower()
    found_sensitive = []

    for param_name in SENSITIVE_PARAMETER_NAMES:
        # Match parameter name as word boundary to avoid false positives
        # e.g., "password" matches but "password_hash" doesn't
        pattern = rf"\b{param_name}\b"
        if re.search(pattern, signature_lower):
            found_sensitive.append(param_name)

    return found_sensitive


def validate_method_exposure(
    method_names: list[str],
    handler_identity: ModelHandlerIdentifier,
    method_signatures: dict[str, str] | None = None,
    file_path: str | None = None,
) -> list[ModelHandlerValidationError]:
    """Validate method names and signatures for security concerns.

    Checks for:
    - Sensitive method names that should be private (prefixed with _)
    - Method signatures containing sensitive parameter names
    - Admin/internal operations exposed publicly

    Args:
        method_names: List of public method names to validate
        handler_identity: Handler identification information
        method_signatures: Optional dict mapping method names to signatures
        file_path: Optional file path for error context

    Returns:
        List of ModelHandlerValidationError for any violations found

    Example:
        >>> from omnibase_infra.models.handlers import ModelHandlerIdentifier
        >>> identity = ModelHandlerIdentifier.from_handler_id("auth-handler")
        >>> errors = validate_method_exposure(
        ...     method_names=["get_api_key", "process_request"],
        ...     handler_identity=identity,
        ...     method_signatures={"get_api_key": "() -> str"},
        ... )
        >>> len(errors)
        1
        >>> errors[0].rule_id
        'SECURITY-001'
    """
    errors: list[ModelHandlerValidationError] = []

    for method_name in method_names:
        # Check for sensitive method names and get appropriate rule ID
        rule_id = _get_sensitivity_rule_id(method_name)
        if rule_id is not None:
            # Customize message and hint based on rule type
            if rule_id == SecurityRuleId.ADMIN_METHOD_PUBLIC:
                message = f"Handler exposes admin/internal method '{method_name}'"
                remediation_hint = f"Prefix method with underscore: '_{method_name}' or move to separate admin module"
                violation_type = "admin_method_public"
            else:
                message = f"Handler exposes sensitive method '{method_name}'"
                remediation_hint = f"Prefix method with underscore: '_{method_name}' to exclude from introspection"
                violation_type = "sensitive_method_exposed"

            error = ModelHandlerValidationError.from_security_violation(
                rule_id=rule_id,
                message=message,
                remediation_hint=remediation_hint,
                handler_identity=handler_identity,
                file_path=file_path,
                details={
                    "method_name": method_name,
                    "violation_type": violation_type,
                },
            )
            errors.append(error)

    # Check method signatures if provided
    if method_signatures:
        for method_name, signature in method_signatures.items():
            sensitive_params = has_sensitive_parameters(signature)
            if sensitive_params:
                error = ModelHandlerValidationError.from_security_violation(
                    rule_id=SecurityRuleId.CREDENTIAL_IN_SIGNATURE,
                    message=f"Method '{method_name}' has sensitive parameters in signature: {', '.join(sensitive_params)}",
                    remediation_hint=f"Use generic parameter names (e.g., 'data' instead of '{sensitive_params[0]}') or make method private",
                    handler_identity=handler_identity,
                    file_path=file_path,
                    details={
                        "method_name": method_name,
                        "signature": signature,
                        "sensitive_parameters": sensitive_params,
                        "violation_type": "credential_in_signature",
                    },
                )
                errors.append(error)

    return errors


def validate_handler_security(
    handler_identity: ModelHandlerIdentifier,
    capabilities: dict[str, object],
    file_path: str | None = None,
) -> list[ModelHandlerValidationError]:
    """Validate handler capabilities for security violations.

    This is the main entry point for security validation of a handler's
    introspection data. It checks:
    - Method exposure (sensitive method names)
    - Method signatures (sensitive parameters)
    - Security best practices

    Args:
        handler_identity: Handler identification information
        capabilities: Capabilities dict from MixinNodeIntrospection.get_capabilities()
        file_path: Optional file path for error context

    Returns:
        List of ModelHandlerValidationError for any violations found

    Example:
        >>> capabilities = {
        ...     "operations": ["get_api_key", "process_request"],
        ...     "protocols": ["ProtocolDatabaseAdapter"],
        ...     "has_fsm": False,
        ...     "method_signatures": {
        ...         "get_api_key": "() -> str",
        ...         "process_request": "(data: dict) -> Result",
        ...     },
        ... }
        >>> errors = validate_handler_security(
        ...     handler_identity=ModelHandlerIdentifier.from_handler_id("auth"),
        ...     capabilities=capabilities,
        ... )
        >>> len(errors) > 0
        True
    """
    operations = capabilities.get("operations", [])
    if not isinstance(operations, list):
        operations = []

    method_signatures = capabilities.get("method_signatures", {})
    if not isinstance(method_signatures, dict):
        method_signatures = {}

    return validate_method_exposure(
        method_names=operations,
        handler_identity=handler_identity,
        method_signatures=method_signatures,
        file_path=file_path,
    )


def convert_to_validation_error(
    rule_id: str,
    message: str,
    remediation_hint: str,
    handler_identity: ModelHandlerIdentifier,
    file_path: str | None = None,
    line_number: int | None = None,
    details: dict[str, object] | None = None,
) -> ModelHandlerValidationError:
    """Convert a security issue to ModelHandlerValidationError.

    This is a convenience function for creating security validation errors
    with consistent structure. Use this when integrating security checks
    into existing validation pipelines.

    Args:
        rule_id: Unique identifier for the validation rule
        message: Human-readable error message
        remediation_hint: Actionable fix suggestion
        handler_identity: Handler identification information
        file_path: Optional file path where error occurred
        line_number: Optional line number (1-indexed)
        details: Optional additional context

    Returns:
        ModelHandlerValidationError configured for security violations

    Example:
        >>> error = convert_to_validation_error(
        ...     rule_id=SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
        ...     message="Handler exposes 'get_password' method",
        ...     remediation_hint="Prefix with underscore: '_get_password'",
        ...     handler_identity=ModelHandlerIdentifier.from_handler_id("auth"),
        ...     file_path="nodes/auth/handlers/handler_authenticate.py",
        ...     line_number=42,
        ... )
        >>> error.error_type == EnumHandlerErrorType.SECURITY_VALIDATION_ERROR
        True
    """
    return ModelHandlerValidationError.from_security_violation(
        rule_id=rule_id,
        message=message,
        remediation_hint=remediation_hint,
        handler_identity=handler_identity,
        file_path=file_path,
        line_number=line_number,
        details=details,
    )


__all__ = [
    "SENSITIVE_METHOD_PATTERNS",
    "SENSITIVE_PARAMETER_NAMES",
    "SecurityRuleId",
    "convert_to_validation_error",
    "has_sensitive_parameters",
    "is_sensitive_method_name",
    "validate_handler_security",
    "validate_method_exposure",
]
