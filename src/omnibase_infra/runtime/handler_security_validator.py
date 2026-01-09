# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Security Validator for OMN-1137.

This module provides the HandlerSecurityValidator class that validates
handler security metadata before loading. The validator enforces rules
about which handler types must have or must not have security metadata.

Security Validation Rules:
    SECURITY-305: EFFECT handlers MUST have security metadata
        (secret_scopes, allowed_domains, or non-default data_classification)
    SECURITY-306: COMPUTE handlers MUST NOT have security metadata
    SECURITY-307: Secret scopes must be valid (non-empty strings)
    SECURITY-308: Domain patterns must be valid URL patterns

Handler Type Security Requirements:
    | Handler Type | Security Metadata Required? |
    |--------------|----------------------------|
    | EFFECT | Yes - must have at least one of: secret_scopes, allowed_domains, or data_classification |
    | COMPUTE | No - must not have any security metadata |
    | NONDETERMINISTIC_COMPUTE | Yes - treated like EFFECT for security purposes |

Architecture:
    This validator operates at handler loading time, before handlers are
    registered. It validates that handler security policies match their
    declared behavioral category (handler_type_category).

See Also:
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - EnumHandlerTypeCategory: Handler behavioral classification
    - EnumSecurityRuleId: Security validation rule identifiers
    - RegistrationSecurityValidator: Environment-level security validation

.. versionadded:: 0.6.4
    Created as part of OMN-1137 handler security metadata validation.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from omnibase_core.enums import EnumDataClassification

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumSecurityRuleId
from omnibase_infra.models.security import (
    ModelHandlerSecurityPolicy,
    ModelSecurityError,
    ModelSecurityValidationResult,
    ModelSecurityWarning,
)

# Default data classification that doesn't count as "having security metadata"
_DEFAULT_CLASSIFICATION = EnumDataClassification.INTERNAL

# Pattern for validating domain patterns
# Supports: hostname, hostname:port, wildcards like *.example.com
_DOMAIN_PATTERN = re.compile(
    r"^"
    r"(\*\.)?[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?"  # First label (optional wildcard)
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*"  # Additional labels
    r"(:\d{1,5})?"  # Optional port
    r"$"
)


class HandlerSecurityValidator:
    """Validates handler security metadata before loading.

    This validator ensures that handler security policies are appropriate
    for their declared behavioral category:
    - EFFECT handlers MUST have security metadata (they perform I/O)
    - COMPUTE handlers MUST NOT have security metadata (they are pure)
    - NONDETERMINISTIC_COMPUTE handlers are treated like EFFECT

    The validator also checks that security metadata values are valid:
    - Secret scopes must be non-empty strings
    - Domain patterns must be valid URL patterns

    Example:
        >>> from omnibase_infra.enums import EnumHandlerTypeCategory
        >>> from omnibase_infra.models.security import ModelHandlerSecurityPolicy
        >>>
        >>> validator = HandlerSecurityValidator()
        >>>
        >>> # Valid COMPUTE handler (no security metadata)
        >>> policy = ModelHandlerSecurityPolicy()
        >>> result = validator.validate(
        ...     handler_name="my_compute_handler",
        ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
        ...     security_policy=policy,
        ... )
        >>> result.valid
        True
        >>>
        >>> # Invalid EFFECT handler (missing security metadata)
        >>> result = validator.validate(
        ...     handler_name="my_effect_handler",
        ...     handler_type=EnumHandlerTypeCategory.EFFECT,
        ...     security_policy=policy,
        ... )
        >>> result.valid
        False
        >>> result.errors[0].code
        'EFFECT_MISSING_SECURITY_METADATA'

    Attributes:
        None - this validator is stateless.

    .. versionadded:: 0.6.4
    """

    def validate(
        self,
        handler_name: str,
        handler_type: EnumHandlerTypeCategory,
        security_policy: ModelHandlerSecurityPolicy,
    ) -> ModelSecurityValidationResult:
        """Validate handler security metadata.

        Validates that the handler's security policy is appropriate for
        its declared behavioral category.

        Rules:
            - EFFECT handlers MUST have security metadata
            - COMPUTE handlers MUST NOT have security metadata
            - NONDETERMINISTIC_COMPUTE treated like EFFECT for security
            - Secret scopes must be valid (non-empty strings)
            - Domain patterns must be valid URL patterns

        Args:
            handler_name: Name of the handler being validated.
            handler_type: Behavioral classification of the handler.
            security_policy: Handler's declared security policy.

        Returns:
            ModelSecurityValidationResult with validation outcome.

        Example:
            >>> validator = HandlerSecurityValidator()
            >>> policy = ModelHandlerSecurityPolicy(
            ...     secret_scopes=frozenset({"database/readonly"}),
            ... )
            >>> result = validator.validate(
            ...     handler_name="db_handler",
            ...     handler_type=EnumHandlerTypeCategory.EFFECT,
            ...     security_policy=policy,
            ... )
            >>> result.valid
            True
        """
        errors: list[ModelSecurityError] = []
        warnings: list[ModelSecurityWarning] = []

        # Check if handler has any security metadata
        has_security_metadata = self._has_security_metadata(security_policy)

        # Validate based on handler type
        if handler_type == EnumHandlerTypeCategory.COMPUTE:
            # COMPUTE handlers MUST NOT have security metadata
            if has_security_metadata:
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.COMPUTE_HAS_SECURITY_METADATA.value,
                        field="security_policy",
                        message=(
                            f"COMPUTE handler '{handler_name}' has security metadata "
                            "but COMPUTE handlers must be pure (no I/O, no secrets). "
                            "Security metadata found: "
                            f"{self._describe_security_metadata(security_policy)}"
                        ),
                        severity="error",
                    )
                )
        elif handler_type in (
            EnumHandlerTypeCategory.EFFECT,
            EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
        ):
            # EFFECT and NONDETERMINISTIC_COMPUTE handlers MUST have security metadata
            if not has_security_metadata:
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.EFFECT_MISSING_SECURITY_METADATA.value,
                        field="security_policy",
                        message=(
                            f"EFFECT handler '{handler_name}' missing required "
                            "security metadata. EFFECT handlers must declare at least "
                            "one of: secret_scopes, allowed_domains, or a non-default "
                            "data_classification."
                        ),
                        severity="error",
                    )
                )

        # Validate secret scopes (if present)
        scope_errors = self.validate_secret_scopes(
            list(security_policy.secret_scopes),
            handler_name,
        )
        errors.extend(scope_errors)

        # Validate domain patterns (if present)
        domain_errors = self.validate_domains(
            list(security_policy.allowed_domains),
            handler_name,
        )
        errors.extend(domain_errors)

        # Return result
        if errors:
            return ModelSecurityValidationResult.failure(
                handler_name=handler_name,
                handler_type=handler_type,
                errors=tuple(errors),
                warnings=tuple(warnings),
            )
        return ModelSecurityValidationResult.success(
            handler_name=handler_name,
            handler_type=handler_type,
            warnings=tuple(warnings),
        )

    def validate_secret_scopes(
        self,
        secret_scopes: list[str],
        handler_name: str | None = None,
    ) -> list[ModelSecurityError]:
        """Validate that required secrets are valid.

        Secret scopes must be non-empty strings without leading/trailing
        whitespace. This method does not check if secrets are available
        (that is done by RegistrationSecurityValidator).

        Args:
            secret_scopes: List of secret scope identifiers to validate.
            handler_name: Optional handler name for error context.

        Returns:
            List of validation errors for invalid secret scopes.

        Example:
            >>> validator = HandlerSecurityValidator()
            >>> errors = validator.validate_secret_scopes(["database/readonly"])
            >>> len(errors)
            0
            >>> errors = validator.validate_secret_scopes(["", "  "])
            >>> len(errors)
            2
        """
        errors: list[ModelSecurityError] = []
        context = f" in handler '{handler_name}'" if handler_name else ""

        for i, scope in enumerate(secret_scopes):
            if not scope or not scope.strip():
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_SECRET_SCOPE.value,
                        field=f"secret_scopes[{i}]",
                        message=(
                            f"Invalid secret scope at index {i}{context}: "
                            "scope must be a non-empty string without "
                            "leading/trailing whitespace."
                        ),
                        severity="error",
                    )
                )
            elif scope != scope.strip():
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_SECRET_SCOPE.value,
                        field=f"secret_scopes[{i}]",
                        message=(
                            f"Invalid secret scope at index {i}{context}: "
                            f"scope '{scope}' has leading/trailing whitespace."
                        ),
                        severity="error",
                    )
                )

        return errors

    def validate_domains(
        self,
        allowed_domains: list[str],
        handler_name: str | None = None,
    ) -> list[ModelSecurityError]:
        """Validate domain allowlist patterns.

        Domain patterns must be valid hostnames, optionally with port
        numbers. Wildcard prefixes (*.example.com) are supported.

        Valid patterns:
            - api.example.com
            - api.example.com:8080
            - *.example.com
            - localhost
            - localhost:3000

        Invalid patterns:
            - Empty strings
            - Strings with leading/trailing whitespace
            - Full URLs (use hostname only)
            - Invalid hostname characters

        Args:
            allowed_domains: List of domain patterns to validate.
            handler_name: Optional handler name for error context.

        Returns:
            List of validation errors for invalid domain patterns.

        Example:
            >>> validator = HandlerSecurityValidator()
            >>> errors = validator.validate_domains(["api.example.com"])
            >>> len(errors)
            0
            >>> errors = validator.validate_domains(["https://api.example.com"])
            >>> len(errors)
            1
        """
        errors: list[ModelSecurityError] = []
        context = f" in handler '{handler_name}'" if handler_name else ""

        for i, domain in enumerate(allowed_domains):
            # Skip wildcard "*" - this is valid (means allow all)
            if domain == "*":
                continue

            # Check for empty or whitespace-only
            if not domain or not domain.strip():
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            "domain must be a non-empty string."
                        ),
                        severity="error",
                    )
                )
                continue

            # Check for leading/trailing whitespace
            if domain != domain.strip():
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            f"domain '{domain}' has leading/trailing whitespace."
                        ),
                        severity="error",
                    )
                )
                continue

            # Check if it looks like a full URL (has scheme)
            if "://" in domain:
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            f"'{domain}' appears to be a full URL. "
                            "Use hostname only (e.g., 'api.example.com')."
                        ),
                        severity="error",
                    )
                )
                continue

            # Validate domain pattern format
            if not _DOMAIN_PATTERN.match(domain):
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            f"'{domain}' is not a valid hostname pattern. "
                            "Use format: hostname or hostname:port "
                            "(e.g., 'api.example.com' or '*.example.com:8080')."
                        ),
                        severity="error",
                    )
                )

        return errors

    def _has_security_metadata(
        self,
        security_policy: ModelHandlerSecurityPolicy,
    ) -> bool:
        """Check if handler has any security metadata declared.

        Security metadata includes:
        - secret_scopes (non-empty)
        - allowed_domains (non-empty)
        - data_classification (non-default, i.e., not INTERNAL)

        Args:
            security_policy: Handler's declared security policy.

        Returns:
            True if handler has any security metadata, False otherwise.
        """
        # Has secret scopes?
        if security_policy.secret_scopes:
            return True

        # Has allowed domains?
        if security_policy.allowed_domains:
            return True

        # Has non-default data classification?
        if security_policy.data_classification != _DEFAULT_CLASSIFICATION:
            return True

        return False

    def _describe_security_metadata(
        self,
        security_policy: ModelHandlerSecurityPolicy,
    ) -> str:
        """Create human-readable description of security metadata.

        Args:
            security_policy: Handler's declared security policy.

        Returns:
            String describing what security metadata is present.
        """
        parts: list[str] = []

        if security_policy.secret_scopes:
            scopes = ", ".join(sorted(security_policy.secret_scopes))
            parts.append(f"secret_scopes=[{scopes}]")

        if security_policy.allowed_domains:
            domains = ", ".join(security_policy.allowed_domains)
            parts.append(f"allowed_domains=[{domains}]")

        if security_policy.data_classification != _DEFAULT_CLASSIFICATION:
            parts.append(
                f"data_classification={security_policy.data_classification.value}"
            )

        return ", ".join(parts) if parts else "(none)"


def validate_handler_security(
    handler_name: str,
    handler_type: EnumHandlerTypeCategory,
    security_policy: ModelHandlerSecurityPolicy,
) -> ModelSecurityValidationResult:
    """Validate handler security metadata.

    Convenience function for one-shot validation without creating a
    validator instance.

    Args:
        handler_name: Name of the handler being validated.
        handler_type: Behavioral classification of the handler.
        security_policy: Handler's declared security policy.

    Returns:
        ModelSecurityValidationResult with validation outcome.

    Example:
        >>> from omnibase_infra.enums import EnumHandlerTypeCategory
        >>> from omnibase_infra.models.security import ModelHandlerSecurityPolicy
        >>> from omnibase_infra.runtime import validate_handler_security
        >>>
        >>> policy = ModelHandlerSecurityPolicy()
        >>> result = validate_handler_security(
        ...     handler_name="my_compute",
        ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
        ...     security_policy=policy,
        ... )
        >>> result.valid
        True

    .. versionadded:: 0.6.4
    """
    validator = HandlerSecurityValidator()
    return validator.validate(handler_name, handler_type, security_policy)


__all__ = [
    "HandlerSecurityValidator",
    "validate_handler_security",
]
