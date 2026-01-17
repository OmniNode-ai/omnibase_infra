# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security Metadata Validator for OMN-1137.

This module provides the SecurityMetadataValidator class that validates
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

Usage in Handler Loading Flow:
    The SecurityMetadataValidator is designed to integrate at the following
    points in the handler loading flow:

    1. **HandlerBootstrapSource (Recommended)**: During handler descriptor
       registration in ``_register_handler()``. This is the earliest point
       where handler metadata is available.

       Location: ``omnibase_infra/runtime/handler_bootstrap_source.py``

       Example integration::

           from omnibase_infra.runtime import validate_handler_security
           from omnibase_infra.models.security import ModelHandlerSecurityPolicy

           def _register_handler(self, descriptor: ModelHandlerDescriptor) -> None:
               # Extract security policy from descriptor (when available)
               security_policy = descriptor.security_policy or ModelHandlerSecurityPolicy()
               handler_type = EnumHandlerTypeCategory(descriptor.handler_kind.upper())

               # Validate security metadata before registration
               result = validate_handler_security(
                   handler_name=descriptor.handler_id,
                   handler_type=handler_type,
                   security_policy=security_policy,
               )

               if not result.valid:
                   for error in result.errors:
                       logger.error(
                           f"Security validation failed for {descriptor.handler_id}: "
                           f"[{error.code}] {error.message}"
                       )
                   raise SecurityValidationError(result)

               # Proceed with registration
               self._descriptors[descriptor.handler_id] = descriptor

    2. **wire_default_handlers() / wire_handlers_from_contract()**: During
       handler class registration with the RegistryProtocolBinding.

       Location: ``omnibase_infra/runtime/wiring.py``

       Example integration::

           from omnibase_infra.runtime import SecurityMetadataValidator

           def wire_handlers_from_contract(contract_config):
               validator = SecurityMetadataValidator()

               for handler_config in handlers_config:
                   # Validate security metadata if present in config
                   if "security" in handler_config:
                       security_policy = ModelHandlerSecurityPolicy(**handler_config["security"])
                       handler_type = EnumHandlerTypeCategory(handler_config.get("kind", "EFFECT"))

                       result = validator.validate(
                           handler_name=handler_type_str,
                           handler_type=handler_type,
                           security_policy=security_policy,
                       )

                       if not result.valid:
                           raise ProtocolConfigurationError(
                               f"Handler {handler_type_str} failed security validation"
                           )

                   handler_registry.register(handler_type, handler_cls)

    3. **wire_registration_handlers()**: During container-based handler
       instance registration.

       Location: ``omnibase_infra/runtime/container_wiring.py``

       Note: At this level, handlers are being instantiated rather than
       registered by type. Security validation should ideally happen earlier
       (in bootstrap or wiring), but can be added here as a safety check.

Integration Status:
    **NOT YET INTEGRATED** - The SecurityMetadataValidator is currently
    standalone and must be called manually. Future work (tracked separately)
    will integrate this validator into the handler loading flow at the
    HandlerBootstrapSource level.

    TODO(OMN-1137): Integrate SecurityMetadataValidator into HandlerBootstrapSource
    when ModelHandlerDescriptor includes security_policy field.

See Also:
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - EnumHandlerTypeCategory: Handler behavioral classification
    - EnumSecurityRuleId: Security validation rule identifiers
    - RegistrationSecurityValidator: Environment-level security validation
    - HandlerBootstrapSource: Bootstrap handler descriptor registration
    - wire_default_handlers: Default handler wiring function
    - wire_handlers_from_contract: Contract-based handler wiring

.. versionadded:: 0.6.4
    Created as part of OMN-1137 handler security metadata validation.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from omnibase_core.enums import EnumDataClassification
from omnibase_infra.enums import (
    EnumHandlerTypeCategory,
    EnumSecurityRuleId,
    EnumValidationSeverity,
)
from omnibase_infra.models.security import (
    ModelHandlerSecurityPolicy,
    ModelSecurityError,
    ModelSecurityValidationResult,
    ModelSecurityWarning,
)

# Default data classification that doesn't count as "having security metadata"
_DEFAULT_CLASSIFICATION = EnumDataClassification.INTERNAL

# Maximum length for a DNS label (RFC 1035)
_MAX_DNS_LABEL_LENGTH = 63

# Maximum total domain length (RFC 1035: 253 characters)
_MAX_DOMAIN_LENGTH = 253

# Valid port range (1-65535)
_MIN_PORT = 1
_MAX_PORT = 65535

# Known URL schemes for more robust URL detection
# urlparse() may interpret "hostname:port" as having a "scheme" of the hostname,
# so we only flag domains as URLs if they have a known URL scheme
_KNOWN_URL_SCHEMES = frozenset(
    {
        "http",
        "https",
        "ftp",
        "ftps",
        "sftp",
        "ssh",
        "file",
        "mailto",
        "tel",
        "data",
        "ws",
        "wss",
        "git",
        "svn",
        "s3",
        "gcs",
    }
)

# Pattern for validating domain patterns
# Supports: hostname, hostname:port, wildcards like *.example.com
_DOMAIN_PATTERN = re.compile(
    r"^"
    r"(\*\.)?[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?"  # First label (optional wildcard)
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*"  # Additional labels
    r"(:\d{1,5})?"  # Optional port
    r"$"
)


class SecurityMetadataValidator:
    """Validates handler security metadata before loading.

    This validator ensures that handler security policies are appropriate
    for their declared behavioral category:
    - EFFECT handlers MUST have security metadata (they perform I/O)
    - COMPUTE handlers MUST NOT have security metadata (they are pure)
    - NONDETERMINISTIC_COMPUTE handlers are treated like EFFECT

    The validator also checks that security metadata values are valid:
    - Secret scopes must be non-empty strings
    - Domain patterns must be valid URL patterns
    - Port numbers must be in valid range (1-65535)
    - DNS labels must be max 63 characters each
    - Total domain length must be max 253 characters (RFC 1035)

    Usage in Handler Loading Flow:
        This validator is designed to be called during handler registration,
        before handlers are loaded into the runtime. The recommended integration
        points are:

        1. **HandlerBootstrapSource._register_handler()** (recommended):
           Validate when handler descriptors are registered at bootstrap.

        2. **wire_handlers_from_contract()**: Validate when handlers are
           wired from contract configuration.

        See the module docstring for detailed integration examples and code.

        **Current Status**: NOT YET INTEGRATED. The validator must be called
        manually. See TODO(OMN-1137) for integration tracking.

    Example:
        >>> from omnibase_infra.enums import EnumHandlerTypeCategory
        >>> from omnibase_infra.models.security import ModelHandlerSecurityPolicy
        >>>
        >>> validator = SecurityMetadataValidator()
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

    See Also:
        - Module docstring: Full integration examples and code snippets
        - HandlerBootstrapSource: Bootstrap handler descriptor registration
        - wire_handlers_from_contract: Contract-based handler wiring

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
            - Port numbers must be in range 1-65535
            - DNS labels must be max 63 characters each
            - Total domain length must be max 253 characters (RFC 1035)

        Args:
            handler_name: Name of the handler being validated.
            handler_type: Behavioral classification of the handler.
            security_policy: Handler's declared security policy.

        Returns:
            ModelSecurityValidationResult with validation outcome.

        Example:
            >>> validator = SecurityMetadataValidator()
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
                        severity=EnumValidationSeverity.ERROR,
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
                        severity=EnumValidationSeverity.ERROR,
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
                subject=handler_name,
                handler_type=handler_type,
                errors=tuple(errors),
                warnings=tuple(warnings),
            )
        return ModelSecurityValidationResult.success(
            subject=handler_name,
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
            >>> validator = SecurityMetadataValidator()
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
                        severity=EnumValidationSeverity.ERROR,
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
                        severity=EnumValidationSeverity.ERROR,
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
            - Port numbers outside 1-65535 range
            - DNS labels longer than 63 characters
            - Total domain length exceeding 253 characters (RFC 1035)

        Args:
            allowed_domains: List of domain patterns to validate.
            handler_name: Optional handler name for error context.

        Returns:
            List of validation errors for invalid domain patterns.

        Example:
            >>> validator = SecurityMetadataValidator()
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
                        severity=EnumValidationSeverity.ERROR,
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
                        severity=EnumValidationSeverity.ERROR,
                    )
                )
                continue

            # Check if it looks like a full URL (has known scheme)
            # Using urlparse().scheme with a known schemes set is more robust
            # than checking for "://" as it handles schemes like "mailto:"
            # that don't use "://". We check against known schemes to avoid
            # false positives with "hostname:port" patterns where urlparse
            # would interpret "hostname" as the scheme.
            parsed = urlparse(domain)
            if parsed.scheme and parsed.scheme.lower() in _KNOWN_URL_SCHEMES:
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            f"'{domain}' appears to be a full URL "
                            f"(detected scheme: '{parsed.scheme}'). "
                            "Use hostname only (e.g., 'api.example.com')."
                        ),
                        severity=EnumValidationSeverity.ERROR,
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
                        severity=EnumValidationSeverity.ERROR,
                    )
                )
                continue

            # Extract hostname and port for further validation
            hostname: str = domain
            port_str: str | None = None
            if ":" in domain:
                # Split on last colon to get port
                parts = domain.rsplit(":", 1)
                hostname = parts[0]
                port_str = parts[1]

            # Validate port range (1-65535)
            if port_str is not None:
                try:
                    port = int(port_str)
                    if port < _MIN_PORT or port > _MAX_PORT:
                        errors.append(
                            ModelSecurityError(
                                code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                                field=f"allowed_domains[{i}]",
                                message=(
                                    f"Invalid domain pattern at index {i}{context}: "
                                    f"port {port} is out of valid range "
                                    f"({_MIN_PORT}-{_MAX_PORT})."
                                ),
                                severity=EnumValidationSeverity.ERROR,
                            )
                        )
                        continue
                except ValueError:
                    # Port is not a valid integer - already caught by regex
                    pass

            # Validate total domain length (max 253 characters, RFC 1035)
            # Note: We validate the hostname part only, without port
            hostname_for_length = hostname
            if hostname_for_length.startswith("*."):
                # Wildcard prefix doesn't count toward length limit
                # but the rest of the domain does
                hostname_for_length = hostname_for_length[2:]

            if len(hostname_for_length) > _MAX_DOMAIN_LENGTH:
                errors.append(
                    ModelSecurityError(
                        code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                        field=f"allowed_domains[{i}]",
                        message=(
                            f"Invalid domain pattern at index {i}{context}: "
                            f"total domain length ({len(hostname_for_length)} characters) "
                            f"exceeds maximum of {_MAX_DOMAIN_LENGTH} characters (RFC 1035)."
                        ),
                        severity=EnumValidationSeverity.ERROR,
                    )
                )
                continue

            # Validate DNS label lengths (max 63 characters each)
            # Remove wildcard prefix if present for label validation
            hostname_for_labels = hostname
            if hostname_for_labels.startswith("*."):
                hostname_for_labels = hostname_for_labels[2:]

            labels = hostname_for_labels.split(".")
            for label_index, label in enumerate(labels):
                if len(label) > _MAX_DNS_LABEL_LENGTH:
                    # Format label position as ordinal (1st, 2nd, 3rd, etc.)
                    position = label_index + 1  # 1-indexed for human readability
                    ordinal = self._ordinal(position)

                    # Show truncated label if very long, full label if reasonable
                    label_display = (
                        f"'{label[:30]}...'" if len(label) > 35 else f"'{label}'"
                    )

                    errors.append(
                        ModelSecurityError(
                            code=EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value,
                            field=f"allowed_domains[{i}]",
                            message=(
                                f"Invalid domain pattern at index {i}{context}: "
                                f"DNS label {label_display} ({len(label)} chars) "
                                f"exceeds maximum {_MAX_DNS_LABEL_LENGTH} chars "
                                f"at {ordinal} label in domain pattern."
                            ),
                            severity=EnumValidationSeverity.ERROR,
                        )
                    )
                    break  # Only report first label error per domain

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

    def _ordinal(self, n: int) -> str:
        """Convert an integer to its ordinal string representation.

        Args:
            n: The integer to convert (1-indexed).

        Returns:
            Ordinal string like "1st", "2nd", "3rd", "4th", etc.

        Example:
            >>> validator = SecurityMetadataValidator()
            >>> validator._ordinal(1)
            '1st'
            >>> validator._ordinal(2)
            '2nd'
            >>> validator._ordinal(3)
            '3rd'
            >>> validator._ordinal(11)
            '11th'
            >>> validator._ordinal(21)
            '21st'
        """
        # Special cases for 11th, 12th, 13th
        if 11 <= (n % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


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
    validator = SecurityMetadataValidator()
    return validator.validate(handler_name, handler_type, security_policy)


__all__ = [
    "SecurityMetadataValidator",
    "validate_handler_security",
]
