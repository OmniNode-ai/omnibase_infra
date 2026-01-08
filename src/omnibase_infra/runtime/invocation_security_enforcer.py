# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Invocation-time security enforcement for handlers.

Enforces handler security policies at invocation time. Part of the
two-layer security validation system (OMN-1098).

Security Rules Enforced:
    SECURITY-310: Outbound domain access denied
    SECURITY-311: Secret scope access denied
    SECURITY-312: Data classification constraint violation

Usage:
    The enforcer is created with a handler's security policy and optionally
    a correlation ID for tracing. All check methods are stateless after
    initialization.

Example:
    >>> from omnibase_infra.runtime.invocation_security_enforcer import (
    ...     InvocationSecurityEnforcer,
    ...     SecurityViolationError,
    ... )
    >>> from omnibase_infra.models.security import ModelHandlerSecurityPolicy
    >>> from omnibase_core.enums import EnumDataClassification
    >>>
    >>> policy = ModelHandlerSecurityPolicy(
    ...     secret_scopes=frozenset({"api-keys"}),
    ...     allowed_domains=["api.example.com"],
    ...     data_classification=EnumDataClassification.INTERNAL,
    ... )
    >>> enforcer = InvocationSecurityEnforcer(policy)
    >>> enforcer.check_domain_access("api.example.com")  # OK
    >>> enforcer.check_domain_access("api.other.com")  # Raises SecurityViolationError
"""

from __future__ import annotations

import re
from uuid import UUID, uuid4

from omnibase_core.enums import EnumDataClassification

from omnibase_infra.enums import EnumSecurityRuleId
from omnibase_infra.models.security import ModelHandlerSecurityPolicy

# Security level mapping for data classifications
# Higher values indicate more sensitive data
_CLASSIFICATION_SECURITY_LEVELS: dict[EnumDataClassification, int] = {
    EnumDataClassification.PUBLIC: 0,
    EnumDataClassification.INTERNAL: 1,
    EnumDataClassification.CONFIDENTIAL: 2,
    EnumDataClassification.RESTRICTED: 3,
    EnumDataClassification.SECRET: 4,
    EnumDataClassification.TOP_SECRET: 5,
    # Additional classifications mapped to appropriate levels
    EnumDataClassification.OPEN: 0,  # Same as PUBLIC
    EnumDataClassification.PRIVATE: 1,  # Same as INTERNAL
    EnumDataClassification.SENSITIVE: 2,  # Same as CONFIDENTIAL
    EnumDataClassification.CLASSIFIED: 4,  # Same as SECRET
    EnumDataClassification.UNCLASSIFIED: 0,  # Same as PUBLIC
}


def _get_security_level(classification: EnumDataClassification) -> int:
    """Get numeric security level for a data classification.

    Args:
        classification: The data classification enum value.

    Returns:
        Numeric security level (0 = lowest, 5 = highest).

    Raises:
        ValueError: If classification is not recognized.
    """
    if classification not in _CLASSIFICATION_SECURITY_LEVELS:
        msg = f"Unknown data classification: {classification}"
        raise ValueError(msg)
    return _CLASSIFICATION_SECURITY_LEVELS[classification]


class SecurityViolationError(Exception):
    """Raised when a security constraint is violated at invocation time.

    This exception is raised when a handler attempts to access resources
    or process data that violates its declared security policy.

    Attributes:
        rule_id: Security rule that was violated (EnumSecurityRuleId).
        message: Human-readable error message describing the violation.
        correlation_id: Tracing correlation ID for observability.

    Security Rules:
        SECURITY-310: Domain access denied - handler tried to access
            a domain not in its allowlist.
        SECURITY-311: Secret scope access denied - handler tried to
            access a secret scope not declared in its policy.
        SECURITY-312: Classification constraint violation - handler
            tried to process data above its classification level.

    Example:
        >>> try:
        ...     enforcer.check_domain_access("forbidden.com")
        ... except SecurityViolationError as e:
        ...     print(f"Rule: {e.rule_id}, Message: {e.message}")
        ...     print(f"Correlation ID: {e.correlation_id}")
    """

    def __init__(
        self,
        rule_id: EnumSecurityRuleId,
        message: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Initialize a security violation error.

        Args:
            rule_id: The security rule that was violated.
            message: Human-readable description of the violation.
            correlation_id: Optional tracing ID for observability.
        """
        self.rule_id = rule_id
        self.message = message
        self.correlation_id = correlation_id
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error string including rule ID."""
        return f"[{self.rule_id}] {self.message}"


class InvocationSecurityEnforcer:
    """Enforces handler security policies at invocation time.

    This class is stateless after initialization. It validates that
    runtime operations comply with the handler's declared security policy.

    The enforcer performs three types of security checks:
        1. Domain access control - validates outbound network access
        2. Secret scope access control - validates secret access requests
        3. Data classification enforcement - validates data sensitivity levels

    Thread Safety:
        This class is thread-safe. All instance attributes are immutable
        after initialization, and all methods are read-only operations.

    Attributes:
        _policy: The handler's immutable security policy.
        _correlation_id: Correlation ID for error tracing.
        _domain_patterns: Pre-compiled regex patterns for domain matching.

    Example:
        >>> policy = ModelHandlerSecurityPolicy(
        ...     secret_scopes=frozenset({"api-keys"}),
        ...     allowed_domains=["*.example.com"],
        ...     data_classification=EnumDataClassification.INTERNAL,
        ... )
        >>> enforcer = InvocationSecurityEnforcer(policy)
        >>> enforcer.check_domain_access("api.example.com")  # OK
        >>> enforcer.check_secret_scope_access("api-keys")  # OK
        >>> enforcer.check_classification_constraint(EnumDataClassification.PUBLIC)  # OK
    """

    def __init__(
        self,
        handler_policy: ModelHandlerSecurityPolicy,
        correlation_id: UUID | None = None,
    ) -> None:
        """Initialize enforcer with handler's security policy.

        Creates an immutable enforcer instance. Domain patterns are
        pre-compiled at initialization time for efficient matching.

        Args:
            handler_policy: Handler's declared security policy. This is
                stored as an immutable reference.
            correlation_id: Optional tracing ID for error reporting. If not
                provided, a new UUID4 will be auto-generated.
        """
        self._policy = handler_policy
        # Auto-generate correlation ID if not provided
        self._correlation_id = correlation_id if correlation_id is not None else uuid4()
        # Pre-compile domain patterns for efficiency
        # Make a defensive copy of allowed_domains to ensure immutability
        self._domain_patterns = self._compile_domain_patterns(
            list(handler_policy.allowed_domains)
        )

    def _compile_domain_patterns(
        self, domains: list[str]
    ) -> tuple[re.Pattern[str], ...]:
        """Compile domain patterns for wildcard matching.

        Converts domain allowlist entries into compiled regex patterns
        for efficient matching. Supports two pattern types:
            - Exact match: "api.example.com"
            - Wildcard: "*.example.com" (matches any single subdomain)

        Args:
            domains: List of domain patterns to compile.

        Returns:
            Tuple of compiled regex patterns (immutable).

        Note:
            Wildcard patterns only match a single subdomain level.
            "*.example.com" matches "api.example.com" but NOT
            "a.b.example.com".
        """
        patterns: list[re.Pattern[str]] = []
        for domain in domains:
            if domain.startswith("*."):
                # Convert *.example.com to regex: ^[^.]+\.example\.com$
                # [^.]+ matches one or more non-dot characters (single subdomain)
                suffix = re.escape(domain[2:])
                pattern = re.compile(rf"^[^.]+\.{suffix}$")
            else:
                # Exact match
                pattern = re.compile(rf"^{re.escape(domain)}$")
            patterns.append(pattern)
        return tuple(patterns)  # Return immutable tuple

    def check_domain_access(self, domain: str) -> None:
        """Check if handler is allowed to access the given domain.

        Validates that the target domain matches one of the patterns
        in the handler's allowed_domains list. Supports exact matches
        and wildcard patterns.

        Args:
            domain: Target domain to access (e.g., "api.example.com").

        Returns:
            None if access is allowed.

        Raises:
            SecurityViolationError: If domain access is denied.
                Rule ID: SECURITY-310 (DOMAIN_ACCESS_DENIED)

        Example:
            >>> enforcer.check_domain_access("api.allowed.com")  # OK
            >>> enforcer.check_domain_access("api.forbidden.com")  # Raises
        """
        # Empty allowlist means no outbound access
        if not self._domain_patterns:
            raise SecurityViolationError(
                rule_id=EnumSecurityRuleId.DOMAIN_ACCESS_DENIED,
                message=f"Domain access denied: '{domain}' (no domains allowed)",
                correlation_id=self._correlation_id,
            )

        # Check if domain matches any allowed pattern
        for pattern in self._domain_patterns:
            if pattern.match(domain):
                return  # Access allowed

        # No pattern matched
        raise SecurityViolationError(
            rule_id=EnumSecurityRuleId.DOMAIN_ACCESS_DENIED,
            message=f"Domain access denied: '{domain}' not in allowlist",
            correlation_id=self._correlation_id,
        )

    def check_secret_scope_access(self, scope: str) -> None:
        """Check if handler is allowed to access the given secret scope.

        Validates that the requested secret scope is declared in the
        handler's security policy. Uses exact string matching (no
        wildcards or prefix matching).

        Args:
            scope: Secret scope to access (e.g., "database-creds").

        Returns:
            None if access is allowed.

        Raises:
            SecurityViolationError: If secret scope access denied.
                Rule ID: SECURITY-311 (SECRET_SCOPE_ACCESS_DENIED)

        Example:
            >>> enforcer.check_secret_scope_access("api-keys")  # OK if declared
            >>> enforcer.check_secret_scope_access("undeclared")  # Raises
        """
        if scope not in self._policy.secret_scopes:
            raise SecurityViolationError(
                rule_id=EnumSecurityRuleId.SECRET_SCOPE_ACCESS_DENIED,
                message=f"Secret scope access denied: '{scope}' not declared",
                correlation_id=self._correlation_id,
            )

    def check_classification_constraint(
        self,
        data_classification: EnumDataClassification,
    ) -> None:
        """Check if handler can process data of the given classification.

        Validates that the data classification level does not exceed
        the handler's declared maximum classification level.

        Classification hierarchy (lowest to highest):
            PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED < SECRET < TOP_SECRET

        Args:
            data_classification: Classification of data being processed.

        Returns:
            None if handler can process data at this level.

        Raises:
            SecurityViolationError: If classification exceeds handler's level.
                Rule ID: SECURITY-312 (CLASSIFICATION_CONSTRAINT_VIOLATION)

        Example:
            >>> # Handler declared for INTERNAL
            >>> enforcer.check_classification_constraint(EnumDataClassification.PUBLIC)  # OK
            >>> enforcer.check_classification_constraint(EnumDataClassification.CONFIDENTIAL)  # Raises
        """
        handler_level = _get_security_level(self._policy.data_classification)
        data_level = _get_security_level(data_classification)

        if data_level > handler_level:
            raise SecurityViolationError(
                rule_id=EnumSecurityRuleId.CLASSIFICATION_CONSTRAINT_VIOLATION,
                message=(
                    f"Classification constraint violated: cannot process "
                    f"{data_classification.value} data with handler classified "
                    f"for {self._policy.data_classification.value}"
                ),
                correlation_id=self._correlation_id,
            )


__all__ = [
    "InvocationSecurityEnforcer",
    "SecurityViolationError",
]
