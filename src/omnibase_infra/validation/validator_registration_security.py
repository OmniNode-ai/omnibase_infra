# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration-time security validation for handlers.

Validates handler security policies against environment constraints
at registration time. Part of the two-layer security validation
system (OMN-1098).

Security Rules Enforced:
    SECURITY-300: Secret scope not permitted by environment
    SECURITY-301: Data classification exceeds environment maximum
    SECURITY-302: Adapter handler requesting secrets
    SECURITY-303: Adapter with non-EFFECT handler category
    SECURITY-304: Adapter missing explicit domain allowlist

Architecture:
    This validator operates at registration time, before handlers are
    permitted to execute. It validates handler-declared security policies
    against environment-level constraints. Runtime enforcement (invocation-time
    validation) is handled separately.

    The validator is stateless for the convenience function pattern, but also
    supports a stateful pattern where the environment policy is bound at
    construction time.

See Also:
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - ModelEnvironmentPolicy: Environment-level security constraints
    - EnumSecurityRuleId: Security validation rule identifiers

"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumSecurityRuleId
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.models.security import (
    ModelEnvironmentPolicy,
    ModelHandlerSecurityPolicy,
    get_security_level,
)


class RegistrationSecurityValidator:
    """Validates handler security policies at registration time.

    This validator checks handler-declared security policies against
    environment-level constraints. It can be used in two patterns:

    1. Stateful pattern (bind environment policy at construction):
        >>> validator = RegistrationSecurityValidator(env_policy)
        >>> errors = validator.validate(handler_policy)
        >>> if validator.is_valid(handler_policy): ...

    2. Stateless pattern (use module-level convenience function):
        >>> errors = validate_handler_registration(handler_policy, env_policy)

    The validator is itself stateless (no mutable state), but binding the
    environment policy at construction provides a convenient interface when
    validating multiple handlers against the same environment.

    Attributes:
        environment_policy: The environment policy bound at construction.

    Security Rules:
        SECURITY-300: Secret scope not permitted by environment
        SECURITY-301: Data classification exceeds environment maximum
        SECURITY-302: Adapter handler requesting secrets
        SECURITY-303: Adapter with non-EFFECT handler category
        SECURITY-304: Adapter missing explicit domain allowlist

    """

    def __init__(self, environment_policy: ModelEnvironmentPolicy) -> None:
        """Initialize the validator with environment policy.

        Args:
            environment_policy: Environment-level security constraints.
        """
        self._environment_policy = environment_policy

    @property
    def environment_policy(self) -> ModelEnvironmentPolicy:
        """Get the environment policy bound to this validator."""
        return self._environment_policy

    def validate(
        self,
        handler_policy: ModelHandlerSecurityPolicy,
        handler_identity: ModelHandlerIdentifier | None = None,
    ) -> list[ModelHandlerValidationError]:
        """Validate handler security policy against environment constraints.

        Args:
            handler_policy: Handler-declared security policy.
            handler_identity: Optional handler identity for error context.
                Defaults to "unknown" if not provided.

        Returns:
            List of validation errors (empty if valid).
        """
        return _validate_policies(
            handler_policy, self._environment_policy, handler_identity
        )

    def is_valid(
        self,
        handler_policy: ModelHandlerSecurityPolicy,
        handler_identity: ModelHandlerIdentifier | None = None,
    ) -> bool:
        """Check if handler security policy is valid for this environment.

        Args:
            handler_policy: Handler-declared security policy.
            handler_identity: Optional handler identity for error context.
                Defaults to "unknown" if not provided.

        Returns:
            True if valid (no errors), False otherwise.
        """
        return len(self.validate(handler_policy, handler_identity)) == 0


def _validate_policies(
    handler_policy: ModelHandlerSecurityPolicy,
    env_policy: ModelEnvironmentPolicy,
    handler_identity: ModelHandlerIdentifier | None = None,
) -> list[ModelHandlerValidationError]:
    """Core validation logic for handler security policy against environment.

    This function implements all validation rules. It is used by both the
    RegistrationSecurityValidator class and the convenience function.

    Args:
        handler_policy: Handler-declared security policy.
        env_policy: Environment-level security constraints.
        handler_identity: Optional handler identity for error context.
            Defaults to "unknown" if not provided.

    Returns:
        List of validation errors (empty if valid).
    """
    # Use default "unknown" if no identity provided
    identity = handler_identity or ModelHandlerIdentifier.from_handler_id("unknown")
    errors: list[ModelHandlerValidationError] = []

    # 1. Check secret scopes (SECURITY-300)
    errors.extend(_validate_secret_scopes(handler_policy, env_policy, identity))

    # 2. Check data classification (SECURITY-301)
    errors.extend(_validate_classification(handler_policy, env_policy, identity))

    # 3. Check adapter constraints (SECURITY-302, 303, 304)
    if handler_policy.is_adapter:
        errors.extend(
            _validate_adapter_constraints(handler_policy, env_policy, identity)
        )

    return errors


def _validate_secret_scopes(
    handler_policy: ModelHandlerSecurityPolicy,
    env_policy: ModelEnvironmentPolicy,
    handler_identity: ModelHandlerIdentifier,
) -> list[ModelHandlerValidationError]:
    """Validate secret scopes against environment permissions.

    SECURITY-300: Secret scope not permitted by environment.

    Wildcard Support:
        If permitted_secret_scopes contains "*", all secret scopes are allowed.
        This is intended for development environments where secret isolation
        is less critical.

    Args:
        handler_policy: Handler-declared security policy.
        env_policy: Environment-level security constraints.
        handler_identity: Handler identity for error context.

    Returns:
        List of errors for unpermitted secret scopes.
    """
    errors: list[ModelHandlerValidationError] = []

    # If "*" is in permitted_secret_scopes, all scopes are allowed
    if "*" in env_policy.permitted_secret_scopes:
        return errors

    # Find unpermitted scopes
    unpermitted = handler_policy.secret_scopes - env_policy.permitted_secret_scopes

    # Create error for each unpermitted scope
    for scope in sorted(unpermitted):  # Sort for deterministic output
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED,
            message=(
                f"Secret scope '{scope}' not permitted in "
                f"{env_policy.environment.value} environment"
            ),
            remediation_hint=(
                f"Remove the '{scope}' secret scope from handler policy "
                "or update environment policy to permit this scope"
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)

    return errors


def _validate_classification(
    handler_policy: ModelHandlerSecurityPolicy,
    env_policy: ModelEnvironmentPolicy,
    handler_identity: ModelHandlerIdentifier,
) -> list[ModelHandlerValidationError]:
    """Validate data classification against environment maximum.

    SECURITY-301: Data classification exceeds environment maximum.

    Args:
        handler_policy: Handler-declared security policy.
        env_policy: Environment-level security constraints.
        handler_identity: Handler identity for error context.

    Returns:
        List containing error if classification exceeds max, else empty.
    """
    errors: list[ModelHandlerValidationError] = []

    # Get security levels for comparison
    handler_level = get_security_level(handler_policy.data_classification)
    max_level = get_security_level(env_policy.max_data_classification)

    # Check if handler classification exceeds environment maximum
    if handler_level > max_level:
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX,
            message=(
                f"Handler data classification '{handler_policy.data_classification.value}' "
                f"exceeds environment maximum '{env_policy.max_data_classification.value}' "
                f"in {env_policy.environment.value} environment"
            ),
            remediation_hint=(
                f"Reduce handler data classification to "
                f"'{env_policy.max_data_classification.value}' or below, "
                "or deploy to an environment with higher classification limits"
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)

    return errors


def _validate_adapter_constraints(
    handler_policy: ModelHandlerSecurityPolicy,
    env_policy: ModelEnvironmentPolicy,
    handler_identity: ModelHandlerIdentifier,
) -> list[ModelHandlerValidationError]:
    """Validate adapter-specific security constraints.

    Adapters are special handlers that interact with external systems.
    They have stricter security constraints:
        SECURITY-302: Adapter requesting secrets (unless override allowed)
        SECURITY-303: Adapter with non-EFFECT category
        SECURITY-304: Adapter missing domain allowlist (if required)

    Args:
        handler_policy: Handler-declared security policy (must have is_adapter=True).
        env_policy: Environment-level security constraints.
        handler_identity: Handler identity for error context.

    Returns:
        List of errors for adapter constraint violations.
    """
    errors: list[ModelHandlerValidationError] = []

    # SECURITY-302: Adapter requesting secrets
    if handler_policy.secret_scopes and not env_policy.adapter_secrets_override_allowed:
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS,
            message=(
                "Adapter handler is requesting secret scopes "
                f"({', '.join(sorted(handler_policy.secret_scopes))}) "
                "but adapters should not have direct secret access"
            ),
            remediation_hint=(
                "Remove secret scopes from adapter handler policy and use "
                "platform secret management (Vault) instead, or enable "
                "adapter_secrets_override_allowed in environment policy"
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)

    # SECURITY-303: Adapter with non-EFFECT category (or missing category)
    # Adapters MUST explicitly set handler_type_category=EFFECT
    if handler_policy.handler_type_category is None:
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
            message=(
                "Invalid adapter handler configuration: "
                "expected handler_type_category=EFFECT, got None"
            ),
            remediation_hint=(
                "Set handler_type_category=EnumHandlerTypeCategory.EFFECT for adapter handlers, "
                "or remove is_adapter=True if this is not an adapter"
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)
    elif handler_policy.handler_type_category != EnumHandlerTypeCategory.EFFECT:
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
            message=(
                f"Adapter handler has category '{handler_policy.handler_type_category.value}' "
                "but adapters must be EFFECT category (external I/O handlers)"
            ),
            remediation_hint=(
                "Change handler_type_category to EFFECT for adapter handlers, "
                "or remove is_adapter=True if this is not an adapter"
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)

    # SECURITY-304: Adapter missing domain allowlist
    # Empty allowed_domains or containing "*" wildcard is treated as missing
    # when explicit domain allowlist is required
    has_explicit_domains = (
        bool(handler_policy.allowed_domains)
        and "*" not in handler_policy.allowed_domains
    )
    if env_policy.require_explicit_domain_allowlist and not has_explicit_domains:
        # Determine appropriate message based on violation type
        if not handler_policy.allowed_domains:
            violation_detail = "missing explicit domain allowlist"
        else:
            violation_detail = (
                "using wildcard '*' in domain allowlist (explicit domains required)"
            )

        error = ModelHandlerValidationError.from_security_violation(
            rule_id=EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST,
            message=(
                f"Adapter handler {violation_detail} "
                f"which is required in {env_policy.environment.value} environment"
            ),
            remediation_hint=(
                "Add allowed_domains to handler policy specifying which "
                "external domains the adapter may communicate with. "
                "Wildcard '*' is not permitted when explicit domain allowlist is required."
            ),
            handler_identity=handler_identity,
        )
        errors.append(error)

    return errors


def validate_handler_registration(
    handler_policy: ModelHandlerSecurityPolicy,
    env_policy: ModelEnvironmentPolicy,
    handler_identity: ModelHandlerIdentifier | None = None,
) -> list[ModelHandlerValidationError]:
    """Validate handler security policy against environment constraints.

    Convenience function for one-shot validation without creating a
    validator instance. For validating multiple handlers against the
    same environment, consider using RegistrationSecurityValidator directly.

    Args:
        handler_policy: Handler-declared security policy.
        env_policy: Environment-level security constraints.
        handler_identity: Optional handler identity for error context.
            Defaults to "unknown" if not provided.

    Returns:
        List of validation errors (empty if valid).

    Example:
        >>> from omnibase_infra.models.security import (
        ...     ModelHandlerSecurityPolicy,
        ...     ModelEnvironmentPolicy,
        ... )
        >>> from omnibase_core.enums import EnumDataClassification
        >>> from omnibase_infra.enums import EnumEnvironment
        >>>
        >>> handler_policy = ModelHandlerSecurityPolicy(
        ...     secret_scopes=frozenset({"database-creds"}),
        ...     data_classification=EnumDataClassification.INTERNAL,
        ... )
        >>> env_policy = ModelEnvironmentPolicy(
        ...     environment=EnumEnvironment.PRODUCTION,
        ...     permitted_secret_scopes=frozenset({"api-keys"}),
        ...     max_data_classification=EnumDataClassification.CONFIDENTIAL,
        ... )
        >>> errors = validate_handler_registration(handler_policy, env_policy)
        >>> len(errors)  # 1 error for unpermitted secret scope
        1
    """
    return _validate_policies(handler_policy, env_policy, handler_identity)


__all__ = [
    "RegistrationSecurityValidator",
    "validate_handler_registration",
]
