# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for registration-time security validation.

These tests verify the behavior of RegistrationSecurityValidator, which enforces
security policies at handler registration time (before handlers can execute).

Ticket: OMN-1098

Test coverage:
    - Secret scope validation (permitted vs unpermitted scopes)
    - Data classification constraint enforcement
    - Adapter-specific constraints (is_adapter=True handlers)
    - Domain allowlist requirements for adapters
    - Wildcard semantics for secret scopes and domain allowlists

See Also:
    - docs/patterns/security_patterns.md
    - OMN-1098 Linear ticket for implementation details
"""

from __future__ import annotations

# Core enum imports
from omnibase_core.enums import EnumDataClassification

# Infra enum imports
from omnibase_infra.enums import EnumHandlerTypeCategory
from omnibase_infra.enums.enum_environment import EnumEnvironment
from omnibase_infra.enums.enum_security_rule_id import EnumSecurityRuleId

# Security policy models
from omnibase_infra.models.security import (
    ModelEnvironmentPolicy,
    ModelHandlerSecurityPolicy,
)

# Registration security validator
from omnibase_infra.validation.validator_registration_security import (
    RegistrationSecurityValidator,
    validate_handler_registration,
)


class TestSecretScopeValidation:
    """Tests for secret scope validation at registration.

    These tests verify that handlers declaring unpermitted secret scopes
    are rejected at registration time, preventing unauthorized secret access.
    """

    def test_secret_scope_violation_at_registration(self) -> None:
        """Handler declaring unpermitted secret scope should fail registration.

        Expected Error: SECURITY-300

        This test verifies that when a handler requests a secret scope
        (database-creds) that is not permitted by the environment policy,
        the registration validation fails with an appropriate error.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds", "api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(
                {"api-keys"}
            ),  # database-creds NOT permitted
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 1
        assert "database-creds" in secret_errors[0].message

    def test_multiple_secret_scope_violations(self) -> None:
        """Handler with multiple unpermitted scopes should report all violations.

        Verifies that when multiple secret scopes are unpermitted, each
        violation is captured in the error list.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds", "vault-keys", "api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),  # Only api-keys permitted
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        # Should report violations for both database-creds and vault-keys
        assert len(secret_errors) >= 2
        error_messages = " ".join(e.message for e in secret_errors)
        assert "database-creds" in error_messages
        assert "vault-keys" in error_messages

    def test_all_secret_scopes_permitted_passes(self) -> None:
        """Handler with all permitted secret scopes should pass validation."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys", "database-creds"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 0

    def test_empty_secret_scopes_passes(self) -> None:
        """Handler requesting no secrets should pass secret scope validation."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),  # No secrets requested
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 0

    def test_wildcard_secret_scope_permits_any_scope(self) -> None:
        """Wildcard '*' in permitted_secret_scopes should allow any requested scope.

        When the environment policy permits '*', handlers can request any secret
        scope without validation errors. This is useful for development environments.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(
                {"database-creds", "vault-keys", "api-keys", "custom-secret"}
            ),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset({"*"}),  # Wildcard permits everything
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT - No secret scope errors should occur
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 0


class TestClassificationConstraintEnforcement:
    """Tests for data classification validation at registration.

    These tests verify that handlers cannot declare a data classification
    that exceeds the maximum allowed by the environment policy.
    """

    def test_classification_exceeds_environment_max(self) -> None:
        """Handler with classification exceeding environment max should fail.

        Expected Error: SECURITY-301

        This test verifies that when a handler declares a data classification
        (SECRET) that exceeds the environment's maximum allowed classification
        (CONFIDENTIAL), the registration validation fails.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.SECRET,  # Too high
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.STAGING,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,  # Lower than SECRET
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        class_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX
        ]
        assert len(class_errors) == 1

    def test_classification_within_limit_passes(self) -> None:
        """Handler with classification within limit should pass."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        class_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX
        ]
        assert len(class_errors) == 0

    def test_classification_equals_max_passes(self) -> None:
        """Handler with classification equal to max should pass."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.CONFIDENTIAL,  # Exactly at max
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,  # Same as handler
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        class_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX
        ]
        assert len(class_errors) == 0

    def test_top_secret_exceeds_confidential(self) -> None:
        """TOP_SECRET classification should fail against CONFIDENTIAL max."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.TOP_SECRET,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.STAGING,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        class_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX
        ]
        assert len(class_errors) == 1


class TestAdapterSecurityConstraints:
    """Tests for is_adapter=True security constraints.

    Adapters are special handlers that interact with external systems.
    They have stricter security constraints:
    - Cannot request secrets directly (must use Vault integration)
    - Must be EFFECT category (external I/O)
    - Must declare explicit domain allowlists when required
    """

    def test_adapter_rejected_when_requesting_secrets(self) -> None:
        """Adapter handler requesting secrets should fail registration.

        Expected Error: SECURITY-302

        Adapters should not have direct secret access - they should use
        the platform's secret management (Vault) instead.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(
                {"some-secret"}
            ),  # Adapters shouldn't request secrets
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"some-secret"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            adapter_secrets_override_allowed=False,  # Don't allow adapters to have secrets
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        adapter_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS
        ]
        assert len(adapter_errors) == 1

    def test_adapter_with_secrets_allowed_when_override_enabled(self) -> None:
        """Adapter can have secrets when environment allows override."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"some-secret"}),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset({"some-secret"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            adapter_secrets_override_allowed=True,  # Allow adapters to have secrets in dev
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        adapter_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS
        ]
        assert len(adapter_errors) == 0

    def test_adapter_with_none_handler_type_category_raises_error(self) -> None:
        """Adapter with handler_type_category=None should fail registration.

        Expected Error: SECURITY-303

        Adapters MUST explicitly set handler_type_category=EFFECT.
        Leaving it as None bypasses the validation and contradicts the
        documented requirement that adapters must be EFFECT category.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=None,  # Missing - should be EFFECT
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=False,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        category_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY
        ]
        assert len(category_errors) == 1
        # Verify error message mentions the missing category (standardized format)
        assert "got none" in category_errors[0].message.lower()

    def test_adapter_with_non_effect_category_raises_error(self) -> None:
        """Adapter with non-EFFECT handler category should fail registration.

        Expected Error: SECURITY-303

        Adapters by definition perform external I/O and must be classified
        as EFFECT handlers. Using COMPUTE category would be architecturally incorrect.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.COMPUTE,  # Wrong - adapters must be EFFECT
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        category_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY
        ]
        assert len(category_errors) == 1

    def test_adapter_with_nondeterministic_compute_raises_error(self) -> None:
        """Adapter with NONDETERMINISTIC_COMPUTE category should also fail."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        category_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY
        ]
        assert len(category_errors) == 1

    def test_adapter_missing_domain_allowlist(self) -> None:
        """Adapter without explicit domain allowlist should fail when required.

        Expected Error: SECURITY-304

        In production environments, adapters must declare which external
        domains they communicate with for security auditing.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],  # Empty - no explicit allowlist
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,  # Require explicit domains
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 1

    def test_adapter_domain_allowlist_not_required_in_dev(self) -> None:
        """Adapter can skip domain allowlist in development when not required."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],  # Empty - but not required in dev
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=False,  # Not required in dev
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 0

    def test_wildcard_domain_rejected_when_explicit_allowlist_required(self) -> None:
        """Wildcard '*' in allowed_domains should be rejected when explicit allowlist required.

        Expected Error: SECURITY-304

        When require_explicit_domain_allowlist=True, adapters must specify actual
        domain names. Using '*' as a wildcard defeats the purpose of the allowlist
        and should be rejected as equivalent to having no allowlist.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["*"],  # Wildcard - not an explicit allowlist
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,  # Requires explicit domains
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT - Wildcard should be treated as missing explicit allowlist
        assert len(errors) >= 1
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 1

    def test_valid_adapter_passes_validation(self) -> None:
        """Properly configured adapter should pass all validation."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),  # No secrets
            allowed_domains=["api.example.com"],  # Explicit allowlist
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,  # Correct category
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT - Should pass all adapter-specific checks
        adapter_errors = [
            e
            for e in errors
            if e.rule_id
            in {
                EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS,
                EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
                EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST,
            }
        ]
        assert len(adapter_errors) == 0


class TestNonAdapterConstraints:
    """Tests verifying non-adapter handlers are not subject to adapter constraints."""

    def test_non_adapter_can_request_secrets(self) -> None:
        """Non-adapter handler can request secrets if environment permits."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,  # Not an adapter
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"database-creds"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            adapter_secrets_override_allowed=False,  # Only affects adapters
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT - Should not get adapter-specific errors
        adapter_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS
        ]
        assert len(adapter_errors) == 0

    def test_non_adapter_can_be_compute(self) -> None:
        """Non-adapter handler can use COMPUTE category."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,  # Not an adapter
            handler_type_category=EnumHandlerTypeCategory.COMPUTE,  # OK for non-adapters
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        category_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY
        ]
        assert len(category_errors) == 0


class TestRegistrationSecurityValidatorClass:
    """Tests for the RegistrationSecurityValidator class interface."""

    def test_validator_instantiation(self) -> None:
        """Validator should be instantiable with environment policy."""
        # ARRANGE
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        validator = RegistrationSecurityValidator(env_policy)

        # ASSERT
        assert validator is not None
        assert validator.environment_policy == env_policy

    def test_validator_validate_method(self) -> None:
        """Validator.validate() should return errors list."""
        # ARRANGE
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )
        validator = RegistrationSecurityValidator(env_policy)

        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds"}),  # Not permitted
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        # ACT
        errors = validator.validate(handler_policy)

        # ASSERT
        assert isinstance(errors, list)
        assert len(errors) >= 1

    def test_validator_is_valid_method(self) -> None:
        """Validator.is_valid() should return boolean."""
        # ARRANGE
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )
        validator = RegistrationSecurityValidator(env_policy)

        valid_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        invalid_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds"}),  # Not permitted
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        # ACT & ASSERT
        assert validator.is_valid(valid_policy) is True
        assert validator.is_valid(invalid_policy) is False


class TestErrorStructure:
    """Tests verifying error objects have required structure."""

    def test_error_has_rule_id(self) -> None:
        """Validation errors should have rule_id attribute."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"unpermitted-secret"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),  # Nothing permitted
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        assert hasattr(errors[0], "rule_id")
        assert errors[0].rule_id is not None

    def test_error_has_message(self) -> None:
        """Validation errors should have message attribute."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"unpermitted-secret"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        assert hasattr(errors[0], "message")
        assert isinstance(errors[0].message, str)
        assert len(errors[0].message) > 0

    def test_error_has_remediation_hint(self) -> None:
        """Validation errors should have remediation_hint attribute."""
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"unpermitted-secret"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        assert len(errors) >= 1
        assert hasattr(errors[0], "remediation_hint")


class TestSecurityRuleIdValues:
    """Tests verifying security rule ID values follow conventions."""

    def test_registration_rule_ids_in_300_range(self) -> None:
        """Registration security rules should be in SECURITY-300 to SECURITY-399 range."""
        # ARRANGE - Get all registration-related rule IDs
        rule_ids = [
            EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED,
            EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX,
            EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS,
            EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
            EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST,
        ]

        # ASSERT - All should be in 300-399 range
        for rule_id in rule_ids:
            prefix, _, number = rule_id.partition("-")
            assert prefix == "SECURITY"
            assert 300 <= int(number) <= 399, f"Rule {rule_id} not in 300-399 range"

    def test_rule_ids_are_unique(self) -> None:
        """All registration security rule IDs should be unique."""
        rule_ids = [
            EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED,
            EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX,
            EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS,
            EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
            EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST,
        ]

        assert len(rule_ids) == len(set(rule_ids)), "Duplicate rule IDs found"


class TestWildcardSemantics:
    """Tests for wildcard handling in security policies.

    Wildcards provide a way to express "allow all" semantics for development
    environments where strict security constraints are less critical.
    """

    def test_secret_scope_wildcard_allows_all_scopes(self) -> None:
        """Environment with "*" in permitted_secret_scopes should allow all scopes.

        This is the standard pattern for development environments where
        secret isolation is less critical.

        See Also:
            ModelEnvironmentPolicy docstring example showing:
            permitted_secret_scopes=frozenset({"*"})  # All scopes
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database-creds", "vault-keys", "api-keys"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset({"*"}),  # Wildcard allows all
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 0, "Wildcard should allow all secret scopes"

    def test_secret_scope_wildcard_with_other_scopes(self) -> None:
        """Wildcard with other scopes should still allow all scopes.

        If "*" is present in the set, all other values are effectively ignored.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"unpermitted-scope"}),
            allowed_domains=[],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset({"*", "api-keys"}),  # Wildcard + specific
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        secret_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED
        ]
        assert len(secret_errors) == 0, (
            "Wildcard should allow all even with other scopes"
        )

    def test_domain_allowlist_wildcard_rejected_when_explicit_required(self) -> None:
        """Adapter with "*" in allowed_domains should fail when explicit required.

        When require_explicit_domain_allowlist=True, using ["*"] is treated
        as equivalent to missing the domain allowlist.

        Expected Error: SECURITY-304
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["*"],  # Wildcard should be rejected
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 1, "Wildcard domain should be rejected"
        assert "wildcard" in domain_errors[0].message.lower()

    def test_domain_allowlist_wildcard_allowed_when_not_required(self) -> None:
        """Adapter with "*" in allowed_domains should pass when not required.

        In development environments, wildcard domain access may be acceptable.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["*"],  # Wildcard
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.DEVELOPMENT,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=False,  # Not required
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 0, "Wildcard should be allowed when not required"

    def test_domain_allowlist_with_wildcard_and_specific_rejected(self) -> None:
        """Adapter with ["*", "api.example.com"] should fail when explicit required.

        If "*" appears anywhere in the list, the whole list is treated as
        non-explicit, regardless of other entries.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.example.com", "*"],  # Wildcard mixed with specific
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 1, (
            "Wildcard should be rejected even with other domains"
        )

    def test_explicit_domains_passes_when_required(self) -> None:
        """Adapter with explicit domains should pass when required.

        This verifies the positive case - explicit domains without wildcards.
        """
        # ARRANGE
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=["api.example.com", "cdn.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=True,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            require_explicit_domain_allowlist=True,
        )

        # ACT
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT
        domain_errors = [
            e
            for e in errors
            if e.rule_id == EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST
        ]
        assert len(domain_errors) == 0, "Explicit domains should pass"
