# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for two-layer handler security validation (OMN-1098).

These tests demonstrate the full two-layer security validation flow:
1. Registration-time validation prevents misconfigured handlers
2. Invocation-time enforcement validates runtime operations

The key insight is: if registration validation rejects a handler,
invocation-time enforcement will never be needed for that handler.
This is defense in depth.

Test Categories:
    - TestTwoLayerSecurityFlow: Full registration → invocation flow
    - TestDefenseInDepthIntegration: Defense in depth scenarios
    - TestSharedClassificationLevels: Verifies shared classification mapping
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from omnibase_core.enums import EnumDataClassification

from omnibase_infra.enums import EnumEnvironment, EnumSecurityRuleId
from omnibase_infra.models.security import (
    CLASSIFICATION_SECURITY_LEVELS,
    ModelEnvironmentPolicy,
    ModelHandlerSecurityPolicy,
    get_security_level,
)
from omnibase_infra.runtime.invocation_security_enforcer import (
    InvocationSecurityEnforcer,
    SecurityViolationError,
)
from omnibase_infra.validation.validator_registration_security import (
    RegistrationSecurityValidator,
    validate_handler_registration,
)


class TestTwoLayerSecurityFlow:
    """Tests demonstrating the complete two-layer security validation flow.

    These tests show how registration-time validation and invocation-time
    enforcement work together to provide defense in depth.
    """

    def test_registration_blocks_invalid_handler_so_invocation_never_needed(
        self,
    ) -> None:
        """Registration blocking prevents bad handlers from running.

        When registration validation rejects a handler, that handler
        will never be instantiated or invoked. This test documents
        the expected flow: registration rejection = no runtime risk.
        """
        # ARRANGE - Handler with disallowed secret scope
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"forbidden-scope"}),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"allowed-scope-only"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT - Registration validation
        errors = validate_handler_registration(handler_policy, env_policy)

        # ASSERT - Registration should fail
        assert len(errors) == 1
        assert errors[0].rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED

        # Therefore, InvocationSecurityEnforcer should never be created
        # for this handler in production. The following documents this:
        # enforcer = InvocationSecurityEnforcer(handler_policy)  # NOT CREATED

    def test_valid_registration_allows_compliant_invocation(self) -> None:
        """Valid registration allows handler to operate within policy.

        When registration validation passes, the handler is permitted
        to operate at runtime. Invocation-time enforcement then ensures
        the handler stays within its declared policy.
        """
        # ARRANGE - Valid handler policy within environment constraints
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=["api.example.com", "storage.example.com"],
            data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys", "database-creds"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # ACT - Registration validation passes
        errors = validate_handler_registration(handler_policy, env_policy)
        assert len(errors) == 0  # Registration successful

        # Now handler can be instantiated
        enforcer = InvocationSecurityEnforcer(handler_policy, correlation_id=uuid4())

        # ASSERT - Compliant operations succeed
        enforcer.check_domain_access("api.example.com")
        enforcer.check_secret_scope_access("api-keys")
        enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)

    def test_valid_registration_but_invocation_violation(self) -> None:
        """Handler may register successfully but violate policy at runtime.

        This is why two-layer validation is necessary: a handler may
        declare a policy at registration time but then attempt to
        exceed that policy at runtime. Invocation enforcement catches this.
        """
        # ARRANGE - Valid handler policy
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=["api.example.com"],
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys", "database-creds"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # Registration passes
        errors = validate_handler_registration(handler_policy, env_policy)
        assert len(errors) == 0

        # Handler instantiated
        enforcer = InvocationSecurityEnforcer(handler_policy, correlation_id=uuid4())

        # ASSERT - Runtime violations are caught
        # Handler tries to access domain it didn't declare
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_domain_access("api.unauthorized.com")
        assert exc_info.value.rule_id == EnumSecurityRuleId.DOMAIN_ACCESS_DENIED

        # Handler tries to access secret it didn't declare
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_secret_scope_access("database-creds")
        assert exc_info.value.rule_id == EnumSecurityRuleId.SECRET_SCOPE_ACCESS_DENIED

        # Handler tries to process data above its classification
        with pytest.raises(SecurityViolationError) as exc_info:
            enforcer.check_classification_constraint(
                EnumDataClassification.CONFIDENTIAL
            )
        assert (
            exc_info.value.rule_id
            == EnumSecurityRuleId.CLASSIFICATION_CONSTRAINT_VIOLATION
        )


class TestDefenseInDepthIntegration:
    """Tests demonstrating defense in depth through two-layer validation.

    Defense in depth means: even if one layer fails or is bypassed,
    the other layer still provides protection.
    """

    def test_classification_validated_at_both_layers(self) -> None:
        """Classification is validated at registration AND invocation.

        Registration: Validates handler's declared classification <= env max
        Invocation: Validates actual data classification <= handler's declared
        """
        # Handler declares CONFIDENTIAL classification
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=(),
            data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # Environment allows up to CONFIDENTIAL
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # LAYER 1: Registration validation passes
        errors = validate_handler_registration(handler_policy, env_policy)
        assert len(errors) == 0

        # LAYER 2: Invocation enforcement
        enforcer = InvocationSecurityEnforcer(handler_policy, correlation_id=uuid4())

        # Processing at or below declared level: OK
        enforcer.check_classification_constraint(EnumDataClassification.PUBLIC)
        enforcer.check_classification_constraint(EnumDataClassification.INTERNAL)
        enforcer.check_classification_constraint(EnumDataClassification.CONFIDENTIAL)

        # Processing above declared level: BLOCKED
        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.RESTRICTED)

    def test_multiple_registration_errors_block_all_issues(self) -> None:
        """Registration can catch multiple security issues at once.

        All issues must be fixed before handler can register.
        """
        # Handler with multiple policy violations
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"forbidden-1", "forbidden-2"}),
            allowed_domains=(),
            data_classification=EnumDataClassification.TOP_SECRET,
            is_adapter=True,
            # Missing handler_type_category (adapter needs EFFECT)
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
            adapter_secrets_override_allowed=False,
            require_explicit_domain_allowlist=True,
        )

        # Registration catches all issues
        errors = validate_handler_registration(handler_policy, env_policy)

        # Should have 6 errors total:
        # - 2x SECRET_SCOPE_NOT_PERMITTED (one per forbidden scope)
        # - 1x CLASSIFICATION_EXCEEDS_MAX
        # - 1x ADAPTER_REQUESTING_SECRETS
        # - 1x ADAPTER_NON_EFFECT_CATEGORY (missing handler_type_category)
        # - 1x ADAPTER_MISSING_DOMAIN_ALLOWLIST (empty allowed_domains)
        assert len(errors) == 6

        # Verify all expected error types are present (prevents rule regressions)
        error_rules = {e.rule_id for e in errors}
        expected_rules = {
            EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED,
            EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX,
            EnumSecurityRuleId.ADAPTER_REQUESTING_SECRETS,
            EnumSecurityRuleId.ADAPTER_NON_EFFECT_CATEGORY,
            EnumSecurityRuleId.ADAPTER_MISSING_DOMAIN_ALLOWLIST,
        }
        assert error_rules == expected_rules, (
            f"Expected rules {expected_rules}, got {error_rules}"
        )


class TestSharedClassificationLevels:
    """Tests verifying the shared classification level mapping.

    Both registration validation and invocation enforcement must use
    the same classification level mapping to ensure consistent security
    decisions. This is enforced by using a shared module.
    """

    def test_classification_levels_are_consistent(self) -> None:
        """Verify classification levels are accessible from shared module.

        The CLASSIFICATION_SECURITY_LEVELS mapping is used by both:
        - RegistrationSecurityValidator
        - InvocationSecurityEnforcer

        This test ensures the mapping is correct and accessible.
        """
        # Verify hierarchy order
        assert get_security_level(EnumDataClassification.PUBLIC) == 0
        assert get_security_level(EnumDataClassification.INTERNAL) == 2
        assert get_security_level(EnumDataClassification.CONFIDENTIAL) == 4
        assert get_security_level(EnumDataClassification.SECRET) == 6
        assert get_security_level(EnumDataClassification.TOP_SECRET) == 7

        # Verify ordering relationships
        assert get_security_level(EnumDataClassification.PUBLIC) < get_security_level(
            EnumDataClassification.INTERNAL
        )
        assert get_security_level(EnumDataClassification.INTERNAL) < get_security_level(
            EnumDataClassification.CONFIDENTIAL
        )
        assert get_security_level(
            EnumDataClassification.CONFIDENTIAL
        ) < get_security_level(EnumDataClassification.SECRET)
        assert get_security_level(EnumDataClassification.SECRET) < get_security_level(
            EnumDataClassification.TOP_SECRET
        )

    def test_all_classifications_have_levels(self) -> None:
        """Verify all data classification values are mapped to security levels.

        Missing mappings would cause KeyError at runtime.
        """
        # Get all classification values from the mapping
        mapped_classifications = set(CLASSIFICATION_SECURITY_LEVELS.keys())

        # Verify important classifications are mapped
        important_classifications = {
            EnumDataClassification.PUBLIC,
            EnumDataClassification.INTERNAL,
            EnumDataClassification.CONFIDENTIAL,
            EnumDataClassification.RESTRICTED,
            EnumDataClassification.SECRET,
            EnumDataClassification.TOP_SECRET,
        }

        for classification in important_classifications:
            assert classification in mapped_classifications, (
                f"Classification {classification} not mapped to security level"
            )
            # Should not raise KeyError
            level = get_security_level(classification)
            assert isinstance(level, int)

    def test_registration_and_invocation_use_same_levels(self) -> None:
        """Verify registration and invocation produce consistent results.

        Using the same classification level mapping ensures that:
        - A handler allowed to register for CONFIDENTIAL data
        - Can process CONFIDENTIAL data at runtime
        - And is blocked from RESTRICTED data at both layers
        """
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=(),
            data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # Environment max is exactly what handler declares
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset(),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # Registration: CONFIDENTIAL handler in CONFIDENTIAL-max env
        errors = validate_handler_registration(handler_policy, env_policy)
        assert len(errors) == 0  # Should pass

        # Invocation: processing CONFIDENTIAL data
        enforcer = InvocationSecurityEnforcer(handler_policy, correlation_id=uuid4())
        enforcer.check_classification_constraint(
            EnumDataClassification.CONFIDENTIAL
        )  # Should pass

        # Both layers agree on RESTRICTED being above CONFIDENTIAL
        with pytest.raises(SecurityViolationError):
            enforcer.check_classification_constraint(EnumDataClassification.RESTRICTED)

        # Verify this would also fail registration if handler declared RESTRICTED
        restricted_handler = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=(),
            data_classification=EnumDataClassification.RESTRICTED,
        )
        errors = validate_handler_registration(restricted_handler, env_policy)
        assert len(errors) == 1
        assert errors[0].rule_id == EnumSecurityRuleId.CLASSIFICATION_EXCEEDS_MAX


class TestValidatorStatefulVsStateless:
    """Tests demonstrating both validator usage patterns.

    The RegistrationSecurityValidator supports two patterns:
    1. Stateful: Bind environment policy at construction
    2. Stateless: Pass both policies to convenience function
    """

    def test_stateful_validator_pattern(self) -> None:
        """Test stateful pattern with environment bound at construction."""
        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # Create validator bound to environment
        validator = RegistrationSecurityValidator(env_policy)

        # Validate multiple handlers against same environment
        valid_handler = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=(),
            data_classification=EnumDataClassification.INTERNAL,
        )
        invalid_handler = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"forbidden-scope"}),
            allowed_domains=(),
            data_classification=EnumDataClassification.INTERNAL,
        )

        assert validator.is_valid(valid_handler)
        assert not validator.is_valid(invalid_handler)

        # Detailed errors for invalid handler
        errors = validator.validate(invalid_handler)
        assert len(errors) == 1
        assert errors[0].rule_id == EnumSecurityRuleId.SECRET_SCOPE_NOT_PERMITTED

    def test_stateless_convenience_function_pattern(self) -> None:
        """Test stateless pattern using convenience function."""
        handler_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api-keys"}),
            allowed_domains=(),
            data_classification=EnumDataClassification.INTERNAL,
        )

        env_policy = ModelEnvironmentPolicy(
            environment=EnumEnvironment.PRODUCTION,
            permitted_secret_scopes=frozenset({"api-keys"}),
            max_data_classification=EnumDataClassification.CONFIDENTIAL,
        )

        # One-shot validation
        errors = validate_handler_registration(handler_policy, env_policy)
        assert len(errors) == 0


__all__: list[str] = [
    "TestDefenseInDepthIntegration",
    "TestSharedClassificationLevels",
    "TestTwoLayerSecurityFlow",
    "TestValidatorStatefulVsStateless",
]
