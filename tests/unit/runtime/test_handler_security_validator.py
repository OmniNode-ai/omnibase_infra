# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerSecurityValidator.

Tests for HandlerSecurityValidator (OMN-1137).
This validator enforces security metadata requirements based on handler type:

- EFFECT handlers: MUST have security metadata (they perform I/O)
- COMPUTE handlers: MUST NOT have security metadata (pure, deterministic)
- NONDETERMINISTIC_COMPUTE handlers: MUST have security metadata (non-deterministic)

Security Metadata Validation:
    - Valid secret scopes (format, permissions)
    - Valid domain patterns (format, allowlist)
    - Data classification levels

See Also:
    - ModelHandlerSecurityPolicy: Security policy model
    - EnumHandlerTypeCategory: Handler behavioral classification
    - EnumSecurityRuleId: Security validation rule identifiers
"""

from __future__ import annotations

import pytest
from omnibase_core.enums import EnumDataClassification

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumSecurityRuleId
from omnibase_infra.models.security import ModelHandlerSecurityPolicy
from omnibase_infra.runtime import HandlerSecurityValidator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def validator() -> HandlerSecurityValidator:
    """Create a HandlerSecurityValidator instance."""
    return HandlerSecurityValidator()


@pytest.fixture
def valid_effect_security_policy() -> ModelHandlerSecurityPolicy:
    """Create a valid security policy for EFFECT handlers."""
    return ModelHandlerSecurityPolicy(
        secret_scopes=frozenset({"database/readonly"}),
        allowed_domains=("api.internal.example.com",),
        data_classification=EnumDataClassification.INTERNAL,
        is_adapter=False,
        handler_type_category=EnumHandlerTypeCategory.EFFECT,
    )


@pytest.fixture
def empty_security_policy() -> ModelHandlerSecurityPolicy:
    """Create an empty security policy (no security requirements)."""
    return ModelHandlerSecurityPolicy(
        secret_scopes=frozenset(),
        allowed_domains=(),
        data_classification=EnumDataClassification.INTERNAL,
        is_adapter=False,
        handler_type_category=None,
    )


@pytest.fixture
def compute_security_policy_with_secrets() -> ModelHandlerSecurityPolicy:
    """Create a security policy with secrets (invalid for COMPUTE)."""
    return ModelHandlerSecurityPolicy(
        secret_scopes=frozenset({"database/readonly"}),
        allowed_domains=("api.example.com",),
        data_classification=EnumDataClassification.INTERNAL,
        is_adapter=False,
        handler_type_category=EnumHandlerTypeCategory.COMPUTE,
    )


@pytest.fixture
def nondeterministic_compute_security_policy() -> ModelHandlerSecurityPolicy:
    """Create a security policy with NONDETERMINISTIC_COMPUTE category."""
    return ModelHandlerSecurityPolicy(
        secret_scopes=frozenset({"uuid-generator"}),
        allowed_domains=(),
        data_classification=EnumDataClassification.INTERNAL,
        is_adapter=False,
        handler_type_category=EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
    )


# =============================================================================
# Test Classes - HandlerSecurityValidator
# =============================================================================


@pytest.mark.unit
class TestHandlerSecurityValidator:
    """Unit tests for HandlerSecurityValidator.

    Tests security validation rules based on handler type category.

    Security Rules:
        - EFFECT handlers MUST declare security requirements
        - COMPUTE handlers MUST NOT have security requirements
        - NONDETERMINISTIC_COMPUTE handlers should have security metadata
    """

    def test_effect_handler_without_security_returns_error(
        self,
        validator: HandlerSecurityValidator,
        empty_security_policy: ModelHandlerSecurityPolicy,
    ) -> None:
        """EFFECT handler without security metadata should fail.

        EFFECT handlers perform external I/O (database, HTTP, etc.) and
        MUST declare their security requirements. A handler without any
        security metadata is a configuration error.
        """
        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=empty_security_policy,
        )

        # Assert
        assert result.has_errors
        assert not result.valid
        assert len(result.errors) == 1
        assert (
            result.errors[0].code
            == EnumSecurityRuleId.EFFECT_MISSING_SECURITY_METADATA.value
        )
        assert "EFFECT" in result.errors[0].message
        assert "security metadata" in result.errors[0].message.lower()

    def test_compute_handler_with_security_returns_error(
        self,
        validator: HandlerSecurityValidator,
        compute_security_policy_with_secrets: ModelHandlerSecurityPolicy,
    ) -> None:
        """COMPUTE handler with security metadata should fail.

        COMPUTE handlers are pure, deterministic functions with no side effects.
        They MUST NOT declare security requirements (secret scopes, domains).
        Having security metadata indicates a misconfigured handler.
        """
        # Act
        result = validator.validate(
            handler_name="compute-handler",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
            security_policy=compute_security_policy_with_secrets,
        )

        # Assert
        assert result.has_errors
        assert not result.valid
        assert len(result.errors) >= 1
        assert (
            result.errors[0].code
            == EnumSecurityRuleId.COMPUTE_HAS_SECURITY_METADATA.value
        )
        assert "COMPUTE" in result.errors[0].message

    def test_valid_effect_security_metadata_passes(
        self,
        validator: HandlerSecurityValidator,
        valid_effect_security_policy: ModelHandlerSecurityPolicy,
    ) -> None:
        """Valid security metadata should pass validation.

        A properly configured EFFECT handler with valid security metadata
        (secret scopes in correct format, valid domain patterns, appropriate
        data classification) should pass all validation checks.
        """
        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=valid_effect_security_policy,
        )

        # Assert
        assert not result.has_errors
        assert result.valid
        assert len(result.errors) == 0

    def test_invalid_secret_scopes_returns_error(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """Invalid secret scopes should fail.

        Secret scopes must be non-empty strings without leading/trailing whitespace.
        """
        # Arrange
        invalid_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"", "  "}),
            allowed_domains=("api.example.com",),
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=invalid_policy,
        )

        # Assert
        assert result.has_errors
        scope_errors = [
            e
            for e in result.errors
            if e.code == EnumSecurityRuleId.INVALID_SECRET_SCOPE.value
        ]
        assert len(scope_errors) >= 1

    def test_invalid_domains_returns_error(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """Invalid domain patterns should fail.

        Domain patterns must be valid hostnames or patterns. Full URLs
        are not allowed.
        """
        # Arrange
        invalid_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database/readonly"}),
            allowed_domains=(
                "https://example.com",  # Has protocol - invalid
                "",  # Empty - invalid
            ),
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=invalid_policy,
        )

        # Assert
        assert result.has_errors
        domain_errors = [
            e
            for e in result.errors
            if e.code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        ]
        assert len(domain_errors) >= 2

    def test_nondeterministic_compute_requires_security(
        self,
        validator: HandlerSecurityValidator,
        empty_security_policy: ModelHandlerSecurityPolicy,
    ) -> None:
        """NONDETERMINISTIC_COMPUTE without security should fail.

        NONDETERMINISTIC_COMPUTE handlers (UUID generation, datetime.now(),
        random.choice()) are treated like EFFECT for security purposes.
        """
        # Act
        result = validator.validate(
            handler_name="nondeterministic-compute-handler",
            handler_type=EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
            security_policy=empty_security_policy,
        )

        # Assert
        assert result.has_errors
        assert len(result.errors) >= 1
        # Should have the same error as EFFECT handler
        assert (
            result.errors[0].code
            == EnumSecurityRuleId.EFFECT_MISSING_SECURITY_METADATA.value
        )

    def test_nondeterministic_compute_with_valid_security_passes(
        self,
        validator: HandlerSecurityValidator,
        nondeterministic_compute_security_policy: ModelHandlerSecurityPolicy,
    ) -> None:
        """NONDETERMINISTIC_COMPUTE with valid security metadata should pass."""
        # Act
        result = validator.validate(
            handler_name="nondeterministic-compute-handler",
            handler_type=EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
            security_policy=nondeterministic_compute_security_policy,
        )

        # Assert
        assert not result.has_errors
        assert result.valid


@pytest.mark.unit
class TestHandlerSecurityValidatorEdgeCases:
    """Edge case tests for HandlerSecurityValidator."""

    def test_compute_handler_with_only_data_classification_passes(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """COMPUTE handler with only data classification should pass.

        Data classification alone (at default level) does not constitute
        "security metadata" that triggers errors. COMPUTE handlers CAN have
        data classification but MUST NOT have secret_scopes or allowed_domains.
        """
        # Arrange
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),  # Empty - OK for COMPUTE
            allowed_domains=(),  # Empty - OK for COMPUTE
            data_classification=EnumDataClassification.INTERNAL,  # Default - OK
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.COMPUTE,
        )

        # Act
        result = validator.validate(
            handler_name="compute-handler",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
            security_policy=policy,
        )

        # Assert - COMPUTE with only default data_classification should pass
        assert not result.has_errors
        assert result.valid

    def test_wildcard_domain_validation(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """Wildcard domain '*' should be validated correctly.

        The wildcard domain '*' is valid and means "allow all domains".
        """
        # Arrange
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"api/general"}),
            allowed_domains=("*",),  # Wildcard - valid format
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=policy,
        )

        # Assert - Wildcard is syntactically valid
        assert not result.has_errors
        assert result.valid

    def test_multiple_validation_errors_aggregated(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """Multiple validation errors should all be collected and returned.

        When multiple validation errors occur, all errors should be
        collected and returned rather than failing on the first error.
        """
        # Arrange - Policy with multiple issues
        invalid_policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"", "  "}),  # Two invalid scopes
            allowed_domains=("https://bad.com", ""),  # Two invalid domains
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        # Act
        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=invalid_policy,
        )

        # Assert - Should collect all errors
        assert result.has_errors
        # At least 4 errors: 2 scope errors + 2 domain errors
        assert len(result.errors) >= 4

    def test_validate_secret_scopes_standalone(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """validate_secret_scopes can be used standalone."""
        # Act
        errors = validator.validate_secret_scopes(
            ["database/readonly", "", "  "],
            handler_name="test-handler",
        )

        # Assert
        assert len(errors) == 2  # Empty and whitespace-only
        for error in errors:
            assert error.code == EnumSecurityRuleId.INVALID_SECRET_SCOPE.value

    def test_validate_domains_standalone(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """validate_domains can be used standalone."""
        # Act
        errors = validator.validate_domains(
            ["api.example.com", "https://bad.com", "", "*.example.com:8080"],
            handler_name="test-handler",
        )

        # Assert - Two invalid: full URL and empty string
        assert len(errors) == 2
        for error in errors:
            assert error.code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value

    def test_valid_domain_patterns(
        self,
        validator: HandlerSecurityValidator,
    ) -> None:
        """Valid domain patterns should not produce errors."""
        # Arrange - All valid patterns
        valid_domains = [
            "api.example.com",
            "localhost",
            "localhost:3000",
            "*.example.com",
            "*.example.com:8080",
            "*",  # Wildcard
        ]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - No errors
        assert len(errors) == 0


@pytest.mark.unit
class TestValidateHandlerSecurityFunction:
    """Tests for the validate_handler_security convenience function."""

    def test_convenience_function_works(self) -> None:
        """validate_handler_security function should work like validator.validate()."""
        from omnibase_infra.runtime import validate_handler_security

        # Arrange
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=(),
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.COMPUTE,
        )

        # Act
        result = validate_handler_security(
            handler_name="compute-handler",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
            security_policy=policy,
        )

        # Assert
        assert result.valid
        assert not result.has_errors


__all__ = [
    "TestHandlerSecurityValidator",
    "TestHandlerSecurityValidatorEdgeCases",
    "TestValidateHandlerSecurityFunction",
]
