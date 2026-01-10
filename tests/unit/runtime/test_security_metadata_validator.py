# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for SecurityMetadataValidator.

Tests for SecurityMetadataValidator (OMN-1137).
This validator enforces security metadata requirements based on handler type:

- EFFECT handlers: MUST have security metadata (they perform I/O)
- COMPUTE handlers: MUST NOT have security metadata (pure, deterministic)
- NONDETERMINISTIC_COMPUTE handlers: MUST have security metadata (non-deterministic)

Security Metadata Validation:
    - Valid secret scopes (format, permissions)
    - Valid domain patterns (format, allowlist)
    - Valid port ranges (1-65535)
    - Valid DNS label lengths (max 63 characters)
    - Total domain length (max 253 characters, RFC 1035)
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
from omnibase_infra.runtime import SecurityMetadataValidator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def validator() -> SecurityMetadataValidator:
    """Create a SecurityMetadataValidator instance."""
    return SecurityMetadataValidator()


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
# Test Classes - SecurityMetadataValidator
# =============================================================================


@pytest.mark.unit
class TestSecurityMetadataValidator:
    """Unit tests for SecurityMetadataValidator.

    Tests security validation rules based on handler type category.

    Security Rules:
        - EFFECT handlers MUST declare security requirements
        - COMPUTE handlers MUST NOT have security requirements
        - NONDETERMINISTIC_COMPUTE handlers should have security metadata
    """

    def test_effect_handler_without_security_returns_error(
        self,
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
class TestSecurityMetadataValidatorEdgeCases:
    """Edge case tests for SecurityMetadataValidator."""

    def test_compute_handler_with_only_data_classification_passes(
        self,
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
        validator: SecurityMetadataValidator,
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
class TestSecurityMetadataValidatorPortValidation:
    """Tests for port range validation in SecurityMetadataValidator."""

    def test_valid_port_range_passes(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Valid port numbers (1-65535) should pass validation."""
        valid_domains_with_ports = [
            "localhost:1",  # Minimum valid port
            "localhost:80",
            "localhost:443",
            "localhost:3000",
            "localhost:8080",
            "localhost:65535",  # Maximum valid port
            "api.example.com:8080",
            "*.example.com:443",
        ]

        # Act
        errors = validator.validate_domains(
            valid_domains_with_ports, handler_name="test"
        )

        # Assert - No errors
        assert len(errors) == 0

    def test_port_below_minimum_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Port 0 should fail validation (must be 1-65535)."""
        # Arrange
        invalid_domains = ["localhost:0"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "port 0" in errors[0].message.lower()
        assert "out of valid range" in errors[0].message.lower()

    def test_port_above_maximum_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Port above 65535 should fail validation."""
        # Arrange
        invalid_domains = ["localhost:70000"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "70000" in errors[0].message
        assert "out of valid range" in errors[0].message.lower()

    def test_port_65536_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Port 65536 (just above max) should fail validation."""
        # Arrange
        invalid_domains = ["localhost:65536"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "65536" in errors[0].message
        assert "out of valid range" in errors[0].message.lower()

    def test_full_validation_with_invalid_port(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Full validation should catch invalid port in EFFECT handler."""
        # Arrange
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database/readonly"}),
            allowed_domains=("api.example.com:70000",),  # Invalid port
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

        # Assert
        assert result.has_errors
        assert len(result.errors) == 1
        assert result.errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "70000" in result.errors[0].message


@pytest.mark.unit
class TestSecurityMetadataValidatorDNSLabelValidation:
    """Tests for DNS label length validation in SecurityMetadataValidator."""

    def test_valid_dns_labels_pass(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """DNS labels with 63 characters or less should pass validation."""
        # Arrange - Label with exactly 63 characters
        label_63_chars = "a" * 63
        valid_domains = [
            f"{label_63_chars}.example.com",  # 63 char label
            "short.example.com",  # Normal length
            "a.b.c.d.example.com",  # Multiple short labels
        ]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - No errors
        assert len(errors) == 0

    def test_dns_label_over_63_chars_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """DNS labels with more than 63 characters should fail validation."""
        # Arrange - Label with 64 characters (1 over limit)
        label_64_chars = "a" * 64
        invalid_domains = [f"{label_64_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "exceeds maximum" in errors[0].message
        assert "63 chars" in errors[0].message
        assert "64 chars" in errors[0].message
        # Verify label position is included
        assert "1st label" in errors[0].message

    def test_dns_label_100_chars_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """DNS labels with 100 characters should fail validation."""
        # Arrange - Very long label
        label_100_chars = "x" * 100
        invalid_domains = [f"{label_100_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "exceeds maximum" in errors[0].message
        assert "100 chars" in errors[0].message
        # Verify label position and truncation of long labels
        assert "1st label" in errors[0].message
        # Labels > 35 chars are truncated with '...'
        assert "..." in errors[0].message

    def test_second_label_too_long_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Second DNS label with more than 63 characters should fail."""
        # Arrange - Second label is too long
        label_64_chars = "b" * 64
        invalid_domains = [f"api.{label_64_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "exceeds maximum" in errors[0].message
        # Verify the position indicates it's the second label
        assert "2nd label" in errors[0].message

    def test_wildcard_domain_label_validation(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """DNS label validation should work correctly with wildcard prefix."""
        # Arrange - Wildcard with long label
        label_64_chars = "c" * 64
        invalid_domains = [f"*.{label_64_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "exceeds maximum" in errors[0].message
        # Wildcard is stripped before label validation, so this is the 1st label
        assert "1st label" in errors[0].message

    def test_full_validation_with_long_dns_label(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Full validation should catch long DNS label in EFFECT handler."""
        # Arrange
        label_70_chars = "z" * 70
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database/readonly"}),
            allowed_domains=(f"{label_70_chars}.example.com",),  # Too long label
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

        # Assert
        assert result.has_errors
        assert len(result.errors) == 1
        assert result.errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "exceeds maximum" in result.errors[0].message
        assert "70 chars" in result.errors[0].message
        assert "1st label" in result.errors[0].message


@pytest.mark.unit
class TestSecurityMetadataValidatorTotalDomainLength:
    """Tests for total domain length validation (RFC 1035: max 253 characters)."""

    def test_domain_with_exactly_253_chars_passes(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Domain with exactly 253 characters should pass validation.

        RFC 1035 specifies a maximum total domain name length of 253 characters.
        A domain at exactly this limit should be valid.
        """
        # Arrange - Create a domain with exactly 253 characters
        # Format: label1.label2.label3...labelN
        # Each label can be max 63 chars, so we use multiple labels
        # 63 + 1 + 63 + 1 + 63 + 1 + 61 = 253 (including dots)
        label_63 = "a" * 63
        label_61 = "b" * 61
        domain_253 = f"{label_63}.{label_63}.{label_63}.{label_61}"
        assert len(domain_253) == 253  # Verify our test setup

        valid_domains = [domain_253]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - No errors
        assert len(errors) == 0

    def test_domain_with_254_chars_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Domain with 254 characters should fail validation.

        RFC 1035 specifies a maximum total domain name length of 253 characters.
        A domain exceeding this limit by even 1 character should be invalid.
        """
        # Arrange - Create a domain with exactly 254 characters
        label_63 = "a" * 63
        label_62 = "b" * 62
        domain_254 = f"{label_63}.{label_63}.{label_63}.{label_62}"
        assert len(domain_254) == 254  # Verify our test setup

        invalid_domains = [domain_254]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "254 characters" in errors[0].message
        assert "253 characters" in errors[0].message
        assert "RFC 1035" in errors[0].message

    def test_domain_with_300_chars_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Domain with 300 characters should fail validation."""
        # Arrange - Create a domain with 300+ characters
        # Use multiple 63-char labels plus a shorter one
        label_63 = "x" * 63
        label_43 = "y" * 43
        # 63 + 1 + 63 + 1 + 63 + 1 + 63 + 1 + 43 = 299, add one more char
        domain_300 = f"{label_63}.{label_63}.{label_63}.{label_63}.{'z' * 44}"
        assert len(domain_300) == 300  # Verify our test setup

        invalid_domains = [domain_300]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "300 characters" in errors[0].message
        assert "253 characters" in errors[0].message

    def test_wildcard_domain_length_excludes_prefix(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Wildcard prefix '*.' should not count toward total domain length.

        When a domain has a wildcard prefix like '*.example.com', only
        'example.com' (the actual resolvable part) should be counted.
        """
        # Arrange - Create a wildcard domain where the base is exactly 253 chars
        label_63 = "a" * 63
        label_61 = "b" * 61
        base_domain = f"{label_63}.{label_63}.{label_63}.{label_61}"
        assert len(base_domain) == 253

        wildcard_domain = f"*.{base_domain}"
        assert len(wildcard_domain) == 255  # 2 extra chars for "*."

        valid_domains = [wildcard_domain]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - Should pass because base domain is exactly 253
        assert len(errors) == 0

    def test_wildcard_domain_over_253_fails(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Wildcard domain with base exceeding 253 chars should fail."""
        # Arrange - Create a wildcard domain where the base is 254 chars
        label_63 = "a" * 63
        label_62 = "b" * 62
        base_domain = f"{label_63}.{label_63}.{label_63}.{label_62}"
        assert len(base_domain) == 254

        wildcard_domain = f"*.{base_domain}"

        invalid_domains = [wildcard_domain]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "254 characters" in errors[0].message
        assert "253 characters" in errors[0].message

    def test_domain_with_port_length_validation(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Port should not count toward total domain length.

        Domain length validation should only consider the hostname part,
        not the port suffix.
        """
        # Arrange - Use a standard domain with port
        # This tests that port is not included in length calculation
        hostname = "api.internal.example.com"
        domain_with_port = f"{hostname}:8080"

        valid_domains = [domain_with_port]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - Should pass because hostname is well under 253
        assert len(errors) == 0

    def test_full_validation_with_long_domain(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Full validation should catch domain exceeding 253 chars in EFFECT handler."""
        # Arrange - Create a domain with 260 characters
        # 63 + 1 + 63 + 1 + 63 + 1 + 63 + 1 + 4 = 260
        label_63 = "z" * 63
        label_4 = "w" * 4
        domain_260 = f"{label_63}.{label_63}.{label_63}.{label_63}.{label_4}"
        assert len(domain_260) == 260

        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database/readonly"}),
            allowed_domains=(domain_260,),
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

        # Assert
        assert result.has_errors
        assert len(result.errors) == 1
        assert result.errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "260 characters" in result.errors[0].message
        assert "253 characters" in result.errors[0].message


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


@pytest.mark.unit
class TestSecurityMetadataValidatorOrdinalHelper:
    """Tests for the _ordinal helper method used in DNS label error messages."""

    def test_ordinal_formatting_basic(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Test basic ordinal formatting (1st, 2nd, 3rd, 4th)."""
        assert validator._ordinal(1) == "1st"
        assert validator._ordinal(2) == "2nd"
        assert validator._ordinal(3) == "3rd"
        assert validator._ordinal(4) == "4th"
        assert validator._ordinal(5) == "5th"

    def test_ordinal_formatting_teens(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Test ordinal formatting for teens (11th, 12th, 13th)."""
        # Special cases: 11th, 12th, 13th
        assert validator._ordinal(11) == "11th"
        assert validator._ordinal(12) == "12th"
        assert validator._ordinal(13) == "13th"

    def test_ordinal_formatting_twenties(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Test ordinal formatting for numbers ending in 1, 2, 3 in twenties."""
        assert validator._ordinal(21) == "21st"
        assert validator._ordinal(22) == "22nd"
        assert validator._ordinal(23) == "23rd"
        assert validator._ordinal(24) == "24th"

    def test_third_label_position_in_error_message(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Test that the 3rd label position is correctly identified in error messages."""
        # Arrange - Third label is too long
        label_64_chars = "d" * 64
        invalid_domains = [f"api.internal.{label_64_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert "3rd label" in errors[0].message

    def test_fourth_label_position_in_error_message(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Test that the 4th label position is correctly identified in error messages."""
        # Arrange - Fourth label is too long
        label_64_chars = "e" * 64
        invalid_domains = [f"api.v1.internal.{label_64_chars}.example.com"]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert
        assert len(errors) == 1
        assert "4th label" in errors[0].message


@pytest.mark.unit
class TestSecurityMetadataValidatorURLSchemeDetection:
    """Tests for URL scheme detection using urlparse in SecurityMetadataValidator.

    The validator uses urlparse().scheme for more robust URL detection,
    which handles various URL schemes including those that don't use "://".
    """

    def test_https_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """HTTPS URLs should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["https://api.example.com"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'https'" in errors[0].message

    def test_http_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """HTTP URLs should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["http://api.example.com"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'http'" in errors[0].message

    def test_ftp_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """FTP URLs should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["ftp://files.example.com"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'ftp'" in errors[0].message

    def test_mailto_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """mailto: URLs (without //) should be detected and rejected.

        This tests the robustness of urlparse over simple "://" checking,
        as mailto: URLs use a colon but not double slashes.
        """
        # Act
        errors = validator.validate_domains(
            ["mailto:test@example.com"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'mailto'" in errors[0].message

    def test_file_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """file:// URLs should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["file:///path/to/file"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'file'" in errors[0].message

    def test_ssh_url_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """ssh:// URLs should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["ssh://user@host.example.com"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message
        assert "detected scheme: 'ssh'" in errors[0].message

    def test_valid_domains_not_false_positive(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Valid domain patterns should not trigger false positives.

        Domains with colons (for ports) should not be mistaken for URLs.
        """
        # Arrange - All valid patterns that should NOT be detected as URLs
        valid_domains = [
            "api.example.com",
            "localhost",
            "localhost:3000",
            "api.example.com:8080",
            "*.example.com",
            "*.example.com:443",
            "internal.service.local",
            "*",
        ]

        # Act
        errors = validator.validate_domains(valid_domains, handler_name="test")

        # Assert - No false positives
        assert len(errors) == 0

    def test_multiple_url_schemes_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """Multiple URLs with different schemes should all be detected."""
        # Arrange
        invalid_domains = [
            "https://secure.example.com",
            "http://api.example.com",
            "ftp://files.example.com",
            "mailto:admin@example.com",
        ]

        # Act
        errors = validator.validate_domains(invalid_domains, handler_name="test")

        # Assert - All 4 should be detected
        assert len(errors) == 4
        schemes_detected = [e.message for e in errors]
        assert any("'https'" in msg for msg in schemes_detected)
        assert any("'http'" in msg for msg in schemes_detected)
        assert any("'ftp'" in msg for msg in schemes_detected)
        assert any("'mailto'" in msg for msg in schemes_detected)

    def test_url_with_path_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """URLs with paths should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["https://api.example.com/v1/users"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message

    def test_url_with_query_string_detected(
        self,
        validator: SecurityMetadataValidator,
    ) -> None:
        """URLs with query strings should be detected and rejected."""
        # Act
        errors = validator.validate_domains(
            ["https://api.example.com?key=value"],
            handler_name="test",
        )

        # Assert
        assert len(errors) == 1
        assert errors[0].code == EnumSecurityRuleId.INVALID_DOMAIN_PATTERN.value
        assert "appears to be a full URL" in errors[0].message


__all__ = [
    "TestSecurityMetadataValidator",
    "TestSecurityMetadataValidatorEdgeCases",
    "TestSecurityMetadataValidatorPortValidation",
    "TestSecurityMetadataValidatorDNSLabelValidation",
    "TestSecurityMetadataValidatorTotalDomainLength",
    "TestValidateHandlerSecurityFunction",
    "TestSecurityMetadataValidatorOrdinalHelper",
    "TestSecurityMetadataValidatorURLSchemeDetection",
]
