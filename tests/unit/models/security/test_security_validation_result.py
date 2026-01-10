# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for security validation result models.

Tests for security validation result models (OMN-1137).
These models capture the results of handler security validation including:

- Validation success/failure status
- List of validation errors with structured information
- Helper methods for result interrogation (has_errors, valid)
- Error aggregation and filtering

See Also:
    - ModelSecurityError: Individual validation error model
    - SecurityMetadataValidator: The validator that produces these results
    - EnumSecurityRuleId: Security validation rule identifiers
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumValidationSeverity
from omnibase_infra.models.security import (
    ModelSecurityError,
    ModelSecurityValidationResult,
    ModelSecurityWarning,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_error() -> ModelSecurityError:
    """Create a sample security error."""
    return ModelSecurityError(
        code="EFFECT_MISSING_SECURITY",
        field="security_policy",
        message="EFFECT handler must declare security metadata",
        severity=EnumValidationSeverity.ERROR,
    )


@pytest.fixture
def sample_warning() -> ModelSecurityWarning:
    """Create a sample security warning."""
    return ModelSecurityWarning(
        code="WILDCARD_DOMAIN_WARNING",
        field="allowed_domains",
        message="Handler uses wildcard domain '*' which may be restricted",
    )


# =============================================================================
# Test Classes - ModelSecurityError
# =============================================================================


@pytest.mark.unit
class TestModelSecurityError:
    """Unit tests for ModelSecurityError model."""

    def test_error_creation(self) -> None:
        """Security error can be created with required fields."""
        error = ModelSecurityError(
            code="TEST_ERROR",
            field="test_field",
            message="Test error message",
            severity=EnumValidationSeverity.ERROR,
        )

        assert error.code == "TEST_ERROR"
        assert error.field == "test_field"
        assert error.message == "Test error message"
        assert error.severity == EnumValidationSeverity.ERROR

    def test_error_default_severity(self) -> None:
        """Security error defaults to ERROR severity."""
        error = ModelSecurityError(
            code="TEST_ERROR",
            field="test_field",
            message="Test error message",
        )

        assert error.severity == EnumValidationSeverity.ERROR

    def test_error_is_frozen(self, sample_error: ModelSecurityError) -> None:
        """Security error should be immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_error.message = "Modified message"  # type: ignore[misc]

    def test_error_forbids_extra_fields(self) -> None:
        """Security error should not allow extra fields."""
        with pytest.raises(ValidationError):
            ModelSecurityError(
                code="TEST_ERROR",
                field="test_field",
                message="Test message",
                extra_field="not allowed",  # type: ignore[call-arg]
            )


# =============================================================================
# Test Classes - ModelSecurityWarning
# =============================================================================


@pytest.mark.unit
class TestModelSecurityWarning:
    """Unit tests for ModelSecurityWarning model."""

    def test_warning_creation(self) -> None:
        """Security warning can be created with required fields."""
        warning = ModelSecurityWarning(
            code="TEST_WARNING",
            field="test_field",
            message="Test warning message",
        )

        assert warning.code == "TEST_WARNING"
        assert warning.field == "test_field"
        assert warning.message == "Test warning message"

    def test_warning_is_frozen(self, sample_warning: ModelSecurityWarning) -> None:
        """Security warning should be immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_warning.message = "Modified message"  # type: ignore[misc]


# =============================================================================
# Test Classes - ModelSecurityValidationResult
# =============================================================================


@pytest.mark.unit
class TestModelSecurityValidationResult:
    """Unit tests for ModelSecurityValidationResult model."""

    def test_valid_result_has_no_errors(self) -> None:
        """Valid result should have empty errors list."""
        result = ModelSecurityValidationResult(
            valid=True,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(),
            warnings=(),
        )

        assert result.errors == ()
        assert len(result.errors) == 0
        assert result.valid
        assert not result.has_errors

    def test_has_errors_returns_true_when_errors_present(
        self,
        sample_error: ModelSecurityError,
    ) -> None:
        """has_errors should return True when errors exist."""
        result = ModelSecurityValidationResult(
            valid=False,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(sample_error,),
            warnings=(),
        )

        assert result.has_errors
        assert not result.valid
        assert len(result.errors) == 1
        assert result.errors[0].code == "EFFECT_MISSING_SECURITY"

    def test_result_has_warnings_property(
        self,
        sample_warning: ModelSecurityWarning,
    ) -> None:
        """Result should track warnings separately from errors."""
        result = ModelSecurityValidationResult(
            valid=True,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(),
            warnings=(sample_warning,),
        )

        assert result.has_warnings
        assert len(result.warnings) == 1
        assert result.warnings[0].code == "WILDCARD_DOMAIN_WARNING"

    def test_error_count_property(
        self,
        sample_error: ModelSecurityError,
    ) -> None:
        """error_count should return the number of errors."""
        result = ModelSecurityValidationResult(
            valid=False,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(sample_error, sample_error),
            warnings=(),
        )

        assert result.error_count == 2

    def test_warning_count_property(
        self,
        sample_warning: ModelSecurityWarning,
    ) -> None:
        """warning_count should return the number of warnings."""
        result = ModelSecurityValidationResult(
            valid=True,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
            errors=(),
            warnings=(sample_warning, sample_warning, sample_warning),
        )

        assert result.warning_count == 3

    def test_bool_returns_valid_state(self) -> None:
        """__bool__ should return the valid state."""
        valid_result = ModelSecurityValidationResult(
            valid=True,
            subject="test",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
            errors=(),
            warnings=(),
        )
        assert bool(valid_result) is True
        assert valid_result  # Truthy

        invalid_result = ModelSecurityValidationResult(
            valid=False,
            subject="test",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(
                ModelSecurityError(
                    code="TEST",
                    field="test",
                    message="test",
                    severity=EnumValidationSeverity.ERROR,
                ),
            ),
            warnings=(),
        )
        assert bool(invalid_result) is False
        assert not invalid_result  # Falsy


@pytest.mark.unit
class TestModelSecurityValidationResultFactoryMethods:
    """Tests for factory methods on ModelSecurityValidationResult."""

    def test_success_factory_method(self) -> None:
        """success() factory should create valid result."""
        result = ModelSecurityValidationResult.success(
            subject="my_handler",
            handler_type=EnumHandlerTypeCategory.COMPUTE,
        )

        assert result.valid
        assert not result.has_errors
        assert result.subject == "my_handler"
        assert result.handler_type == EnumHandlerTypeCategory.COMPUTE
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_success_factory_with_warnings(
        self,
        sample_warning: ModelSecurityWarning,
    ) -> None:
        """success() factory can include warnings."""
        result = ModelSecurityValidationResult.success(
            subject="my_handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            warnings=(sample_warning,),
        )

        assert result.valid
        assert not result.has_errors
        assert result.has_warnings
        assert result.warning_count == 1

    def test_failure_factory_method(
        self,
        sample_error: ModelSecurityError,
    ) -> None:
        """failure() factory should create invalid result."""
        result = ModelSecurityValidationResult.failure(
            subject="bad_handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(sample_error,),
        )

        assert not result.valid
        assert result.has_errors
        assert result.subject == "bad_handler"
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT
        assert len(result.errors) == 1

    def test_failure_factory_requires_errors(self) -> None:
        """failure() factory should raise if errors is empty."""
        with pytest.raises(ValueError, match="errors must be non-empty"):
            ModelSecurityValidationResult.failure(
                subject="bad_handler",
                handler_type=EnumHandlerTypeCategory.EFFECT,
                errors=(),
            )

    def test_failure_factory_with_warnings(
        self,
        sample_error: ModelSecurityError,
        sample_warning: ModelSecurityWarning,
    ) -> None:
        """failure() factory can include warnings."""
        result = ModelSecurityValidationResult.failure(
            subject="bad_handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(sample_error,),
            warnings=(sample_warning,),
        )

        assert not result.valid
        assert result.has_errors
        assert result.has_warnings
        assert result.error_count == 1
        assert result.warning_count == 1


@pytest.mark.unit
class TestModelSecurityValidationResultImmutability:
    """Tests for immutability of security validation result models."""

    def test_result_is_frozen(self) -> None:
        """SecurityValidationResult should be immutable (frozen)."""
        result = ModelSecurityValidationResult(
            valid=True,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(),
            warnings=(),
        )

        with pytest.raises(ValidationError):
            result.valid = False  # type: ignore[misc]

    def test_errors_tuple_is_immutable(
        self,
        sample_error: ModelSecurityError,
    ) -> None:
        """errors tuple should not be modifiable."""
        result = ModelSecurityValidationResult(
            valid=False,
            subject="test-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            errors=(sample_error,),
            warnings=(),
        )

        # Tuples are immutable, can't append
        with pytest.raises(AttributeError):
            result.errors.append(sample_error)  # type: ignore[attr-defined]


@pytest.mark.unit
class TestModelSecurityValidationResultIntegration:
    """Integration tests for validation result with real validator."""

    def test_result_from_validator(self) -> None:
        """Result from validator should have correct structure."""
        from omnibase_core.enums import EnumDataClassification

        from omnibase_infra.models.security import ModelHandlerSecurityPolicy
        from omnibase_infra.runtime import SecurityMetadataValidator

        validator = SecurityMetadataValidator()
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset({"database/readonly"}),
            allowed_domains=("api.example.com",),
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=EnumHandlerTypeCategory.EFFECT,
        )

        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=policy,
        )

        # Should be valid
        assert isinstance(result, ModelSecurityValidationResult)
        assert result.valid
        assert not result.has_errors
        assert result.subject == "effect-handler"
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT

    def test_failed_result_from_validator(self) -> None:
        """Failed result from validator should have errors."""
        from omnibase_core.enums import EnumDataClassification

        from omnibase_infra.models.security import ModelHandlerSecurityPolicy
        from omnibase_infra.runtime import SecurityMetadataValidator

        validator = SecurityMetadataValidator()
        # Empty policy for EFFECT handler should fail
        policy = ModelHandlerSecurityPolicy(
            secret_scopes=frozenset(),
            allowed_domains=(),
            data_classification=EnumDataClassification.INTERNAL,
            is_adapter=False,
            handler_type_category=None,
        )

        result = validator.validate(
            handler_name="effect-handler",
            handler_type=EnumHandlerTypeCategory.EFFECT,
            security_policy=policy,
        )

        # Should be invalid
        assert isinstance(result, ModelSecurityValidationResult)
        assert not result.valid
        assert result.has_errors
        assert result.error_count >= 1
        # Check error structure
        error = result.errors[0]
        assert isinstance(error, ModelSecurityError)
        assert error.code
        assert error.field
        assert error.message


__all__ = [
    "TestModelSecurityError",
    "TestModelSecurityWarning",
    "TestModelSecurityValidationResult",
    "TestModelSecurityValidationResultFactoryMethods",
    "TestModelSecurityValidationResultImmutability",
    "TestModelSecurityValidationResultIntegration",
]
