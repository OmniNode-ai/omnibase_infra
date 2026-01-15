# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security validation result model for security validation.

This module defines the ModelSecurityValidationResult model for capturing
complete security validation outcomes, including all errors and warnings.
This model is used by SecurityMetadataValidator (OMN-1137) to return
structured validation outcomes.

Example:
    >>> from omnibase_infra.enums import EnumHandlerTypeCategory
    >>> from omnibase_infra.models.security import (
    ...     ModelSecurityError,
    ...     ModelSecurityWarning,
    ... )
    >>> result = ModelSecurityValidationResult(
    ...     valid=False,
    ...     subject="my_component",
    ...     handler_type=EnumHandlerTypeCategory.EFFECT,
    ...     errors=[
    ...         ModelSecurityError(
    ...             code="MISSING_SECRET_SCOPE",
    ...             field="secret_scopes",
    ...             message="Required secret scope not declared",
    ...             severity="error",
    ...         )
    ...     ],
    ...     warnings=[],
    ... )
    >>> result.has_errors
    True
    >>> result.error_count
    1

See Also:
    - ModelSecurityError: Security validation error model
    - ModelSecurityWarning: Security validation warning model
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - RegistrationSecurityValidator: Registration-time security validation
    - EnumSecurityRuleId: Security validation rule identifiers
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.security.model_security_error import ModelSecurityError
from omnibase_infra.models.security.model_security_warning import ModelSecurityWarning


class ModelSecurityValidationResult(BaseModel):
    """Result of security validation.

    Captures the complete outcome of validating a component's security
    policy, including all errors and warnings encountered. This model
    provides a structured interface for validation consumers.

    The ``valid`` attribute indicates whether the target passed security
    validation. A target is valid only if there are no errors; warnings
    do not affect validity.

    Attributes:
        valid: Whether the target passed security validation (no errors).
        subject: Name of the validated component or target.
        handler_type: Behavioral classification of the target.
        errors: List of security validation errors.
        warnings: List of security validation warnings.

    Warning:
        **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
        return ``True`` only when ``valid`` is True. This differs from typical
        Pydantic model behavior where ``bool(model)`` always returns ``True`` for
        any valid model instance. Use explicit attribute access if needed::

            if result:  # Uses __bool__, True only if valid
                proceed_with_registration()

            if result is not None:  # Always True for constructed instance
                inspect_result()

    Example:
        >>> from omnibase_infra.enums import EnumHandlerTypeCategory
        >>> result = ModelSecurityValidationResult(
        ...     valid=True,
        ...     subject="safe_component",
        ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
        ...     errors=[],
        ...     warnings=[],
        ... )
        >>> bool(result)
        True
        >>> result.has_errors
        False
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    valid: bool = Field(
        description="Whether the target passed security validation",
    )
    subject: str = Field(
        description="Identifier of the validated component (e.g., handler name)",
    )
    handler_type: EnumHandlerTypeCategory = Field(
        description="Behavioral classification of the handler",
    )
    errors: tuple[ModelSecurityError, ...] = Field(
        default_factory=tuple,
        description="List of security validation errors",
    )
    warnings: tuple[ModelSecurityWarning, ...] = Field(
        default_factory=tuple,
        description="List of security validation warnings",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any validation errors exist.

        Returns:
            True if there are one or more errors, False otherwise.

        Example:
            >>> result = ModelSecurityValidationResult(
            ...     valid=False,
            ...     subject="test",
            ...     handler_type=EnumHandlerTypeCategory.EFFECT,
            ...     errors=(ModelSecurityError(
            ...         code="TEST", field="test", message="test", severity="error"
            ...     ),),
            ...     warnings=(),
            ... )
            >>> result.has_errors
            True
        """
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any validation warnings exist.

        Returns:
            True if there are one or more warnings, False otherwise.

        Example:
            >>> result = ModelSecurityValidationResult(
            ...     valid=True,
            ...     subject="test",
            ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
            ...     errors=(),
            ...     warnings=(ModelSecurityWarning(
            ...         code="TEST", field="test", message="test"
            ...     ),),
            ... )
            >>> result.has_warnings
            True
        """
        return len(self.warnings) > 0

    @property
    def error_count(self) -> int:
        """Get the number of validation errors.

        Returns:
            Count of errors in this result.

        Example:
            >>> result = ModelSecurityValidationResult(
            ...     valid=True,
            ...     subject="test",
            ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
            ...     errors=(),
            ...     warnings=(),
            ... )
            >>> result.error_count
            0
        """
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get the number of validation warnings.

        Returns:
            Count of warnings in this result.

        Example:
            >>> result = ModelSecurityValidationResult(
            ...     valid=True,
            ...     subject="test",
            ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
            ...     errors=(),
            ...     warnings=(),
            ... )
            >>> result.warning_count
            0
        """
        return len(self.warnings)

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``valid`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic conditional checks::

                if result:
                    # Validation passed - proceed with registration
                    register_handler()
                else:
                    # Validation failed - handle errors
                    for error in result.errors:
                        log_error(error)

        Returns:
            True if validation passed, False otherwise.
        """
        return self.valid

    @classmethod
    def success(
        cls,
        subject: str,
        handler_type: EnumHandlerTypeCategory,
        warnings: tuple[ModelSecurityWarning, ...] = (),
    ) -> ModelSecurityValidationResult:
        """Create a successful validation result.

        Factory method for creating a valid result. May include warnings
        but will have no errors.

        Args:
            subject: Name of the validated component or target.
            handler_type: Behavioral classification of the target.
            warnings: Optional warnings to include in the result.

        Returns:
            ModelSecurityValidationResult with valid=True and no errors.

        Example:
            >>> result = ModelSecurityValidationResult.success(
            ...     subject="my_component",
            ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
            ... )
            >>> result.valid
            True
            >>> result.has_errors
            False
        """
        return cls(
            valid=True,
            subject=subject,
            handler_type=handler_type,
            errors=(),
            warnings=warnings,
        )

    @classmethod
    def failure(
        cls,
        subject: str,
        handler_type: EnumHandlerTypeCategory,
        errors: tuple[ModelSecurityError, ...],
        warnings: tuple[ModelSecurityWarning, ...] = (),
    ) -> ModelSecurityValidationResult:
        """Create a failed validation result.

        Factory method for creating an invalid result with errors.

        Args:
            subject: Name of the validated component or target.
            handler_type: Behavioral classification of the target.
            errors: Validation errors (must be non-empty for failure).
            warnings: Optional warnings to include in the result.

        Returns:
            ModelSecurityValidationResult with valid=False and errors.

        Raises:
            ProtocolConfigurationError: If errors is empty (failures must have errors).

        Example:
            >>> error = ModelSecurityError(
            ...     code="TEST",
            ...     field="test",
            ...     message="Test error",
            ...     severity="error",
            ... )
            >>> result = ModelSecurityValidationResult.failure(
            ...     subject="bad_component",
            ...     handler_type=EnumHandlerTypeCategory.EFFECT,
            ...     errors=(error,),
            ... )
            >>> result.valid
            False
            >>> result.has_errors
            True
        """
        if not errors:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="failure",
            )
            raise ProtocolConfigurationError(
                "errors must be non-empty for failure results", context=context
            )
        return cls(
            valid=False,
            subject=subject,
            handler_type=handler_type,
            errors=errors,
            warnings=warnings,
        )


__all__ = [
    "ModelSecurityValidationResult",
]
