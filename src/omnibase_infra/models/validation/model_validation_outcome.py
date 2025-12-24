# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation outcome model for representing validation results.

This model replaces the tuple pattern `tuple[bool, str | None]` that was used
for validation result returns. By using a single model type, we eliminate
the tuple+union pattern while providing richer validation context.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 2 (OMN-1002).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelValidationOutcome(BaseModel):
    """Outcome of a validation operation.

    This model normalizes validation results into a consistent structure,
    replacing the `tuple[bool, str | None]` pattern with a more expressive
    model that can carry additional context.

    Attributes:
        is_valid: Whether the validation passed.
        error_message: Error message if validation failed, None if passed.

    Example:
        >>> # Successful validation
        >>> outcome = ModelValidationOutcome.success()
        >>> outcome.is_valid
        True
        >>> outcome.error_message is None
        True

        >>> # Failed validation
        >>> outcome = ModelValidationOutcome.failure("Invalid category")
        >>> outcome.is_valid
        False
        >>> outcome.error_message
        'Invalid category'

    .. versionadded:: 0.6.0
    """

    is_valid: bool = Field(
        description="Whether the validation passed.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if validation failed, None if passed.",
    )

    @classmethod
    def success(cls) -> ModelValidationOutcome:
        """Create a successful validation outcome.

        Returns:
            ModelValidationOutcome with is_valid=True and no error message.

        Example:
            >>> outcome = ModelValidationOutcome.success()
            >>> outcome.is_valid
            True

        .. versionadded:: 0.6.0
        """
        return cls(is_valid=True, error_message=None)

    @classmethod
    def failure(cls, error_message: str) -> ModelValidationOutcome:
        """Create a failed validation outcome.

        Args:
            error_message: Description of why validation failed.

        Returns:
            ModelValidationOutcome with is_valid=False and the error message.

        Example:
            >>> outcome = ModelValidationOutcome.failure("Missing required field")
            >>> outcome.is_valid
            False
            >>> outcome.error_message
            'Missing required field'

        .. versionadded:: 0.6.0
        """
        return cls(is_valid=False, error_message=error_message)

    @classmethod
    def from_legacy_result(
        cls, result: tuple[bool, str | None]
    ) -> ModelValidationOutcome:
        """Create from legacy tuple-based validation result.

        This factory method handles the conversion from the old tuple pattern
        to the new model structure.

        Args:
            result: Legacy validation result as (is_valid, error_message).

        Returns:
            ModelValidationOutcome with equivalent values.

        Example:
            >>> ModelValidationOutcome.from_legacy_result((True, None)).is_valid
            True
            >>> ModelValidationOutcome.from_legacy_result(
            ...     (False, "Error")
            ... ).error_message
            'Error'

        .. versionadded:: 0.6.0
        """
        is_valid, error_message = result
        return cls(is_valid=is_valid, error_message=error_message)

    def to_legacy_result(self) -> tuple[bool, str | None]:
        """Convert back to legacy tuple format.

        This method enables gradual migration by allowing conversion back
        to the original format where needed.

        Returns:
            Tuple of (is_valid, error_message).

        Example:
            >>> ModelValidationOutcome.success().to_legacy_result()
            (True, None)
            >>> ModelValidationOutcome.failure("Error").to_legacy_result()
            (False, 'Error')

        .. versionadded:: 0.6.0
        """
        return (self.is_valid, self.error_message)

    def __bool__(self) -> bool:
        """Allow using outcome in boolean context.

        Returns:
            True if validation passed, False otherwise.

        Example:
            >>> if ModelValidationOutcome.success():
            ...     print("Valid!")
            Valid!

        .. versionadded:: 0.6.0
        """
        return self.is_valid

    def raise_if_invalid(self, exception_type: type[Exception] = ValueError) -> None:
        """Raise an exception if validation failed.

        Args:
            exception_type: Type of exception to raise. Defaults to ValueError.

        Raises:
            exception_type: If validation failed, with error_message as the message.

        Example:
            >>> outcome = ModelValidationOutcome.failure("Bad value")
            >>> outcome.raise_if_invalid()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ValueError: Bad value

        .. versionadded:: 0.6.0
        """
        if not self.is_valid:
            raise exception_type(self.error_message)
