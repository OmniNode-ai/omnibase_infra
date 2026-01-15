# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation outcome model for representing validation results.

This model replaces the tuple pattern `tuple[bool, str | None]` that was used
for validation result returns. By using a single model type, we eliminate
the tuple+union pattern while providing richer validation context.

Note:
    The ``error_message`` field uses an empty string sentinel value instead
    of None to eliminate the ``str | None`` union. Use the ``has_error``
    property to check if an error message exists before accessing it.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 2 (OMN-1002).

.. versionchanged:: 0.6.1
    Changed ``error_message`` from ``str | None`` to ``str`` with empty string
    sentinel for complete union reduction (OMN-1002).
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class ModelValidationOutcome(BaseModel):
    """Outcome of a validation operation.

    This model normalizes validation results into a consistent structure,
    replacing the `tuple[bool, str | None]` pattern with a more expressive
    model that can carry additional context.

    Attributes:
        is_valid: Whether the validation passed.
        error_message: Error message if validation failed, empty string if passed.
            Use the ``has_error`` property to check if an error message exists.

    Example:
        >>> # Successful validation
        >>> outcome = ModelValidationOutcome.success()
        >>> outcome.is_valid
        True
        >>> outcome.has_error
        False
        >>> outcome.error_message
        ''

        >>> # Failed validation
        >>> outcome = ModelValidationOutcome.failure("Invalid category")
        >>> outcome.is_valid
        False
        >>> outcome.has_error
        True
        >>> outcome.error_message
        'Invalid category'

    .. versionadded:: 0.6.0
    .. versionchanged:: 0.6.1
        Changed ``error_message`` from ``str | None`` to ``str`` with empty string
        sentinel value. Added ``has_error`` property.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    is_valid: bool = Field(
        description="Whether the validation passed.",
    )
    error_message: str = Field(
        default="",
        description="Error message if validation failed, empty string if passed. "
        "Use has_error property to check if an error exists.",
    )

    @property
    def has_error(self) -> bool:
        """Check if an error message exists.

        This property should be used to check for an error before accessing
        ``error_message`` to ensure meaningful results.

        Returns:
            True if error_message is non-empty, False otherwise.

        Example:
            >>> ModelValidationOutcome.success().has_error
            False
            >>> ModelValidationOutcome.failure("Error").has_error
            True

        .. versionadded:: 0.6.1
            Added as part of union reduction (OMN-1002) to complement
            the ``error_message`` field's change from ``str | None`` to ``str``.
        """
        return bool(self.error_message)

    @classmethod
    def success(cls) -> ModelValidationOutcome:
        """Create a successful validation outcome.

        Returns:
            ModelValidationOutcome with is_valid=True and empty error message.

        Example:
            >>> outcome = ModelValidationOutcome.success()
            >>> outcome.is_valid
            True
            >>> outcome.has_error
            False

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Returns empty string instead of None for error_message.
        """
        return cls(is_valid=True, error_message="")

    @classmethod
    def failure(cls, error_message: str) -> ModelValidationOutcome:
        """Create a failed validation outcome.

        Args:
            error_message: Description of why validation failed. Must be non-empty.

        Returns:
            ModelValidationOutcome with is_valid=False and the error message.

        Raises:
            ProtocolConfigurationError: If error_message is empty.

        Example:
            >>> outcome = ModelValidationOutcome.failure("Missing required field")
            >>> outcome.is_valid
            False
            >>> outcome.has_error
            True
            >>> outcome.error_message
            'Missing required field'

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Added validation that error_message must be non-empty.
        """
        if not error_message:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="failure",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                "error_message must be non-empty for failure outcomes", context=context
            )
        return cls(is_valid=False, error_message=error_message)

    @classmethod
    def from_legacy_result(
        cls, result: tuple[bool, str | None]
    ) -> ModelValidationOutcome:
        """Create from legacy tuple-based validation result.

        This factory method handles the conversion from the old tuple pattern
        to the new model structure. None values in the error_message are
        converted to empty strings.

        Args:
            result: Legacy validation result as (is_valid, error_message).
                None error_message values are converted to empty strings.

        Returns:
            ModelValidationOutcome with equivalent values.

        Example:
            >>> ModelValidationOutcome.from_legacy_result((True, None)).is_valid
            True
            >>> ModelValidationOutcome.from_legacy_result((True, None)).has_error
            False
            >>> ModelValidationOutcome.from_legacy_result(
            ...     (False, "Error")
            ... ).error_message
            'Error'

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Converts None error_message to empty string.
        """
        is_valid, error_message = result
        return cls(is_valid=is_valid, error_message=error_message or "")

    def to_legacy_result(self) -> tuple[bool, str | None]:
        """Convert back to legacy tuple format.

        This method enables gradual migration by allowing conversion back
        to the original format where needed. Empty string error_message is
        converted back to None.

        Returns:
            Tuple of (is_valid, error_message). Empty string is converted to None.

        Example:
            >>> ModelValidationOutcome.success().to_legacy_result()
            (True, None)
            >>> ModelValidationOutcome.failure("Error").to_legacy_result()
            (False, 'Error')

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Converts empty string error_message to None for backwards compatibility.
        """
        return (self.is_valid, self.error_message or None)

    def __bool__(self) -> bool:
        """Allow using outcome in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``is_valid`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic conditional checks for validation results::

                if outcome:
                    # Validation passed - proceed
                    continue_processing()
                else:
                    # Validation failed - handle error
                    print(outcome.error_message)

            If you need to check model validity instead, use explicit attribute access::

                # Check for validation success (uses __bool__)
                if outcome:
                    ...

                # Check model is valid (always True for constructed instance)
                if outcome is not None:
                    ...

                # Explicit validation check (preferred for clarity)
                if outcome.is_valid:
                    ...

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
            # Use the error_message directly; has_error will be True for
            # properly constructed failure outcomes
            raise exception_type(self.error_message)
