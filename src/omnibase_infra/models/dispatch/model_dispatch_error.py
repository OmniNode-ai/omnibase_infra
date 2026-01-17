# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Error Model for Error Information Grouping.

This module provides the ModelDispatchError model that groups error-related
fields (error_message, error_code, error_details) into a single sub-model.
This reduces the number of optional fields in parent models like ModelDispatchResult.

Design Pattern:
    ModelDispatchError is a pure data model that captures error information for
    failed dispatch operations:
    - error_message: Human-readable error description
    - error_code: Typed error code from EnumCoreErrorCode
    - error_details: Additional JSON-serializable context

    This model uses sentinel values where possible to minimize union count in
    the codebase (OMN-1002).

Sentinel Values:
    - Empty string ("") for error_message means "not set"
    - None for error_code (unavoidable for enum type safety)
    - Empty dict for error_details means "no additional details"

Thread Safety:
    ModelDispatchError is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Union Reduction:
    This model replaces three separate optional fields with a single
    ``ModelDispatchError | None`` field in parent models, reducing union count
    by 2 per parent model that uses it.

JsonType Recursion Fix (OMN-1274):
    The ``error_details`` field uses ``dict[str, object]`` instead of the
    recursive ``JsonType`` type alias. Here is why:

    **The Original Problem:**
    ``JsonType`` was a recursive type alias::

        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None

    Pydantic 2.x performs eager schema generation at **class definition time**.
    When building JSON schemas for validation, it recursively expands type aliases,
    causing ``RecursionError: maximum recursion depth exceeded``::

        JsonType -> dict[str, JsonType] | list[JsonType] | ...
                 -> dict[str, dict[str, JsonType] | ...] | ...
                 -> ... (infinite expansion in _generate_schema.py)

    **Why dict[str, object] is Correct for error_details:**
    Error details are structured diagnostic information, always represented as
    key-value dictionaries (e.g., ``{"host": "db.example.com", "retry_count": 3}``).
    They do NOT need to support arrays or primitives at the root level.

    Using ``dict[str, object]`` provides:
    - Correct semantics: Error details are always dictionaries
    - Type safety: Pydantic validates the outer structure
    - No recursion: ``object`` avoids the recursive type expansion

    **Caveats:**
    - Values are typed as ``object`` (no static type checking on values)
    - For fields needing full JSON support (any JSON value), use ``JsonType``
      from ``omnibase_core.types`` (now fixed via TypeAlias pattern)

Example:
    >>> from omnibase_core.enums import EnumCoreErrorCode
    >>> from omnibase_infra.models.dispatch import ModelDispatchError
    >>>
    >>> # Create error info
    >>> error = ModelDispatchError(
    ...     error_message="Database connection failed",
    ...     error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
    ...     error_details={"host": "db.example.com", "retry_count": 3},
    ... )
    >>>
    >>> # Check if error has specific info
    >>> error.has_message
    True
    >>> error.has_code
    True

See Also:
    omnibase_infra.models.dispatch.ModelDispatchResult: Uses this for error info
    omnibase_core.enums.EnumCoreErrorCode: Error code enumeration
    ADR: ``docs/decisions/adr-any-type-pydantic-workaround.md`` (historical)
    Pydantic issue: https://github.com/pydantic/pydantic/issues/3278

.. versionadded:: 0.7.0
    Added as part of OMN-1004 Optional Field Audit (Task 4.2).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumCoreErrorCode

# Sentinel values for "not set" state
_SENTINEL_STR: str = ""


class ModelDispatchError(BaseModel):
    """
    Error information for failed dispatch operations.

    Groups error_message, error_code, and error_details into a single model to
    reduce the number of optional fields in parent models. Uses sentinel values
    where possible to minimize union count.

    Sentinel Values:
        - Empty string ("") for error_message means "not set"
        - None for error_code (unavoidable for enum type safety)
        - Empty dict for error_details means "no additional details"

    Null Coercion:
        Constructors accept ``None`` for string fields and convert to sentinel.

    Attributes:
        error_message: Human-readable error description. Empty string if not set.
        error_code: Typed error code from EnumCoreErrorCode. None if not set.
        error_details: Additional JSON-serializable error context.

    Example:
        >>> from omnibase_core.enums import EnumCoreErrorCode
        >>> error = ModelDispatchError(
        ...     error_message="Connection failed",
        ...     error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
        ... )
        >>> error.has_message
        True
        >>> error.has_details  # Empty dict
        False

    .. versionadded:: 0.7.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Error Fields ----
    error_message: str = Field(
        default=_SENTINEL_STR,
        description="Human-readable error description. Empty string if not set.",
    )
    error_code: EnumCoreErrorCode | None = Field(
        default=None,
        description="Typed error code from EnumCoreErrorCode. None if not set.",
    )
    error_details: dict[str, object] = Field(
        default_factory=dict,
        description="Additional JSON-serializable error context.",
    )

    # ---- Validators for None-to-Sentinel Conversion ----
    @field_validator("error_message", mode="before")
    @classmethod
    def _convert_none_to_str_sentinel(cls, v: object) -> str:
        """Convert None to empty string sentinel for null coercion."""
        if v is None:
            return _SENTINEL_STR
        if isinstance(v, str):
            return v
        return str(v)

    # ---- Sentinel Check Properties ----
    @property
    def has_message(self) -> bool:
        """Check if error_message is set (not empty string)."""
        return self.error_message != _SENTINEL_STR

    @property
    def has_code(self) -> bool:
        """Check if error_code is set (not None)."""
        return self.error_code is not None

    @property
    def has_details(self) -> bool:
        """Check if error_details has any entries."""
        return len(self.error_details) > 0

    @property
    def is_empty(self) -> bool:
        """Check if all error fields are unset."""
        return not self.has_message and not self.has_code and not self.has_details

    # ---- Factory Methods ----
    @classmethod
    def empty(cls) -> ModelDispatchError:
        """Create an empty error with all fields unset.

        Returns:
            ModelDispatchError with all fields set to sentinels/None.

        Example:
            >>> error = ModelDispatchError.empty()
            >>> error.is_empty
            True

        .. versionadded:: 0.7.0
        """
        return cls()

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        code: EnumCoreErrorCode | None = None,
        details: dict[str, object] | None = None,
    ) -> ModelDispatchError:
        """Create error info from an exception.

        Extracts the error message from the exception and optionally
        includes an error code and additional details.

        Args:
            exception: The exception to extract error info from.
            code: Optional error code. Defaults to None.
            details: Optional additional details. Defaults to empty dict.

        Returns:
            ModelDispatchError populated from the exception.

        Example:
            >>> try:
            ...     raise ValueError("Something went wrong")
            ... except ValueError as e:
            ...     error = ModelDispatchError.from_exception(e)
            >>> error.error_message
            'Something went wrong'

        .. versionadded:: 0.7.0
        """
        return cls(
            error_message=str(exception),
            error_code=code,
            error_details=details or {},
        )

    @classmethod
    def from_message(
        cls,
        message: str,
        code: EnumCoreErrorCode | None = None,
        details: dict[str, object] | None = None,
    ) -> ModelDispatchError:
        """Create error info from a message string.

        Convenience factory for creating error from a message string.

        Args:
            message: The error message.
            code: Optional error code. Defaults to None.
            details: Optional additional details. Defaults to empty dict.

        Returns:
            ModelDispatchError with provided values.

        Example:
            >>> from omnibase_core.enums import EnumCoreErrorCode
            >>> error = ModelDispatchError.from_message(
            ...     "Connection timeout",
            ...     code=EnumCoreErrorCode.TIMEOUT_ERROR,
            ... )
            >>> error.has_code
            True

        .. versionadded:: 0.7.0
        """
        return cls(
            error_message=message,
            error_code=code,
            error_details=details or {},
        )

    def with_details(self, **kwargs: object) -> ModelDispatchError:
        """Create a copy with additional error details.

        Merges the provided kwargs with existing error_details.

        Args:
            **kwargs: Additional error details to add.

        Returns:
            New ModelDispatchError with merged details.

        Example:
            >>> error = ModelDispatchError(error_message="Failed")
            >>> error = error.with_details(retry_count=3, host="db.example.com")
            >>> error.error_details["retry_count"]
            3

        .. versionadded:: 0.7.0
        """
        merged_details = {**self.error_details, **kwargs}
        return self.model_copy(update={"error_details": merged_details})

    def to_dict(self) -> dict[str, str | int | dict[str, object]]:
        """Convert to dictionary with only set fields.

        Returns a dictionary containing only fields that are set (non-sentinel),
        suitable for logging or JSON serialization.

        Returns:
            Dictionary with error information.

        Example:
            >>> from omnibase_core.enums import EnumCoreErrorCode
            >>> error = ModelDispatchError(
            ...     error_message="Failed",
            ...     error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            ... )
            >>> d = error.to_dict()
            >>> "error_message" in d
            True
            >>> "error_details" in d  # Empty dict excluded
            False

        .. versionadded:: 0.7.0
        """
        result: dict[str, str | int | dict[str, object]] = {}
        if self.has_message:
            result["error_message"] = self.error_message
        if self.has_code:
            # NOTE: has_code guard ensures error_code is not None,
            # but mypy cannot narrow the Optional[Enum] type in this context.
            result["error_code"] = self.error_code.name  # type: ignore[union-attr]  # NOTE: property guard narrowing limitation
        if self.has_details:
            result["error_details"] = self.error_details
        return result


__all__ = ["ModelDispatchError"]
