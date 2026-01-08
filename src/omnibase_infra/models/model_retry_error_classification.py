# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Retry Error Classification Model.

This module provides a Pydantic model for encapsulating the result of error
classification during retry handling in infrastructure handlers.

The model determines:
- Whether to retry the operation
- Whether to record circuit breaker failure
- The appropriate error type to raise after retries exhausted
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_retry_error_category import EnumRetryErrorCategory


class ModelRetryErrorClassification(BaseModel):
    """Result of error classification for retry handling.

    This model encapsulates the decision-making result when classifying
    exceptions during retry loops in infrastructure handlers.

    Attributes:
        category: The error category for retry decision making.
        should_retry: Whether the error is eligible for retry.
        record_circuit_failure: Whether to record circuit breaker failure.
        error_message: Sanitized error message for logging and retry state.

    Example:
        >>> from omnibase_infra.enums import EnumRetryErrorCategory
        >>> classification = ModelRetryErrorClassification(
        ...     category=EnumRetryErrorCategory.TIMEOUT,
        ...     should_retry=True,
        ...     record_circuit_failure=True,
        ...     error_message="Operation timed out after 30s",
        ... )
        >>> classification.should_retry
        True

        The model is immutable (frozen=True) - attempting to modify raises an error:

        >>> classification.should_retry = False  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: ...

    Note:
        The error_message field should contain SANITIZED error descriptions only.
        Never include credentials, tokens, or other sensitive data in this field.
        See CLAUDE.md "Error Sanitization Guidelines" for the security policy.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    category: EnumRetryErrorCategory = Field(
        description="The error category for retry decision making",
    )
    should_retry: bool = Field(
        description="Whether the error is eligible for retry",
    )
    record_circuit_failure: bool = Field(
        description="Whether to record circuit breaker failure",
    )
    error_message: str = Field(
        description="Sanitized error message for logging and retry state",
    )


__all__: list[str] = ["ModelRetryErrorClassification"]
