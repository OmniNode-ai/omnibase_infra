# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Backend Result Model for Registry Effect Operations.

This module provides ModelBackendResult, representing the result of an individual
backend operation (Consul or PostgreSQL) within the dual-registration workflow.

Architecture:
    ModelBackendResult captures the outcome of a single backend operation:
    - success: Whether the operation completed successfully
    - error: Error message if the operation failed (sanitized)
    - duration_ms: Time taken for the operation

    This model is used within ModelRegistryResponse to report per-backend status,
    enabling partial failure detection and targeted retry strategies.

Security:
    Error messages MUST be sanitized before inclusion. Never include:
    - Credentials, connection strings, or secrets
    - Internal IP addresses or hostnames
    - PII (names, emails, etc.)

    See CLAUDE.md "Error Sanitization Guidelines" for complete rules.

Related:
    - ModelRegistryResponse: Uses this model for consul_result and postgres_result
    - NodeRegistryEffect: Effect node that produces these results
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Compiled regex for secret pattern detection (module-level for performance)
# Used by ModelBackendResult._warn_on_potential_secrets validator
_SECRET_PATTERNS: re.Pattern[str] = re.compile(
    r"(password\s*=|api_key\s*=|token\s*=|secret\s*=|credentials|"
    r"connection_string\s*=.*@)",
    re.IGNORECASE,
)

_logger = logging.getLogger(__name__)


class ModelBackendResult(BaseModel):
    """Result of an individual backend operation.

    Captures the outcome of a single backend operation (Consul or PostgreSQL)
    within the dual-registration workflow. Used to enable partial failure
    detection and targeted retry strategies.

    Immutability:
        This model uses frozen=True to ensure backend results are immutable
        once created, supporting safe concurrent access and comparison.

    Attributes:
        success: Whether the backend operation completed successfully.
        error: Sanitized error message if success is False.
        error_code: Optional error code for programmatic handling.
        duration_ms: Time taken for the operation in milliseconds.
        backend_id: Optional identifier for the backend instance.

    Design Note - No ``retries`` Field:
        This model intentionally does NOT include a ``retries`` field because:

        1. **Effect layer dispatches once**: The effect node dispatches to handlers
           exactly once per operation. It does not implement retry loops.
        2. **Handlers own retry logic**: Handlers implement their own retry behavior
           using the ``retry_policy`` configuration from the contract. Retry count
           is internal handler state, not exposed in results.
        3. **Caller-controlled retries**: Callers can use the ``retry_partial_failure``
           operation for explicit retries after partial failures.

        **Important**: At the effect layer, a ``retries`` field would always be 0.
        Retry counts are only meaningful when aggregated by the orchestrator layer,
        which tracks how many times ``retry_partial_failure`` was called.

        For observability of retry attempts:
        - Handlers should emit metrics/logs during internal retry loops
        - Use ``correlation_id`` to correlate retry attempts across logs
        - Orchestrator layer can track ``retry_partial_failure`` operation calls

        See: ``contract.yaml`` error_handling.retry_policy for handler configuration.

    Example:
        >>> result = ModelBackendResult(
        ...     success=True,
        ...     duration_ms=45.2,
        ...     backend_id="consul",
        ... )
        >>> result.success
        True

    Example (failure case):
        >>> result = ModelBackendResult(
        ...     success=False,
        ...     error="Connection refused to database host",
        ...     error_code="DATABASE_CONNECTION_ERROR",
        ...     duration_ms=5000.0,
        ...     backend_id="postgres",
        ... )
        >>> result.success
        False
        >>> result.error
        'Connection refused to database host'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the backend operation completed successfully",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if success is False",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for programmatic handling (e.g., DATABASE_CONNECTION_ERROR)",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_id: str | None = Field(
        default=None,
        description="Optional identifier for the backend instance",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing the operation",
    )

    @field_validator("error", mode="after")
    @classmethod
    def _warn_on_potential_secrets(cls, value: str | None) -> str | None:
        """Defense-in-depth validator to detect potential secret exposure in error messages.

        This validator checks if the error string contains common patterns that might
        indicate accidental secret exposure. It logs a warning if detected but does NOT
        block or modify the value - this is defense-in-depth, not enforcement.

        The actual sanitization should happen at the error source (see CLAUDE.md
        "Error Sanitization Guidelines"). This validator serves as a safety net
        to detect when sanitization was missed.

        Detected Patterns:
            - password= (case insensitive)
            - api_key= (case insensitive)
            - token= (case insensitive)
            - secret= (case insensitive)
            - credentials (case insensitive)
            - connection_string=...@ (indicates embedded credentials)

        Args:
            value: The error string to check, or None.

        Returns:
            The original value unchanged.

        Security:
            This validator intentionally does NOT raise an exception or modify
            the value. It only logs a warning to alert operators that a potential
            secret may have been exposed. The rationale:
            1. Blocking could cause unexpected runtime failures
            2. Modification could mask debugging information
            3. Warning provides observability without breaking functionality
        """
        if value is None:
            return value

        if _SECRET_PATTERNS.search(value):
            _logger.warning(
                "Potential secret pattern detected in error message. "
                "Error messages should be sanitized at the source. "
                "See CLAUDE.md 'Error Sanitization Guidelines' for rules. "
                "Pattern detected in error field of ModelBackendResult."
            )

        return value


__all__ = ["ModelBackendResult"]
