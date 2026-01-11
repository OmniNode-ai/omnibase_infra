# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Error Model for Handler Discovery Operations.

This module provides the ModelDiscoveryError model for tracking errors
encountered during handler discovery operations.

See Also:
    - ModelDiscoveryResult: Aggregate discovery results
    - ModelDiscoveryWarning: Non-fatal warnings

.. versionadded:: 0.7.0
    Created as part of OMN-1133 contract-based handler discovery.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelDiscoveryError(BaseModel):
    """Represents an error encountered during handler discovery.

    Captures detailed information about a discovery failure including the
    structured error code, human-readable message, and context about which
    contract or handler caused the error.

    Attributes:
        error_code: Structured error code (e.g., "MODULE_NOT_FOUND", "CLASS_NOT_FOUND").
        message: Human-readable error description.
        contract_path: Path to the contract file that caused the error, if applicable.
        handler_name: Name of the handler that caused the error, if applicable.
        details: Additional structured details about the error.

    Example:
        >>> from pathlib import Path
        >>> error = ModelDiscoveryError(
        ...     error_code="MODULE_NOT_FOUND",
        ...     message="Could not find module 'myapp.handlers.auth'",
        ...     contract_path=Path("/app/handlers/auth/contract.yaml"),
        ...     handler_name="HandlerAuth",
        ... )
        >>> error.error_code
        'MODULE_NOT_FOUND'

    .. versionadded:: 0.7.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    error_code: str = Field(
        ...,
        min_length=1,
        description="Structured error code (e.g., MODULE_NOT_FOUND, CLASS_NOT_FOUND)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error description",
    )
    contract_path: Path | None = Field(
        default=None,
        description="Path to the contract file that caused the error",
    )
    handler_name: str | None = Field(
        default=None,
        description="Name of the handler that caused the error",
    )
    details: dict[str, object] | None = Field(
        default=None,
        description="Additional structured details about the error",
    )


__all__ = ["ModelDiscoveryError"]
