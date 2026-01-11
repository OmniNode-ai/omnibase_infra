# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Warning Model for Handler Discovery Operations.

This module provides the ModelDiscoveryWarning model for tracking non-fatal
warnings encountered during handler discovery operations.

See Also:
    - ModelDiscoveryResult: Aggregate discovery results
    - ModelDiscoveryError: Fatal discovery errors

.. versionadded:: 0.7.0
    Created as part of OMN-1133 contract-based handler discovery.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelDiscoveryWarning(BaseModel):
    """Represents a non-fatal warning during handler discovery.

    Captures warnings that don't prevent discovery from completing but
    indicate potential issues that should be addressed.

    Attributes:
        warning_code: Structured warning code for categorization.
        message: Human-readable warning description.
        contract_path: Path to the contract file that triggered the warning.
        handler_name: Name of the handler related to the warning.

    Example:
        >>> from pathlib import Path
        >>> warning = ModelDiscoveryWarning(
        ...     warning_code="DEPRECATED_FIELD",
        ...     message="Contract uses deprecated 'handler_class' field",
        ...     contract_path=Path("/app/handlers/legacy/contract.yaml"),
        ... )
        >>> warning.warning_code
        'DEPRECATED_FIELD'

    .. versionadded:: 0.7.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    warning_code: str = Field(
        ...,
        min_length=1,
        description="Structured warning code for categorization",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable warning description",
    )
    contract_path: Path | None = Field(
        default=None,
        description="Path to the contract file that triggered the warning",
    )
    handler_name: str | None = Field(
        default=None,
        description="Name of the handler related to the warning",
    )


__all__ = ["ModelDiscoveryWarning"]
