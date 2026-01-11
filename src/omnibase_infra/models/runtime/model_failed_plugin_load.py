# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Failed Plugin Load Model for Observability Logging.

This module provides ModelFailedPluginLoad which tracks plugin loading failures
for observability purposes.

See Also:
    - PluginLoader: Uses this model for failure tracking
    - ModelPluginLoadSummary: Summary model that includes failed loads

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Plugin Loader observability logging.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelFailedPluginLoad(BaseModel):
    """Represents a plugin that failed to load.

    Captures the contract path and error details for plugins that could not
    be loaded during a batch operation. Used for diagnostics and debugging.

    Attributes:
        contract_path: Path to the contract file that failed to load.
        error_message: Human-readable error description.
        error_code: Structured error code (e.g., HANDLER_LOADER_002).

    Example:
        >>> failed = ModelFailedPluginLoad(
        ...     contract_path=Path("/app/plugins/auth/contract.yaml"),
        ...     error_message="Invalid YAML syntax",
        ...     error_code="HANDLER_LOADER_002",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    contract_path: Path = Field(
        ...,
        description="Path to the contract file that failed to load",
    )
    error_message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error description",
    )
    error_code: str | None = Field(
        default=None,
        description="Structured error code (e.g., HANDLER_LOADER_002)",
    )


__all__ = ["ModelFailedPluginLoad"]
