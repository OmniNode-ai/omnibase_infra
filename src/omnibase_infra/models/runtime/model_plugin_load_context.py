# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Plugin Load Context Model for Summary Logging.

This module provides ModelPluginLoadContext which groups the parameters needed
for logging a plugin load summary.

See Also:
    - PluginLoader: Uses this model to pass context to _log_load_summary
    - ModelPluginLoadSummary: The summary model created from this context

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Plugin Loader observability logging.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.runtime.model_failed_plugin_load import (
    ModelFailedPluginLoad,
)
from omnibase_infra.models.runtime.model_loaded_handler import ModelLoadedHandler


class ModelPluginLoadContext(BaseModel):
    """Context for logging a plugin load summary.

    Groups related parameters needed to create a load summary, reducing
    the number of function parameters.

    Attributes:
        operation: The type of load operation (e.g., 'load_from_directory').
        source: The source path or patterns used for discovery.
        total_discovered: Total number of contract files discovered.
        handlers: List of successfully loaded handlers.
        failed_plugins: List of plugins that failed to load.
        duration_seconds: Total operation time in seconds.
        correlation_id: Correlation ID for distributed tracing.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    operation: str = Field(
        ...,
        min_length=1,
        description="Type of load operation performed",
    )
    source: str = Field(
        ...,
        description="Source path or patterns used for discovery",
    )
    total_discovered: int = Field(
        ...,
        ge=0,
        description="Total number of contract files discovered",
    )
    handlers: list[ModelLoadedHandler] = Field(
        default_factory=list,
        description="List of successfully loaded handlers",
    )
    failed_plugins: list[ModelFailedPluginLoad] = Field(
        default_factory=list,
        description="List of plugins that failed to load",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total operation time in seconds",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )


__all__ = ["ModelPluginLoadContext"]
