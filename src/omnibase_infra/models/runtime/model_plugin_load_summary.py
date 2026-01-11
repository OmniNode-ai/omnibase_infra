# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Plugin Load Summary Model for Observability Logging.

This module provides ModelPluginLoadSummary which tracks the results of batch
plugin loading operations for observability purposes.

The summary captures:
- Total plugins discovered and loaded
- Details of any failed loads
- Load duration for performance monitoring
- Correlation ID for distributed tracing

See Also:
    - PluginLoader: Uses this model for summary logging
    - ModelLoadedHandler: Individual plugin metadata

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Plugin Loader observability logging.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.runtime.model_failed_plugin_load import (
    ModelFailedPluginLoad,
)


class ModelPluginLoadSummary(BaseModel):
    """Summary of a batch plugin loading operation.

    Captures comprehensive metrics about a batch loading operation for
    observability, debugging, and performance monitoring.

    Attributes:
        operation: The type of load operation (e.g., 'load_from_directory',
            'discover_and_load').
        source: The source path or patterns used for discovery.
        total_discovered: Total number of contract files discovered.
        total_loaded: Number of plugins successfully loaded.
        total_failed: Number of plugins that failed to load.
        loaded_plugins: List of successfully loaded plugin details (name, class, module).
        failed_plugins: List of failed plugin details with error information.
        duration_seconds: Total time taken for the operation in seconds.
        correlation_id: Correlation ID for distributed tracing.
        completed_at: Timestamp when the operation completed.

    Example:
        >>> from uuid import uuid4
        >>> summary = ModelPluginLoadSummary(
        ...     operation="load_from_directory",
        ...     source="/app/plugins",
        ...     total_discovered=5,
        ...     total_loaded=4,
        ...     total_failed=1,
        ...     loaded_plugins=[
        ...         {"name": "auth.plugin", "class": "AuthPlugin", "module": "app.plugins.auth"},
        ...     ],
        ...     failed_plugins=[ModelFailedPluginLoad(...)],
        ...     duration_seconds=0.23,
        ...     correlation_id=uuid4(),
        ...     completed_at=datetime.now(UTC),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
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
    total_loaded: int = Field(
        ...,
        ge=0,
        description="Number of plugins successfully loaded",
    )
    total_failed: int = Field(
        ...,
        ge=0,
        description="Number of plugins that failed to load",
    )
    loaded_plugins: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of loaded plugin details (name, class, module)",
    )
    failed_plugins: list[ModelFailedPluginLoad] = Field(
        default_factory=list,
        description="List of failed plugin details with error information",
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
    completed_at: datetime = Field(
        ...,
        description="Timestamp when the operation completed",
    )


__all__ = ["ModelPluginLoadSummary"]
