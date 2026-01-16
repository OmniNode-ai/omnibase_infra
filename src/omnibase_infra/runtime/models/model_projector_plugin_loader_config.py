# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Plugin Loader Configuration Model.

This module provides the Pydantic model for projector plugin loader configuration.
All fields are strongly typed to eliminate Any usage and enable proper validation.

Example:
    >>> config = ModelProjectorPluginLoaderConfig(
    ...     graceful_mode=True,
    ...     base_paths=[Path("/workspace/projectors")],
    ... )
    >>> loader = ProjectorPluginLoader(config=config)
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectorPluginLoaderConfig(BaseModel):
    """Configuration model for ProjectorPluginLoader.

    This model encapsulates the configuration parameters for the projector
    plugin loader, separating config from runtime dependencies (container,
    schema_manager, pool).

    Attributes:
        graceful_mode: If True, collect errors and continue discovery.
            If False (default), raise on first error.
        base_paths: Optional list of base paths for security validation.
            Symlinks are only allowed if they resolve within these paths.
            If None, uses the paths provided to discovery methods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    graceful_mode: bool = Field(
        default=False,
        description="If True, collect errors and continue discovery. "
        "If False (default), raise on first error.",
    )
    base_paths: list[Path] | None = Field(
        default=None,
        description="Optional list of base paths for security validation. "
        "Symlinks are only allowed if they resolve within these paths.",
    )
