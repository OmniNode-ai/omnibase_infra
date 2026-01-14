# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Persistence Handler Configuration Model.

This module provides the Pydantic model for HandlerManifestPersistence initialization
configuration, defining storage path and optional settings.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestPersistenceConfig(BaseModel):
    """Configuration for HandlerManifestPersistence initialization.

    This model defines the configuration parameters required to initialize
    the manifest persistence handler, including the storage path for manifests.

    Attributes:
        storage_path: Base directory for manifest storage. Manifests will be
            stored in date-partitioned subdirectories: {storage_path}/{year}/{month}/{day}/
        max_file_size: Maximum manifest file size in bytes for read operations.
            If None, uses default (50 MB).
        correlation_id: Optional correlation ID for initialization tracing.

    Example:
        >>> config = ModelManifestPersistenceConfig(
        ...     storage_path="/data/manifests",
        ...     max_file_size=100 * 1024 * 1024,  # 100 MB
        ... )
        >>> print(config.storage_path)
        '/data/manifests'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    storage_path: str = Field(
        min_length=1,
        description="Base directory path for manifest storage. "
        "Manifests are stored in date-partitioned subdirectories.",
    )
    max_file_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum manifest file size in bytes for read operations. "
        "If None, uses default (50 MB).",
    )
    correlation_id: UUID | str | None = Field(
        default=None,
        description="Optional correlation ID for initialization tracing.",
    )


__all__: list[str] = ["ModelManifestPersistenceConfig"]
