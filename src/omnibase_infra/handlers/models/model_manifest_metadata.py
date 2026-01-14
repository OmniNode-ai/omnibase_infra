# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Metadata Model.

This module provides the lightweight metadata model for manifest queries
when full manifest data is not needed.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestMetadata(BaseModel):
    """Lightweight metadata for a stored manifest.

    Used when querying manifests with metadata_only=True to avoid
    loading full manifest data. Contains only the essential fields
    needed for listing, filtering, and identification.

    Attributes:
        manifest_id: The unique identifier of the manifest.
        created_at: Timestamp when the manifest was created/stored.
        correlation_id: The correlation ID from the manifest, if present.
            Used for tracing related operations across the system.
        node_id: The node ID from the manifest, if present.
            Identifies which node generated this manifest.

    Example:
        >>> from datetime import datetime, timezone
        >>> metadata = ModelManifestMetadata(
        ...     manifest_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        ...     created_at=datetime.now(timezone.utc),
        ...     correlation_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        ...     node_id="registration-orchestrator",
        ... )
        >>> print(metadata.manifest_id)
        UUID('550e8400-e29b-41d4-a716-446655440000')
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifest_id: UUID = Field(
        description="Unique identifier of the manifest.",
    )
    created_at: datetime = Field(
        description="Timestamp when the manifest was created/stored.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID from the manifest for operation tracing.",
    )
    node_id: str | None = Field(
        default=None,
        description="Node ID identifying which node generated this manifest.",
    )
    file_path: str = Field(
        min_length=1,
        description="Absolute path to the manifest file.",
    )
    file_size: int = Field(
        ge=0,
        description="Size of the manifest file in bytes.",
    )


__all__: list[str] = ["ModelManifestMetadata"]
