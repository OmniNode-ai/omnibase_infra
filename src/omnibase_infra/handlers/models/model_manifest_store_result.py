# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Store Result Model.

This module provides the Pydantic model for the manifest.store operation
result, containing storage metadata and idempotency status.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestStoreResult(BaseModel):
    """Result from manifest.store operation.

    Contains metadata about the completed store operation, including
    the storage location and whether a new manifest was created.

    Attributes:
        manifest_id: The unique identifier of the stored manifest,
            extracted from the input manifest's manifest_id field.
        file_path: The resolved absolute filesystem path where the manifest
            was stored (e.g., /var/lib/onex/manifests/2025/01/14/{id}.json).
        created: True if a new manifest was created, False if a manifest
            with the same ID already existed (idempotent operation).
            When False, the existing manifest is not overwritten.
        bytes_written: Number of bytes written to storage. Zero if the
            manifest already existed (idempotent, no overwrite).

    Example:
        >>> result = ModelManifestStoreResult(
        ...     manifest_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        ...     file_path="/var/lib/onex/manifests/2025/01/14/550e8400.json",
        ...     created=True,
        ...     bytes_written=4096,
        ... )
        >>> print(result.created)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifest_id: UUID = Field(
        description="Unique identifier of the stored manifest.",
    )
    file_path: str = Field(
        min_length=1,
        description="Resolved absolute path where the manifest was stored.",
    )
    created: bool = Field(
        description="True if new manifest created, False if already existed (idempotent).",
    )
    bytes_written: int = Field(
        ge=0,
        description="Number of bytes written. Zero if manifest already existed.",
    )


__all__: list[str] = ["ModelManifestStoreResult"]
