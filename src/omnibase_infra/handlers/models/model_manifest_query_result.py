# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Query Result Model.

This module provides the Pydantic model for the manifest.query operation
result, containing the list of matching manifests or metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.model_manifest_metadata import ModelManifestMetadata


class ModelManifestQueryResult(BaseModel):
    """Result from manifest.query operation.

    Contains matching manifests based on the query filters. Uses separate
    fields for metadata-only and full-manifest results to maintain type safety.

    Attributes:
        manifests: List of matching manifest metadata (when metadata_only=True).
            Empty list when metadata_only=False.
        manifest_data: List of full manifest dictionaries (when metadata_only=False).
            Empty list when metadata_only=True.
        total_count: Number of results returned.
        metadata_only: Whether results contain only metadata.

    Note:
        This model implements custom __bool__ behavior. The boolean value
        of an instance reflects whether any manifests were found, enabling
        idiomatic conditional checks like:

            result = handler.query(payload)
            if result:  # True only when total_count > 0
                for manifest in result.manifests:
                    process(manifest)

    Example:
        >>> result = ModelManifestQueryResult(
        ...     manifests=[ModelManifestMetadata(...)],
        ...     manifest_data=[],
        ...     total_count=1,
        ...     metadata_only=True,
        ... )
        >>> print(bool(result))
        True

        >>> empty_result = ModelManifestQueryResult(
        ...     manifests=[],
        ...     manifest_data=[],
        ...     total_count=0,
        ...     metadata_only=False,
        ... )
        >>> print(bool(empty_result))
        False
    """

    model_config = ConfigDict(
        strict=False,  # Allow dict with various value types
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifests: list[ModelManifestMetadata] = Field(
        default_factory=list,
        description="List of matching manifest metadata (when metadata_only=True).",
    )
    manifest_data: list[dict[str, object]] = Field(
        default_factory=list,
        description="List of full manifest dictionaries (when metadata_only=False).",
    )
    total_count: int = Field(
        ge=0,
        description="Number of results returned.",
    )
    metadata_only: bool = Field(
        description="True if results contain metadata only, False for full manifests.",
    )

    def __bool__(self) -> bool:
        """Return True if any manifests were found.

        Warning:
            This differs from standard Pydantic model behavior where
            bool(model) always returns True. This model returns True
            only when total_count > 0, enabling idiomatic conditional checks.

        Returns:
            True if total_count > 0, False otherwise.
        """
        return self.total_count > 0


__all__: list[str] = ["ModelManifestQueryResult"]
