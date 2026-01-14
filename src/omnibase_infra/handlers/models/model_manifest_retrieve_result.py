# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Retrieve Result Model.

This module provides the Pydantic model for the manifest.retrieve operation
result, containing the retrieved manifest or indication of not found.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestRetrieveResult(BaseModel):
    """Result from manifest.retrieve operation.

    Contains the retrieved manifest data and metadata, or indicates that
    the requested manifest was not found.

    Attributes:
        manifest_id: The unique identifier of the manifest that was queried.
        manifest: The serialized ModelExecutionManifest as a dictionary,
            or None if the manifest was not found. When present, this can
            be passed to ModelExecutionManifest.model_validate() to
            reconstruct the original manifest object.
        file_path: The absolute path where the manifest was found,
            or None if the manifest was not found.
        found: True if the manifest was found and retrieved, False if
            no manifest exists with the requested ID.

    Note:
        This model implements custom __bool__ behavior. The boolean value
        of an instance reflects the 'found' status, enabling idiomatic
        conditional checks like:

            result = handler.retrieve(manifest_id)
            if result:  # True only when found=True
                process_manifest(result.manifest)

    Example:
        >>> from uuid import UUID
        >>> result = ModelManifestRetrieveResult(
        ...     manifest_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        ...     manifest={"manifest_id": "...", "node_id": "..."},
        ...     file_path="/data/manifests/2025/01/15/550e8400.json",
        ...     found=True,
        ... )
        >>> print(bool(result))
        True

        >>> not_found = ModelManifestRetrieveResult(
        ...     manifest_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        ...     manifest=None,
        ...     file_path=None,
        ...     found=False,
        ... )
        >>> print(bool(not_found))
        False
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifest_id: UUID = Field(
        description="Unique identifier of the manifest that was queried.",
    )
    manifest: dict[str, object] | None = Field(
        description="Serialized manifest dictionary, or None if not found.",
    )
    file_path: str | None = Field(
        default=None,
        description="Absolute path where manifest was found, or None if not found.",
    )
    found: bool = Field(
        description="True if manifest was found, False otherwise.",
    )

    def __bool__(self) -> bool:
        """Return True if manifest was found.

        Warning:
            This differs from standard Pydantic model behavior where
            bool(model) always returns True. This model returns True
            only when found=True, enabling idiomatic conditional checks.

        Returns:
            True if found=True, False otherwise.
        """
        return self.found


__all__: list[str] = ["ModelManifestRetrieveResult"]
