# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Retrieve Payload Model.

This module provides the Pydantic model for the manifest.retrieve operation
payload, specifying which manifest to retrieve by ID.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestRetrievePayload(BaseModel):
    """Payload for manifest.retrieve operation.

    Specifies the manifest to retrieve by its unique identifier.

    Attributes:
        manifest_id: The unique identifier of the manifest to retrieve.
            This corresponds to the manifest_id field in the stored
            ModelExecutionManifest.

    Example:
        >>> payload = ModelManifestRetrievePayload(
        ...     manifest_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifest_id: UUID = Field(
        description="Unique identifier of the manifest to retrieve.",
    )


__all__: list[str] = ["ModelManifestRetrievePayload"]
