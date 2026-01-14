# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Store Payload Model.

This module provides the Pydantic model for the manifest.store operation
payload, containing the serialized ModelExecutionManifest to persist.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestStorePayload(BaseModel):
    """Payload for manifest.store operation.

    Contains the serialized execution manifest to be persisted. The manifest
    is stored as a dictionary to allow flexible schema evolution while
    maintaining the ability to deserialize back to ModelExecutionManifest.

    Attributes:
        manifest: The serialized ModelExecutionManifest as a dictionary.
            This should be the result of calling .model_dump() on a
            ModelExecutionManifest instance. The manifest_id field within
            the manifest is used as the storage key.

    Example:
        >>> from omnibase_core.models import ModelExecutionManifest
        >>> manifest = ModelExecutionManifest(...)
        >>> payload = ModelManifestStorePayload(
        ...     manifest=manifest.model_dump()
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    manifest: dict[str, object] = Field(
        description="Serialized ModelExecutionManifest dictionary from model_dump().",
    )


__all__: list[str] = ["ModelManifestStorePayload"]
