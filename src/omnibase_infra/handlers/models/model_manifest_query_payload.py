# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Query Payload Model.

This module provides the Pydantic model for the manifest.query operation
payload, supporting flexible filtering by correlation ID, node ID, and
time ranges.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestQueryPayload(BaseModel):
    """Payload for manifest.query operation.

    Specifies filter criteria for querying stored manifests. All filter
    fields are optional and combined with AND logic when multiple are
    specified.

    Attributes:
        correlation_id: Filter by correlation ID. Only manifests with
            this exact correlation_id will be returned.
        node_id: Filter by node ID. Only manifests with this exact
            node_id will be returned.
        created_after: Filter by creation time. Only manifests created
            at or after this timestamp will be returned.
        created_before: Filter by creation time. Only manifests created
            at or before this timestamp will be returned.
        limit: Maximum number of manifests to return. Defaults to 100.
            Use to prevent excessive memory usage on large result sets.
        metadata_only: If True, return only lightweight metadata instead
            of full manifest data. Useful for listing/browsing operations.

    Example:
        >>> from datetime import datetime, timezone
        >>> payload = ModelManifestQueryPayload(
        ...     node_id="mynode",
        ...     created_after=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ...     limit=50,
        ... )
        >>> print(payload.metadata_only)
        False

        >>> metadata_payload = ModelManifestQueryPayload(
        ...     metadata_only=True,
        ...     limit=1000,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID | None = Field(
        default=None,
        description="Filter by exact correlation ID match.",
    )
    node_id: str | None = Field(
        default=None,
        description="Filter by exact node ID match.",
    )
    created_after: datetime | None = Field(
        default=None,
        description="Filter for manifests created at or after this timestamp.",
    )
    created_before: datetime | None = Field(
        default=None,
        description="Filter for manifests created at or before this timestamp.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of manifests to return (1-10000).",
    )
    metadata_only: bool = Field(
        default=False,
        description="If True, return only metadata instead of full manifests.",
    )


__all__: list[str] = ["ModelManifestQueryPayload"]
