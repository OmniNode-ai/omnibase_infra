# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog response model.

Defines the full catalog response published on ``topic-catalog-response``
in reply to a catalog query. Contains the list of topic entries, catalog
metadata, and optional warnings for partial-success scenarios.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation

.. versionadded:: 0.9.0
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelTopicCatalogResponse(BaseModel):
    """Full catalog response with topic entries and metadata.

    Published in response to a ``ModelTopicCatalogQuery``. The
    ``correlation_id`` matches the original query for request-response
    pairing. The ``topics`` tuple contains all matching entries after
    applying query filters.

    Attributes:
        correlation_id: Matches the query's correlation_id for pairing.
        topics: Tuple of topic catalog entries matching the query filters.
        catalog_version: Monotonically increasing version for cache
            invalidation. Incremented on every catalog change.
        node_count: Total number of registered nodes at response time.
        generated_at: UTC timestamp when the response was generated.
        warnings: Tuple of warning messages for partial-success scenarios
            (e.g., ``("Topic X metadata unavailable",)``).
        schema_version: Schema version for forward compatibility.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> response = ModelTopicCatalogResponse(
        ...     correlation_id=uuid4(),
        ...     topics=(),
        ...     catalog_version=1,
        ...     node_count=5,
        ...     generated_at=datetime.now(timezone.utc),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Matches the query's correlation_id for request-response pairing.",
    )
    topics: tuple[ModelTopicCatalogEntry, ...] = Field(
        default_factory=tuple,
        description="Topic catalog entries matching the query filters.",
    )
    catalog_version: int = Field(
        ...,
        ge=0,
        description=(
            "Monotonically increasing catalog version for cache invalidation."
        ),
    )
    node_count: int = Field(
        ...,
        ge=0,
        description="Total number of registered nodes at response time.",
    )
    generated_at: datetime = Field(
        ...,
        description="UTC timestamp when the response was generated.",
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Warning messages for partial-success scenarios.",
    )
    schema_version: int = Field(
        default=1,
        ge=1,
        description="Schema version for forward compatibility.",
    )

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that generated_at is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)


__all__: list[str] = ["ModelTopicCatalogResponse"]
