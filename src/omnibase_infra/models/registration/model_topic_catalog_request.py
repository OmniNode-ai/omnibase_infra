# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Topic catalog request model for introspection-based catalog queries.

Defines the request payload published on ``request-introspection.v1``
to trigger a catalog response listing all registered ``onex.evt.*`` topics.

Related Tickets:
    - OMN-2923: Catalog responder for topic-catalog-request.v1

.. versionadded:: 0.11.0
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelTopicCatalogRequest(BaseModel):
    """Request model for topic catalog introspection queries.

    Published on ``onex.cmd.platform.request-introspection.v1`` to trigger
    the registration orchestrator to return a snapshot of all ``onex.evt.*``
    topics from registered nodes.

    Attributes:
        correlation_id: Correlation ID echoed back in the response for
            request-response pairing.
        requested_at: UTC timestamp when the request was created.
        requester: Optional identifier of the requesting client.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> request = ModelTopicCatalogRequest(
        ...     correlation_id=uuid4(),
        ...     requested_at=datetime.now(timezone.utc),
        ...     requester="omnidash",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed back in the response for request-response pairing.",
    )
    requested_at: datetime = Field(
        ...,
        description="UTC timestamp when the request was created (must be timezone-aware).",
    )
    requester: str | None = Field(
        default=None,
        description="Optional identifier of the requesting client.",
    )

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that requested_at is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)


__all__: list[str] = ["ModelTopicCatalogRequest"]
