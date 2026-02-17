# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog change notification model.

Defines the change event published on ``topic-catalog-changed`` when topics
are added or removed from the catalog. Contains delta tuples sorted
alphabetically for deterministic ordering (D7).

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation

.. versionadded:: 0.9.0
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelTopicCatalogChanged(BaseModel):
    """Change notification for the topic catalog.

    Published when the catalog changes (topics added or removed). The delta
    tuples are sorted alphabetically for deterministic ordering per D7.

    Attributes:
        correlation_id: Unique identifier for this change event.
        catalog_version: New catalog version after this change.
        topics_added: Alphabetically sorted tuple of topic suffixes added.
        topics_removed: Alphabetically sorted tuple of topic suffixes removed.
        trigger_node_id: ID of the node that triggered the change, if known.
        trigger_reason: Human-readable reason for the change.
        changed_at: UTC timestamp when the change occurred.
        schema_version: Schema version for forward compatibility.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> changed = ModelTopicCatalogChanged(
        ...     correlation_id=uuid4(),
        ...     catalog_version=2,
        ...     topics_added=("onex.evt.platform.new-topic.v1",),
        ...     topics_removed=(),
        ...     trigger_reason="Node registered with new topic",
        ...     changed_at=datetime.now(timezone.utc),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for this change event.",
    )
    catalog_version: int = Field(
        ...,
        ge=0,
        description="New catalog version after this change.",
    )
    topics_added: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Alphabetically sorted tuple of topic suffixes added.",
    )
    topics_removed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Alphabetically sorted tuple of topic suffixes removed.",
    )
    trigger_node_id: str | None = Field(
        default=None,
        max_length=256,
        description="ID of the node that triggered the change, if known.",
    )
    trigger_reason: str = Field(
        default="",
        max_length=1024,
        description="Human-readable reason for the change.",
    )
    changed_at: datetime = Field(
        ...,
        description="UTC timestamp when the change occurred.",
    )
    schema_version: int = Field(
        default=1,
        ge=1,
        description="Schema version for forward compatibility.",
    )

    @field_validator("changed_at")
    @classmethod
    def validate_changed_at_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that changed_at is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)

    @model_validator(mode="after")
    def sort_delta_tuples(self) -> ModelTopicCatalogChanged:
        """Sort topics_added and topics_removed alphabetically (D7).

        Ensures deterministic ordering regardless of the order in which
        topics are discovered or removed. Uses ``object.__setattr__``
        because the model is frozen.

        Returns:
            Self with sorted delta tuples.
        """
        object.__setattr__(self, "topics_added", tuple(sorted(self.topics_added)))
        object.__setattr__(self, "topics_removed", tuple(sorted(self.topics_removed)))
        return self


__all__: list[str] = ["ModelTopicCatalogChanged"]
