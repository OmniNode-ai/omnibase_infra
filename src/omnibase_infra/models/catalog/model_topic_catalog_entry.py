# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog entry model.

Defines a single topic entry in the catalog response. Each entry contains
both the canonical topic suffix and the resolved topic name, along with
publisher/subscriber counts and computed active status.

Design Decisions:
    - D1: Both ``topic_suffix`` (canonical identity) and ``topic_name`` (resolved)
      are included in every entry for debugging and operational visibility.
    - D4: ``is_active`` is computed as ``publisher_count > 0 or subscriber_count > 0``
      and is never accepted as input. Enforced via ``@model_validator(mode="after")``.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation

.. versionadded:: 0.9.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelTopicCatalogEntry(BaseModel):
    """Single topic entry in the catalog.

    Contains identity, metadata, and runtime state for one topic. The
    ``is_active`` field is always computed from publisher/subscriber counts
    and cannot be set directly.

    Attributes:
        topic_suffix: Canonical ONEX 5-segment topic suffix
            (e.g., ``onex.evt.platform.node-registration.v1``).
        topic_name: Resolved Kafka topic name including any tenant/namespace
            prefix.
        description: Human-readable description of the topic purpose.
        partitions: Number of partitions configured for this topic.
        publisher_count: Number of active publishers on this topic.
        subscriber_count: Number of active subscribers on this topic.
        is_active: Computed field: ``True`` if the topic has at least one
            publisher or subscriber. Never accepted as input.
        tags: Optional tags for categorization (e.g., ``("platform", "lifecycle")``).

    Example:
        >>> entry = ModelTopicCatalogEntry(
        ...     topic_suffix="onex.evt.platform.node-registration.v1",
        ...     topic_name="onex.evt.platform.node-registration.v1",
        ...     description="Node registration events",
        ...     partitions=6,
        ...     publisher_count=2,
        ...     subscriber_count=3,
        ... )
        >>> entry.is_active
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    topic_suffix: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Canonical ONEX 5-segment topic suffix.",
    )
    topic_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Resolved Kafka topic name.",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Human-readable description of the topic purpose.",
    )
    partitions: int = Field(
        ...,
        ge=1,
        description="Number of partitions configured for this topic.",
    )
    publisher_count: int = Field(
        default=0,
        ge=0,
        description="Number of active publishers on this topic.",
    )
    subscriber_count: int = Field(
        default=0,
        ge=0,
        description="Number of active subscribers on this topic.",
    )
    is_active: bool = Field(
        default=False,
        description=(
            "Computed field: True if publisher_count > 0 or subscriber_count > 0. "
            "Never accepted as input -- overwritten by model_validator."
        ),
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Optional tags for categorization.",
    )

    @model_validator(mode="after")
    def compute_is_active(self) -> ModelTopicCatalogEntry:
        """Compute ``is_active`` from publisher and subscriber counts.

        This validator enforces D4: ``is_active`` is always derived, never
        accepted as user input. Even if a caller passes ``is_active=True``
        with zero counts, this validator will correct it to ``False``.

        Returns:
            Self with ``is_active`` set to the computed value.
        """
        computed = self.publisher_count > 0 or self.subscriber_count > 0
        # Use object.__setattr__ because model is frozen
        object.__setattr__(self, "is_active", computed)
        return self


__all__: list[str] = ["ModelTopicCatalogEntry"]
