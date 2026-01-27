# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Event Bus Configuration Model.

This module provides the model for a node's resolved event bus configuration,
containing lists of topics the node subscribes to and publishes to.

Key Design Decisions:
    1. Topics stored as environment-qualified strings (e.g., "dev.onex.evt.intent-classified.v1")
    2. Metadata fields are tooling-only; routing uses ONLY topic strings
    3. Property methods extract topic strings only for routing lookups
    4. Model is frozen (immutable) with extra="forbid" for safety

Example:
    >>> from omnibase_infra.models.registration import (
    ...     ModelEventBusTopicEntry,
    ...     ModelNodeEventBusConfig,
    ... )
    >>> entry = ModelEventBusTopicEntry(
    ...     topic="dev.onex.evt.intent-classified.v1",
    ...     event_type="ModelIntentClassified",
    ... )
    >>> config = ModelNodeEventBusConfig(subscribe_topics=[entry])
    >>> config.subscribe_topic_strings
    ['dev.onex.evt.intent-classified.v1']
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration.model_event_bus_topic_entry import (
    ModelEventBusTopicEntry,
)


class ModelNodeEventBusConfig(BaseModel):
    """Resolved event bus configuration for registry storage.

    This model holds the resolved, environment-qualified topic strings
    that a node subscribes to and publishes to. It is designed for
    storage in the registry to enable dynamic topic-based routing.

    The property methods (subscribe_topic_strings, publish_topic_strings)
    extract only the topic strings for routing lookups, ignoring all
    metadata fields.

    Attributes:
        subscribe_topics: List of topics the node subscribes to.
        publish_topics: List of topics the node publishes to.

    Example:
        >>> config = ModelNodeEventBusConfig(
        ...     subscribe_topics=[
        ...         ModelEventBusTopicEntry(topic="dev.onex.evt.input.v1"),
        ...     ],
        ...     publish_topics=[
        ...         ModelEventBusTopicEntry(topic="dev.onex.evt.output.v1"),
        ...     ],
        ... )
        >>> config.subscribe_topic_strings
        ['dev.onex.evt.input.v1']
        >>> config.publish_topic_strings
        ['dev.onex.evt.output.v1']
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subscribe_topics: list[ModelEventBusTopicEntry] = Field(
        default_factory=list,
        description="List of topics the node subscribes to.",
    )
    publish_topics: list[ModelEventBusTopicEntry] = Field(
        default_factory=list,
        description="List of topics the node publishes to.",
    )

    @property
    def subscribe_topic_strings(self) -> list[str]:
        """Extract topic strings only, for routing lookups.

        Returns:
            List of environment-qualified topic strings from subscribe_topics.
            Metadata fields are ignored.
        """
        return [entry.topic for entry in self.subscribe_topics]

    @property
    def publish_topic_strings(self) -> list[str]:
        """Extract topic strings only, for routing lookups.

        Returns:
            List of environment-qualified topic strings from publish_topics.
            Metadata fields are ignored.
        """
        return [entry.topic for entry in self.publish_topics]


__all__ = ["ModelNodeEventBusConfig"]
