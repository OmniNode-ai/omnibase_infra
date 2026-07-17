# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Generated managed-staging canary catalog model (OMN-14727, B7).

The concrete canary catalog: the topics to create + the consumer groups to use,
all minted under the canary prefix. ``topics`` carry the **full** concrete
canary topic name in :attr:`ModelTopicSpec.suffix` (prefix already applied)
because these specs are what a topic provisioner would create on the cluster.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.topics.model_topic_spec import ModelTopicSpec


class ModelCanaryCatalog(BaseModel):
    """The generated canary catalog: topics to create + consumer groups to use.

    Attributes:
        epoch: Epoch token this catalog was minted under.
        topic_prefix: Common prefix on every topic in this catalog.
        group_prefix: Common prefix on every group in this catalog.
        topics: Per-topic creation specs (full canary names).
        groups: Full canary consumer group names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    epoch: str
    topic_prefix: str
    group_prefix: str
    topics: tuple[ModelTopicSpec, ...] = Field(default_factory=tuple)
    groups: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def topic_names(self) -> tuple[str, ...]:
        """Full canary topic names (the concrete Kafka topics to create)."""
        return tuple(spec.suffix for spec in self.topics)


__all__: list[str] = ["ModelCanaryCatalog"]
