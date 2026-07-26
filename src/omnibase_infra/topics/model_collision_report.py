# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Zero-collision readback report model (OMN-14727, B7).

Result of checking a generated canary catalog against a snapshot of the
cluster's existing topics + consumer groups: exact-name collisions and prefix
conflicts, plus a derived ``is_clean`` verdict.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCollisionReport(BaseModel):
    """Result of a zero-collision readback of a catalog against existing state.

    Attributes:
        topic_prefix: Canary topic prefix that was checked.
        group_prefix: Canary group prefix that was checked.
        checked_topic_count: Number of canary topics checked.
        checked_group_count: Number of canary groups checked.
        existing_topic_count: Number of existing topics in the snapshot.
        existing_group_count: Number of existing consumer groups in the snapshot.
        colliding_topics: Generated topic names that exactly match an existing
            topic.
        colliding_groups: Generated group names that exactly match an existing
            consumer group.
        prefix_conflicting_topics: Existing topics already living under the
            canary topic prefix (a namespace conflict).
        prefix_conflicting_groups: Existing consumer groups already living under
            the canary group prefix.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic_prefix: str
    group_prefix: str
    checked_topic_count: int
    checked_group_count: int
    existing_topic_count: int
    existing_group_count: int
    colliding_topics: tuple[str, ...] = Field(default_factory=tuple)
    colliding_groups: tuple[str, ...] = Field(default_factory=tuple)
    prefix_conflicting_topics: tuple[str, ...] = Field(default_factory=tuple)
    prefix_conflicting_groups: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def is_clean(self) -> bool:
        """``True`` iff the canary namespace is fully disjoint from existing state."""
        return not (
            self.colliding_topics
            or self.colliding_groups
            or self.prefix_conflicting_topics
            or self.prefix_conflicting_groups
        )


__all__: list[str] = ["ModelCollisionReport"]
