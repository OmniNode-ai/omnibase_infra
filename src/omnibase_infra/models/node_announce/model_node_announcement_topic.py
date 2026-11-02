# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The durable node-announcement topic of one workload (OMN-21590).

One exclusive-writer topic per workload carries that workload's announcements; the
workload sits in the event-name segment, as the state topics do, so every expansion
passes core's five-segment topic validator. The lane overlay lists the workloads whose
announcements the registry reads.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_core.validation import validate_topic_suffix

_KIND = "evt"
_PRODUCER = "platform"
_EVENT_PREFIX = "node-announcement-from-"


class ModelNodeAnnouncementTopic(BaseModel):
    """A workload id and the topic suffix it expands to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Composed from its segments, as the generated-tool suffixes are, so the topic-literal
    # and topic-naming gates see segments and the placeholder never reads as a topic.
    TEMPLATE: ClassVar[str] = f"onex.{_KIND}.{_PRODUCER}.{_EVENT_PREFIX}{{workload}}.v1"

    workload: str

    @field_validator("workload")
    @classmethod
    def _expansion_is_a_valid_topic(cls, value: str) -> str:
        if not value:
            raise ValueError("workload must be a non-empty string")
        # Each workload owns its topic. Normalizing here would silently alias
        # distinct workload identities (for example, a_b and a-b).
        if value != value.strip() or "_" in value:
            raise ValueError("workload must already be in canonical kebab-case")
        workload = value
        result = validate_topic_suffix(cls.TEMPLATE.format(workload=workload))
        if not result.is_valid:
            raise ValueError(
                f"workload {value!r} expands to an invalid topic: {result.error}"
            )
        return workload

    @property
    def suffix(self) -> str:
        """The validated topic suffix for this workload."""
        return self.TEMPLATE.format(workload=self.workload)

    @classmethod
    def suffix_for(cls, workload: str) -> str:
        """Expand the template for one workload; raise ``ValueError`` when it is invalid."""
        return cls(workload=workload).suffix
