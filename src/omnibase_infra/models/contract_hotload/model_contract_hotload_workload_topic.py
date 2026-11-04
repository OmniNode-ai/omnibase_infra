# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The per-workload hot-loading topics a runtime writes (OMN-21226, plan 3.2).

Each workload writes two exclusive-writer topics, as the work-ledger pattern does: an
applied topic with the registry changes it applied or refused, and a state topic with
its instance state records. The workload sits in the event-name segment, so every
expansion keeps the five segments core's topic validator requires, and a reader takes
the workload from the topic and never from the payload.

The state topic holds the latest record of each instance, so it is compacted: a record
that ages off it is a missing instance. The applied topic is an event log and is time
retained. Both declarations are here because a per-workload topic has no contract
section to carry a ``topic_config`` and the provisioner takes a caller-supplied spec.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

_KIND = "evt"
_PRODUCER = "platform"
_APPLIED_PREFIX = "contract-hotload-applied-from-"
_STATE_PREFIX = "contract-hotload-state-from-"

#: One partition: the state topic is keyed by instance and ordered per key, and the
#: applied topic is low-volume (registry changes, not commands).
_PARTITIONS = 1


class ModelContractHotloadWorkloadTopic(BaseModel):
    """A workload id, which of its two topics, and what that topic expands to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Composed from its segments, as the node-announcement template is, so the
    # topic-literal and topic-naming gates see segments and the placeholder never
    # reads as a topic.
    APPLIED_TEMPLATE: ClassVar[str] = (
        f"onex.{_KIND}.{_PRODUCER}.{_APPLIED_PREFIX}{{workload}}.v1"
    )
    STATE_TEMPLATE: ClassVar[str] = (
        f"onex.{_KIND}.{_PRODUCER}.{_STATE_PREFIX}{{workload}}.v1"
    )

    workload: str
    channel: Literal["applied", "state"]

    @field_validator("workload")
    @classmethod
    def _workload_is_canonical_kebab_case(cls, value: str) -> str:
        if not value:
            raise ValueError("workload must be a non-empty string")
        # Each workload owns its topic. Normalizing here would silently alias
        # distinct workload identities (for example, a_b and a-b).
        if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value) is None:
            raise ValueError("workload must already be in canonical kebab-case")
        return value

    @property
    def template(self) -> str:
        """The template of this channel."""
        return self.STATE_TEMPLATE if self.channel == "state" else self.APPLIED_TEMPLATE

    @property
    def suffix(self) -> str:
        """The validated topic suffix; raises ``ValueError`` when it is invalid."""
        suffix = self.template.format(workload=self.workload)
        result = validate_topic_suffix(suffix)
        if not result.is_valid:
            raise ValueError(
                f"workload {self.workload!r} expands to an invalid topic: {result.error}"
            )
        return suffix

    @property
    def spec(self) -> ModelTopicSpec:
        """The creation spec: compacted for the state topic, time retained otherwise."""
        policy = "compact" if self.channel == "state" else "delete"
        return ModelTopicSpec(
            suffix=self.suffix,
            partitions=_PARTITIONS,
            kafka_config={"cleanup.policy": policy},
        )

    @classmethod
    def applied(cls, workload: str) -> ModelContractHotloadWorkloadTopic:
        """The applied topic of one workload."""
        return cls(workload=workload, channel="applied")

    @classmethod
    def state(cls, workload: str) -> ModelContractHotloadWorkloadTopic:
        """The state topic of one workload."""
        return cls(workload=workload, channel="state")
