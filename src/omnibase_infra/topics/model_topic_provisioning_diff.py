# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Desired-vs-live topic diff, shared by every provisioning path (OMN-15395).

A full startup provisioning pass used to issue one ``CreateTopics`` call per
known topic — ~1,280 authorizations across ~1,026 topic ARNs on every boot,
whether or not anything was missing, which is what repeatedly tripped the AWS
unauthorized-API-call alarm. The provisioner never asked the broker what
already existed; it relied on ``TopicAlreadyExistsError`` as flow control.

:func:`build_provisioning_diff` is the single, pure, transport-free set
comparison every provisioning path now runs *before* issuing creates. It is
also the topic-side engine behind
``managed_staging_topic_checker.build_topic_diff`` (which layers the canary
namespace's prefix/consumer-group semantics on top) so there is exactly one
diff implementation, not two.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class ModelTopicProvisioningDiff(BaseModel):
    """Desired topic set diffed against a live broker snapshot.

    Attributes:
        desired_topics: Every topic name the provisioning pass wants to exist.
        missing_topics: Desired names absent from the broker — the ONLY names a
            provisioning pass may issue ``CreateTopics`` for.
        present_topics: Desired names already on the broker. Never re-created
            and never mutated; spec drift on these is reported, not repaired.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    desired_topics: tuple[str, ...] = Field(default_factory=tuple)
    missing_topics: tuple[str, ...] = Field(default_factory=tuple)
    present_topics: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def has_missing(self) -> bool:
        """``True`` iff at least one desired topic is absent from the broker."""
        return bool(self.missing_topics)


def build_provisioning_diff(
    desired_topics: Iterable[str],
    existing_topics: Iterable[str],
) -> ModelTopicProvisioningDiff:
    """Diff a desired topic set against a live broker snapshot.

    Pure and transport-free so provisioning behaviour is unit-testable without
    a broker.

    Args:
        desired_topics: Topic names the caller wants to exist. Order is
            preserved (callers provision in contract-declared priority order);
            duplicates are collapsed.
        existing_topics: Topic names currently on the broker.

    Returns:
        A :class:`ModelTopicProvisioningDiff` splitting the desired set into
        missing (create these) and present (leave alone).
    """
    desired = tuple(dict.fromkeys(desired_topics))
    existing = set(existing_topics)
    return ModelTopicProvisioningDiff(
        desired_topics=desired,
        missing_topics=tuple(name for name in desired if name not in existing),
        present_topics=tuple(name for name in desired if name in existing),
    )


__all__: list[str] = [
    "ModelTopicProvisioningDiff",
    "build_provisioning_diff",
]
