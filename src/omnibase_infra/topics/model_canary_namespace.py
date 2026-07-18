# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative managed-staging canary namespace model (OMN-14727, B7).

Declares the canary topic/group prefix, the MSK IAM resource patterns the prefix
must fall inside, and the conservative sizing / group-start policy. The
``model_validator`` fails **closed** if either prefix is not authorized by its
IAM patterns -- turning the ticket's "a prefix outside them fails AUTH, not
collision" warning into a mechanical, un-bypassable invariant.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def iam_pattern_authorizes(name: str, patterns: Iterable[str]) -> bool:
    """Return whether ``name`` is authorized by any MSK IAM resource pattern.

    MSK IAM resource patterns are globs where ``*`` is a trailing wildcard and
    every other character (including ``.``) is literal. ``"onex.*"`` therefore
    authorizes any name beginning with the literal prefix ``"onex."``.

    A pattern without a trailing ``*`` must match ``name`` exactly.

    Args:
        name: The topic name, consumer group name, or candidate prefix to test.
        patterns: IAM resource patterns (e.g. ``("onex.*", "omninode.*")``).

    Returns:
        ``True`` if at least one pattern authorizes ``name``, else ``False``.
    """
    for pattern in patterns:
        if pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


class ModelCanaryNamespace(BaseModel):
    """Declarative canary namespace: the prefix + IAM patterns + sizing policy.

    Loaded from ``managed_staging_canary_catalog_namespace.yaml``.

    Attributes:
        ticket: Owning Linear ticket.
        epoch: Managed-staging epoch token (e.g. ``"mstg1"``). Bump to mint a
            fresh, provably disjoint namespace.
        description: Human-readable namespace description.
        topic_prefix: Common prefix prepended to every candidate topic suffix.
            MUST be authorized by ``iam_topic_patterns`` and end with ``.``.
        group_prefix: Common prefix prepended to every canary consumer group.
            MUST be authorized by ``iam_group_patterns`` and end with ``.``.
        iam_topic_patterns: MSK IAM topic resource patterns delivered by A1.
        iam_group_patterns: MSK IAM group resource patterns delivered by A1.
        group_start_policy: Consumer group start/reset policy for the canary.
        default_partitions: Conservative per-topic partition count. Final sizing
            is the SS2.8 partition-pressure decision applied at Phase 3.
        default_replication_factor: Per-topic replication factor.
        candidate_contract_roots: Repo-relative roots scanned to extract the
            candidate's contract-owned topic suffixes + subscribing nodes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket: str = Field(..., min_length=1)
    epoch: str = Field(..., min_length=1)
    description: str = ""
    topic_prefix: str = Field(..., min_length=1)
    group_prefix: str = Field(..., min_length=1)
    iam_topic_patterns: tuple[str, ...] = Field(..., min_length=1)
    iam_group_patterns: tuple[str, ...] = Field(..., min_length=1)
    group_start_policy: Literal["earliest", "latest", "none"] = "earliest"
    default_partitions: int = Field(default=1, ge=1)
    default_replication_factor: int = Field(default=2, ge=1)
    candidate_contract_roots: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _prefixes_are_authorized_and_terminated(self) -> Self:
        """Fail closed unless both prefixes are IAM-authorized and dot-terminated."""
        if not self.topic_prefix.endswith("."):
            raise ValueError(
                f"topic_prefix must end with '.' for clean namespacing, "
                f"got {self.topic_prefix!r}"
            )
        if not self.group_prefix.endswith("."):
            raise ValueError(
                f"group_prefix must end with '.' for clean namespacing, "
                f"got {self.group_prefix!r}"
            )
        if not iam_pattern_authorizes(self.topic_prefix, self.iam_topic_patterns):
            raise ValueError(
                f"topic_prefix {self.topic_prefix!r} is outside the MSK IAM "
                f"topic patterns {self.iam_topic_patterns!r}; such a prefix "
                f"fails AUTH (AccessDenied), not collision. Choose a prefix that "
                f"starts with an authorized pattern root (e.g. 'onex.')."
            )
        if not iam_pattern_authorizes(self.group_prefix, self.iam_group_patterns):
            raise ValueError(
                f"group_prefix {self.group_prefix!r} is outside the MSK IAM "
                f"group patterns {self.iam_group_patterns!r}; such a prefix "
                f"fails AUTH (AccessDenied), not collision."
            )
        return self


__all__: list[str] = ["ModelCanaryNamespace", "iam_pattern_authorizes"]
