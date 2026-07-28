# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Typed diff report for the managed-staging canary catalog vs. a live broker
(OMN-15283).

:func:`~omnibase_infra.topics.managed_staging_topic_checker.build_topic_diff`
compares a generated :class:`~omnibase_infra.topics.model_canary_catalog.ModelCanaryCatalog`
against a snapshot of a broker's existing topics + consumer groups and reports
three disjoint buckets per resource kind:

* **missing** -- catalog-listed names absent from the broker (the fail-closed
  gate signal);
* **present** -- catalog-listed names already on the broker;
* **out_of_catalog** -- existing names that live under the catalog's prefix but
  are not catalog-listed (a namespace conflict / stray name -- reported, never
  mutated).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelManagedStagingTopicDiff(BaseModel):
    """Typed diff of a managed-staging canary catalog against a live broker.

    Attributes:
        topic_prefix: Canary topic prefix the diff was computed against.
        group_prefix: Canary group prefix the diff was computed against.
        missing_topics: Catalog topics absent from the broker snapshot.
        present_topics: Catalog topics already present on the broker.
        out_of_catalog_topics: Existing topics under ``topic_prefix`` that are
            not catalog-listed.
        missing_groups: Catalog groups absent from the broker snapshot.
        present_groups: Catalog groups already present on the broker.
        out_of_catalog_groups: Existing groups under ``group_prefix`` that are
            not catalog-listed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic_prefix: str
    group_prefix: str
    missing_topics: tuple[str, ...] = Field(default_factory=tuple)
    present_topics: tuple[str, ...] = Field(default_factory=tuple)
    out_of_catalog_topics: tuple[str, ...] = Field(default_factory=tuple)
    missing_groups: tuple[str, ...] = Field(default_factory=tuple)
    present_groups: tuple[str, ...] = Field(default_factory=tuple)
    out_of_catalog_groups: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def is_fully_present(self) -> bool:
        """``True`` iff every catalog-listed topic and group is on the broker."""
        return not (self.missing_topics or self.missing_groups)

    @property
    def has_out_of_catalog(self) -> bool:
        """``True`` iff any stray prefix-matching name was found on the broker."""
        return bool(self.out_of_catalog_topics or self.out_of_catalog_groups)


__all__: list[str] = ["ModelManagedStagingTopicDiff"]
