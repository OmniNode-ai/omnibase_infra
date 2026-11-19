# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A topic a contract or topics.yaml declares, validated by the contract topic extractor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, field_validator

# OMN-25548. A declared topic may be one tenant's physical copy of a canonical topic:
# ``tenant-<slug>.<canonical topic>``. The slug is 1 to 32 characters, lowercase
# letters, digits and inner hyphens. The extractor and the boot provisioner share
# this one pattern.
TENANT_SLUG_PATTERN = r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?"


class ModelContractTopicEntry(BaseModel):
    """A single validated topic extracted from one or more contract.yaml files.

    Per-topic provisioning config (``partitions``, ``replication_factor``,
    ``kafka_config``) is carried from a contract's optional ``topic_config``
    block (OMN-13238). When a contract does not declare ``topic_config`` for a
    topic, these fields stay ``None`` and the provisioner falls back to the
    ``ModelTopicSpec`` defaults.
    """

    topic: str
    kind: Literal["evt", "cmd", "intent", "dlq", "snapshot"]
    producer: str
    event_name: str
    version: str  # e.g. "v1"
    source_contracts: tuple[Path, ...]
    provisioning_priority: int = 100
    partitions: int | None = None
    replication_factor: int | None = None
    kafka_config: Mapping[str, str] | None = None
    # OMN-25548: the tenant slug of a ``tenant-<slug>.``-prefixed wire topic;
    # empty for a canonical topic. ``kind``/``producer``/``event_name``/``version``
    # always describe the canonical topic after the prefix.
    tenant_slug: str = ""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator("kafka_config", mode="before")
    @classmethod
    def _freeze_kafka_config(
        cls, v: Mapping[str, str] | None
    ) -> Mapping[str, str] | None:
        """Freeze a mutable dict supplied at construction time."""
        if isinstance(v, dict):
            return MappingProxyType(dict(v))
        return v

    def merge_sources(self, other: ModelContractTopicEntry) -> ModelContractTopicEntry:
        """Return a new entry with source_contracts merged (deduped, sorted).

        Per-topic config is merged by taking the first non-``None`` value for
        each field, so a topic declared in two contracts (one carrying
        ``topic_config``, one not) keeps the declared config. When BOTH carry
        conflicting config, ``self`` wins (deterministic by topic-string sort
        order in the caller).
        """
        combined = tuple(
            sorted(set(self.source_contracts) | set(other.source_contracts))
        )
        return self.model_copy(
            update={
                "source_contracts": combined,
                "provisioning_priority": min(
                    self.provisioning_priority,
                    other.provisioning_priority,
                ),
                "partitions": (
                    self.partitions if self.partitions is not None else other.partitions
                ),
                "replication_factor": (
                    self.replication_factor
                    if self.replication_factor is not None
                    else other.replication_factor
                ),
                "kafka_config": (
                    self.kafka_config
                    if self.kafka_config is not None
                    else other.kafka_config
                ),
            }
        )
