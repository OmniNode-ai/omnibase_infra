# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic per-topic creation spec for ONEX platform topics.

Each topic in the platform registry has a ModelTopicSpec that defines its
suffix (full ONEX 5-segment topic name), partition count, replication factor,
and optional Kafka config overrides (e.g., compaction settings for snapshot
topics).

Design Notes:
    ModelSnapshotTopicConfig cannot be reused here because its validator
    rejects non-compact cleanup policies. ModelTopicSpec is a lightweight
    model that supports any cleanup policy and optional config overrides.

Related:
    - platform_topic_suffixes.py: Registry of all platform topic specs
    - service_topic_manager.py: TopicProvisioner consumes specs for creation
    - OMN-2115: Bus audit layer 1 - generic bus health diagnostics

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Canonical defaults for platform topic creation.
# These live here (not in service_topic_manager) to avoid a circular import:
#   topics/__init__ -> model_topic_spec -> service_topic_manager -> topics/__init__
# service_topic_manager re-imports these constants for its own fallback path.
DEFAULT_EVENT_TOPIC_PARTITIONS: int = 6
DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR: int = 1


class ModelTopicSpec(BaseModel):
    """Per-topic creation spec: suffix + partitions + optional Kafka config overrides.

    Attributes:
        suffix: Full ONEX 5-segment topic name (e.g., "onex.evt.platform.node-registration.v1").  # onex-topic-allow: pending contract auto-wiring
        partitions: Number of partitions for the topic.
        replication_factor: Replication factor for the topic.
        kafka_config: Optional Kafka topic config overrides (e.g., {"cleanup.policy": "compact"}).
        provisioning_priority: Lower values are provisioned first.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    suffix: str
    partitions: int = DEFAULT_EVENT_TOPIC_PARTITIONS
    replication_factor: int = DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR
    kafka_config: Mapping[str, str] | None = Field(default=None)
    provisioning_priority: int = 100

    @field_validator("kafka_config", mode="before")
    @classmethod
    def freeze_kafka_config(
        cls, v: Mapping[str, str] | None
    ) -> Mapping[str, str] | None:
        """Freeze mutable dict passed at construction time."""
        if isinstance(v, dict):
            return MappingProxyType(v)
        return v


__all__: list[str] = [
    "DEFAULT_EVENT_TOPIC_PARTITIONS",
    "DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR",
    "ModelTopicSpec",
]
