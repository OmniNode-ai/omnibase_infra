# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic per-topic creation spec for ONEX platform topics.

Each topic in the platform registry has a ModelTopicSpec that defines its
suffix (full ONEX 5-segment topic name), partition count, replication factor,
and optional Kafka config overrides (e.g., compaction settings for snapshot
topics).

Design Notes:
    ModelSnapshotTopicConfig cannot be reused here because its validator
    rejects non-compact cleanup policies. ModelTopicSpec is a lightweight
    dataclass that supports any cleanup policy and optional config overrides.

Related:
    - platform_topic_suffixes.py: Registry of all platform topic specs
    - service_topic_manager.py: TopicProvisioner consumes specs for creation
    - OMN-2115: Bus audit layer 1 - generic bus health diagnostics

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# Canonical defaults for platform topic creation.
# These live here (not in service_topic_manager) to avoid a circular import:
#   topics/__init__ -> model_topic_spec -> service_topic_manager -> topics/__init__
# service_topic_manager re-imports these constants for its own fallback path.
DEFAULT_EVENT_TOPIC_PARTITIONS: int = 6
DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR: int = 1


@dataclass(frozen=True)
class ModelTopicSpec:
    """Per-topic creation spec: suffix + partitions + optional Kafka config overrides.

    Attributes:
        suffix: Full ONEX 5-segment topic name (e.g., "onex.evt.platform.node-registration.v1").
        partitions: Number of partitions for the topic.
        replication_factor: Replication factor for the topic.
        kafka_config: Optional Kafka topic config overrides (e.g., {"cleanup.policy": "compact"}).
    """

    suffix: str
    partitions: int = DEFAULT_EVENT_TOPIC_PARTITIONS
    replication_factor: int = DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR
    kafka_config: Mapping[str, str] | None = field(default=None)

    def __post_init__(self) -> None:
        """Freeze mutable kafka_config dict passed at construction time."""
        if isinstance(self.kafka_config, dict):
            object.__setattr__(
                self, "kafka_config", MappingProxyType(self.kafka_config)
            )


__all__: list[str] = [
    "DEFAULT_EVENT_TOPIC_PARTITIONS",
    "DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR",
    "ModelTopicSpec",
]
