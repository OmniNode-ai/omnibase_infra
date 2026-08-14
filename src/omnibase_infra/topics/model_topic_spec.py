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

# Canonical partition default for platform topic creation.
# This lives here (not in service_topic_manager) to avoid a circular import:
#   topics/__init__ -> model_topic_spec -> service_topic_manager -> topics/__init__
#
# OMN-15395: there is deliberately NO replication-factor constant here any more.
# ``DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1`` used to be applied silently
# whenever the owning contract declared nothing, which is how 519 RF1 topics
# were created on MSK against a broker whose own default is RF2. Replication is
# now resolved exclusively by ``ModelTopicProvisioningPolicy``, which fails
# closed on a managed cluster instead of defaulting.
DEFAULT_EVENT_TOPIC_PARTITIONS: int = 6


class ModelTopicSpec(BaseModel):
    """Per-topic creation spec: suffix + partitions + optional Kafka config overrides.

    Attributes:
        suffix: Full ONEX 5-segment topic name (e.g., "onex.evt.platform.node-registration.v1").  # onex-topic-allow: pending contract auto-wiring
        partitions: Number of partitions for the topic.
        replication_factor: Replication factor declared by the owning contract.
            ``None`` means **the contract declared none** — it does NOT mean 1.
            Resolution to an explicit value (or a fail-closed refusal) is
            :class:`~omnibase_infra.topics.model_topic_provisioning_policy.ModelTopicProvisioningPolicy`'s
            job and happens on the creation path, never here (OMN-15395).
        kafka_config: Optional Kafka topic config overrides (e.g., {"cleanup.policy": "compact"}).
        provisioning_priority: Lower values are provisioned first.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    suffix: str
    partitions: int = DEFAULT_EVENT_TOPIC_PARTITIONS
    replication_factor: int | None = Field(default=None, ge=1)
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
    "ModelTopicSpec",
]
