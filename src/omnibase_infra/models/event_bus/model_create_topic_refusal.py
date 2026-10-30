# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A broker's refusal of one topic inside a ``CreateTopics`` response (OMN-17427).

``aiokafka`` 0.13 returns the ``CreateTopicsResponse`` and raises nothing for a
per-topic error: a topic refused for ``INVALID_PARTITIONS`` (the broker at its
partition ceiling), ``TOPIC_ALREADY_EXISTS`` or ``INVALID_REPLICATION_FACTOR``
comes back as a ``response.topic_errors`` entry and the call returns as if it
had worked. The provisioner read none of them, so every refused topic was
recorded as created and every ``except`` written for these errors was
unreachable against the real client.

On 2026-10-30 an abandoned pre-PR slot held 2262 partitions of the shared dev
broker, which held 4827 under a ceiling of 7000. A slot boot asked for 2267
topics, the broker refused the 93 past the ceiling (``Can not increase
partition count due to core limit``, broker log
``topic_invalid_partitions_core_limit``) and the runtime booted with 2174 of
them, its consumers reporting ``topic_absent`` (``error_code=3``) for the rest.

This model reads the refusal and turns it into the exception the provisioner's
existing handlers already classify. A response with no ``topic_errors`` (a
fake, ``None``) is no refusal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

_TOPIC_ALREADY_EXISTS = 36
_INVALID_PARTITIONS = 37
_INVALID_REPLICATION_FACTOR = 38


class ModelCreateTopicRefusal(BaseModel):
    """One topic's non-zero answer from a ``CreateTopics`` response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str
    errno: int
    broker_message: str | None = None

    @classmethod
    def from_response(
        cls, response: object, topic: str
    ) -> ModelCreateTopicRefusal | None:
        """Return the broker's refusal of ``topic``, or ``None`` when it accepted."""
        entries = getattr(response, "topic_errors", None)
        if not isinstance(entries, (list, tuple)):
            return None
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            if entry[0] != topic or not isinstance(entry[1], int) or entry[1] == 0:
                continue
            note = entry[2] if len(entry) > 2 and isinstance(entry[2], str) else None
            return cls(topic=topic, errno=entry[1], broker_message=note)
        return None

    @property
    def is_already_exists(self) -> bool:
        return self.errno == _TOPIC_ALREADY_EXISTS

    @property
    def is_partition_ceiling(self) -> bool:
        """``INVALID_PARTITIONS`` whose message names a limit.

        Redpanda names the limit (``core limit``; memory and file-descriptor
        limits read alike). The same code also answers a nonsensical partition
        count, which is no ceiling, so the message decides.
        """
        return (
            self.errno == _INVALID_PARTITIONS
            and "limit" in (self.broker_message or "").lower()
        )

    def as_exception(
        self,
        *,
        created: int,
        planned: int,
        metadata: Mapping[str, Mapping[str, object]],
    ) -> Exception:
        """The exception the provisioner's handlers classify.

        A partition ceiling is fail-closed (a ``TopicReplicationPolicyError``,
        already re-raised past every best-effort boundary in the runtime): a
        topic that does not exist behind a warning that reads as success leaves
        the runtime booting with a partial topic set. An unhostable replica
        count keeps its wire name so the OMN-15395 D5 classifier matches it.
        Anything else is an ordinary create failure.
        """
        text = f"[Error {self.errno}] {self.topic}: {self.broker_message}"
        if self.errno == _INVALID_REPLICATION_FACTOR:
            return RuntimeError(f"INVALID_REPLICATION_FACTOR {text}")
        if not self.is_partition_ceiling:
            return RuntimeError(text)
        held: Counter[str] = Counter()
        for name, entry in metadata.items():
            parts = entry.get("partitions")
            held[name.split(".", 1)[0]] += len(parts) if isinstance(parts, list) else 1
        top = sorted(held.items(), key=lambda item: (-item[1], item[0]))[:6]
        return TopicReplicationPolicyError(
            f"Broker refused to create topic {self.topic!r}: it is at its "
            f"partition ceiling ({self.broker_message}). This topic and every "
            f"later one in this pass will not exist: {created} created this "
            f"pass, {planned - created} not. Partitions held before this pass, "
            f"by namespace: {', '.join(f'{ns}={n}' for ns, n in top) or 'none read'}"
            ". Free partitions (tear down or reap an abandoned pre-PR slot with "
            "prepr_teardown_slot.py reap) rather than raising "
            "topic_partitions_per_shard, which is sized against broker memory "
            "(OMN-20266).",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="create_topic",
                target_name=self.topic,
            ),
            topic=self.topic,
            created_this_pass=created,
        )
