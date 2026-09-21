# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep the control-topic lag current WHILE a rebuild runs (OMN-18990).

WHY A SECOND CLIENT RATHER THAN THE POLLING ONE
------------------------------------------------
``LagSampler`` is a cache. The only writer was ``DeployConsumer._sample_lag``,
called from inside ``poll_and_accept``, and this agent's run loop is strictly
serial: it awaits ``poll_and_accept``, and when that returns a command it
awaits ``_execute_command``, a full rebuild measured at a 1402s mean. No poll
happens during the rebuild, so no sample was taken during it, and ``/queue``
served the PRE-rebuild lag as a confident integer for the whole window in
which later merges queue up. The measurement was blind during the only period
it exists to measure.

``queue_depth.LagSampler``'s own docstring says why the obvious repair is
refused: it holds no kafka client, "because a client touched from the loop
thread while the worker is polling it is a data race". That reasoning is
correct and is preserved here rather than traded away. This refresher owns a
SEPARATE, group-less consumer used for nothing but ``end_offsets``, so there
is no shared client to race on and no possibility of this observation
disturbing the group's position, assignment or commits.

WHY ``end_offsets`` AND NOT ``highwater``
------------------------------------------
``highwater`` is a cached field updated by fetch responses. A refresher that
read it without polling would return the watermark as of the last poll --
which is the same staleness, moved one layer down and harder to see.
``end_offsets`` is a ListOffsets round trip and answers about now.

The committed side comes from the sampler, which already tracks what this
process has committed through (``note_commit``). That is deliberate: reading
committed offsets from the broker would be a second coordinator round trip for
a number this process already knows exactly.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It observes. It does not poll, consume, commit, seek, join the group, or touch
the agent's own consumer. A refresher that could advance an offset would be
able to lose a command in order to make its own number smaller.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final

from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.queue_depth import LagSampler, ModelControlTopicLag

logger = logging.getLogger(__name__)

#: Seconds between refreshes. Half of ``MAX_LAG_SAMPLE_AGE_SECONDS``, so one
#: missed round trip does not flip a healthy agent to unreadable while a
#: refresher that has genuinely stopped is caught inside the bound.
DEFAULT_REFRESH_INTERVAL_SECONDS: Final = 60.0

#: How the basis is spelled when this module took the measurement, rather than
#: the poll path. Named on the wire so a reader can tell which surface answered.
BASIS: Final = "committed+end_offsets"

ConsumerFactory = Callable[[], Any]


class LagRefresher:
    """Samples the control-topic lag out of band, into the shared sampler.

    Construction does NOT connect. The client is built on the first refresh
    and kept, so an agent whose broker is briefly unreachable at startup does
    not fail to start over an observation.
    """

    def __init__(
        self,
        kafka_config: ModelDeployAgentKafkaConfig,
        sampler: LagSampler,
        topic: str,
        *,
        consumer_factory: ConsumerFactory | None = None,
    ) -> None:
        self._kafka_config = kafka_config
        self._sampler = sampler
        self._topic = topic
        self._consumer_factory = consumer_factory or self._build_consumer
        self._consumer: Any | None = None
        self._lock = threading.Lock()

    def _build_consumer(self) -> Any:
        from kafka import KafkaConsumer

        # No group_id: this client joins nothing, is assigned nothing and
        # commits nothing. It exists to ask the broker one question.
        return KafkaConsumer(
            **self._kafka_config.consumer_kwargs(),
            group_id=None,
            enable_auto_commit=False,
        )

    def refresh(self) -> None:
        """Take one sample. Never raises: an observation is not worth a job."""
        with self._lock:
            try:
                self._refresh_locked()
            except Exception as exc:  # noqa: BLE001 - unreadable is a verdict, not a crash
                logger.debug("out-of-band lag refresh failed: %s", exc)
                self._drop_consumer()
                self._sampler.record(
                    ModelControlTopicLag.unknown(
                        f"the out-of-band control-topic lag refresh failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )

    def _refresh_locked(self) -> None:
        from kafka import TopicPartition

        if self._consumer is None:
            self._consumer = self._consumer_factory()
        consumer = self._consumer
        partitions = consumer.partitions_for_topic(self._topic)
        if not partitions:
            self._sampler.record(
                ModelControlTopicLag.unknown(
                    f"the broker reports no partitions for {self._topic}, so "
                    "the records beyond this agent cannot be counted"
                )
            )
            return
        topic_partitions = [TopicPartition(self._topic, p) for p in sorted(partitions)]
        end_offsets = consumer.end_offsets(topic_partitions)
        total = 0
        for topic_partition in topic_partitions:
            end = end_offsets.get(topic_partition)
            if end is None:
                self._sampler.record(
                    ModelControlTopicLag.unknown(
                        f"the broker returned no end offset for "
                        f"{topic_partition}, so the records beyond this agent "
                        "cannot be counted"
                    )
                )
                return
            committed = self._sampler.committed(topic_partition)
            if committed is None:
                # This process has not committed through anything yet, so it
                # has no offset of its own to measure against. Reported rather
                # than guessed: the poll path's position fallback is not
                # available here, and inventing one would be the undercount
                # this ticket removes.
                self._sampler.record(
                    ModelControlTopicLag.unknown(
                        f"this process has committed no offset on "
                        f"{topic_partition} yet, so its place in the topic "
                        "cannot be read out of band"
                    )
                )
                return
            total += max(0, int(end) - int(committed))
        self._sampler.record(
            ModelControlTopicLag(
                value=total,
                basis=BASIS,
                observed_at=datetime.now(UTC),
            )
        )

    def _drop_consumer(self) -> None:
        consumer, self._consumer = self._consumer, None
        if consumer is None:
            return
        try:
            consumer.close()
        except Exception:  # noqa: BLE001 - closing a broken client is best effort
            pass

    def close(self) -> None:
        with self._lock:
            self._drop_consumer()
