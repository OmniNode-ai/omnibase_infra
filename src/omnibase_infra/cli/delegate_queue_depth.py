# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observe how deep the command queue is when a delegation's wait expires.

OMN-18852. The CLI's hard-timeout backstop reported *that* a delegation did
not terminalize and for how long, which is true and was not the question the
caller had. Measured on the ``.201`` dev lane 2026-09-19: nine delegations in
26 minutes, queue wait growing monotonically from 3 s to 445 s, three callers
timing out at ~306 s. Every one of the three got a correct answer -- produced
and published after it had exited. A control run spent 179 s of its 181 s wall
clock queued behind an inference that itself took 1.559 s.

A bare "exceeded hard timeout of 310s" tells that caller nothing about which
of those it hit. "There were 7 records ahead of yours" does, and it is the
difference between retrying and waiting.

**The number is observed, never supplied.** It is the consumer group's own
uncommitted backlog on the command topic -- broker-reported committed offsets
against broker-reported log-end offsets, through the same
``ServiceConsumerLagObserver`` / ``AdapterKafkaAdminLag`` pair the
topic-migration drain gate uses. Nothing here accepts a depth from the caller,
and nothing here guesses one.

**What it is NOT, stated rather than implied.** It is not this record's own
queue position. The publish path discards its ``ModelPublishReceipt``
(``RuntimeLocal._run_event_driven`` calls ``await bus.publish(...)`` and keeps
nothing), so the caller's own ``(partition, offset)`` is not knowable here at
all. What is knowable is the group's backlog, which INCLUDES this caller's
record while it is unconsumed; ``records_ahead`` is that backlog minus the
caller's own one record. Under auto-commit the committed offset also trails
the live fetch position by up to the commit interval, so the figure can
overstate slightly. It is an observation of the queue, reported as such.

**An unresolvable depth says so.** Every failure -- no broker, an in-process
bus, no consumer group known, an admin client that will not start, a probe
that runs long -- returns ``records_ahead=None`` with a reason naming the
cause. Printing a misleading zero would tell the caller the queue was empty,
which is the one thing the evidence says it was not.
"""

from __future__ import annotations

import asyncio
import logging

from aiokafka import AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient

from omnibase_infra.cli.model_delegate_queue_depth import ModelDelegateQueueDepth
from omnibase_infra.migration.adapter_kafka_admin_lag import AdapterKafkaAdminLag
from omnibase_infra.migration.service_consumer_lag_observer import (
    ServiceConsumerLagObserver,
)

__all__ = ["QUEUE_DEPTH_PROBE_SECONDS", "observe_delegate_queue_depth"]

logger = logging.getLogger(__name__)

#: Wall-clock bound on the probe. The caller has ALREADY blown its deadline
#: when this runs, so the probe must be a short tail on an established
#: failure, never a second wait. Exceeding it is an unresolved depth with a
#: reason, not a longer hang.
QUEUE_DEPTH_PROBE_SECONDS = 8.0


async def _observe_async(
    *,
    broker: str,
    command_topic: str,
    consumer_group: str,
) -> int:
    """Total uncommitted backlog for ``consumer_group`` on ``command_topic``."""
    admin = AIOKafkaAdminClient(bootstrap_servers=broker)
    consumer = AIOKafkaConsumer(bootstrap_servers=broker, enable_auto_commit=False)
    await admin.start()
    try:
        await consumer.start()
        try:
            # ``AIOKafkaAdminClient`` satisfies the committed-offset half of
            # ``ProtocolKafkaAdminLike`` structurally; the adapter supplies the
            # ``list_offsets`` half the pinned 0.13.0 client omits (OMN-12632),
            # and is itself the full protocol surface the observer needs.
            observer = ServiceConsumerLagObserver(AdapterKafkaAdminLag(admin, consumer))
            lag = await observer.observe(consumer_group)
            if not lag.has_partitions_for_topic(command_topic):
                raise ValueError(
                    f"group {consumer_group!r} has no observed partitions on "
                    f"{command_topic!r}"
                )
            return lag.lag_for_topic(command_topic)
        finally:
            await consumer.stop()
    finally:
        await admin.close()


def observe_delegate_queue_depth(
    *,
    bus: str,
    broker: str,
    command_topic: str,
    consumer_groups: tuple[str, ...],
    probe_seconds: float = QUEUE_DEPTH_PROBE_SECONDS,
) -> ModelDelegateQueueDepth:
    """Resolve the command queue's depth, or say why it could not be resolved.

    Called from the CLI's synchronous timeout handler, after the runtime's own
    event loop has already exited, so it owns the loop it runs on.

    Args:
        bus: The RESOLVED transport, as the refusal reports it.
        broker: Broker address the run was bound to; empty in-process.
        command_topic: Topic the command was published to.
        consumer_groups: Groups observed STABLE on ``command_topic`` when the
            locus was resolved, before the publish. The first is probed: on
            this path the delegate-skill orchestrator is the single consumer,
            and a group that has since died yields an unresolved depth with a
            reason rather than a substituted sibling's number.
        probe_seconds: Bound on the whole probe.

    Returns:
        A depth, or ``records_ahead=None`` with ``unresolved_reason`` set.
        Never raises: this runs on a path that has already failed, and a probe
        that turns a typed refusal into a traceback is worse than no probe.
    """
    if bus != "kafka":
        return ModelDelegateQueueDepth(
            unresolved_reason=(
                f"bus is {bus!r}, an in-process transport with no shared "
                "command queue to measure"
            )
        )
    if not broker:
        return ModelDelegateQueueDepth(
            unresolved_reason="the run resolved no broker address to probe"
        )
    if not command_topic:
        return ModelDelegateQueueDepth(
            unresolved_reason="the run resolved no command topic to probe"
        )
    if not consumer_groups:
        return ModelDelegateQueueDepth(
            unresolved_reason=(
                "no consumer group was observed STABLE on "
                f"{command_topic!r} when this run was dispatched, so there is "
                "no committed offset to measure a backlog against"
            )
        )

    consumer_group = consumer_groups[0]
    try:
        backlog = asyncio.run(
            asyncio.wait_for(
                _observe_async(
                    broker=broker,
                    command_topic=command_topic,
                    consumer_group=consumer_group,
                ),
                timeout=probe_seconds,
            )
        )
    except TimeoutError:
        return ModelDelegateQueueDepth(
            consumer_group=consumer_group,
            unresolved_reason=(
                f"the broker did not answer a lag query within {probe_seconds:g}s"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - boundary: a failed probe is a reason
        logger.debug("delegate queue-depth probe failed", exc_info=True)
        return ModelDelegateQueueDepth(
            consumer_group=consumer_group,
            unresolved_reason=(
                f"the lag query failed: {type(exc).__name__}: {exc}"[:300]
            ),
        )

    return ModelDelegateQueueDepth(
        consumer_group=consumer_group,
        backlog_records=backlog,
        # The backlog includes this caller's own unconsumed record; what the
        # caller asked is how many are IN FRONT of it.
        records_ahead=max(backlog - 1, 0),
    )
