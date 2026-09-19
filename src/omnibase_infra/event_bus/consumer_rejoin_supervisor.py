# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded-poll supervisor that rejoins a wedged consumer group (OMN-18640).

**The defect this closes.** ``EventBusKafka._consume_loop`` consumed with
``async for msg in consumer``. When the dev-lane broker was recreated at
2026-09-18T23:16:46Z the client logged ``NodeNotReadyError``, marked its
coordinator dead, and then logged ``GroupCoordinatorNotAvailableError`` 21,200
times over 97 minutes. Not one of those reached the loop: aiokafka retries the
coordinator inside its own background task, so ``__anext__`` simply never
returned. The loop was not failing -- it was waiting, forever, on a group it
would never rejoin, and only recreating the CONTAINER cleared it. The same
shape had occurred 25 hours earlier from a different trigger.

**The fix.** Poll with a deadline instead of waiting forever. Every poll that
returns nothing is an opportunity to ask whether silence means "caught up" or
"wedged", and the answer is measurable: compare the consumer's own fetch
position against the log end offsets reported by the partition LEADERS, which
answer independently of the group coordinator. A consumer that is behind and
not moving, for long enough, with the brokers answering, is closed, recreated
and rejoined from its committed offsets.

**What is deliberately bounded.** A rejoin is ordered only after N consecutive
confirmations and never more often than the cooldown, so a fault that survives
recreation degrades to periodic retries rather than a recreate loop. Both
numbers are declared configuration with no environment fallback.

**What this does not do.** It does not fix a broker that is down: an
unreachable end-offset probe returns "not stalled" and leaves the consumer to
aiokafka's own reconnect. It does not change delivery semantics; the
replacement consumer joins the same group and resumes from the same committed
offsets, which is at-least-once exactly as before.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.consumer_rejoin_policy import (
    ModelConsumerRejoinPolicy,
    evaluate_consumer_stall,
)
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from omnibase_infra.models.health.model_consumer_group_rejoin_event import (
    ModelConsumerGroupRejoinEvent,
)
from omnibase_infra.models.health.model_consumer_poll_observation import (
    ModelConsumerPollObservation,
)
from omnibase_infra.models.health.model_consumer_stall_verdict import (
    ModelConsumerStallVerdict,
)
from omnibase_infra.models.health.model_consumer_sync_status import (
    ModelConsumerSyncStatus,
)
from omnibase_infra.protocols.protocol_rejoinable_consumer import (
    FetchedRecord,
    ProtocolRejoinableConsumer,
)

logger = logging.getLogger(__name__)

# Upper bound on the rejoin events retained in memory for the readiness
# surface to read. Small on purpose: a runtime that has wedged more times than
# this in one lifetime has a problem the newest entries already describe.
REJOIN_HISTORY_CAPACITY: Final[int] = 32

# The reasons under which a group is BEHIND AND NOT ADVANCING, which is the
# question readiness asks. ``NOT_STALLED_WITHIN_COOLDOWN`` is in this set on
# purpose: it means the stall signature still holds and the rejoin path is
# merely not allowed to act yet. The cooldown bounds the REMEDY; it is not a
# statement about the group, and a readiness surface that took it as one would
# read green through the whole of every cooldown (OMN-18640 AC1).
_STALL_SIGNATURES: Final[frozenset[EnumConsumerStallReason]] = frozenset(
    {
        EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
        EnumConsumerStallReason.STALLED_NO_ASSIGNMENT,
        EnumConsumerStallReason.NOT_STALLED_WITHIN_COOLDOWN,
    }
)


class ModelConsumerPollBatch:
    """Records from one bounded poll, plus what the supervisor did about it.

    Not a pydantic model: it carries live ``ConsumerRecord`` objects and the
    consumer handle itself, neither of which is serialisable, and it never
    leaves the process.
    """

    __slots__ = ("consumer", "records", "rejoined", "verdict")

    def __init__(
        self,
        *,
        records: Mapping[TopicPartition, Sequence[FetchedRecord]],
        consumer: ProtocolRejoinableConsumer,
        rejoined: bool,
        verdict: ModelConsumerStallVerdict | None,
    ) -> None:
        self.records = records
        self.consumer = consumer
        self.rejoined = rejoined
        self.verdict = verdict


class ConsumerRejoinSupervisor:
    """Owns the bounded poll and the stall/rejoin decision for one consumer.

    One instance per ``(topic, group_id)``. The caller drives it by awaiting
    :meth:`next_batch` in place of iterating the consumer, and must use the
    consumer handle the returned batch carries: after a rejoin the previous
    handle is stopped and must not be touched again.
    """

    def __init__(
        self,
        *,
        topic: str,
        group_id: str,
        policy: ModelConsumerRejoinPolicy,
        poll_timeout_ms: int,
        recreate_consumer: Callable[[], Awaitable[ProtocolRejoinableConsumer]],
        emit_event: Callable[[ModelConsumerGroupRejoinEvent], Awaitable[None]]
        | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialise the supervisor.

        Args:
            topic: Topic being consumed.
            group_id: Effective consumer group id.
            policy: Stall thresholds.
            poll_timeout_ms: Deadline for one ``getmany``. This is what turns
                an unbounded wait into a loop that can ask questions.
            recreate_consumer: Builds and STARTS a replacement consumer for the
                same topic and group. The supervisor stops the old one first.
            emit_event: Optional sink for the typed rejoin event. Emission
                failure is tolerated; it must never take down the consumer it
                is reporting on.
            clock: Monotonic clock, injectable so the replay test can drive
                97 minutes of wedge without waiting for them.
        """
        self._topic = topic
        self._group_id = group_id
        self._policy = policy
        self._poll_timeout_ms = poll_timeout_ms
        self._recreate_consumer = recreate_consumer
        self._emit_event = emit_event
        self._clock = clock

        self._last_record_at: float = clock()
        self._last_rejoin_at: float | None = None
        self._consecutive_stalls: int = 0
        self._rejoin_events: list[ModelConsumerGroupRejoinEvent] = []

        # --- readiness state (OMN-18640 AC1) ----------------------------
        # Held separately from the rejoin state above because a rejoin RESETS
        # ``_last_record_at``, so a readiness surface derived from it would go
        # green the instant a rejoin was attempted -- whether or not the
        # rejoin worked. ``_stall_started_at`` survives rejoins and survives
        # the cooldown, and is cleared only by evidence of sync.
        self._evaluated: bool = False
        self._latest_reason: EnumConsumerStallReason = (
            EnumConsumerStallReason.NOT_STALLED_PROGRESSING
        )
        self._latest_measurement: ModelConsumerPollObservation | None = None
        self._stall_started_at: float | None = None
        self._rejoin_count: int = 0
        self._last_rejoin_failed: bool = False

    @property
    def rejoin_events(self) -> tuple[ModelConsumerGroupRejoinEvent, ...]:
        """Typed rejoin records, oldest first, for a readiness dimension.

        Exposed in-process and ungated so a readiness probe can read it
        without depending on the feature-flagged consumer-health emitter. A
        recovery surface that only exists when an optional flag is on is the
        failure mode this ticket is about, one layer up.
        """
        return tuple(self._rejoin_events)

    def sync_status(self) -> ModelConsumerSyncStatus:
        """Report whether this group is in sync, for the readiness dimension.

        Cheap and synchronous on purpose: it reports what the poll loop has
        already measured and issues no request of its own. A readiness probe
        that put a ``ListOffsets`` on the wire for every wired topic would be
        a second, differently-timed opinion about the same fact, and the two
        would eventually disagree in front of an operator.
        """
        now = self._clock()
        measurement = self._latest_measurement
        return ModelConsumerSyncStatus(
            topic=self._topic,
            consumer_group=self._group_id,
            evaluated=self._evaluated,
            reason=self._latest_reason,
            # From the last evaluation that actually PROBED the leaders. An
            # evaluation inside the stall window short-circuits before the
            # probe and reports a placeholder zero; carrying that forward
            # would erase a real backlog every time the loop ticked.
            backlog_records=measurement.backlog_records if measurement else 0,
            seconds_since_last_record=max(0.0, now - self._last_record_at),
            stalled_seconds=(
                0.0
                if self._stall_started_at is None
                else max(0.0, now - self._stall_started_at)
            ),
            assigned_partitions=(measurement.assigned_partitions if measurement else 0),
            broker_reachable=measurement.broker_reachable if measurement else True,
            rejoin_count=self._rejoin_count,
            last_rejoin_failed=self._last_rejoin_failed,
            unready_after_seconds=self._policy.sync_unready_seconds,
        )

    def _note_evaluation(self, reason: EnumConsumerStallReason, now: float) -> None:
        """Advance the readiness stall clock from one evaluation's reason.

        Only two outcomes CLEAR a stall, and neither is merely the absence of
        one. ``NOT_STALLED_IDLE`` is a measurement: the fetch position is at
        the end of every assigned partition. A delivered record (handled in
        :meth:`next_batch`) is the other. ``NOT_STALLED_PROGRESSING`` is
        deliberately NOT one of them -- after a rejoin resets the clock it
        means only "inside the window", which is the absence of a measurement,
        and treating it as evidence is how a surface reports recovery from a
        rejoin that did nothing. ``NOT_STALLED_BROKER_UNREACHABLE`` is a
        failed measurement and likewise clears nothing.
        """
        self._evaluated = True
        self._latest_reason = reason
        if reason in _STALL_SIGNATURES:
            if self._stall_started_at is None:
                self._stall_started_at = now
        elif reason is EnumConsumerStallReason.NOT_STALLED_IDLE:
            self._stall_started_at = None

    def _note_flow(self) -> None:
        """Records moved. That is the only unambiguous proof of sync."""
        self._stall_started_at = None
        self._last_rejoin_failed = False
        self._evaluated = True
        self._latest_reason = EnumConsumerStallReason.NOT_STALLED_PROGRESSING

    async def next_batch(
        self, consumer: ProtocolRejoinableConsumer
    ) -> ModelConsumerPollBatch:
        """Poll once with a deadline, and rejoin if the consumer is wedged.

        Args:
            consumer: The consumer handle to poll.

        Returns:
            The records fetched (possibly empty), the consumer handle to use
            for the next call, and the stall verdict when one was evaluated.
        """
        records = await consumer.getmany(timeout_ms=self._poll_timeout_ms)
        if records:
            self._last_record_at = self._clock()
            self._consecutive_stalls = 0
            self._note_flow()
            return ModelConsumerPollBatch(
                records=records,
                consumer=consumer,
                rejoined=False,
                verdict=None,
            )

        verdict, observation = await self._evaluate(consumer)
        if not verdict.should_rejoin:
            return ModelConsumerPollBatch(
                records={},
                consumer=consumer,
                rejoined=False,
                verdict=verdict,
            )

        replacement = await self._rejoin(consumer, verdict, observation)
        return ModelConsumerPollBatch(
            records={},
            consumer=replacement,
            rejoined=replacement is not consumer,
            verdict=verdict,
        )

    async def _evaluate(
        self, consumer: ProtocolRejoinableConsumer
    ) -> tuple[ModelConsumerStallVerdict, ModelConsumerPollObservation]:
        """Measure the consumer and classify it."""
        now = self._clock()
        silent_for = max(0.0, now - self._last_record_at)

        # Short-circuit before probing the broker: an evaluation inside the
        # window cannot produce a stall whatever the offsets say, and probing
        # on every empty poll would put a ListOffsets request on the wire at
        # the poll cadence for every wired topic in the runtime.
        if silent_for < self._policy.stall_seconds:
            observation = ModelConsumerPollObservation(
                topic=self._topic,
                consumer_group=self._group_id,
                seconds_since_last_record=silent_for,
                assigned_partitions=len(consumer.assignment()),
                broker_reachable=True,
                backlog_records=0,
                seconds_since_last_rejoin=self._since_last_rejoin(now),
            )
        else:
            observation = await self._observe(consumer, silent_for, now)
            self._latest_measurement = observation

        verdict = evaluate_consumer_stall(
            observation,
            self._policy,
            prior_consecutive_stalls=self._consecutive_stalls,
        )
        self._consecutive_stalls = verdict.consecutive_stalls
        self._note_evaluation(verdict.reason, now)
        return verdict, observation

    async def _observe(
        self,
        consumer: ProtocolRejoinableConsumer,
        silent_for: float,
        now: float,
    ) -> ModelConsumerPollObservation:
        """Probe leaders for end offsets and compute the backlog."""
        assignment = consumer.assignment()
        if not assignment:
            # No assignment to probe. Reachability is asserted True so the
            # no-assignment stall class is reachable; a client with no
            # assignment and an unreachable broker is covered by the same
            # remedy, and recreating it is the correct action either way.
            return ModelConsumerPollObservation(
                topic=self._topic,
                consumer_group=self._group_id,
                seconds_since_last_record=silent_for,
                assigned_partitions=0,
                broker_reachable=True,
                backlog_records=0,
                seconds_since_last_rejoin=self._since_last_rejoin(now),
            )

        try:
            end_offsets = await consumer.end_offsets(list(assignment))
            positions = {tp: await consumer.position(tp) for tp in assignment}
        except Exception as probe_error:  # noqa: BLE001 — boundary: an unreachable probe is a verdict, not a crash
            logger.debug(
                "consumer_stall_probe_failed topic=%s group=%s error=%s",
                self._topic,
                self._group_id,
                probe_error,
                exc_info=True,
            )
            return ModelConsumerPollObservation(
                topic=self._topic,
                consumer_group=self._group_id,
                seconds_since_last_record=silent_for,
                assigned_partitions=len(assignment),
                broker_reachable=False,
                backlog_records=0,
                seconds_since_last_rejoin=self._since_last_rejoin(now),
            )

        backlog = 0
        for tp in assignment:
            end = end_offsets.get(tp)
            position = positions.get(tp)
            if end is None or position is None:
                continue
            backlog += max(0, int(end) - int(position))

        return ModelConsumerPollObservation(
            topic=self._topic,
            consumer_group=self._group_id,
            seconds_since_last_record=silent_for,
            assigned_partitions=len(assignment),
            broker_reachable=True,
            backlog_records=backlog,
            seconds_since_last_rejoin=self._since_last_rejoin(now),
        )

    def _since_last_rejoin(self, now: float) -> float | None:
        if self._last_rejoin_at is None:
            return None
        return max(0.0, now - self._last_rejoin_at)

    async def _rejoin(
        self,
        consumer: ProtocolRejoinableConsumer,
        verdict: ModelConsumerStallVerdict,
        observation: ModelConsumerPollObservation,
    ) -> ProtocolRejoinableConsumer:
        """Close the wedged consumer and start a replacement.

        Returns the replacement on success, or the original handle when the
        replacement could not be started -- the caller keeps polling either
        way, and the cooldown spaces out the next attempt.
        """
        now = self._clock()
        self._last_rejoin_at = now
        self._consecutive_stalls = 0

        logger.error(
            "consumer_group_rejoin_forced topic=%s group=%s reason=%s "
            "silent_for=%.1fs backlog=%d -- the consumer held its group and "
            "stopped fetching while the partition leaders were answering; "
            "closing and rejoining from committed offsets (OMN-18640)",
            self._topic,
            self._group_id,
            verdict.reason.value,
            observation.seconds_since_last_record,
            observation.backlog_records,
            extra={
                "topic": self._topic,
                "group_id": self._group_id,
                "reason": verdict.reason.value,
            },
        )

        try:
            await consumer.stop()
        except Exception as stop_error:  # noqa: BLE001 — boundary: a wedged client may also fail to close
            logger.warning(
                "consumer_group_rejoin_stop_failed topic=%s group=%s error=%s",
                self._topic,
                self._group_id,
                stop_error,
                exc_info=True,
            )

        replacement: ProtocolRejoinableConsumer = consumer
        succeeded = False
        failure_detail = ""
        try:
            replacement = await self._recreate_consumer()
            succeeded = True
        except Exception as recreate_error:
            failure_detail = f"{type(recreate_error).__name__}: {recreate_error}"[:500]
            logger.exception(
                "consumer_group_rejoin_failed topic=%s group=%s error=%s",
                self._topic,
                self._group_id,
                recreate_error,
            )

        self._rejoin_count += 1
        # A failed rejoin is held until flow RESUMES, not until the next
        # evaluation looks calmer: the rejoin below resets _last_record_at on
        # success, so without this the readiness surface would clear itself on
        # the strength of a recovery that never happened (OMN-18640 AC1).
        self._last_rejoin_failed = not succeeded

        if succeeded:
            self._last_record_at = self._clock()

        await self._record(verdict, observation, succeeded, failure_detail)
        return replacement

    async def _record(
        self,
        verdict: ModelConsumerStallVerdict,
        observation: ModelConsumerPollObservation,
        succeeded: bool,
        failure_detail: str,
    ) -> None:
        """Append the typed event and hand it to the optional sink."""
        event = ModelConsumerGroupRejoinEvent(
            topic=self._topic,
            consumer_group=self._group_id,
            reason=verdict.reason,
            stalled_seconds=observation.seconds_since_last_record,
            backlog_records=observation.backlog_records,
            consecutive_stalls=verdict.required_consecutive_stalls,
            rejoin_succeeded=succeeded,
            failure_detail=failure_detail,
        )
        self._rejoin_events.append(event)
        if len(self._rejoin_events) > REJOIN_HISTORY_CAPACITY:
            del self._rejoin_events[
                0 : len(self._rejoin_events) - REJOIN_HISTORY_CAPACITY
            ]

        if self._emit_event is None:
            return
        try:
            await self._emit_event(event)
        except Exception:  # noqa: BLE001 — boundary: best-effort emission must not break recovery
            logger.debug(
                "consumer_group_rejoin_emit_failed topic=%s group=%s",
                self._topic,
                self._group_id,
                exc_info=True,
            )


__all__ = [
    "REJOIN_HISTORY_CAPACITY",
    "ConsumerRejoinSupervisor",
    "ModelConsumerPollBatch",
]
