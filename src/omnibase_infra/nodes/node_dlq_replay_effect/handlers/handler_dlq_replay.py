# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for the contract-native DLQ replay node (OMN-12619).

Drives the relocated replay engine: consume DLQ messages from a persistent
consumer group, decide eligibility with the reused ``should_replay()``, replay
eligible messages exactly once to their original topic, and QUARANTINE
non-replayable messages to ``onex.dlq.omnibase-infra.quarantine.v1`` instead of the legacy
skip-and-drop path that silently lost messages.

Truthfulness invariants:
    - A replay attempt that raises is recorded as FAILED (never COMPLETED).
    - A QUARANTINED outcome is recorded only after the quarantine publish
      succeeds; a failed quarantine publish is recorded as FAILED.
    - Tracking (``dlq_replay_history``) records every terminal outcome --
      including the OMN-17896 unparseable-record path, which reached its own
      durable quarantine without ever calling the recorder until OMN-18111.
    - An audit write is a side effect of an ALREADY-DURABLE outcome and can
      therefore never change one: a tracking failure costs an audit row, never
      a verdict and never a committable offset (OMN-18111).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from weakref import WeakKeyDictionary

from aiokafka.errors import KafkaError

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.dlq.models.model_dlq_replay_record import ModelDlqReplayRecord
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import (
    DlqDependencyLifecycleTimeoutError,
    DlqTopicFixedPointError,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
    generate_replay_correlation_id,
    should_replay,
    unwrap_nested_dlq_record,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_commit_ledger import (
    ModelDlqCommitLedger,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_result import (
    ModelDlqReplayResult,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_run_result import (
    ModelDlqReplayRunResult,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
    ModelUnparseableDlqRecord,
)

if TYPE_CHECKING:
    from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking

logger = logging.getLogger(__name__)

HANDLER_ID_DLQ_REPLAY: str = "dlq-replay-handler"


class ProtocolDlqBacklogProbe(Protocol):
    """Reads the replay group's undrained count per topic without joining it.

    ``DlqGroupBacklogProbe`` in ``engine_dlq_replay`` is the runtime
    implementation (OMN-19085).
    """

    async def undrained(self, topics: Sequence[str]) -> Mapping[str, int]: ...


class DlqConsumerDrainState:
    """What every run over one consumer shares: its mutex and its failure count.

    ``failed_attempts`` and ``halted`` are OMN-19241. They live beside the run
    mutex, not on a handler, because the three dispatchers are three handler
    instances over the same consumers; a count kept per handler would let each
    of them spend the whole bound. Both are read and written only while the
    mutex is held.
    """

    __slots__ = ("failed_attempts", "halted", "lock")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        # (partition, offset) -> consecutive runs whose handling of it FAILED.
        self.failed_attempts: dict[tuple[int, int], int] = {}
        # partition -> the offset whose record halted it.
        self.halted: dict[int, int] = {}


_DRAIN_STATES: WeakKeyDictionary[object, DlqConsumerDrainState] = WeakKeyDictionary()
"""One run mutex per shared consumer object (OMN-18084).

``service_kernel`` keys runtime dependencies by handler NAME, so the three
per-topic dispatcher entries OMN-18013 split this node's routing into all
resolve to the same ``dependencies["HandlerDlqReplay"]`` mapping — one
``DLQConsumer`` behind three ``HandlerDlqReplay`` instances. Nothing then
serialised them, and the run's dependency start-up skipped a dependency
already flagged ``_started``, so it did not put that dependency in the list its
own ``finally`` stopped while the peer that DID start it stopped it
unconditionally. One dispatcher's teardown therefore lands under another's
drain, and the victim's next read of the consumer raises
``RuntimeError("Consumer not started")``.

Keyed on the consumer INSTANCE rather than held on the handler, because the
consumer is the object that is actually shared: dispatchers that own separate
consumers must not serialise against each other, and two handlers over one
consumer must. Weak keys so a discarded consumer takes its lock with it.

This also stops two runs iterating one ``AIOKafkaConsumer`` concurrently, which
was never safe independently of the lifecycle flag.
"""


def _drain_state_for(consumer: object) -> DlqConsumerDrainState:
    state = _DRAIN_STATES.get(consumer)
    if state is None:
        state = DlqConsumerDrainState()
        _DRAIN_STATES[consumer] = state
    return state


def _run_lock_for(consumer: object) -> asyncio.Lock:
    """Return the mutex guarding one shared consumer's start/drain/stop."""
    return _drain_state_for(consumer).lock


class DlqDependencyLease:
    """Reference count over one dependency that runs share (OMN-19241).

    The run mutex above is keyed on the consumer, and since OMN-18119 each
    topic has its own, so runs over different topics proceed concurrently. The
    replay and quarantine producers are NOT per topic: ``service_kernel``
    builds one of each and hands them to every dispatcher. Before this, a run
    that found a producer stopped started it and stopped it in its ``finally``,
    and a run that found it started borrowed it and stopped nothing -- so the
    starter finishing first stopped the producer under the borrower. Measured
    on the .201 dev lane 2026-09-23: every quarantine from 10:32:48Z raised
    ``Quarantine producer not started``.

    A lease stops a dependency only when the LAST holder releases it, and only
    if a lease started it. A dependency the caller started itself (as
    ``scripts/dlq_replay.py`` does) is never stopped here.

    Chosen over the two alternatives. A run mutex keyed on the producers would
    serialise every topic behind one lock, and a run that cannot take the mutex
    within its budget skips the topic (OMN-17137), so a busy topic would starve
    the other two. Per-run producers would need a factory in the kernel's
    dependency mapping and in the script; this keeps both call sites unchanged.
    """

    __slots__ = ("holders", "lock", "owned")

    def __init__(self) -> None:
        # Held across start() and stop() only, never across a drain.
        self.lock = asyncio.Lock()
        self.holders = 0
        self.owned = False


_DEPENDENCY_LEASES: WeakKeyDictionary[object, DlqDependencyLease] = WeakKeyDictionary()


def _lease_for(dependency: object) -> DlqDependencyLease:
    lease = _DEPENDENCY_LEASES.get(dependency)
    if lease is None:
        lease = DlqDependencyLease()
        _DEPENDENCY_LEASES[dependency] = lease
    return lease


def _halted_coordinate(topic: str, partition: int, offset: int) -> str:
    return f"{topic}/{partition}@{offset}"


class HandlerDlqReplay:
    """EFFECT handler that replays or quarantines DLQ messages.

    Dependencies (constructor-injected):
        consumers: One DLQ consumer per DECLARED subscribe topic, keyed by
            topic, all on the same persistent group. OMN-18119: this was a
            single ``consumer`` and the contract declares THREE subscribe
            topics, so two of the three were drained by nothing at all.
        producer: Replays eligible messages to the original topic.
        quarantine_producer: Publishes non-replayable messages to quarantine.
        tracking: Optional ``ServiceDlqTracking`` for dlq_replay_history.
        backlog_probe: Optional reader of the replay group's undrained count
            per topic (OMN-19085). When it reports a topic fully committed, the
            run skips that topic's consumer start -- a group join that costs
            the broker's initial rebalance delay -- because there is nothing
            on it to drain. Absent, or when a read fails, every topic is
            drained exactly as before.
    """

    def __init__(
        self,
        *,
        consumers: Mapping[str, DLQConsumer],
        producer: DLQProducer,
        quarantine_producer: DLQQuarantineProducer,
        tracking: ServiceDlqTracking | None = None,
        backlog_probe: ProtocolDlqBacklogProbe | None = None,
    ) -> None:
        if not consumers:
            raise ValueError(
                "HandlerDlqReplay requires at least one DLQ consumer; an empty "
                "mapping would drain nothing while reporting clean runs"
            )
        self._consumers: dict[str, DLQConsumer] = dict(consumers)
        self._producer = producer
        self._quarantine_producer = quarantine_producer
        self._tracking = tracking
        self._backlog_probe = backlog_probe
        # Bounds, filters and the quarantine topic are identical across the
        # per-topic configs -- only ``dlq_topic`` differs, and every use of it
        # takes the DRAINED consumer's own config rather than this one.
        self._config: ModelDlqReplayEngineConfig = next(
            iter(self._consumers.values())
        ).config
        self._start_index = 0

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, envelope: ModelEventEnvelope[ModelDlqReplayRunResult]
    ) -> ModelHandlerOutput[ModelDlqReplayRunResult]:
        """Canonical entry point: drain the DLQ and return the run result.

        The envelope carries the correlation context for the run. The payload
        type is the run result for causality typing; the run itself is driven
        by the injected engine and ``self._config``.

        OMN-15021: ``envelope`` is NOT guaranteed to be a ``ModelEventEnvelope``
        instance at runtime despite the type hint. Because this method's first
        parameter is literally named ``envelope``,
        ``handler_wiring._handler_accepts_event_envelope`` classifies this
        handler as envelope-accepting and the auto-wiring dispatch callback
        (``_make_dispatch_callback``, ``event_model is None`` / operation_match
        branch) hands it the RAW materialized dispatch dict
        (``{"payload": ..., "__bindings": ..., "__debug_trace": ...}`` --
        ``ModelMaterializedDispatch``) unchanged, never a hydrated envelope. A
        bare ``envelope.correlation_id`` attribute access previously crashed
        with ``AttributeError: 'dict' object has no attribute 'correlation_id'``
        on every dispatch delivered this way -- observed live 2026-07-24 when
        ``onex.dlq.omnibase-infra.events.v1`` took its first real traffic under
        ``ONEX_BOUNDARY_DLQ_ENABLED``. Extraction below tolerates both shapes
        and NEVER raises; a genuinely undecodable envelope still produces a
        working ``handle()`` (fresh generated ids) with a loud WARNING log --
        never a silent swallow. This does not affect DLQ record fidelity: the
        actual per-message decode is ``run()``'s own typed Kafka consumption
        via ``ModelDlqMessage.from_kafka_message``, which is independently
        defensive and unaffected by this entry point's id extraction.
        """
        correlation_id = _extract_envelope_correlation_id(envelope)
        envelope_id = _extract_envelope_id(envelope)
        if correlation_id is None or envelope_id is None:
            generated_id = uuid4()
            logger.warning(
                "HandlerDlqReplay.handle() received a dispatch envelope that is "
                "not a hydrated ModelEventEnvelope (type=%s) -- this is the "
                "runtime auto-wiring boundary's materialized dispatch-dict shape "
                "(OMN-15021), not malformed DLQ content. Falling back to a "
                "generated id for the missing field(s) instead of crashing; the "
                "DLQ drain itself is unaffected.",
                type(envelope).__name__,
            )
            if correlation_id is None:
                correlation_id = generated_id
            if envelope_id is None:
                envelope_id = generated_id
        run_result = await self.run()
        return ModelHandlerOutput.for_compute(
            input_envelope_id=envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_DLQ_REPLAY,
            result=run_result,
        )

    async def run(self) -> ModelDlqReplayRunResult:
        """Drain EVERY declared DLQ topic for one BOUNDED batch.

        This handler is auto-wired as a PER-MESSAGE trigger on the DLQ topics
        (``event_bus.subscribe_topics`` in ``contract.yaml``) but drains via a
        persistent, whole-topic-shaped consumer group.

        OMN-16422 bounded every invocation two ways -- record count
        (``max_records_per_run``, additionally clamped by an explicit ``limit``)
        and wall-clock (``max_run_duration_seconds``) -- and commits
        incrementally every ``commit_every_n_records`` records, so a bounded
        invocation always returns quickly (keeping the OUTER trigger consumer's
        poll loop alive) and always makes committed progress even while the
        topic is self-feeding.

        OMN-17137: the wall-clock bound MUST be enforced on the WAIT for the
        next record, not only between records. ``DLQConsumer.consume_messages()``
        iterates an ``AIOKafkaConsumer`` whose ``__anext__`` blocks until a
        record arrives; ``consumer_timeout_ms`` is aiokafka's background
        fetching wait, not an idle-iteration timeout. A deadline checked only
        inside the loop body is never re-evaluated once the topic goes idle, so
        ``run()`` parked in ``getone()`` forever and held the outer trigger
        consumer until aiokafka evicted it. Every acquisition below is driven
        through ``asyncio.wait_for``.

        OMN-18084: runs over ONE shared consumer are SERIALISED, because
        ``service_kernel`` keys dependencies by handler name and the three
        per-topic dispatcher entries all resolve to one dependency mapping.

        OMN-18119 -- what changed, and the three bounds that make it safe.
        The contract declares THREE subscribe topics and this handler took one
        consumer, so ``config.dlq_topic`` decided the whole drain and the other
        two topics were consumed by nothing. Live on the .201 dev lane the
        replay group held a committed offset for exactly ONE topic-partition
        while 1,661 records sat in the commands DLQ. The handler now holds one
        consumer per declared topic and drains each in turn:

        * **The wall clock is SHARED across topics**, not multiplied by them. A
          run still returns inside ``max_run_duration_seconds`` regardless of
          how many topics are declared, which is what keeps OMN-17137 closed.
        * **The record budget is PER TOPIC.** A shared record budget would be
          spent entirely by the first topic in the order, and on this lane the
          first topic is a 700,000-record backlog -- the other two would never
          see a single record.
        * **An idle topic costs one short probe.** The FIRST record on a topic
          is awaited for at most ``idle_probe_seconds``; an empty topic breaks
          there instead of burning the run's remaining budget. Once a record
          arrives the topic continues under the shared remaining budget.

        The start of the order ROTATES by one each run. Without that, a topic
        whose predecessor can always fill its own budget never gets reached at
        all while a backlog is hot -- the starvation is not hypothetical, it is
        the steady state of this lane today.
        """
        results: list[ModelDlqReplayResult] = []
        topics = list(self._consumers)
        start = self._start_index % len(topics)
        self._start_index = (start + 1) % len(topics)
        order = topics[start:] + topics[:start]

        deadline = time.monotonic() + self._config.max_run_duration_seconds
        backlog = await self._read_backlog(order, deadline)
        for topic in order:
            # OMN-19085: a topic the replay group has fully committed has
            # nothing to drain, so its consumer is not started. Starting it is
            # a join of the shared onex-dlq-replay group, which waits the
            # broker's initial rebalance delay (3 s on redpanda) every time,
            # and the trigger that asked for this run is itself a record on a
            # declared topic -- so a zero read now proves that record was
            # already handled. Only an explicit 0 skips; a missing answer
            # drains.
            if backlog is not None and backlog.get(topic) == 0:
                logger.debug(
                    "DLQ replay group has no undrained record on %s; skipping "
                    "its consumer start for this run (OMN-19085).",
                    topic,
                )
                continue
            if time.monotonic() >= deadline:
                logger.debug(
                    "DLQ replay run exhausted its wall-clock budget before "
                    "reaching %s; it leads the order next run (OMN-18119).",
                    topic,
                )
                break
            consumer = self._consumers[topic]
            # OMN-18084: the whole start -> drain -> stop sequence is one
            # critical section per shared consumer. Acquiring per consumer
            # rather than around the dispatch keeps the scope exactly the
            # lifecycle that is shared; a dispatcher whose consumer nobody
            # else holds never waits.
            #
            # OMN-17137 (second pass): that acquisition is itself an await, and
            # an UNBOUNDED one. A peer run wedged in its own teardown holds the
            # mutex forever, so every later run queued behind it inherits the
            # wedge -- which is how ONE stuck consumer took down both trafficked
            # topics on the stability lane while the untrafficked third stayed
            # Stable. A run that cannot take the mutex inside its remaining
            # budget gives the topic up for this pass; the rotation puts it at
            # the head of the order next time.
            lock = _run_lock_for(consumer)
            lock_wait = deadline - time.monotonic()
            try:
                await asyncio.wait_for(lock.acquire(), timeout=lock_wait)
            except TimeoutError:
                logger.warning(
                    "DLQ replay run could not take the run mutex for %s within "
                    "its remaining %.2fs budget -- a peer run still holds it. "
                    "Skipping this topic rather than parking the outer trigger "
                    "consumer (OMN-17137).",
                    topic,
                    max(lock_wait, 0.0),
                )
                continue
            try:
                results.extend(await self._run_locked(consumer, deadline))
            finally:
                lock.release()

        return self._summarize(results, tuple(order), self._halted_partitions())

    async def _read_backlog(
        self, topics: Sequence[str], deadline: float
    ) -> Mapping[str, int] | None:
        """The replay group's undrained count per topic, or None if unknown.

        OMN-19085. Bounded by ``backlog_probe_timeout_seconds`` AND by what is
        left of the run's shared wall clock, so the read can shorten a run but
        never lengthen it past ``max_run_duration_seconds`` (OMN-17137). Every
        failure answers None, which makes the run drain every topic exactly as
        it did before the probe existed: an unreadable backlog must cost time,
        never a record.
        """
        if self._backlog_probe is None:
            return None
        budget = min(
            self._config.backlog_probe_timeout_seconds,
            deadline - time.monotonic(),
        )
        if budget <= 0.0:
            return None
        try:
            return await asyncio.wait_for(
                self._backlog_probe.undrained(topics), timeout=budget
            )
        except (TimeoutError, KafkaError, OSError) as exc:
            logger.warning(
                "DLQ replay could not read the replay group's backlog within "
                "%.2fs (%s: %s); draining every declared topic this run "
                "(OMN-19085).",
                budget,
                type(exc).__name__,
                exc,
            )
            return None
        except Exception:
            # Deliberately broad, and deliberately NOT a quarantine or a verdict
            # path: the only consequence of catching here is the pre-OMN-19085
            # behaviour (drain every topic). Letting an unexpected probe error
            # escape would fail the whole dispatch, and the boundary answers a
            # failed dispatch on this node by dead-lettering its trigger onto
            # the topic it drains -- the OMN-18084 amplifier. Logged with the
            # traceback on every trigger, so a defect here is loud, not silent.
            logger.exception(
                "DLQ replay backlog probe raised unexpectedly; draining every "
                "declared topic this run (OMN-19085)."
            )
            return None

    async def _run_locked(
        self, consumer: DLQConsumer, deadline: float
    ) -> list[ModelDlqReplayResult]:
        """Drain ONE topic, holding that consumer's run mutex.

        ``deadline`` is the RUN's shared wall clock, not this topic's own.
        """
        config = consumer.config
        state = _drain_state_for(consumer)
        leased_dependencies = await self._acquire_runtime_dependencies(consumer)
        try:
            results: list[ModelDlqReplayResult] = []
            count = 0
            uncommitted = 0
            withheld = 0
            # OMN-17896: the offsets whose handling actually COMPLETED, per
            # partition, as a contiguous prefix. A partition is BLOCKED by the
            # first record on it that did not complete, so a later success can
            # never commit past an earlier failure. Both commits below use
            # this map; a bare ``commit()`` commits the consumer's POSITION —
            # every record the iterator has already handed out, including one
            # whose handling never finished — so any early exit advanced past
            # the in-flight record.
            ledger = ModelDlqCommitLedger()
            # OMN-19241: a halted partition starts the run blocked, so nothing
            # on it is attempted and nothing on it is committed.
            for partition in state.halted:
                ledger.block((config.dlq_topic, partition))
            limit = config.limit
            max_records = config.max_records_per_run
            effective_limit = max_records if limit is None else min(limit, max_records)
            commit_every = config.commit_every_n_records

            messages = consumer.consume_messages()
            try:
                while count < effective_limit:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        logger.warning(
                            "DLQ replay run hit its wall-clock bound (%.1fs) after "
                            "%d records on %s -- returning this bounded batch "
                            "instead of continuing to drain (OMN-16422).",
                            config.max_run_duration_seconds,
                            count,
                            config.dlq_topic,
                        )
                        break

                    # OMN-17137: bound the WAIT, not just the gap between
                    # records. Without this timeout an idle topic parks the
                    # run in aiokafka's unbounded ``getone()`` and the
                    # deadline above is never reached again.
                    #
                    # OMN-18119: the FIRST record of a topic gets the short
                    # idle probe rather than the whole remaining budget, so a
                    # declared-but-empty topic costs ``idle_probe_seconds``
                    # and the topics behind it in the order still get drained.
                    wait = (
                        remaining
                        if count
                        else min(config.idle_probe_seconds, remaining)
                    )
                    try:
                        message = await asyncio.wait_for(anext(messages), timeout=wait)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        if count:
                            logger.warning(
                                "DLQ replay run hit its wall-clock bound (%.1fs) "
                                "after %d records on %s while WAITING for the "
                                "next record -- the topic went idle mid-drain. "
                                "Returning this bounded batch rather than parking "
                                "the outer trigger consumer (OMN-17137).",
                                config.max_run_duration_seconds,
                                count,
                                config.dlq_topic,
                            )
                        else:
                            logger.debug(
                                "DLQ topic %s produced no record within the "
                                "%.2fs idle probe; moving to the next declared "
                                "topic (OMN-18119).",
                                config.dlq_topic,
                                wait,
                            )
                        break

                    count += 1
                    # OMN-19241: nothing BEHIND a failed record on its partition
                    # is handled in the same run. Its offset cannot be committed
                    # over the failed one, so the next run reads it again; a
                    # record replayed here would be replayed once per run for as
                    # long as the failure lasts. It is left for redelivery.
                    if self._ledger_key(message, config.dlq_topic) in ledger.blocked:
                        withheld += 1
                        continue

                    if isinstance(message, ModelUnparseableDlqRecord):
                        result = await self._quarantine_unparseable(message)
                    else:
                        result = await self._process_message(message, config)
                    results.append(result)
                    self._mark_offset(message, result, ledger, config.dlq_topic)
                    self._count_failure(message, result, state, config)
                    uncommitted += 1

                    if (
                        not config.dry_run
                        and uncommitted >= commit_every
                        and ledger.has_committable_offsets
                    ):
                        await consumer.commit_offsets(ledger.completed)
                        uncommitted = 0
            finally:
                # Release the iterator deterministically. After a timeout the
                # cancelled ``__anext__`` has already finalized an async
                # generator, so this is a no-op there; it matters for the
                # count-bound and StopAsyncIteration exits.
                aclose = getattr(messages, "aclose", None)
                if aclose is not None:
                    await aclose()

            if uncommitted and not config.dry_run and ledger.has_committable_offsets:
                await consumer.commit_offsets(ledger.completed)

            if withheld:
                logger.warning(
                    "DLQ replay withheld %d record(s) on %s behind a record that "
                    "failed or a halted partition (%s); they are redelivered, "
                    "not replayed twice (OMN-19241).",
                    withheld,
                    config.dlq_topic,
                    sorted(ledger.blocked),
                )
            return results
        finally:
            await self._release_runtime_dependencies(leased_dependencies)

    def _count_failure(
        self,
        message: DlqDrainRecord,
        result: ModelDlqReplayResult,
        state: DlqConsumerDrainState,
        config: ModelDlqReplayEngineConfig,
    ) -> None:
        """Bound how often one record may fail, and say so when it trips.

        OMN-19241. A FAILED record withholds its offset (OMN-17896), which is
        right, but the next run reads it again and fails again for as long as
        the cause lasts, and on the .201 dev lane that was every run for hours
        with nothing but a traceback per attempt. At
        ``max_record_failure_attempts`` the partition is halted instead: it is
        not attempted again in this process, one CRITICAL line names the
        coordinate, and every run result carries it. Restarting the runtime
        after fixing the cause clears it.
        """
        coordinate = (message.dlq_partition, message.dlq_offset)
        if result.status != EnumReplayStatus.FAILED:
            state.failed_attempts.pop(coordinate, None)
            return
        attempts = state.failed_attempts.get(coordinate, 0) + 1
        bound = config.max_record_failure_attempts
        if attempts < bound:
            state.failed_attempts[coordinate] = attempts
            logger.error(
                "DLQ record %s failed (attempt %d of %d): %s. Its offset is "
                "withheld and it is retried on the next run (OMN-19241).",
                _halted_coordinate(config.dlq_topic, *coordinate),
                attempts,
                bound,
                result.message,
            )
            return
        state.failed_attempts.pop(coordinate, None)
        state.halted[message.dlq_partition] = message.dlq_offset
        logger.critical(
            "DLQ replay HALTED partition %s: the record there failed %d times "
            "(last: %s). Nothing on this partition is replayed, quarantined or "
            "committed until the runtime restarts; fix the cause, then "
            "restart. Consumer lag on it will grow (OMN-19241).",
            _halted_coordinate(config.dlq_topic, *coordinate),
            attempts,
            result.message,
        )

    def _halted_partitions(self) -> tuple[str, ...]:
        return tuple(
            _halted_coordinate(topic, partition, offset)
            for topic, consumer in self._consumers.items()
            for partition, offset in sorted(_drain_state_for(consumer).halted.items())
        )

    async def _acquire_runtime_dependencies(
        self, consumer: DLQConsumer
    ) -> list[object]:
        """Lease this run's Kafka dependencies, starting any that are stopped.

        Only the consumer for the topic being drained is started; the peers for
        the other declared topics stay closed until their own turn, so a run
        holds one Kafka consumer at a time rather than one per declared topic
        (OMN-18119).

        OMN-17137 (second pass): every ``start()`` here is bounded by
        ``dependency_lifecycle_timeout_seconds``. ``AIOKafkaConsumer.start()``
        joins a consumer group, and on this node's shared ``onex-dlq-replay``
        group that join lands in a permanent rebalance. An unbounded join holds
        the outer trigger consumer exactly as an unbounded record wait did.

        OMN-19241: the producers are shared by every dispatcher, so each
        dependency is LEASED rather than started-and-stopped by whoever got
        there first (see ``DlqDependencyLease``).
        """
        leased: list[object] = []
        try:
            for dependency in (
                consumer,
                self._producer,
                self._quarantine_producer,
            ):
                await self._acquire_lease(dependency)
                leased.append(dependency)
            return leased
        except Exception:
            await self._release_runtime_dependencies(leased)
            raise

    async def _acquire_lease(self, dependency: object) -> None:
        """Take one holder on ``dependency``, starting it if it is stopped.

        Both awaits are bounded by ``dependency_lifecycle_timeout_seconds``.
        The lease lock is held only across a peer's ``start()`` or ``stop()``,
        each already bounded, so its wait is too.
        """
        budget = self._config.dependency_lifecycle_timeout_seconds
        lease = _lease_for(dependency)
        name = type(dependency).__name__
        try:
            await asyncio.wait_for(lease.lock.acquire(), timeout=budget)
        except TimeoutError as exc:
            raise DlqDependencyLifecycleTimeoutError(
                f"the lease on {name} was not free within {budget:.2f}s; ending "
                f"this DLQ replay run so the outer trigger consumer keeps "
                f"polling (OMN-19241)",
                dependency=name,
                operation="lease",
                timeout_seconds=budget,
            ) from exc
        try:
            start = getattr(dependency, "start", None)
            if start is not None and not getattr(dependency, "_started", False):
                try:
                    await asyncio.wait_for(start(), timeout=budget)
                except TimeoutError as exc:
                    raise DlqDependencyLifecycleTimeoutError(
                        f"{name}.start() did not complete within {budget:.2f}s; "
                        f"ending this DLQ replay run so the outer trigger "
                        f"consumer keeps polling (OMN-17137)",
                        dependency=name,
                        operation="start",
                        timeout_seconds=budget,
                    ) from exc
                lease.owned = True
            lease.holders += 1
        finally:
            lease.lock.release()

    async def _release_runtime_dependencies(self, dependencies: list[object]) -> None:
        """Release this run's leases; the LAST holder stops what a lease started.

        OMN-17137 (second pass) -- the stop is the await the live wedge was in,
        and it is the one place a timeout must NOT propagate. Measured on the
        .201 dev lane 2026-09-16: the three per-topic DLQ consumers share one
        Kafka group and this handler starts and stops one of them on every
        trigger message, so the group rebalances without pause (generation
        227,675 eight minutes after a cold boot). ``AIOKafkaConsumer.stop()``
        issued into that storm awaits a coordinator close that never settles.
        Because this runs in ``_run_locked``'s ``finally``, that unbounded await
        was holding BOTH the run mutex and the outer trigger consumer's serial
        poll loop; aiokafka evicted the outer consumer at
        ``max_poll_interval_ms`` and nothing rejoined it, because a rejoin only
        happens on the next poll and the loop was still inside this ``await``.

        A timed-out stop is reported at ERROR and ABANDONED rather than raised:
        the drain it is tearing down has already produced its committed,
        durable result, and turning a completed batch into an exception would
        discard that result and redeliver every record in it. The dependency is
        left for the next lease's ``_started`` check to reconcile -- degraded,
        and strictly better than a runtime that never polls again.
        """
        budget = self._config.dependency_lifecycle_timeout_seconds
        for dependency in reversed(dependencies):
            lease = _lease_for(dependency)
            name = type(dependency).__name__
            try:
                await asyncio.wait_for(lease.lock.acquire(), timeout=budget)
            except TimeoutError:
                # The holder count still has to drop, or no later release can
                # ever reach zero. A bare decrement is safe on one event loop.
                lease.holders -= 1
                logger.exception(
                    "DLQ replay release of %s could not take its lease within "
                    "%.2fs; the holder was dropped without a stop (OMN-19241).",
                    name,
                    budget,
                )
                continue
            try:
                lease.holders -= 1
                if lease.holders > 0 or not lease.owned:
                    continue
                lease.owned = False
                stop = getattr(dependency, "stop", None)
                if stop is None:
                    continue
                try:
                    await asyncio.wait_for(stop(), timeout=budget)
                except TimeoutError:
                    logger.exception(
                        "DLQ replay teardown of %s did not complete within "
                        "%.2fs and was ABANDONED. This is the OMN-17137 wedge: "
                        "an unbounded stop() holds the outer trigger consumer's "
                        "poll loop until aiokafka evicts it at "
                        "max_poll_interval_ms, and nothing rejoins because a "
                        "rejoin needs a poll. The drain's committed offsets "
                        "stand.",
                        name,
                        budget,
                    )
            finally:
                lease.lock.release()

    @staticmethod
    def _ledger_key(message: DlqDrainRecord, dlq_topic: str) -> tuple[str, int]:
        # ``ModelDlqMessage`` carries no DLQ topic of its own — a consumer
        # drains exactly one, named by ITS OWN config, which is why the caller
        # passes it rather than reading a handler-wide primary (OMN-18119) —
        # while the unparseable record carries its own so the two shapes key
        # identically.
        topic = (
            message.dlq_topic
            if isinstance(message, ModelUnparseableDlqRecord)
            else dlq_topic
        )
        return (topic, message.dlq_partition)

    def _mark_offset(
        self,
        message: DlqDrainRecord,
        result: ModelDlqReplayResult,
        ledger: ModelDlqCommitLedger,
        dlq_topic: str,
    ) -> None:
        """Record whether this record's offset may be committed (OMN-17896).

        A record COMPLETED when it reached a durable terminal outcome — it was
        replayed, it was durably quarantined, or the run is a dry run and
        published nothing at all. A ``FAILED`` result means the record is
        durable NOWHERE: neither replayed nor quarantined. Advancing past it
        would be the silent drop §4 rule 1 of the lab repair plan forbids,
        reached through the quarantine path rather than through a ``continue``.
        Its partition is blocked for the rest of the batch so no later success
        can commit over it.
        """
        key = self._ledger_key(message, dlq_topic)
        if result.status == EnumReplayStatus.FAILED:
            ledger.block(key)
            return
        ledger.mark_completed(key, message.dlq_offset + 1)

    async def _quarantine_unparseable(
        self, record: ModelUnparseableDlqRecord
    ) -> ModelDlqReplayResult:
        """Durably quarantine a record that could not be parsed (OMN-17896).

        The publish is CONFIRMED — the broker's own record metadata is bound
        here — before the caller is allowed to mark the offset committable. A
        failed publish yields ``FAILED``, which blocks the partition, so the
        record is redelivered rather than acked on a quarantine that never
        happened.

        OMN-18111: every exit below that published (or tried to) also writes an
        audit row. Before that ticket this method wrote none at all, so the
        records whose original body could not be established — exactly the ones
        a reclassification owner needs an audit trail for — were the ones
        ``dlq_replay_history`` was guaranteed to be silent about.
        """
        quarantine_correlation_id = generate_replay_correlation_id()

        if self._config.dry_run:
            return ModelDlqReplayResult(
                correlation_id=quarantine_correlation_id,
                original_topic=record.dlq_topic,
                status=EnumReplayStatus.PENDING,
                message=f"DRY RUN - would quarantine unparseable: {record.reason}",
                replay_correlation_id=quarantine_correlation_id,
            )

        try:
            confirmation = (
                await self._quarantine_producer.quarantine_unparseable_record(
                    record, quarantine_correlation_id
                )
            )
        except Exception as exc:
            logger.exception(
                "FAILED to quarantine unparseable DLQ record at %s/%s/%s -- the "
                "offset is WITHHELD so the record is redelivered",
                record.dlq_topic,
                record.dlq_partition,
                record.dlq_offset,
            )
            await self._record_unparseable(
                record,
                EnumReplayStatus.FAILED,
                quarantine_correlation_id,
                error_message=f"Quarantine of unparseable record failed: {exc}",
            )
            return ModelDlqReplayResult(
                correlation_id=quarantine_correlation_id,
                original_topic=record.dlq_topic,
                status=EnumReplayStatus.FAILED,
                message=f"Quarantine of unparseable record failed: {exc}",
                replay_correlation_id=quarantine_correlation_id,
            )

        if confirmation is None:
            logger.error(
                "Quarantine publish for the unparseable DLQ record at %s/%s/%s "
                "returned no confirmation; treating it as NOT durable and "
                "withholding the offset (OMN-17896)",
                record.dlq_topic,
                record.dlq_partition,
                record.dlq_offset,
            )
            await self._record_unparseable(
                record,
                EnumReplayStatus.FAILED,
                quarantine_correlation_id,
                error_message="Quarantine publish returned no confirmation",
            )
            return ModelDlqReplayResult(
                correlation_id=quarantine_correlation_id,
                original_topic=record.dlq_topic,
                status=EnumReplayStatus.FAILED,
                message="Quarantine of unparseable record was not confirmed",
                replay_correlation_id=quarantine_correlation_id,
            )

        await self._record_unparseable(
            record,
            EnumReplayStatus.QUARANTINED,
            quarantine_correlation_id,
            error_message=record.reason,
        )
        logger.info(
            "QUARANTINED unparseable DLQ record at %s/%s/%s (%s)",
            record.dlq_topic,
            record.dlq_partition,
            record.dlq_offset,
            record.reason,
        )
        return ModelDlqReplayResult(
            correlation_id=quarantine_correlation_id,
            original_topic=record.dlq_topic,
            status=EnumReplayStatus.QUARANTINED,
            message=f"Quarantined unparseable record: {record.reason}",
            replay_correlation_id=quarantine_correlation_id,
        )

    async def _process_message(
        self, message: ModelDlqMessage, config: ModelDlqReplayEngineConfig
    ) -> ModelDlqReplayResult:
        # OMN-18084: resolve a nested dead letter to the record it actually
        # wraps BEFORE deciding anything about it. A record whose original_topic
        # is itself a DLQ topic has a dead-letter envelope for a body; replaying
        # it strips one layer and writes the rest back onto the topic it came
        # from. Unwrapping hands every clause below the record that really
        # failed, and the refusal path terminalises rather than falling through
        # to a publish.
        try:
            message, unwrap_depth = unwrap_nested_dlq_record(message)
        except DlqTopicFixedPointError as exc:
            return await self._quarantine(message, str(exc))

        eligible, reason = should_replay(message, config)

        if unwrap_depth:
            # The note goes on the reason (and so into the quarantine record and
            # dlq_replay_history) rather than only into a log line, so the
            # nesting a record arrived with is recoverable after the fact.
            reason = (
                f"[unwrapped {unwrap_depth} dead-letter envelope layer(s) to "
                f"{message.original_topic} (OMN-18084)] {reason}"
            )
            logger.info(
                "UNWRAPPED %d dead-letter envelope layer(s) on %s/%s to %s "
                "(eligible=%s)",
                unwrap_depth,
                config.dlq_topic,
                message.dlq_offset,
                message.original_topic,
                eligible,
            )

        if not eligible:
            return await self._quarantine(message, reason)

        replay_correlation_id = generate_replay_correlation_id()

        if self._config.dry_run:
            return ModelDlqReplayResult(
                correlation_id=message.correlation_id,
                original_topic=message.original_topic,
                status=EnumReplayStatus.PENDING,
                message="DRY RUN - would replay",
                replay_correlation_id=replay_correlation_id,
            )

        try:
            await self._producer.replay_message(message, replay_correlation_id)
        except Exception as exc:
            await self._record(
                message,
                EnumReplayStatus.FAILED,
                replay_correlation_id,
                error_message=str(exc),
            )
            logger.exception(
                "FAILED replay for %s -> %s",
                message.correlation_id,
                message.original_topic,
            )
            return ModelDlqReplayResult(
                correlation_id=message.correlation_id,
                original_topic=message.original_topic,
                status=EnumReplayStatus.FAILED,
                message=f"Replay failed: {exc}",
                replay_correlation_id=replay_correlation_id,
            )

        await self._record(message, EnumReplayStatus.COMPLETED, replay_correlation_id)
        return ModelDlqReplayResult(
            correlation_id=message.correlation_id,
            original_topic=message.original_topic,
            status=EnumReplayStatus.COMPLETED,
            message="Replayed successfully",
            replay_correlation_id=replay_correlation_id,
        )

    async def _quarantine(
        self, message: ModelDlqMessage, reason: str
    ) -> ModelDlqReplayResult:
        """Route a non-replayable message to quarantine (never drop it)."""
        quarantine_correlation_id = generate_replay_correlation_id()

        if self._config.dry_run:
            return ModelDlqReplayResult(
                correlation_id=message.correlation_id,
                original_topic=message.original_topic,
                status=EnumReplayStatus.PENDING,
                message=f"DRY RUN - would quarantine: {reason}",
                replay_correlation_id=quarantine_correlation_id,
            )

        try:
            confirmation = await self._quarantine_producer.quarantine_message(
                message, reason, quarantine_correlation_id
            )
        except Exception as exc:
            await self._record(
                message,
                EnumReplayStatus.FAILED,
                quarantine_correlation_id,
                error_message=f"Quarantine publish failed: {exc}",
            )
            logger.exception(
                "FAILED to quarantine non-replayable message %s",
                message.correlation_id,
            )
            return ModelDlqReplayResult(
                correlation_id=message.correlation_id,
                original_topic=message.original_topic,
                status=EnumReplayStatus.FAILED,
                message=f"Quarantine failed: {exc}",
                replay_correlation_id=quarantine_correlation_id,
            )

        if confirmation is None:
            # OMN-17896: a publish that returns no confirmation has not been
            # shown to be durable, and an offset may only advance on a
            # CONFIRMED publication. Recording FAILED here blocks the
            # partition, so the record is redelivered rather than acked on a
            # quarantine nobody can prove happened.
            await self._record(
                message,
                EnumReplayStatus.FAILED,
                quarantine_correlation_id,
                error_message="Quarantine publish returned no confirmation",
            )
            logger.error(
                "Quarantine publish for %s returned no confirmation; treating "
                "it as NOT durable and withholding the offset",
                message.correlation_id,
            )
            return ModelDlqReplayResult(
                correlation_id=message.correlation_id,
                original_topic=message.original_topic,
                status=EnumReplayStatus.FAILED,
                message="Quarantine was not confirmed",
                replay_correlation_id=quarantine_correlation_id,
            )

        await self._record(
            message,
            EnumReplayStatus.QUARANTINED,
            quarantine_correlation_id,
            error_message=reason,
        )
        logger.info("QUARANTINED %s (%s)", message.correlation_id, reason)
        return ModelDlqReplayResult(
            correlation_id=message.correlation_id,
            original_topic=message.original_topic,
            status=EnumReplayStatus.QUARANTINED,
            message=f"Quarantined: {reason}",
            replay_correlation_id=quarantine_correlation_id,
        )

    async def _record(
        self,
        message: ModelDlqMessage,
        status: EnumReplayStatus,
        replay_correlation_id: UUID,
        error_message: str | None = None,
    ) -> None:
        if self._tracking is None or not self._tracking.is_tracking_enabled:
            return
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=message.correlation_id,
            replay_correlation_id=replay_correlation_id,
            original_topic=message.original_topic,
            target_topic=message.original_topic,
            replay_status=status,
            replay_timestamp=datetime.now(UTC),
            success=status == EnumReplayStatus.COMPLETED,
            error_message=error_message,
            dlq_offset=message.dlq_offset,
            dlq_partition=message.dlq_partition,
            retry_count=message.retry_count,
        )
        await self._write_audit_row(record)

    async def _record_unparseable(
        self,
        record: ModelUnparseableDlqRecord,
        status: EnumReplayStatus,
        quarantine_correlation_id: UUID,
        error_message: str,
    ) -> None:
        """Write the audit row for a record that could not be parsed (OMN-18111).

        Three fields cannot be read off an unparseable record, and each is
        answered with the truth rather than with a plausible-looking value:

        ``original_message_id``
            There is no correlation id to carry — establishing one is exactly
            what failed. A UUIDv5 over the record's DLQ COORDINATE
            (``topic/partition/offset``) is used instead: it is derived from
            something real, it is stable, and two rows for one coordinate are
            recognisably the same record, which is the identity a
            reclassification owner works from.
        ``original_topic``
            Unknown. The only topic that is true of this record is the DLQ it
            was read from, so that is what the row says.
        ``target_topic``
            The quarantine topic, because that is where the record actually
            went. Naming a replay target it was never published to would make
            the row a nicer-looking lie.

        ``retry_count`` is 0: an unparseable record carries no readable retry
        count, and the column is NOT NULL.
        """
        if self._tracking is None or not self._tracking.is_tracking_enabled:
            return
        coordinate = f"{record.dlq_topic}/{record.dlq_partition}/{record.dlq_offset}"
        await self._write_audit_row(
            ModelDlqReplayRecord(
                id=uuid4(),
                original_message_id=uuid5(NAMESPACE_URL, coordinate),
                replay_correlation_id=quarantine_correlation_id,
                original_topic=record.dlq_topic,
                target_topic=self._config.quarantine_topic,
                replay_status=status,
                replay_timestamp=datetime.now(UTC),
                success=False,
                error_message=error_message,
                dlq_offset=record.dlq_offset,
                dlq_partition=record.dlq_partition,
                retry_count=0,
            )
        )

    async def _write_audit_row(self, record: ModelDlqReplayRecord) -> None:
        """Persist one audit row, ISOLATED from the outcome it describes.

        OMN-18111. Every caller reaches this only after the outcome is already
        durable: the replay publish or the quarantine publish has been
        CONFIRMED by the broker. Letting a tracking failure propagate from here
        would therefore convert a confirmed quarantine into a ``FAILED``
        result, block the partition in ``_mark_offset``, and have the record
        redelivered and re-quarantined on the next run — reintroducing the
        OMN-18084 amplification through the audit path, with a database outage
        as its trigger.

        So the row is best-effort and the failure is LOUD: an ERROR log naming
        the outcome that went unrecorded, never a silent swallow. A row lost
        here is an audit gap; a verdict lost here would be a live loop.
        """
        if self._tracking is None:
            return
        try:
            await self._tracking.record_replay_attempt(record)
        except Exception:  # boundary: an audit write may not change a durable outcome
            logger.exception(
                "FAILED to write the dlq_replay_history audit row for the %s "
                "outcome at %s/%s (replay_correlation_id=%s). The outcome "
                "itself is durable and STANDS; only the audit row is lost "
                "(OMN-18111).",
                record.replay_status.value,
                record.dlq_partition,
                record.dlq_offset,
                record.replay_correlation_id,
            )

    def _summarize(
        self,
        results: list[ModelDlqReplayResult],
        topics_drained: tuple[str, ...],
        halted_partitions: tuple[str, ...],
    ) -> ModelDlqReplayRunResult:
        """Aggregate one run across every topic it visited (OMN-18119).

        ``dlq_topic`` names the topic the run STARTED on, which rotates; the
        full visit order is in ``topics_drained``. Counts and ``results`` span
        every topic, so a caller reading only the counts is not told a
        single-topic story about a multi-topic run.
        """

        def _count(status: EnumReplayStatus) -> int:
            return sum(1 for r in results if r.status == status)

        return ModelDlqReplayRunResult(
            dlq_topic=topics_drained[0],
            topics_drained=topics_drained,
            total_processed=len(results),
            completed=_count(EnumReplayStatus.COMPLETED),
            quarantined=_count(EnumReplayStatus.QUARANTINED),
            failed=_count(EnumReplayStatus.FAILED),
            pending=_count(EnumReplayStatus.PENDING),
            dry_run=self._config.dry_run,
            halted_partitions=halted_partitions,
            results=tuple(results),
        )


def _extract_envelope_correlation_id(envelope: object) -> UUID | None:
    """Best-effort ``correlation_id`` extraction tolerant of both a real
    ``ModelEventEnvelope`` and the runtime's materialized dispatch-dict shape
    (OMN-15021). Returns ``None`` (never raises) when no usable value is
    present so the caller can decide the fallback + logging policy.
    """
    if isinstance(envelope, ModelEventEnvelope):
        return envelope.correlation_id
    if isinstance(envelope, Mapping):
        candidate: object = envelope.get("correlation_id")
        if candidate is None:
            debug_trace = envelope.get("__debug_trace")
            if isinstance(debug_trace, Mapping):
                candidate = debug_trace.get("correlation_id")
        if candidate is None:
            payload = envelope.get("payload")
            if isinstance(payload, Mapping):
                candidate = payload.get("correlation_id")
        return _coerce_uuid(candidate)
    return _coerce_uuid(getattr(envelope, "correlation_id", None))


def _extract_envelope_id(envelope: object) -> UUID | None:
    """Best-effort ``envelope_id`` extraction (OMN-15021).

    Only a real ``ModelEventEnvelope`` instance carries an ``envelope_id`` --
    the materialized dispatch dict (``ModelMaterializedDispatch``) never does,
    so a dict/mapping input always yields ``None`` here (by design, not a bug).
    """
    if isinstance(envelope, ModelEventEnvelope):
        return envelope.envelope_id
    if isinstance(envelope, Mapping):
        return None
    return _coerce_uuid(getattr(envelope, "envelope_id", None))


def _coerce_uuid(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


__all__ = ["HandlerDlqReplay", "HANDLER_ID_DLQ_REPLAY"]
