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
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from weakref import WeakKeyDictionary

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.dlq.models.model_dlq_replay_record import ModelDlqReplayRecord
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import DlqTopicFixedPointError
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
    generate_replay_correlation_id,
    should_replay,
    unwrap_nested_dlq_record,
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

_RUN_LOCKS: WeakKeyDictionary[object, asyncio.Lock] = WeakKeyDictionary()
"""One run mutex per shared consumer object (OMN-18084).

``service_kernel`` keys runtime dependencies by handler NAME, so the three
per-topic dispatcher entries OMN-18013 split this node's routing into all
resolve to the same ``dependencies["HandlerDlqReplay"]`` mapping — one
``DLQConsumer`` behind three ``HandlerDlqReplay`` instances. Nothing then
serialised them, and ``_ensure_runtime_dependencies_started`` skips a
dependency already flagged ``_started``, so it does not put that dependency in
the list its own ``finally`` stops while the peer that DID start it stops it
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


def _run_lock_for(consumer: object) -> asyncio.Lock:
    """Return the mutex guarding one shared consumer's start/drain/stop."""
    lock = _RUN_LOCKS.get(consumer)
    if lock is None:
        lock = asyncio.Lock()
        _RUN_LOCKS[consumer] = lock
    return lock


class HandlerDlqReplay:
    """EFFECT handler that replays or quarantines DLQ messages.

    Dependencies (constructor-injected):
        consumer: DLQ topic consumer (persistent group).
        producer: Replays eligible messages to the original topic.
        quarantine_producer: Publishes non-replayable messages to quarantine.
        tracking: Optional ``ServiceDlqTracking`` for dlq_replay_history.
    """

    def __init__(
        self,
        *,
        consumer: DLQConsumer,
        producer: DLQProducer,
        quarantine_producer: DLQQuarantineProducer,
        tracking: ServiceDlqTracking | None = None,
    ) -> None:
        self._consumer = consumer
        self._producer = producer
        self._quarantine_producer = quarantine_producer
        self._tracking = tracking
        self._config: ModelDlqReplayEngineConfig = consumer.config

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
        """Consume the DLQ topic for one BOUNDED batch (OMN-16422).

        This handler is auto-wired as a PER-MESSAGE trigger on the DLQ topic
        (``event_bus.subscribe_topics`` in ``contract.yaml``), but drains via
        a persistent, whole-topic-shaped consumer group. Every invocation is
        bounded two ways -- record count (``max_records_per_run``,
        additionally clamped by an explicit ``limit`` when one is set) and
        wall-clock (``max_run_duration_seconds``) -- and commits incrementally
        every ``commit_every_n_records`` processed records. A bounded
        invocation always returns quickly (keeping the OUTER trigger
        consumer's poll loop alive) and always makes committed progress, so
        the persistent group's committed offset advances even while the topic
        is continuously self-feeding (OMN-16422 / OMN-16418).

        OMN-17137: the wall-clock bound MUST be enforced on the WAIT for the
        next record, not only between records. ``DLQConsumer.consume_messages()``
        iterates an ``AIOKafkaConsumer`` whose ``__anext__`` is
        ``while True: return await self.getone()`` -- it blocks indefinitely
        until a record arrives. The ``consumer_timeout_ms=5000`` passed at
        construction does NOT end that iteration: in aiokafka that parameter
        is the *background fetching routine's* max wait (default 200ms), not
        kafka-python's idle-iteration timeout. So a deadline checked only
        inside the loop body is never re-evaluated once the topic goes idle
        mid-drain -- ``run()`` parked in ``getone()`` forever, holding the
        outer trigger consumer's serial ``_consume_loop`` until aiokafka
        evicted it at ``max_poll_interval_ms`` (live: 30 min on the stability
        lane, ``OffsetCommit ... UnknownMemberIdError``, group left at zero
        members = ``Empty``, generation 338 by the time it was measured). The
        acquisition below is therefore driven through ``asyncio.wait_for``
        with the run's REMAINING budget, and both bounds are checked BEFORE
        the next record is requested.

        OMN-18084: runs over ONE shared consumer are SERIALISED. OMN-18013 split
        this node's routing into three per-topic dispatcher entries, and
        ``service_kernel`` keys dependencies by handler name, so all three
        resolve to a single ``DLQConsumer``. Concurrently, that object was
        started and stopped by peers -- one dispatcher's ``finally`` tore down a
        consumer another was mid-drain on, and the victim raised
        ``RuntimeError("Consumer not started")`` on every message. Cost of the
        mutex: a peer waits at most ``max_run_duration_seconds`` (10 s by
        default, so at most ~30 s across three dispatchers), well inside
        ``max_poll_interval_ms``. Two runs iterating one ``AIOKafkaConsumer``
        was never safe anyway, so the serialisation is the correct semantics
        for a shared consumer, not only a lifecycle repair.
        """
        # OMN-18084: the whole start -> drain -> stop sequence is one critical
        # section per shared consumer. Acquiring INSIDE run() rather than around
        # the dispatch keeps the scope exactly the lifecycle that is shared; a
        # dispatcher whose consumer nobody else holds never waits.
        async with _run_lock_for(self._consumer):
            return await self._run_locked()

    async def _run_locked(self) -> ModelDlqReplayRunResult:
        """One bounded drain, holding this consumer's run mutex."""
        started_dependencies = await self._ensure_runtime_dependencies_started()
        try:
            results: list[ModelDlqReplayResult] = []
            count = 0
            uncommitted = 0
            # OMN-17896: the offsets whose handling actually COMPLETED, per
            # partition, as a contiguous prefix. A partition is BLOCKED by the
            # first record on it that did not complete, so a later success can
            # never commit past an earlier failure. Both commits below use
            # this map; a bare ``commit()`` commits the consumer's POSITION —
            # every record the iterator has already handed out, including one
            # whose handling never finished — so any early exit advanced past
            # the in-flight record.
            completed: dict[tuple[str, int], int] = {}
            blocked: set[tuple[str, int]] = set()
            limit = self._config.limit
            max_records = self._config.max_records_per_run
            effective_limit = max_records if limit is None else min(limit, max_records)
            commit_every = self._config.commit_every_n_records
            deadline = time.monotonic() + self._config.max_run_duration_seconds

            messages = self._consumer.consume_messages()
            try:
                while count < effective_limit:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        logger.warning(
                            "DLQ replay run hit its wall-clock bound (%.1fs) after "
                            "%d records -- returning this bounded batch instead of "
                            "continuing to drain (OMN-16422).",
                            self._config.max_run_duration_seconds,
                            count,
                        )
                        break

                    # OMN-17137: bound the WAIT, not just the gap between
                    # records. Without this timeout an idle topic parks the
                    # run in aiokafka's unbounded ``getone()`` and the
                    # deadline above is never reached again.
                    try:
                        message = await asyncio.wait_for(
                            anext(messages), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        logger.warning(
                            "DLQ replay run hit its wall-clock bound (%.1fs) after "
                            "%d records while WAITING for the next record -- the "
                            "topic went idle mid-drain. Returning this bounded "
                            "batch rather than parking the outer trigger consumer "
                            "(OMN-17137).",
                            self._config.max_run_duration_seconds,
                            count,
                        )
                        break

                    if isinstance(message, ModelUnparseableDlqRecord):
                        result = await self._quarantine_unparseable(message)
                    else:
                        result = await self._process_message(message)
                    results.append(result)
                    self._mark_offset(message, result, completed, blocked)
                    count += 1
                    uncommitted += 1

                    if (
                        not self._config.dry_run
                        and uncommitted >= commit_every
                        and completed
                    ):
                        await self._consumer.commit_offsets(completed)
                        uncommitted = 0
            finally:
                # Release the iterator deterministically. After a timeout the
                # cancelled ``__anext__`` has already finalized an async
                # generator, so this is a no-op there; it matters for the
                # count-bound and StopAsyncIteration exits.
                aclose = getattr(messages, "aclose", None)
                if aclose is not None:
                    await aclose()

            if uncommitted and not self._config.dry_run and completed:
                await self._consumer.commit_offsets(completed)

            return self._summarize(results)
        finally:
            await self._stop_runtime_dependencies(started_dependencies)

    async def _ensure_runtime_dependencies_started(self) -> list[object]:
        """Start owned Kafka dependencies lazily when a replay run executes."""
        started: list[object] = []
        try:
            for dependency in (
                self._consumer,
                self._producer,
                self._quarantine_producer,
            ):
                if getattr(dependency, "_started", False):
                    continue
                start = getattr(dependency, "start", None)
                if start is None:
                    continue
                await start()
                started.append(dependency)
            return started
        except Exception:
            await self._stop_runtime_dependencies(started)
            raise

    async def _stop_runtime_dependencies(self, dependencies: list[object]) -> None:
        for dependency in reversed(dependencies):
            stop = getattr(dependency, "stop", None)
            if stop is not None:
                await stop()

    def _mark_offset(
        self,
        message: DlqDrainRecord,
        result: ModelDlqReplayResult,
        completed: dict[tuple[str, int], int],
        blocked: set[tuple[str, int]],
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
        # ``ModelDlqMessage`` carries no DLQ topic of its own — this consumer
        # drains exactly one, named by the config — while the unparseable
        # record carries its own so the two shapes key identically.
        topic = (
            message.dlq_topic
            if isinstance(message, ModelUnparseableDlqRecord)
            else self._config.dlq_topic
        )
        key = (topic, message.dlq_partition)
        if key in blocked:
            return
        if result.status == EnumReplayStatus.FAILED:
            blocked.add(key)
            return
        completed[key] = message.dlq_offset + 1

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

    async def _process_message(self, message: ModelDlqMessage) -> ModelDlqReplayResult:
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

        eligible, reason = should_replay(message, self._config)

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
                self._config.dlq_topic,
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
        self, results: list[ModelDlqReplayResult]
    ) -> ModelDlqReplayRunResult:
        def _count(status: EnumReplayStatus) -> int:
            return sum(1 for r in results if r.status == status)

        return ModelDlqReplayRunResult(
            dlq_topic=self._config.dlq_topic,
            total_processed=len(results),
            completed=_count(EnumReplayStatus.COMPLETED),
            quarantined=_count(EnumReplayStatus.QUARANTINED),
            failed=_count(EnumReplayStatus.FAILED),
            pending=_count(EnumReplayStatus.PENDING),
            dry_run=self._config.dry_run,
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
