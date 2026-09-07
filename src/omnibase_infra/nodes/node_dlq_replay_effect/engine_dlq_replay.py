# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reusable DLQ replay engine for the contract-native replay node (OMN-12619).

This module owns the Kafka I/O surface and the replay eligibility predicate.
It is the relocated, importable home of the engine that previously lived in
``scripts/dlq_replay.py`` (``DLQConsumer``, ``DLQProducer``, ``should_replay``,
``generate_replay_correlation_id``). The CLI script now imports from here, and
the node handler drives this engine end-to-end.

What is net-new for OMN-12619:
    - ``DLQQuarantineProducer``: publishes non-replayable messages to
      ``onex.dlq.omnibase-infra.quarantine.v1`` instead of the legacy skip-and-drop path.

Eligibility (``should_replay``) is unchanged from the legacy implementation —
it is not reimplemented; it is moved.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaConnectionError, KafkaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
from omnibase_infra.event_bus.mixin_kafka_dlq import DLQ_UNREADABLE_VALUE_MARKER
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
from omnibase_infra.nodes.node_dlq_replay_effect.models.enum_dlq_replay_filter_type import (
    EnumDlqReplayFilterType,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
    DlqRecordUnparseableError,
    ModelUnparseableDlqRecord,
)
from omnibase_infra.utils.util_datetime import is_timezone_aware

logger = logging.getLogger(__name__)

# Centralized non-retryable error types (consistent with event_bus_kafka.py).
NON_RETRYABLE_ERRORS = EnumNonRetryableErrorCategory.get_all_values()

# Consumer-group identity for the persistent replay consumer. The runtime kernel
# reads ONEX_GROUP_ID (service_kernel.py:639); this default documents the value
# the dlq-replay-consumer service must set. Replaces the legacy ephemeral
# "dlq-replay-{pid}" group that left no durable read position.
DLQ_REPLAY_CONSUMER_GROUP: str = "onex-dlq-replay"


def generate_replay_correlation_id() -> UUID:
    """Mint a new correlation ID for a replay/quarantine outcome."""
    return uuid4()


def parse_datetime_with_timezone(dt_string: str) -> datetime:
    """Parse an ISO-8601 string, normalising 'Z' and assuming UTC if naive."""
    normalized = dt_string.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if not is_timezone_aware(dt):
        dt = dt.replace(tzinfo=UTC)
    return dt


def safe_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to max_chars (characters, not bytes) with a suffix."""
    if len(text) <= max_chars:
        return text
    suffix_len = len(suffix)
    if max_chars <= suffix_len:
        return suffix[:max_chars]
    return text[: max_chars - suffix_len] + suffix


def sanitize_bootstrap_servers(servers: str) -> str:
    """Return host:port entries safe for logging, or '[redacted]' if unexpected."""
    if "@" in servers or "://" in servers:
        return "[redacted]"
    try:
        sanitized_parts = []
        for raw in servers.split(","):
            part = raw.strip()
            if ":" in part:
                host, port = part.rsplit(":", 1)
                if port.isdigit():
                    sanitized_parts.append(f"{host}:{port}")
                else:
                    return "[redacted]"
            else:
                sanitized_parts.append(part)
        return ",".join(sanitized_parts)
    except Exception:  # noqa: BLE001 — boundary: returns degraded response
        return "[redacted]"


class ModelDlqReplayEngineConfig(BaseModel):
    """Runtime configuration for the DLQ replay engine.

    Carries the eligibility predicate inputs (mirrors the legacy
    ``ModelReplayConfig`` fields that ``should_replay`` reads) plus Kafka
    connection details. ``bootstrap_servers`` and ``consumer_group`` are
    required; there is no silent default for the broker address.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_servers: str = Field(..., min_length=1)
    dlq_topic: str = Field(..., min_length=1)
    quarantine_topic: str = Field(
        default_factory=lambda: build_dlq_topic("quarantine"), min_length=1
    )
    consumer_group: str = Field(default=DLQ_REPLAY_CONSUMER_GROUP, min_length=1)
    max_replay_count: int = Field(default=5, gt=0)
    rate_limit_per_second: float = Field(default=100.0, gt=0.0)
    dry_run: bool = Field(default=False)
    filter_type: EnumDlqReplayFilterType = Field(default=EnumDlqReplayFilterType.ALL)
    filter_topics: tuple[str, ...] = Field(default=())
    filter_error_types: tuple[str, ...] = Field(default=())
    filter_correlation_ids: tuple[UUID, ...] = Field(default=())
    filter_start_time: datetime | None = Field(default=None)
    filter_end_time: datetime | None = Field(default=None)
    add_replay_headers: bool = Field(default=True)
    limit: int | None = Field(default=None, gt=0)
    max_request_size: int = Field(default=10485760, gt=0)
    request_timeout_ms: int = Field(default=30000, gt=0)
    # OMN-16422: this node is auto-wired as a PER-MESSAGE trigger on the DLQ
    # topic (event_bus.subscribe_topics in contract.yaml) but HandlerDlqReplay
    # is a whole-topic batch drain. On a self-feeding DLQ (replayed messages
    # that keep failing land right back on the same topic) an unbounded run()
    # never goes idle, never commits, and starves the outer trigger consumer's
    # heartbeat. These three fields bound every invocation so it always
    # returns quickly and always makes committed progress, regardless of how
    # busy the topic is. Defaults apply to the live runtime construction
    # (service_kernel.py) with no additional plumbing required.
    max_records_per_run: int = Field(default=200, gt=0)
    max_run_duration_seconds: float = Field(default=10.0, gt=0.0)
    commit_every_n_records: int = Field(default=25, gt=0)

    @field_validator("bootstrap_servers")
    @classmethod
    def _bootstrap_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "bootstrap_servers cannot be empty; provide a Kafka broker address"
            )
        return stripped


def _is_boundary_failure_terminal(original_value: str) -> bool:
    """True when the DLQ'd record is itself a ``ModelBoundaryFailureTerminal``.

    Reads the SAME field ``handler_wiring._is_boundary_failure_terminal_record``
    reads — the envelope's ``payload_type`` — so the two guards on the two legs
    of the loop cannot disagree about what a terminal is.

    Anything unparseable is reported as "not a terminal": this predicate may add
    a refusal, never withdraw eligibility from a record it cannot classify.
    """
    from omnibase_infra.runtime.boundary_failure_terminal import (
        ModelBoundaryFailureTerminal,
    )

    try:
        decoded = json.loads(original_value)
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(decoded, dict):
        return False
    return decoded.get("payload_type") == ModelBoundaryFailureTerminal.__name__


def should_replay(
    message: ModelDlqMessage, config: ModelDlqReplayEngineConfig
) -> tuple[bool, str]:
    """Eligibility predicate — relocated from scripts/dlq_replay.py.

    Evaluation order: boundary failure terminal, max-retry count, non-retryable
    error type, time-range (orthogonal), then the type-specific filter. Returns
    (eligible, reason).

    THE TERMINAL CHECK COMES FIRST, AND IT IS PERMANENT (OMN-17895). Replay
    exists to re-attempt work that might succeed on a second look. A boundary
    failure terminal is not work — it is the runtime's own answer that some work
    will never be attempted again. It carries no request payload, so every
    consumer holding a dispatcher for its topic's ``event_type`` fails
    ``model_validate`` on it deterministically; attempt 2 fails exactly as
    attempt 1 did, and each attempt MULTIPLIES because every rejecting consumer
    DLQs it independently.

    Measured on the dev lane 2026-09-04: 11,328 of the 11,340 records on
    ``onex.evt.omnimarket.swarm-fanout-completed.v1`` carried
    ``x-replayed-by=node_dlq_replay_effect``. One burst bucketed by replay count
    was 12 / 24 / 48 / 96 / 192 / 384 — exact doubling per round from two
    rejecting consumers, sum 2^0..2^5 = 63x per terminal. The
    ``max_replay_count`` cap bounded it; it did not prevent it. The identical
    signature was live on ``onex.evt.omnimarket.redeploy-completed.v1`` and
    ``onex.evt.omnibase-infra.runtime-manifest-published.v1`` in the same window,
    so this is a property of the record class, not of one contract.

    Refusal here routes the record to ``quarantine_topic`` on the caller's
    existing non-replayable path (OMN-12619), so it stays durable and
    reclassifiable rather than being dropped.
    """
    # OMN-17896: the empty-body clause comes FIRST because an empty body is
    # not a record that MIGHT succeed on a second look -- it is one that is
    # guaranteed to fail in every consumer of the original topic. Publishing
    # it produced ~4 zero-byte records/s on the dev lane and 5,535 decode
    # errors per five minutes, and no other clause in this predicate would
    # ever have stopped it.
    if not message.original_value.strip():
        return (
            False,
            "Empty original body: the DLQ record carries no original message "
            "body, so replaying it would publish a zero-byte record that no "
            "consumer of the original topic can decode (OMN-17896)",
        )

    if message.original_value == DLQ_UNREADABLE_VALUE_MARKER:
        return (
            False,
            "Unreadable original body: the DLQ record records that the "
            f"original value could not be read ({DLQ_UNREADABLE_VALUE_MARKER}); "
            "those marker bytes are a statement, not a body, and must not be "
            "published to the original topic (OMN-17896)",
        )

    if _is_boundary_failure_terminal(message.original_value):
        return (
            False,
            "Boundary failure terminal: the record is an answer, not a request — "
            "replaying it is deterministically undispatchable and amplifies once "
            "per rejecting consumer (OMN-17895)",
        )

    if message.retry_count >= config.max_replay_count:
        return (
            False,
            f"Exceeded max replay count: {message.retry_count} >= {config.max_replay_count}",
        )

    if message.error_type in NON_RETRYABLE_ERRORS:
        return (False, f"Non-retryable error type: {message.error_type}")

    if config.filter_start_time or config.filter_end_time:
        try:
            failure_dt = parse_datetime_with_timezone(message.failure_timestamp)
            if config.filter_start_time and failure_dt < config.filter_start_time:
                return (False, f"Before start time: {config.filter_start_time}")
            if config.filter_end_time and failure_dt > config.filter_end_time:
                return (False, f"After end time: {config.filter_end_time}")
        except ValueError as exc:
            logger.warning(
                "Failed to parse failure_timestamp, skipping time filter",
                extra={
                    "correlation_id": str(message.correlation_id),
                    "failure_timestamp": message.failure_timestamp,
                    "parse_error": str(exc),
                },
            )

    if config.filter_type == EnumDlqReplayFilterType.BY_TOPIC:
        if message.original_topic not in config.filter_topics:
            return (False, f"Topic not in filter: {message.original_topic}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_ERROR_TYPE:
        if message.error_type not in config.filter_error_types:
            return (False, f"Error type not in filter: {message.error_type}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_CORRELATION_ID:
        if message.correlation_id not in config.filter_correlation_ids:
            return (False, f"Correlation ID not in filter: {message.correlation_id}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_TIME_RANGE:
        # Time windows are applied orthogonally above; this branch makes the
        # time-range-only filter explicit in logs and future maintenance.
        return (True, "Eligible for replay within time range")

    return (True, "Eligible for replay")


class DLQConsumer:
    """Reads and parses messages from a DLQ topic with a persistent group."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._started = False

    async def start(self) -> None:
        logger.info(
            "Starting DLQ consumer for topic: %s",
            self.config.dlq_topic,
            extra={
                "bootstrap_servers": sanitize_bootstrap_servers(
                    self.config.bootstrap_servers
                ),
                "consumer_group": self.config.consumer_group,
            },
        )
        try:
            self._consumer = AIOKafkaConsumer(
                self.config.dlq_topic,
                bootstrap_servers=self.config.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id=self.config.consumer_group,
                # OMN-17137: this is NOT an idle-iteration timeout. In aiokafka
                # ``consumer_timeout_ms`` is the background fetching routine's
                # max wait (default 200ms) -- ``__anext__`` still blocks forever
                # on a quiet topic. Nothing below ``HandlerDlqReplay.run()``
                # bounds the wait for the next record; run() owns that bound.
                consumer_timeout_ms=5000,
                **build_aiokafka_auth_kwargs_from_env(),
            )
            await self._consumer.start()
            self._started = True
        except KafkaConnectionError:
            logger.exception("Failed to connect DLQ consumer to Kafka")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ consumer start")
            raise

    async def stop(self) -> None:
        if self._started and self._consumer is not None:
            try:
                await self._consumer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping DLQ consumer: %s", exc)
            finally:
                self._started = False
                self._consumer = None

    async def commit(self) -> None:
        """Commit the consumer's POSITION — only call after durable handling.

        Retained for callers that genuinely mean "everything handed out of the
        iterator completed". ``HandlerDlqReplay`` no longer uses it: a bare
        position commit advances past any record that was in flight when the
        loop exited (OMN-17896). Use :meth:`commit_offsets` instead.
        """
        if self._started and self._consumer is not None:
            await self._consumer.commit()

    async def commit_offsets(self, offsets: Mapping[tuple[str, int], int]) -> None:
        """Commit an EXPLICIT offset map built from completed records only.

        ``offsets`` maps ``(topic, partition)`` to the next offset to read —
        i.e. one past the last record whose handling completed. Committing this
        rather than the position is what makes a deadline, a cancellation or a
        failed quarantine publish a REDELIVERY rather than a silent loss
        (OMN-17896; §4 must-not item 18 of the lab repair plan).
        """
        if not offsets:
            return
        if self._started and self._consumer is not None:
            await self._consumer.commit(
                {
                    TopicPartition(topic, partition): next_offset
                    for (topic, partition), next_offset in offsets.items()
                }
            )

    async def consume_messages(self) -> AsyncIterator[DlqDrainRecord]:
        """Yield each DLQ record, parsed — or, when it cannot be parsed, as a
        typed ``ModelUnparseableDlqRecord`` carrying its raw bytes.

        OMN-17896. This generator performs NO durable write, deliberately. It
        is driven through ``asyncio.wait_for(anext(...), timeout=remaining)``
        in ``HandlerDlqReplay.run()``, so anything awaited here is inside that
        timeout: a deadline landing mid-publish would CANCEL the publish, the
        handler would catch ``TimeoutError`` as the ordinary idle-topic case,
        break, and commit — losing the very record the publish was meant to
        make durable. So the refusal is carried out as a VALUE and the handler
        quarantines it on its own frame.

        Nothing is skipped. The two bare ``continue`` statements that used to
        drop a null-valued record and an undecodable-JSON record are gone: the
        consumer's position has already advanced past both by the time this
        body runs, so a skip here is a record the trailing commit erases.
        """
        if not self._started or self._consumer is None:
            raise RuntimeError("Consumer not started")
        try:
            async for msg in self._consumer:
                if msg.value is None:
                    logger.warning("Null DLQ message value at offset %s", msg.offset)
                    yield ModelUnparseableDlqRecord(
                        dlq_topic=self.config.dlq_topic,
                        dlq_partition=msg.partition,
                        dlq_offset=msg.offset,
                        raw_value=None,
                        reason=(
                            "DLQ record carries a null value, so no original "
                            "message can be established from it"
                        ),
                    )
                    continue
                try:
                    decoded_value = msg.value.decode("utf-8")
                except UnicodeDecodeError:
                    decoded_value = msg.value.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(decoded_value)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse DLQ message at offset %s: %s", msg.offset, exc
                    )
                    yield ModelUnparseableDlqRecord(
                        dlq_topic=self.config.dlq_topic,
                        dlq_partition=msg.partition,
                        dlq_offset=msg.offset,
                        raw_value=msg.value,
                        reason=f"DLQ record body is not JSON: {exc}",
                    )
                    continue
                try:
                    parsed = ModelDlqMessage.from_kafka_message(
                        payload=payload,
                        dlq_offset=msg.offset,
                        dlq_partition=msg.partition,
                    )
                except (DlqRecordUnparseableError, ValidationError) as exc:
                    # ONLY these two typed refusals. Never a bare
                    # ``except Exception``: that would swallow programming
                    # errors into the quarantine path and hide them.
                    logger.warning(
                        "Refusing unparseable DLQ record at offset %s: %s",
                        msg.offset,
                        exc,
                    )
                    yield ModelUnparseableDlqRecord(
                        dlq_topic=self.config.dlq_topic,
                        dlq_partition=msg.partition,
                        dlq_offset=msg.offset,
                        raw_value=msg.value,
                        reason=f"DLQ record could not be parsed: {exc}",
                    )
                    continue
                yield parsed
        except asyncio.CancelledError:
            logger.info("DLQ consumption cancelled")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ consumption")
            raise


class DLQProducer:
    """Replays messages back to their original topic with replay headers."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._started = False
        self._last_publish: datetime | None = None
        self._interval = 1.0 / config.rate_limit_per_second

    async def start(self) -> None:
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                acks="all",
                enable_idempotence=True,
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
                **build_aiokafka_auth_kwargs_from_env(),
            )
            await self._producer.start()
            self._started = True
        except KafkaConnectionError:
            logger.exception("Failed to connect DLQ producer to Kafka")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ producer start")
            raise

    async def stop(self) -> None:
        if self._started and self._producer is not None:
            try:
                await self._producer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping DLQ producer: %s", exc)
            finally:
                self._started = False
                self._producer = None

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        """Replay a message to its original topic. Raises on publish failure."""
        if not self._started or self._producer is None:
            raise RuntimeError("Producer not started")

        if self._last_publish is not None:
            elapsed = (datetime.now(UTC) - self._last_publish).total_seconds()
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)

        headers: list[tuple[str, bytes]] = []
        if self.config.add_replay_headers:
            headers = [
                ("x-replay-count", str(message.retry_count + 1).encode("utf-8")),
                ("x-replayed-at", datetime.now(UTC).isoformat().encode("utf-8")),
                ("x-replayed-by", b"node_dlq_replay_effect"),
                ("x-original-dlq-offset", str(message.dlq_offset).encode("utf-8")),
                ("x-replay-correlation-id", str(replay_correlation_id).encode("utf-8")),
                ("correlation_id", str(message.correlation_id).encode("utf-8")),
            ]

        # OMN-17896: last-resort refusal. ``should_replay`` already refuses an
        # empty or unreadable body, so this is unreachable in the wired path --
        # it is here so that no future caller can publish a zero-byte record
        # through this producer without saying so out loud.
        if not message.original_value.strip():
            raise DlqRecordUnparseableError(
                "Refusing to replay a record with no original body: encoding it "
                "would publish a zero-byte record onto "
                f"{message.original_topic} that no consumer can decode "
                "(OMN-17896)"
            )

        key = (
            message.original_key.encode("utf-8", errors="replace")
            if message.original_key
            else None
        )
        value = message.original_value.encode("utf-8", errors="replace")

        await self._producer.send_and_wait(
            message.original_topic, value=value, key=key, headers=headers
        )
        self._last_publish = datetime.now(UTC)


class DLQQuarantineProducer:
    """Publishes non-replayable messages to onex.dlq.omnibase-infra.quarantine.v1 (OMN-12619).

    This is the durable replacement for the legacy skip-and-drop path. A
    non-replayable message is wrapped with the quarantine decision (reason,
    original DLQ coordinates, and a quarantine correlation ID) and published so
    a quarantine owner can later reclassify or re-enter it. Publish failures
    propagate so the caller never records a false QUARANTINED success.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._started = False

    async def start(self) -> None:
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                acks="all",
                enable_idempotence=True,
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
                **build_aiokafka_auth_kwargs_from_env(),
            )
            await self._producer.start()
            self._started = True
        except KafkaError:
            logger.exception("Kafka error during quarantine producer start")
            raise

    async def stop(self) -> None:
        if self._started and self._producer is not None:
            try:
                await self._producer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping quarantine producer: %s", exc)
            finally:
                self._started = False
                self._producer = None

    @staticmethod
    def build_quarantine_payload(
        message: ModelDlqMessage,
        reason: str,
        quarantine_correlation_id: UUID,
        source_dlq_topic: str,
    ) -> dict[str, object]:
        """Build the durable quarantine record payload for a PARSED message."""
        return {
            "quarantine_class": "parsed_dlq_record",
            "quarantine_correlation_id": str(quarantine_correlation_id),
            "quarantined_at": datetime.now(UTC).isoformat(),
            "quarantined_by": "node_dlq_replay_effect",
            "reason": reason,
            "original_topic": message.original_topic,
            "original_correlation_id": str(message.correlation_id),
            "error_type": message.error_type,
            "retry_count": message.retry_count,
            "source_dlq_topic": source_dlq_topic,
            "source_dlq_offset": message.dlq_offset,
            "source_dlq_partition": message.dlq_partition,
            "original_payload": message.raw_payload,
        }

    @staticmethod
    def build_unparseable_quarantine_payload(
        record: ModelUnparseableDlqRecord,
        quarantine_correlation_id: UUID,
    ) -> dict[str, object]:
        """Build the durable quarantine payload for a record that would not parse.

        OMN-17896. ``build_quarantine_payload`` reads ``original_topic``,
        ``correlation_id``, ``error_type``, ``retry_count`` and ``raw_payload``
        off a constructed ``ModelDlqMessage`` — the exact object whose
        construction just failed — so it cannot express this record, and
        inventing those five values would be the fabricated default this whole
        change exists to remove. The bytes are carried base64-encoded because
        they are, by definition, not known to be text; the encoding is named in
        the record rather than assumed. ``quarantine_class`` keeps this
        distinguishable from a parsed quarantine so a later reclassifier can
        tell "this could not be parsed" from "this was parsed and refused".
        """
        raw = record.raw_value
        return {
            "quarantine_class": "unparseable_dlq_record",
            "quarantine_correlation_id": str(quarantine_correlation_id),
            "quarantined_at": datetime.now(UTC).isoformat(),
            "quarantined_by": "node_dlq_replay_effect",
            "reason": record.reason,
            "source_dlq_topic": record.dlq_topic,
            "source_dlq_offset": record.dlq_offset,
            "source_dlq_partition": record.dlq_partition,
            "original_raw_value_encoding": "base64",
            "original_raw_value_b64": (
                None if raw is None else base64.b64encode(raw).decode("ascii")
            ),
            "original_raw_value_bytes": None if raw is None else len(raw),
        }

    async def quarantine_message(
        self,
        message: ModelDlqMessage,
        reason: str,
        quarantine_correlation_id: UUID,
    ) -> object:
        """Publish a non-replayable message to the quarantine topic.

        Raises on publish failure; the caller must NOT record QUARANTINED
        success unless this returns normally. Returns the broker's own record
        metadata — the CONFIRMATION of publication — so the caller binds a
        durability fact rather than the mere absence of an exception.
        """
        if not self._started or self._producer is None:
            raise RuntimeError("Quarantine producer not started")

        payload = self.build_quarantine_payload(
            message, reason, quarantine_correlation_id, self.config.dlq_topic
        )
        headers: list[tuple[str, bytes]] = [
            ("x-quarantine-reason", reason.encode("utf-8")),
            (
                "x-quarantine-correlation-id",
                str(quarantine_correlation_id).encode("utf-8"),
            ),
            ("correlation_id", str(message.correlation_id).encode("utf-8")),
        ]
        key = str(message.correlation_id).encode("utf-8")
        value = json.dumps(payload).encode("utf-8")
        return await self._producer.send_and_wait(
            self.config.quarantine_topic, value=value, key=key, headers=headers
        )

    async def quarantine_unparseable_record(
        self,
        record: ModelUnparseableDlqRecord,
        quarantine_correlation_id: UUID,
    ) -> object:
        """Publish an UNPARSEABLE DLQ record, raw bytes intact (OMN-17896).

        Raises on publish failure. Returns the broker's record metadata so the
        caller can bind the confirmation before advancing any offset.
        """
        if not self._started or self._producer is None:
            raise RuntimeError("Quarantine producer not started")

        payload = self.build_unparseable_quarantine_payload(
            record, quarantine_correlation_id
        )
        headers: list[tuple[str, bytes]] = [
            ("x-quarantine-reason", record.reason.encode("utf-8")),
            (
                "x-quarantine-correlation-id",
                str(quarantine_correlation_id).encode("utf-8"),
            ),
            ("x-quarantine-class", b"unparseable_dlq_record"),
        ]
        # No correlation id exists for a record that would not parse, so the
        # key is its DLQ coordinate — deterministic, and never invented.
        key = (
            f"{record.dlq_topic}:{record.dlq_partition}:{record.dlq_offset}"
        ).encode()
        value = json.dumps(payload).encode("utf-8")
        return await self._producer.send_and_wait(
            self.config.quarantine_topic, value=value, key=key, headers=headers
        )


__all__ = [
    "DLQ_REPLAY_CONSUMER_GROUP",
    "DLQ_UNREADABLE_VALUE_MARKER",
    "DLQConsumer",
    "DLQProducer",
    "DLQQuarantineProducer",
    "ModelDlqReplayEngineConfig",
    "generate_replay_correlation_id",
    "parse_datetime_with_timezone",
    "safe_truncate",
    "sanitize_bootstrap_servers",
    "should_replay",
]
