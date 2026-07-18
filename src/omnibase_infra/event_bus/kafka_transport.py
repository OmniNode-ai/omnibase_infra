# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Kafka face for the unified-runtime transport protocols (epic OMN-14717, S3).

Net-new, currently UNUSED in production (ticket OMN-14756; see
``docs/plans/2026-07-17-single-runtime-transport-di-unification-plan.md`` sections
(c)/(d), step S3). This is the concrete Kafka implementation of the two core
transport protocols

* ``omnibase_core.protocols.runtime.protocol_transport_consumer.ProtocolTransportConsumer``
* ``omnibase_core.protocols.runtime.protocol_transport_producer.ProtocolTransportProducer``

so the ONE runtime in core can run on the real broker via DI, with core never
importing Kafka (dependency inversion: this impl lives in infra and points inward).

Scope discipline (plan I6 — "runtime owns policy, transport owns mechanism")
---------------------------------------------------------------------------
This class is ONLY the raw Kafka client behind the protocol. It carries **zero**
dispatch / coercion / fan-out / DLQ / retry logic — ``RuntimeDispatch`` (core, S4)
owns every one of those decisions and expresses them through the two protocol
primitives (``poll`` / ``commit`` / ``nack`` / ``send``). The mapping to Kafka:

* ``poll``   -> ``AIOKafkaConsumer.getmany`` (per-partition, offset-ordered).
* ``commit`` -> ``AIOKafkaConsumer.commit({tp: offset+1})``. Kafka's committed
  offset is the *next* offset to fetch, so committing ``msg.offset + 1`` advances
  the partition high-water mark to ``msg.offset`` and thereby commits every offset
  ``<= msg.offset`` on that partition (plan HOLE 1 / conformance
  ``test_commit_at_k_commits_all_leq_k_on_partition``).
* ``nack``   -> ``AIOKafkaConsumer.seek(tp, msg.offset)``. Does NOT advance the
  committed offset; re-exposes ``msg`` and every later same-partition offset for
  redelivery (Kafka ``seek``; conformance
  ``test_nack_redelivers_from_offset_including_later``).
* ``send``   -> ``AIOKafkaProducer.send_and_wait`` (awaits the broker ack).

Two settings are FORCED for these NEW transport consumers and are deliberately not
read from the shared config (plan S3): ``enable_auto_commit=False`` (the runtime,
not the client, decides when an offset is durable) and ``auto_offset_reset`` defaults
to ``"earliest"`` (a pull-based at-least-once consumer must see the existing backlog
on first boot of a group). The legacy push-callback consumers in ``EventBusKafka``
keep their per-consumer ``self._config.enable_auto_commit`` setting untouched — this
face is a separate class and changes nothing about them.

Substitutability
----------------
``KafkaTransport`` is exercised by the SAME parametrized
``TransportConformanceSuite`` that certifies ``InMemoryTransport`` in core (S2). The
suite asserting identical observable Kafka semantics against both impls is what
licenses "in-memory golden chain => Kafka golden chain".
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import cast

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

__all__ = ["KafkaTransport"]


class KafkaTransport:
    """Concrete Kafka transport: one instance is both consumer and producer.

    Consumer identity (``group`` + assigned ``topics``) is construction-time config
    supplied by the composition root — there is no runtime ``subscribe`` call,
    mirroring the pull-based protocol and the in-memory transport. A producer-only
    instance is constructed with no ``topics``; a consumer instance is constructed
    with the topics it is assigned. A "restart" is simply a new instance on the same
    ``group``: with ``enable_auto_commit=False`` it resumes from the committed
    offset, so every uncommitted offset is redelivered.
    """

    # Bounds for the post-position-change prefetch (see _prime). One batch only —
    # never the whole backlog — so priming is a latency optimization, not a load.
    _PRIME_POLL_MS = 500
    _PRIME_MAX_POLLS = 10
    _PRIME_MAX_RECORDS = 500

    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        group: str = "onex.transport.kafka",
        topics: Sequence[str] = (),
        auto_offset_reset: str = "earliest",
    ) -> None:
        self._config = config
        self._group = group
        self._topics: tuple[str, ...] = tuple(topics)
        self._auto_offset_reset = auto_offset_reset
        self._producer: AIOKafkaProducer | None = None
        self._consumer: AIOKafkaConsumer | None = None
        self._started = False
        # Eagerly-prefetched, not-yet-returned messages (see _prime): serving these
        # first is what lets poll() return promptly right after a position change
        # (assign/seek), instead of racing the fetcher's refetch.
        self._buffer: list[ModelTransportMessage] = []

    @classmethod
    def from_bootstrap(
        cls,
        bootstrap: str,
        *,
        group: str = "onex.transport.kafka",
        topics: Sequence[str] = (),
        auto_offset_reset: str = "earliest",
    ) -> KafkaTransport:
        """Build a transport bound to an explicit bootstrap-servers override.

        Mirrors ``EventBusKafka.from_bootstrap``: the default config (with env
        overrides) is the base, then ``bootstrap_servers`` is set to the supplied
        value so the caller wins over ``KAFKA_BOOTSTRAP_SERVERS``.
        """
        config = ModelKafkaEventBusConfig.default().model_copy(
            update={"bootstrap_servers": bootstrap}
        )
        return cls(
            config=config,
            group=group,
            topics=topics,
            auto_offset_reset=auto_offset_reset,
        )

    # -- auth / client-version plumbing (reused from EventBusKafka's helpers) ----

    def _auth_kwargs(self) -> dict[str, object]:
        return build_aiokafka_auth_kwargs(self._config)

    def _client_version_kwargs(self, client_cls: type[object]) -> dict[str, object]:
        if self._config.api_version is None:
            return {}
        try:
            parameters = inspect.signature(client_cls.__init__).parameters
        except (TypeError, ValueError):
            return {}
        if "api_version" not in parameters:
            return {}
        return {"api_version": self._config.api_version}

    # -- lifecycle --------------------------------------------------------------

    async def start(self) -> None:
        """Start the producer and, if ``topics`` were assigned, the consumer."""
        if self._started:
            return

        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._config.bootstrap_servers,
                acks=self._config.acks_aiokafka,
                enable_idempotence=self._config.enable_idempotence,
                max_request_size=self._config.max_request_size,
                retry_backoff_ms=self._config.reconnect_backoff_ms,
                **self._client_version_kwargs(AIOKafkaProducer),
                **self._auth_kwargs(),
            )
            await self._producer.start()

            if self._topics:
                # Group ``subscribe`` (topics passed positionally): the consumer joins
                # the group and, on join, NATIVELY resumes each partition from its
                # group-committed offset (a fresh group starts at ``auto_offset_reset``).
                # This is what makes "restart resumes from the committed offset" and
                # "uncommitted offsets redeliver on restart" hold, with no manual
                # ``seek`` dance. A single transport instance is the sole group member,
                # so it is assigned every partition of its topics — matching the unified
                # runtime's single-poll-loop-per-topic-set model (single-owner-per-topic
                # is the S6 boot invariant, R1). The one-time join latency is absorbed by
                # ``_prime`` below so the runtime's first ``poll`` still returns promptly.
                self._consumer = AIOKafkaConsumer(
                    *self._topics,
                    bootstrap_servers=self._config.bootstrap_servers,
                    group_id=self._group,
                    # FORCED for the new transport consumers (plan S3): the runtime, not
                    # the client, decides when an offset is durable. Legacy push
                    # consumers keep their per-consumer config setting untouched.
                    enable_auto_commit=False,
                    auto_offset_reset=self._auto_offset_reset,
                    session_timeout_ms=self._config.session_timeout_ms,
                    heartbeat_interval_ms=self._config.heartbeat_interval_ms,
                    max_poll_interval_ms=self._config.max_poll_interval_ms,
                    retry_backoff_ms=self._config.reconnect_backoff_ms,
                    **self._client_version_kwargs(AIOKafkaConsumer),
                    **self._auth_kwargs(),
                )
                await self._consumer.start()
                # Trigger the group join + first fetch now, buffering the first batch, so
                # the runtime's first poll() returns the available records instead of
                # racing the lazy rebalance.
                await self._prime(self._consumer)
        except BaseException:
            await self.close()
            raise

        self._started = True

    async def _prime(self, consumer: AIOKafkaConsumer) -> None:
        """Eagerly fetch ONE batch into ``self._buffer`` after a position change.

        Kafka fetches lazily: the first ``getmany`` after a group join or a ``seek``
        can return empty while the join/refetch is still in flight. A pull runtime
        with a generous poll timeout would just retry, but a short-timeout caller
        (and the shared conformance suite's 50ms drain) would see a spurious empty
        batch. Priming one bounded batch here makes the next ``poll`` return the
        already-available records deterministically. It is a pure latency
        optimization — bounded by ``max_records`` (never the whole backlog) and
        touching neither offsets nor commits.

        The loop tolerates join latency (an early empty ``getmany`` while the
        rebalance completes) without hanging on a genuinely empty position: once the
        consumer has an assignment, two consecutive empty fetches mean "nothing
        available", so priming returns with an empty buffer.
        """
        empty_after_assignment = 0
        for _ in range(self._PRIME_MAX_POLLS):
            raw = await consumer.getmany(
                timeout_ms=self._PRIME_POLL_MS, max_records=self._PRIME_MAX_RECORDS
            )
            if raw:
                for topic_partition in sorted(
                    raw, key=lambda tp: (tp.topic, tp.partition)
                ):
                    for record in raw[topic_partition]:
                        self._buffer.append(self._to_model(topic_partition, record))
                return
            if consumer.assignment():
                empty_after_assignment += 1
                if empty_after_assignment >= 2:
                    return

    async def close(self) -> None:
        """Stop the consumer and producer; safe to call more than once."""
        self._buffer.clear()
        first_error: BaseException | None = None
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except BaseException as exc:  # noqa: BLE001 — cleanup must attempt both clients.
                first_error = exc
            finally:
                self._consumer = None
        if self._producer is not None:
            try:
                await self._producer.stop()
            except BaseException as exc:  # noqa: BLE001 — cleanup must attempt both clients.
                if first_error is None:
                    first_error = exc
            finally:
                self._producer = None
        self._started = False
        if first_error is not None:
            raise first_error

    # -- consumer protocol ------------------------------------------------------

    def _require_consumer(self) -> AIOKafkaConsumer:
        if self._consumer is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="poll",
                target_name="kafka_transport",
            )
            raise ProtocolConfigurationError(
                "KafkaTransport consumer used before start() or without assigned "
                "topics; construct with topics=[...] and await start().",
                context=context,
                parameter="topics",
                value=list(self._topics),
            )
        return self._consumer

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        """Pull up to ``max_messages`` records, in per-partition offset order.

        ``getmany`` returns ``{TopicPartition: [records...]}`` where each partition's
        records are already ascending by offset. Advancing the consumer's fetch
        position is what makes ``nack`` (a ``seek`` back) and restart-redelivery
        observable; the committed offset only moves on an explicit ``commit``.

        Any records eagerly prefetched by ``_prime`` after the last position change
        are served first (in per-partition offset order), so a poll immediately after
        ``start``/``nack`` returns the already-available records without racing the
        fetcher.
        """
        if max_messages < 1:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="poll",
                target_name="kafka_transport",
            )
            raise ProtocolConfigurationError(
                f"max_messages must be a positive integer, got {max_messages}",
                context=context,
                parameter="max_messages",
                value=max_messages,
            )
        consumer = self._require_consumer()
        if self._buffer:
            batch = self._buffer[:max_messages]
            del self._buffer[:max_messages]
            return batch
        raw = await consumer.getmany(timeout_ms=timeout_ms, max_records=max_messages)
        polled: list[ModelTransportMessage] = []
        # Deterministic partition ordering; records within a partition stay
        # offset-ascending as getmany returns them.
        for topic_partition in sorted(raw, key=lambda tp: (tp.topic, tp.partition)):
            for record in raw[topic_partition]:
                polled.append(self._to_model(topic_partition, record))
        return polled

    @staticmethod
    def _to_model(
        topic_partition: TopicPartition, record: object
    ) -> ModelTransportMessage:
        """Map an aiokafka ``ConsumerRecord`` to the transport-agnostic model."""
        raw_headers = getattr(record, "headers", None) or ()
        headers: dict[str, bytes] = {}
        for key, value in raw_headers:
            if value is None:
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="poll",
                    target_name="kafka_transport",
                )
                raise ProtocolConfigurationError(
                    "KafkaTransport cannot map a nullable Kafka header into "
                    "ModelTransportMessage.headers, which requires bytes.",
                    context=context,
                    parameter=f"headers[{key!r}]",
                    value=None,
                )
            headers[key] = value if isinstance(value, bytes) else bytes(value)
        offset = int(record.offset)  # type: ignore[attr-defined]
        return ModelTransportMessage(
            topic=record.topic,  # type: ignore[attr-defined]
            partition=int(record.partition),  # type: ignore[attr-defined]
            offset=offset,
            key=record.key,  # type: ignore[attr-defined]
            value=record.value,  # type: ignore[attr-defined]
            headers=headers,
            # Opaque cursor: TopicPartition + offset. The runtime never interprets
            # it; commit()/nack() reconstruct the same coordinate from the model.
            ack_token=(topic_partition, offset),
        )

    async def commit(self, message: object) -> None:
        """Advance the partition committed offset to ``message`` (commits all <= it).

        ``message`` is typed ``object`` (a supertype of the protocol's
        ``ProtocolTransportMessage``) so this impl stays contravariantly compatible
        with the protocol while narrowing to the concrete ``ModelTransportMessage``
        it yields from ``poll`` — the same convention as ``InMemoryTransport``.
        Committing ``offset + 1`` (the next offset to fetch) advances the high-water
        mark to ``msg.offset``, committing every earlier offset on the partition.
        """
        msg = cast("ModelTransportMessage", message)
        consumer = self._require_consumer()
        topic_partition = TopicPartition(msg.topic, msg.partition)
        await consumer.commit({topic_partition: msg.offset + 1})

    async def nack(self, message: object) -> None:
        """Re-expose ``message`` and later same-partition offsets (Kafka ``seek``).

        Does NOT advance the committed offset; seeks the live consumer's fetch
        position back to ``msg.offset`` so the next ``poll`` refetches it and every
        later offset on that partition.

        Sibling partitions are NOT touched (OMN-14757). ``_prime`` eagerly prefetches
        one bounded batch across EVERY assigned partition, advancing every partition's
        fetch position. A ``seek`` rewinds only the nacked partition, so its buffered
        records are stale and must be refetched — but the buffered records from OTHER
        partitions are the ONLY remaining copy at their (already-advanced) fetch
        position. Clearing the whole buffer here (the prior behaviour) dropped them:
        they became invisible until a consumer restart even though the runtime never
        touched them — an at-least-once liveness bug and a substitutability break vs
        the per-partition-lazy in-memory transport, whose ``nack`` touches only the
        nacked partition. So we discard ONLY the nacked partition's stale residue,
        RETAIN the sibling residue, and re-prime to refill the sought partition.
        """
        msg = cast("ModelTransportMessage", message)
        consumer = self._require_consumer()
        topic_partition = TopicPartition(msg.topic, msg.partition)
        consumer.seek(topic_partition, msg.offset)
        # Keep prefetched residue from OTHER (topic, partition)s — dropping it strands
        # those messages until restart. Only the nacked partition's buffered records
        # are stale post-seek; drop just those and re-prime to refill from the sought
        # position. Per-partition offset order is preserved: retained siblings stay
        # ascending, and the re-primed nacked-partition records append after them.
        self._buffer = [
            buffered
            for buffered in self._buffer
            if (buffered.topic, buffered.partition) != (msg.topic, msg.partition)
        ]
        await self._prime(consumer)

    # -- producer protocol ------------------------------------------------------

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None:
        """Publish one event to ``topic`` and await the broker acknowledgement."""
        if self._producer is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="send",
                target_name="kafka_transport",
            )
            raise ProtocolConfigurationError(
                "KafkaTransport producer used before start(); await start() first.",
                context=context,
                parameter="producer",
                value=None,
            )
        kafka_headers: list[tuple[str, bytes]] | None = [
            (key_, value_) for key_, value_ in headers.items()
        ] or None
        await self._producer.send_and_wait(
            topic, value=value, key=key, headers=kafka_headers
        )
