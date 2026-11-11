# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Transactional producer over aiokafka, fenced by a caller-fixed id (OMN-21228).

``EventBusKafka.transactional_producer`` builds the underlying
``AIOKafkaProducer`` with its own bootstrap, auth and client-version settings
and hands it here; this module owns what happens next. ``start`` is where the
fence is applied: aiokafka's ``start`` calls ``InitProducerId`` for the
``transactional_id``, the broker bumps the producer epoch and aborts any open
transaction of the previous holder, and the previous holder's next transaction
operation fails with a fencing error that this module maps to
``InfraProducerFencedError``.

One ``publish`` is one transaction. Records are validated and staged first, so a
bad record refuses the whole batch before anything is sent; they are then sent
inside the transaction and become visible to a ``read_committed`` reader only at
commit. There is no retry: a fence cannot be retried, and any other failure
aborts the transaction, so a retry is the caller's decision.

Not an ONEX handler (no ``handler_type``/``handler_category``); same arrangement
as ``kafka_permanent_failure``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.handlers.kafka_publish_headers import headers_for_unheadered_publish
from omnibase_infra.handlers.transactional_producer_fencing import (
    InfraProducerFencedError,
    is_producer_fenced,
)
from omnibase_infra.models.event_bus.model_transactional_record import (
    ModelTransactionalRecord,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer
    from aiokafka.structs import RecordMetadata

    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

# The bus resolves one record to (physical topic, wire headers, event headers).
PrepareRecord = Callable[
    [ModelTransactionalRecord],
    Awaitable[tuple[str, list[tuple[str, bytes]], ModelEventHeaders]],
]


class KafkaTransactionalProducer:
    """One ``transactional_id``, one aiokafka producer, one transaction per publish."""

    def __init__(
        self,
        *,
        transactional_id: str,
        producer: AIOKafkaProducer,
        cluster: str,
        timeout_seconds: float,
        prepare: PrepareRecord,
    ) -> None:
        self._transactional_id = transactional_id
        self._producer = producer
        self._cluster = cluster
        self._timeout_seconds = timeout_seconds
        self._prepare = prepare
        self._closed = False
        self._fenced = False
        # One transaction at a time per producer: the broker allows no more.
        self._publish_lock = asyncio.Lock()

    @property
    def transactional_id(self) -> str:
        return self._transactional_id

    async def start(self) -> None:
        """Initialize the producer id for ``transactional_id``, fencing older holders."""
        try:
            await asyncio.wait_for(
                self._producer.start(), timeout=self._timeout_seconds
            )
        except Exception as exc:
            self._closed = True
            raise InfraConnectionError(
                f"Transactional producer {self._transactional_id!r} failed to initialize",
                context=self._context("start"),
            ) from exc

    def _context(self, operation: str) -> ModelInfraErrorContext:
        return ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation=operation,
            target_name=f"kafka.transactional.{self._transactional_id}",
        )

    async def publish(
        self, records: Sequence[ModelTransactionalRecord]
    ) -> tuple[ModelPublishReceipt, ...]:
        if self._fenced:
            raise self._fenced_error()
        if self._closed:
            raise InfraUnavailableError(
                f"Transactional producer {self._transactional_id!r} is closed",
                context=self._context("publish"),
            )
        if not records:
            raise ProtocolConfigurationError(
                "A transactional publish needs at least one record",
                context=self._context("publish"),
                parameter="records",
                value=0,
            )
        staged = [(record, await self._prepare(record)) for record in records]

        async with self._publish_lock:
            futures: list[asyncio.Future[RecordMetadata]] = []
            try:
                async with self._producer.transaction():
                    for record, (topic, kafka_headers, _) in staged:
                        futures.append(
                            await self._producer.send(
                                topic,
                                value=record.value,
                                key=record.key,
                                headers=kafka_headers,
                            )
                        )
                metadata = await asyncio.wait_for(
                    asyncio.gather(*futures), timeout=self._timeout_seconds
                )
            except Exception as exc:
                for future in futures:
                    if future.done() and not future.cancelled():
                        future.exception()  # consumed: the transaction error is the one reported
                if is_producer_fenced(exc):
                    self._fenced = True
                    raise self._fenced_error() from exc
                raise InfraConnectionError(
                    f"Transactional publish under {self._transactional_id!r} aborted",
                    context=self._context("publish"),
                ) from exc

        now = datetime.now(UTC)
        return tuple(
            ModelPublishReceipt(
                topic=str(getattr(meta, "topic", topic) or topic),
                partition=int(meta.partition),
                offset=int(meta.offset),
                cluster=self._cluster,
                produced_at=now,
                transport=EnumInfraTransportType.KAFKA,
                idempotency_key=headers.idempotency_key,
            )
            for (_, (topic, _, headers)), meta in zip(staged, metadata, strict=True)
        )

    def _fenced_error(self) -> InfraProducerFencedError:
        return InfraProducerFencedError(
            f"Transactional producer {self._transactional_id!r} was fenced by a "
            "newer producer with the same transactional id",
            context=self._context("publish"),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._producer.stop()
        except Exception:
            if not self._fenced:
                raise


async def open_kafka_transactional_producer(
    bus: EventBusKafka, transactional_id: str
) -> KafkaTransactionalProducer:
    """Open and initialize a producer for ``transactional_id`` on ``bus``'s cluster.

    The producer is built with the bus's own bootstrap, auth and client-version
    settings (``EventBusKafka._new_producer``). Initializing it is what fences
    every older producer under the same id.

    Raises:
        InfraUnavailableError: If the bus has not been started.
        InfraConnectionError: If the producer cannot initialize.
    """
    if not bus._started:
        raise InfraUnavailableError(
            "Event bus not started. Call start() first.",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="transactional_producer",
                target_name=f"kafka.{bus._environment}",
            ),
        )

    async def prepare(
        record: ModelTransactionalRecord,
    ) -> tuple[str, list[tuple[str, bytes]], ModelEventHeaders]:
        """Validate one record as ``publish`` does; resolve its wire topic and headers."""
        headers = record.headers or headers_for_unheadered_publish(
            record.topic,
            record.value,
            environment=bus._environment,
            minted_identity_topics=bus._minted_identity_topics,
        )
        bus._validate_topic_name(record.topic, headers.correlation_id)
        await bus._enforce_onex_topic_format(record.topic, headers.correlation_id)
        return (
            bus.resolve_bus_binding(record.topic).physical_topic,
            bus._model_headers_to_kafka(headers),
            headers,
        )

    producer = KafkaTransactionalProducer(
        transactional_id=transactional_id,
        producer=bus._new_producer(transactional_id=transactional_id),
        cluster=bus._bootstrap_servers,
        timeout_seconds=bus._timeout_seconds,
        prepare=prepare,
    )
    await producer.start()
    return producer
