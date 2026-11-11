# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Transactional producer over the in-memory bus (OMN-21228).

The same call as the Kafka producer, with the same fencing rule, so a consumer
of ``ProtocolTransactionalProducer`` is exercised without a broker.

The in-memory bus has no transaction log, so "fenced" is an epoch comparison:
``InmemoryTransactionalEpochs.initialize`` hands out the next epoch for an id,
and a producer holding an older epoch is fenced. The comparison runs inside the
bus's own lock together with the append of the whole batch
(``EventBusInmemory.publish_atomic``, see ``inmemory_bus_publish``), so a fence lands before a batch or after
it, never in the middle: a fenced producer leaves nothing behind.

Not an ONEX handler (no ``handler_type``/``handler_category``); same arrangement
as ``kafka_permanent_failure``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.handlers.transactional_producer_fencing import (
    InfraProducerFencedError,
)
from omnibase_infra.models.event_bus.model_transactional_record import (
    ModelTransactionalRecord,
)

PublishAtomic = Callable[..., Awaitable[tuple[ModelPublishReceipt, ...]]]


class InmemoryTransactionalEpochs:
    """The current epoch of each ``transactional_id`` on one in-memory bus."""

    def __init__(self) -> None:
        self._epochs: dict[str, int] = {}

    def initialize(self, transactional_id: str) -> int:
        """Start a new epoch for ``transactional_id``, fencing the previous one."""
        epoch = self._epochs.get(transactional_id, -1) + 1
        self._epochs[transactional_id] = epoch
        return epoch

    def is_current(self, transactional_id: str, epoch: int) -> bool:
        return self._epochs.get(transactional_id) == epoch


class InmemoryTransactionalProducer:
    """One ``transactional_id`` at one epoch on one in-memory bus."""

    def __init__(
        self,
        *,
        publish_atomic: PublishAtomic,
        epochs: InmemoryTransactionalEpochs,
        transactional_id: str,
    ) -> None:
        self._publish_atomic = publish_atomic
        self._epochs = epochs
        self._transactional_id = transactional_id
        self._epoch = epochs.initialize(transactional_id)
        self._closed = False

    @property
    def transactional_id(self) -> str:
        return self._transactional_id

    def _context(self) -> ModelInfraErrorContext:
        return ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.INMEMORY,
            operation="publish",
            target_name=f"inmemory.transactional.{self._transactional_id}",
        )

    def _require_current(self) -> None:
        if not self._epochs.is_current(self._transactional_id, self._epoch):
            raise InfraProducerFencedError(
                f"Transactional producer {self._transactional_id!r} was fenced by a "
                "newer producer with the same transactional id",
                context=self._context(),
            )

    async def publish(
        self, records: Sequence[ModelTransactionalRecord]
    ) -> tuple[ModelPublishReceipt, ...]:
        if self._closed:
            raise InfraUnavailableError(
                f"Transactional producer {self._transactional_id!r} is closed",
                context=self._context(),
            )
        if not records:
            raise ProtocolConfigurationError(
                "A transactional publish needs at least one record",
                context=self._context(),
                parameter="records",
                value=0,
            )
        return await self._publish_atomic(
            [(r.topic, r.key, r.value, r.headers) for r in records],
            precondition=self._require_current,
        )

    async def close(self) -> None:
        self._closed = True


def open_transactional_producer(
    publish_atomic: PublishAtomic,
    epochs: InmemoryTransactionalEpochs,
    transactional_id: str,
    *,
    started: bool,
) -> InmemoryTransactionalProducer:
    """Open a producer under ``transactional_id``, fencing any older one."""
    if not started:
        raise InfraUnavailableError(
            "Event bus not started. Call start() first.",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INMEMORY,
                operation="transactional_producer",
                target_name="inmemory.transactional",
            ),
        )
    return InmemoryTransactionalProducer(
        publish_atomic=publish_atomic,
        epochs=epochs,
        transactional_id=transactional_id,
    )
