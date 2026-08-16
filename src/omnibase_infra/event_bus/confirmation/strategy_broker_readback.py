# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Readback-confirmed durability (OMN-15861).

Lifts the produce+readback shape already proven in
``omnibase_infra/runtime/gateway_canary_probe.py`` (OMN-15741) -- produce, then
poll an authoritative surface until the record is observed or a
``readback_deadline_seconds`` budget expires -- and turns it from a healthcheck
into the confirmation seam every durable-outbox ack goes through.

The three outcomes are kept distinct on purpose:

* observed within the deadline           -> ``CONFIRMED``
* surface reached, record not there      -> ``UNCONFIRMED`` (retry; keep the record)
* surface unreachable / receipt missing  -> ``UNKNOWN``     (fails closed; keep the record)

``UNCONFIRMED`` and ``UNKNOWN`` produce the same ack decision (do not ack) but
are *reported* differently, because "the broker says no" and "the broker never
answered" need different operator responses.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from omnibase_infra.enums.enum_confirmation_state import EnumConfirmationState
from omnibase_infra.event_bus.models.model_durability_confirmation import (
    ModelDurabilityConfirmation,
)
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.protocols.protocol_readback_source import ProtocolReadbackSource

logger = logging.getLogger(__name__)

STRATEGY_NAME_BROKER_READBACK = "broker_readback"

DEFAULT_READBACK_DEADLINE_SECONDS = 5.0


class BrokerReadbackStrategy:
    """Confirms a receipt by reading the record back off an authoritative surface.

    Args:
        source: The surface to consult. Its ``transport`` must match the
            receipt's, or the confirmation resolves to ``UNKNOWN`` -- confirming
            a Kafka produce against an in-memory history would be a false
            durable claim, so the mismatch is refused rather than ignored.
        readback_deadline_seconds: Wall-clock budget handed to the source. Must
            be ``> 0`` (mirrors the ``gt=0`` validation on
            ``ModelGatewayCanaryConfig.readback_deadline_seconds``).

    Raises:
        ValueError: If ``readback_deadline_seconds`` is not positive.
    """

    def __init__(
        self,
        source: ProtocolReadbackSource,
        *,
        readback_deadline_seconds: float = DEFAULT_READBACK_DEADLINE_SECONDS,
    ) -> None:
        if readback_deadline_seconds <= 0:
            raise ValueError(
                "readback_deadline_seconds must be > 0; a non-positive deadline "
                "makes every confirmation fail closed and silently stalls the outbox"
            )
        self._source = source
        self._deadline_seconds = readback_deadline_seconds

    @property
    def name(self) -> str:
        """Stable identifier recorded on every confirmation."""
        return STRATEGY_NAME_BROKER_READBACK

    async def confirm(
        self, receipt: ModelPublishReceipt | None
    ) -> ModelDurabilityConfirmation:
        """Resolve ``receipt`` by readback. Never raises; unreachable -> ``UNKNOWN``."""
        now = datetime.now(UTC)

        if receipt is None:
            return ModelDurabilityConfirmation(
                state=EnumConfirmationState.UNKNOWN,
                strategy=self.name,
                receipt=None,
                checked_at=now,
                detail=(
                    "publish returned no durability coordinate; nothing to read back"
                ),
            )

        if receipt.transport is not self._source.transport:
            return ModelDurabilityConfirmation(
                state=EnumConfirmationState.UNKNOWN,
                strategy=self.name,
                receipt=receipt,
                checked_at=now,
                detail=(
                    f"readback source transport {self._source.transport.value} does "
                    f"not match receipt transport {receipt.transport.value}; refusing "
                    "to confirm a record against a surface that cannot hold it"
                ),
            )

        try:
            observed = await self._source.observe(
                receipt, deadline_seconds=self._deadline_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- boundary: unreachable != absent
            logger.warning(
                "readback source unreachable for %s: %s",
                receipt.coordinate,
                exc,
                exc_info=True,
                extra={
                    "topic": receipt.topic,
                    "partition": receipt.partition,
                    "offset": receipt.offset,
                },
            )
            return ModelDurabilityConfirmation(
                state=EnumConfirmationState.UNKNOWN,
                strategy=self.name,
                receipt=receipt,
                checked_at=datetime.now(UTC),
                detail=(
                    f"readback source raised {type(exc).__name__}: {exc}; "
                    "indeterminate, failing closed"
                ),
            )

        if observed:
            return ModelDurabilityConfirmation(
                state=EnumConfirmationState.CONFIRMED,
                strategy=self.name,
                receipt=receipt,
                checked_at=datetime.now(UTC),
                detail="",
            )

        return ModelDurabilityConfirmation(
            state=EnumConfirmationState.UNCONFIRMED,
            strategy=self.name,
            receipt=receipt,
            checked_at=datetime.now(UTC),
            detail=(
                f"record at {receipt.coordinate} was not observed within "
                f"{self._deadline_seconds}s of readback"
            ),
        )


__all__: list[str] = [
    "DEFAULT_READBACK_DEADLINE_SECONDS",
    "STRATEGY_NAME_BROKER_READBACK",
    "BrokerReadbackStrategy",
]
