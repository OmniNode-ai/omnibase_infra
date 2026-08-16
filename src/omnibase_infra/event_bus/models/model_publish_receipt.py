# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Durability coordinate returned by ``EventBus.publish`` (OMN-15861).

Before this model existed, ``EventBusKafka.publish`` returned ``None``: the
``record_metadata.partition``/``record_metadata.offset`` the broker handed back
were written to a debug log and then discarded. A caller therefore could not
distinguish "the produce call did not raise" from "the record is on the broker
at a coordinate I can read back" -- and every durable-outbox ack in the platform
was built on the former while claiming the latter.

``ModelPublishReceipt`` is that missing coordinate. It is deliberately *not* a
durability claim on its own: holding a receipt proves only that a produce path
reported a position. Turning a receipt into a durable claim requires a
``ProtocolConfirmationStrategy`` (see
``omnibase_infra.event_bus.confirmation``), which reads the coordinate back from
an authoritative surface. Canonical invariant 7: *a publish return is not
durability.*

Related:
    - ``omnibase_infra.event_bus.confirmation``: strategies that consume a receipt
    - OMN-15861: projection-confirmed durable outbox
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelPublishReceipt(BaseModel):
    """A broker-reported position for one produced record.

    Attributes:
        topic: Topic the record was produced to. Always the *wire* topic, i.e.
            the name the broker knows, after any tenant prefixing.
        partition: Broker-assigned partition index (``>= 0``).
        offset: Broker-assigned offset within ``partition`` (``>= 0``).
        cluster: Identity of the broker/cluster that assigned the coordinate.
            For Kafka this is the configured bootstrap-servers string; for the
            in-memory bus it is ``"inmemory.<environment>"``. Two receipts with
            equal ``(topic, partition, offset)`` but different ``cluster`` are
            different records -- readback MUST compare this field.
        produced_at: UTC timestamp at which the produce path observed the
            coordinate. Timezone-aware; naive datetimes are rejected.
        transport: Which transport produced the coordinate. A confirmation
            strategy uses this to refuse a readback source that does not match.
        idempotency_key: Stable per-logical-event key, when the producer supplied
            one. Carried so a readback or projection lookup can match the record
            by identity rather than by position, which is what makes replay after
            a crash safe. ``None`` when the producer did not supply one -- in
            which case only positional readback is possible.

    Example:
        >>> from datetime import UTC, datetime
        >>> receipt = ModelPublishReceipt(
        ...     topic="onex.evt.emit.v1",
        ...     partition=3,
        ...     offset=41,
        ...     cluster="redpanda:9092",
        ...     produced_at=datetime(2026, 8, 13, tzinfo=UTC),
        ...     transport=EnumInfraTransportType.KAFKA,
        ...     idempotency_key="emit-7f3c",
        ... )
        >>> receipt.coordinate
        'onex.evt.emit.v1/3/41'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1, description="Wire topic name")
    partition: int = Field(..., ge=0, description="Broker-assigned partition")
    offset: int = Field(..., ge=0, description="Broker-assigned offset")
    cluster: str = Field(..., min_length=1, description="Broker/cluster identity")
    produced_at: datetime = Field(
        ..., description="Timezone-aware UTC instant the coordinate was observed"
    )
    transport: EnumInfraTransportType = Field(
        ..., description="Transport that assigned the coordinate"
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Stable logical-event key, when the producer supplied one",
    )

    @field_validator("produced_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes -- an ambiguous instant is not evidence."""
        return validate_timezone_aware_datetime(value)

    @property
    def coordinate(self) -> str:
        """Human-readable ``topic/partition/offset`` for logs and test failures."""
        return f"{self.topic}/{self.partition}/{self.offset}"


__all__: list[str] = ["ModelPublishReceipt"]
