# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The typed refusal, and the typed value it becomes, for an unparseable DLQ
record (OMN-17896).

WHY THIS MODULE EXISTS. ``ModelDlqMessage.from_kafka_message`` used to convert
an absent ``original_message.value`` into an empty string, which the replay
producer then encoded to a ZERO-BYTE record and published back onto the
original topic. Measured on the dev lane 2026-09-07: 200 of 200 sampled
records on ``onex.evt.omniclaude.tool-executed.v1`` carried both ``VALSZ=0``
and ``x-replayed-by=node_dlq_replay_effect``, sustained at roughly 3.5-4.3
records/s, producing 5,535 ``JSONDecodeError`` lines in a five-minute window.

The refusal is only half the repair. Raised where it used to be swallowed, a
refusal must reach the quarantine path, not escape the drain: the parse runs
inside ``DLQConsumer.consume_messages``' generator, whose enclosing ``try``
catches only ``asyncio.CancelledError`` and ``KafkaError``, so a raise there
propagates out of ``anext`` in ``HandlerDlqReplay.run()``, past the quarantine
decision, and out of the run -- the batch aborts, nothing is quarantined, and
the offset never advances. So the generator does NOT publish (a durable write
awaited inside ``asyncio.wait_for(anext(...))`` is cancellable and would be
lost); it yields ``ModelUnparseableDlqRecord``, a typed value the handler
recognises on its own frame and quarantines with the raw bytes intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
        ModelDlqMessage,
    )


class DlqRecordUnparseableError(ValueError):
    """A DLQ record cannot be parsed into a ``ModelDlqMessage``.

    Subclasses ``ValueError`` deliberately: ``ModelDlqMessage._parse_retry_count``
    already raised a bare ``ValueError`` on this same path, so every existing
    caller and test that expects ``ValueError`` keeps working while the
    consumer boundary can catch this ONE named type rather than widening to a
    bare ``except Exception`` that would swallow programming errors too.
    """


class ModelUnparseableDlqRecord(BaseModel):
    """A DLQ record the parse refused, carried to the handler for quarantine.

    Holds the record's RAW BYTES rather than any parsed field. A record whose
    payload cannot be parsed still has bytes, and those bytes are the only
    thing that makes it reclassifiable later; inventing values for the fields
    ``build_quarantine_payload`` reads off a parsed message would be exactly
    the fabricated default this ticket exists to remove.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dlq_topic: str = Field(..., min_length=1, description="DLQ topic read from.")
    dlq_partition: int = Field(..., ge=0, description="Partition of the record.")
    dlq_offset: int = Field(..., ge=0, description="Offset of the record.")
    raw_value: bytes | None = Field(
        ...,
        description=(
            "The record's raw bytes exactly as Kafka delivered them, or None "
            "when the record carried no value at all. Never a substituted "
            "empty body."
        ),
    )
    reason: str = Field(
        ..., min_length=1, description="Why the record could not be parsed."
    )


#: What ``DLQConsumer.consume_messages`` yields and ``HandlerDlqReplay.run``
#: consumes: either a parsed DLQ message, or the typed refusal that carries the
#: raw bytes of one that would not parse. Declared once, as an alias, so the
#: shape is stated in a single place rather than repeated at every seam.
type DlqDrainRecord = ModelDlqMessage | ModelUnparseableDlqRecord

__all__ = [
    "DlqDrainRecord",
    "DlqRecordUnparseableError",
    "ModelUnparseableDlqRecord",
]
