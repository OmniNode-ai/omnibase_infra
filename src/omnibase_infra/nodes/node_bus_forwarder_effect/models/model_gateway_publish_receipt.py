# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where a destination broker actually put a published record (OMN-17201).

``send_and_wait`` resolving proves a broker acknowledged the record. It does
not name the broker. Both .201 lane brokers are the compose service
``redpanda``, both advertise that bare name on their internal listener, and the
gateway forwarder is joined to BOTH lane networks -- so a producer that
bootstraps against the intended lane can be handed ``redpanda:9092`` in the
metadata response, re-resolve it onto the other network, and acknowledge every
subsequent record from the lane it is reading FROM. Nothing raises. The
``Lane mirror delivered`` line is emitted for every record. The destination
lane's high-water mark never moves.

Measured on .201 2026-09-07, read-only, and the reason this model exists: on
the stability lane's ``onex.evt.omniclaude.tool-executed.v1``, a 6-partition
1500-record window sampled at three epochs carried 389 identifiable records and
**zero** duplicate ``message_id`` values before 20:44Z, then 82 and 106
duplicate ids in equal-sized windows after it -- the mirror's own republishes
landing back on its source lane. Over the same period the dev lane's p0 took
4000/4000 records from ``node_dlq_replay_effect`` and **zero** carrying a
``message_id`` at all.

The receipt is what turns "a broker took it" into "THIS coordinate took it".
Coordinates are the exact fact the loop check needs: a destination coordinate
that the source consumer subsequently READS can only exist if the destination
broker is the source broker. That test has no threshold, no heuristic and no
wire change -- the mirror stays a byte-for-byte republisher.

``None`` is a legitimate receipt: the HTTPS ingest publisher (OMN-16459) has no
broker coordinates to report, and a leg that cannot report them must not be
made to invent them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayPublishReceipt(BaseModel):
    """The destination coordinate a broker acknowledged for one publish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(description="Destination topic the broker wrote to")
    partition: int = Field(ge=0, description="Destination partition")
    offset: int = Field(ge=0, description="Destination offset assigned by the broker")

    @property
    def coordinate(self) -> tuple[str, int, int]:
        """``(topic, partition, offset)`` -- the loop check's comparison key."""
        return (self.topic, self.partition, self.offset)


__all__ = ["ModelGatewayPublishReceipt"]
