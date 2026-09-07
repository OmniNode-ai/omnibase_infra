# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror delivery-verification counters for the container healthcheck.

OMN-17201. The lane mirror already awaits a real broker acknowledgement for
every record (``KafkaTransport.send`` is ``send_and_wait``), so the
``Lane mirror delivered`` line is not a fire-and-forget lie in the usual sense.
It is a *weaker* claim than it reads as: an acknowledgement proves that A
broker took the record, never that the INTENDED lane's broker did, and never
that the destination high-water mark moved as a result.

That gap is not theoretical. Measured on .201 2026-09-07, read-only: the
stability lane's ``onex.evt.omniclaude.tool-executed.v1`` p0 carries the same
``message_id`` at offsets 55985 and 55986, 52 ms apart, with an identical
envelope ``timestamp`` header -- the second is the mirror's own republish
landing back on its SOURCE lane, because both lane brokers advertise the bare
name ``redpanda`` and the dual-homed forwarder re-resolved it onto the wrong
network. 754 ``Lane mirror delivered`` lines were emitted while the dev lane's
high-water mark did not take a single one of them.

This model is what makes that shape reportable instead of invisible. It is a
FILE contract for the same reason ``ModelGatewayEgressHealth`` is: the writer
is the long-running forwarder process and the reader is
``onex-gateway-canary-probe``, a separate exec in the same container that
deliberately never talks to the forwarder.

Offsets are kept per ``"<topic>:<partition>"`` for the source and per
``"<lane>|<topic>:<partition>"`` for each destination, both holding the highest
SOURCE offset seen in that role. Two maps rather than one lag number because a
single scalar cannot say WHICH lane stopped confirming, and "which lane" is the
whole operator action.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayLaneMirrorHealth(BaseModel):
    """Lane-mirror delivery counters for one gateway process lifetime.

    Lifetime-of-process, not windowed, exactly like
    ``ModelGatewayEgressHealth``: the window belongs to the reader, which needs
    raw timestamps to tell "nothing is confirming right now" apart from "one
    lane blipped an hour ago and everything has crossed since".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mirrored_total: int = Field(
        default=0,
        ge=0,
        description="Records this process published to EVERY declared mirror lane",
    )
    refused_total: int = Field(
        default=0,
        ge=0,
        description="Records refused for carrying no usable identity (OMN-17919)",
    )
    accepted_then_failed_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Records this leg accepted and then failed to publish -- the class "
            "that ``refused_total`` structurally cannot count, because the "
            "refusal happens before the publish is attempted"
        ),
    )
    loop_detected_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Times a record this process mirrored came back on the SOURCE lane "
            "at a later offset, proving a destination IS the source"
        ),
    )
    consumed_source_offsets: dict[str, int] = Field(
        default_factory=dict,
        description='Highest source offset consumed, keyed "<topic>:<partition>"',
    )
    confirmed_delivered_offsets: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Highest source offset a destination broker ACKNOWLEDGED, keyed "
            '"<lane>|<topic>:<partition>"'
        ),
    )
    last_mirrored_at: datetime | None = Field(default=None)
    last_failure_at: datetime | None = Field(default=None)
    last_failure_lane: str | None = Field(default=None)
    last_loop_detected_at: datetime | None = Field(default=None)
    last_loop_detected_lane: str | None = Field(default=None)


__all__ = ["ModelGatewayLaneMirrorHealth"]
