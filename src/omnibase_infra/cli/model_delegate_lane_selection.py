# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lane a delegation is addressed to, and the broker that lane declares.

OMN-16871. ``onex delegate --bus kafka`` used to resolve its broker ADDRESS
from the ambient ``KAFKA_BOOTSTRAP_SERVERS`` environment variable. On the
launching Mac that variable names the ``.201`` STABILITY-TEST broker, so every
ad hoc delegation issued from a developer shell landed on a governed proof
lane -- the lane whose readiness projection the compose-path prod-promotion
gate resolves its ``stability-proven`` premise from. The bus TYPE was already
config-resolved (OMN-17304: "env vars may bootstrap WHERE configuration is
found, never WHAT the transport is"); the address was the half that ruling had
not reached.

This model is the resolved answer to "which lane, and therefore which broker".
It is produced only from a checked-in lane declaration, never from the
environment, and it carries its own provenance so a receipt can name the file
that answered rather than merely the address.

The transport fields travel with the address on purpose. The two halves were
split once already (OMN-18012) and the result was a plaintext client opening
against a SASL listener for 25 consecutive runs; an address with no declared
protocol is not a usable statement about a lane.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateLaneSelection"]


class ModelDelegateLaneSelection(BaseModel):
    """One delegation's resolved lane target, read from a lane declaration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    lane: str = Field(description="Lane id the caller selected explicitly")
    bootstrap_servers: str = Field(
        description="Broker the declaration binds to this lane, host:port"
    )
    security_protocol: str = Field(
        description="librdkafka security protocol, exactly as the lane declares it"
    )
    sasl_mechanism: str | None = Field(
        default=None,
        description="Declared SASL mechanism; absent on a non-SASL protocol",
    )
    declared_in: Path = Field(
        description="The lane declaration this selection was read out of"
    )
