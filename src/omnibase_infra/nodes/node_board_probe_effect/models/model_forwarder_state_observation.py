# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a reader saw of one gateway forwarder process.

Ticket: OMN-19930
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelForwarderStateObservation(BaseModel):
    """One read of a forwarder: process state, refused topics and cloud leg.

    A failed read is an observation with ``read_ok=False`` and the reason in
    ``read_error``, never an empty observation that could read as healthy.
    ``refused_topics`` are the wire topics refused within the last
    ``window_seconds`` of the forwarder's log and not admitted after.
    ``observed_cloud_broker`` is the bootstrap the forwarder's mounted
    broker-ref map resolves for the contract's cloud broker ref, or ``""``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    read_ok: bool
    read_error: str = ""
    running: bool = False
    started_at: datetime | None = None
    observed_at: datetime
    window_seconds: int = Field(default=0, ge=0)
    refused_topics: tuple[str, ...] = ()
    observed_cloud_broker: str = ""


__all__ = ["ModelForwarderStateObservation"]
