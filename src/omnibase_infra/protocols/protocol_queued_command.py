# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a queued deploy command (OMN-26718)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from omnibase_infra.protocols.protocol_valued import ProtocolValued


class ProtocolQueuedCommand(Protocol):
    """The fields of a queued deploy command the priority request reads."""

    correlation_id: str
    runtime_lane: ProtocolValued
    requested_by: str
    git_ref: str
    scope: ProtocolValued
    build_source: ProtocolValued
    services: Sequence[str]
    rollback: object | None
    image_ref: str | None
    image_digest: str | None
