# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a queued deploy wrapping its command (OMN-26718)."""

from __future__ import annotations

from typing import Protocol

from omnibase_infra.protocols.protocol_queued_command import ProtocolQueuedCommand


class ProtocolQueuedDeploy(Protocol):
    @property
    def command(self) -> ProtocolQueuedCommand: ...
