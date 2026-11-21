# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the subscription wiring tells a member's loop sampler about ticks (OMN-26249)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolTickObserver(Protocol):
    """Told when a member subscribes to the runtime tick and when one is consumed."""

    def mark_tick_subscriber(self) -> None:
        """The member wired a subscription to the runtime tick topic."""
        ...

    def record_tick(self, scheduled_at: datetime, consumed_at: datetime) -> None:
        """A tick scheduled at ``scheduled_at`` was consumed at ``consumed_at``."""
        ...


__all__ = ["ProtocolTickObserver"]
