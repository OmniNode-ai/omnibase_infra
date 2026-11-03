# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for the clock a non-EFFECT node package reads time through.

A handler or dispatcher in a non-EFFECT node package never calls the wall clock
or a monotonic clock itself. It takes a ``ProtocolClock`` through its
constructor; the node's wiring supplies the system clock and tests supply a
fixed one.

Related Tickets:
    - OMN-21698: Registration orchestrator clock injection
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolClock(Protocol):
    """Source of wall-clock readings and monotonic durations."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC datetime."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic reading in seconds, for measuring durations."""
        ...


__all__ = ["ProtocolClock"]
