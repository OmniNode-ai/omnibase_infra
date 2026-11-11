# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""System clock behind ``ProtocolClock``.

The one place the wall clock and the performance counter are read for node
packages that are not EFFECT nodes. Those packages take a ``ProtocolClock``
(through the typed handler request or a dispatcher constructor) and never call
``datetime.now`` or ``time.perf_counter`` in their own modules; tests pass a
fixed clock in place of this one.

Related Tickets:
    - OMN-21698: Registration orchestrator clock injection
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


class ModelSystemClock(BaseModel):
    """``ProtocolClock`` backed by the host's wall and performance clocks.

    Stateless: every instance is equal to every other.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    def now(self) -> datetime:
        """Return the current timezone-aware UTC datetime."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return the performance counter reading in seconds."""
        return time.perf_counter()


SYSTEM_CLOCK: ModelSystemClock = ModelSystemClock()
"""The shared system clock, the default of a dispatcher constructor."""

__all__ = ["SYSTEM_CLOCK", "ModelSystemClock"]
