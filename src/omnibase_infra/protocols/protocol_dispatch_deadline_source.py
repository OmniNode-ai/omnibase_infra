# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event-bus surface the ``dispatch_deadline`` health dimension reads (OMN-19355).

Structural for the same reasons as ``ProtocolConsumerSyncSource``: the health
monitor is handed a ``ProtocolEventBusLike`` and runs against every transport,
including the in-memory bus, which has no consume loop to abandon a dispatch
from. Asking whether a transport can answer is the honest form of the check.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.health.model_dispatch_deadline_status import (
    ModelDispatchDeadlineStatus,
)


@runtime_checkable
class ProtocolDispatchDeadlineSource(Protocol):
    """A transport that can report dispatches it abandoned at their deadline."""

    def dispatch_deadline_status(self) -> ModelDispatchDeadlineStatus:
        """Return the bus's abandoned-dispatch state. Must not perform I/O."""
        ...


__all__ = ["ProtocolDispatchDeadlineSource"]
