# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""States of the runtime shared listener (OMN-22377)."""

from __future__ import annotations

from enum import Enum


class EnumSharedListenerState(str, Enum):
    """Listener state machine of one runtime (rule A5)."""

    DETACHED = "detached"
    """Not accepting on the shared socket."""

    ATTACHING = "attaching"
    """Fetching the socket and starting the site; counts for nothing yet."""

    ATTACHED = "attached"
    """Confirmed acceptor on the shared socket."""

    DRAINING = "draining"
    """Stopped accepting; answering the connections already accepted."""

    HOLDING = "holding"
    """Attached but not ready; the last acceptor holds its requests (A6)."""
