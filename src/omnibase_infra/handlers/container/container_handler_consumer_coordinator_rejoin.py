# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerConsumerCoordinatorRejoin (OMN-21683).

The consumer rejoin supervisor decides rejoins through this handler. This module
is the one place that builds it.
"""

from __future__ import annotations

from omnibase_infra.handlers.handler_consumer_coordinator_rejoin import (
    HandlerConsumerCoordinatorRejoin,
)

__all__: list[str] = ["bind_handler_consumer_coordinator_rejoin"]


def bind_handler_consumer_coordinator_rejoin() -> HandlerConsumerCoordinatorRejoin:
    """Build the consumer coordinator rejoin decision handler."""
    return HandlerConsumerCoordinatorRejoin()
