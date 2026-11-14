# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a runtime event bus does when it loses its Kafka broker.

A runtime that serves other processes must not route their traffic into a
private in-memory bus no other process can reach. Local fallback is only for a
runtime declared single-process, in its ``event_bus.recovery`` configuration or
the shipped runtime_event_bus_recovery.yaml contract.
"""

from __future__ import annotations

from enum import Enum


class EnumEventBusLostBrokerPolicy(str, Enum):
    """Behaviour of the runtime bus while its Kafka broker is unreachable."""

    REFUSE_READY = "refuse_ready"
    """Shared runtime: report not ready, fail publishes, keep retrying Kafka."""

    LOCAL_FALLBACK = "local_fallback"
    """Declared single-process runtime: deliver in memory while Kafka recovers."""


__all__ = ["EnumEventBusLostBrokerPolicy"]
