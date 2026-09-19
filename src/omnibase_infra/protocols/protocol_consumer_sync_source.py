# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event-bus surface the ``consumer_sync`` readiness dimension reads (OMN-18640 AC1).

Declared structurally rather than importing ``EventBusKafka`` into the health
monitor, for two reasons that are both about what the monitor is allowed to
assume.

First, the monitor is handed a ``ProtocolEventBusLike`` and runs against every
transport, including the in-memory bus that has no consumer groups at all.
Asking "can you answer the consumer-sync question?" is the honest form of that
check; an ``isinstance(bus, EventBusKafka)`` would couple a health service to
one concrete transport and would silently answer "no" for any future Kafka
implementation.

Second, ``omnibase_infra.services`` importing ``omnibase_infra.event_bus``
concretely at module scope is a dependency this repo does not otherwise have,
and the health monitor is the one service that must never fail to start
because something it reports on failed to import.

``@runtime_checkable`` is load-bearing here: the check happens at runtime
against whatever object the kernel wired, which is exactly the question being
asked.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.health.model_consumer_sync_status import (
    ModelConsumerSyncStatus,
)


@runtime_checkable
class ProtocolConsumerSyncSource(Protocol):
    """A transport that can report the sync state of the groups it consumes."""

    def consumer_sync_statuses(self) -> tuple[ModelConsumerSyncStatus, ...]:
        """Return one status per ``(topic, consumer group)`` being consumed.

        Must not perform I/O: it reports what the consume loop has already
        measured. An empty tuple means this process consumes nothing yet,
        which is a legitimate boot state and not a finding.
        """
        ...


__all__ = ["ProtocolConsumerSyncSource"]
