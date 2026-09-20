# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a bus that accepts a per-subscription in-flight bound.

OMN-18852. Auto-wiring reaches the bus as ``ProtocolEventBusSubscriber``
(``omnibase_core``), whose ``subscribe`` declares neither
``auto_offset_reset`` nor ``required_for_readiness`` nor any concurrency
argument. Widening that protocol would be a third-repo change for a
transport-local concern, and passing an undeclared keyword through it does not
type-check. So the bound is declared by its own call, made BEFORE ``subscribe``
starts the consume loop, and the capability is tested structurally.

The method name is deliberately distinctive. ``@runtime_checkable`` checks
member PRESENCE and not signatures, so a protocol declaring only ``subscribe``
would match every bus in the repo -- including ``EventBusInmemory``, which has
no poll loop to bound -- and the narrowing would be a lie that fails at the
call. Exactly one class implements ``declare_consume_concurrency``, so
``isinstance`` against this protocol is a genuine capability test.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ProtocolConsumeConcurrencyDeclarer"]


@runtime_checkable
class ProtocolConsumeConcurrencyDeclarer(Protocol):
    """A bus whose consume loop can hold more than one record in flight."""

    def declare_consume_concurrency(
        self,
        *,
        topic: str,
        group_id: str,
        max_in_flight_records: int,
    ) -> None:
        """Bound concurrently in-flight records for one ``(topic, group_id)``.

        Must be called before the ``subscribe`` that starts the consume loop
        for that pair; the loop reads the bound when it starts.

        Args:
            topic: Topic the subscription consumes.
            group_id: The consumer group id the subscription resolves to --
                the same value ``subscribe`` keys its consumer by.
            max_in_flight_records: 1 keeps the inline serial path unchanged;
                greater than 1 dispatches under a semaphore.
        """
        ...
