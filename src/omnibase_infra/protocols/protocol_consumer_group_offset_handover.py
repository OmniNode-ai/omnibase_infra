# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a bus that can seed a new group from a retired one.

OMN-21753, under OMN-21288. Auto-wiring reaches the bus as
``ProtocolEventBusSubscriber`` (``omnibase_core``), whose ``subscribe`` has no
notion of where a new group starts beyond ``auto_offset_reset``. A writer
folded into the runtime needs its first position to be the retired writer
container's committed offsets, so the handover is its own call, made BEFORE
``subscribe`` joins the group. ``HandlerConsumerGroupOffsetHandover`` serves
``EventBusKafka`` and ``EventBusInmemory`` itself (their modules are baselined
and only shrink) and asks any other bus that implements this protocol.

The method name is deliberately distinctive: ``@runtime_checkable`` checks
member presence, not signatures.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)

__all__ = ["ProtocolConsumerGroupOffsetHandover"]


@runtime_checkable
class ProtocolConsumerGroupOffsetHandover(Protocol):
    """A bus that copies a retired group's committed offsets into a new group."""

    async def hand_over_committed_offsets(
        self,
        *,
        topic: str,
        from_group: str,
        to_group: str,
        from_auto_offset_reset: Literal["earliest", "latest"],
    ) -> ModelOffsetHandoverResult:
        """Seed ``to_group`` on ``topic`` from ``from_group``.

        Must be called before the ``subscribe`` that joins ``to_group``,
        while the retired consumer is stopped. Applies
        ``plan_offset_handover``: a partition ``to_group`` already committed
        is kept, so a second call is a no-op.

        Args:
            topic: Canonical topic, as passed to ``subscribe``.
            from_group: The retired consumer's group, used as is.
            to_group: The group id passed to ``subscribe``; the bus applies
                the same scoping it applies there.
            from_auto_offset_reset: Where the retired consumer started on a
                partition it never committed.
        """
        ...
