# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Join a kernel-owned consumer's topics together, off the boot path (OMN-26418).

The kernel's savings correlation step used to ``await event_bus.subscribe(...)``
for its three signal topics one after another before it let the rest of the
boot continue. Each ``subscribe`` starts a Kafka consume loop and waits for the
group join, and a join waits on the broker's rebalance: on the 2026-11-23T10:55Z
dev boot the three joins took 25.1 s end to end (10:55:30.673 to 10:55:55.781),
19 s of it the first join alone, and on the 23:10Z boot of 2026-11-21 they took
35 s. Nothing downstream of that step reads those consumers, so the boot waited
on a join no later step needs.

:func:`subscribe_topics_concurrently` is what the kernel now runs as the first
statement of its savings correlation task: every topic joins at once, one
failed join is logged and never stops another, and the task the kernel already
cancels at shutdown carries the joins, so no new task needs an owner.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
    from omnibase_infra.models import ModelNodeIdentity

logger = logging.getLogger(__name__)

TopicCallback = Callable[["ModelEventMessage"], Awaitable[None]]


async def subscribe_topics_concurrently(
    subscribe: Callable[..., Awaitable[object]],
    subscriptions: Sequence[tuple[str, TopicCallback]],
    *,
    node_identity: ModelNodeIdentity,
    purpose: str,
) -> int:
    """Subscribe every ``(topic, callback)`` pair at once; return how many joined.

    A subscribe that raises is logged at ERROR with its traceback, its topic and
    ``purpose``, and counted out; the other subscribes are unaffected. Cancellation is not
    swallowed: the caller's task is cancelled at shutdown and every join in
    flight is cancelled with it.

    Args:
        subscribe: The bus's ``subscribe(topic, node_identity=, on_message=)``,
            which starts the consumer and returns once its group has joined.
        subscriptions: The topics and the callback each one delivers to.
        node_identity: The identity every consumer group is derived from.
        purpose: What the consumers are for, named in the failure log line.

    Returns:
        The number of topics whose subscribe returned.
    """

    async def _join(topic: str, on_message: TopicCallback) -> bool:
        try:
            await subscribe(topic, node_identity=node_identity, on_message=on_message)
        except Exception:
            logger.exception("Could not subscribe to %s for %s", topic, purpose)
            return False
        return True

    joined = await asyncio.gather(*(_join(t, cb) for t, cb in subscriptions))
    return sum(1 for ok in joined if ok)


__all__ = ["TopicCallback", "subscribe_topics_concurrently"]
