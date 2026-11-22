# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wake a dormant consumer before a message reaches its topic (OMN-26396).

A contract the lane left dormant has no subscription and no provisioned topic. A
message another node publishes to its topic (an inner hop of a workflow, not a
first request through the ingress) would be lost, or wait for ever on a topic
nobody consumes. The wake belongs in the runtime's own publish path, never in a
handler (a handler holds no bus): ``install_publish_gate`` puts a coroutine in
front of the runtime bus's ``publish``, awaited with the topic before the message
is sent. Every publish of that bus passes through it: ``publish_envelope`` (the
result appliers' path) ends in ``publish``, and so does a raw publish.

The gate is installed once, on the bus instance, by the always-wired ingress
node. It adds nothing to the bus class and nothing for a topic that has no
dormant consumer: the front door answers those at once.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable

__all__: list[str] = ["install_publish_gate"]

_GATE_ATTRIBUTE = "_lazy_start_publish_gate"


def install_publish_gate(bus: object, gate: Callable[[str], Awaitable[None]]) -> bool:
    """Put ``gate`` in front of ``bus.publish``; False when the bus has no instance ``publish``.

    Installing twice on one bus keeps the first gate.
    """
    publish = getattr(bus, "publish", None)
    if not callable(publish) or not hasattr(bus, "__dict__"):
        return False
    if getattr(publish, _GATE_ATTRIBUTE, None) is not None:
        return True

    @functools.wraps(publish)
    async def gated(*args: object, **kwargs: object) -> object:
        topic = args[0] if args else kwargs["topic"]
        await gate(str(topic))
        return await publish(*args, **kwargs)

    setattr(gated, _GATE_ATTRIBUTE, gate)
    vars(bus)["publish"] = gated
    return True
