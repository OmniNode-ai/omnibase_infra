# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``check_fence`` port (the ticket's runtime/placement/fence.py; the canonical-file-shape ratchet keeps new code under handlers/): does a fence still name its resource's live grant (OMN-21424)?

An effect target calls it before it acts on a fenced effect. It answers ``ok``
only when a live slot of the resource carries exactly that fence; never ``ok``
for a fence below a live grant's (``stale``), a fence with no live grant at or
above it (``expired``: the resource is free, or the fence's own slot key is gone
while only older grants remain) or an unreachable etcd (``unknown``). The prefix
read is linearizable, so a fence missing from it names no live grant.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease.handler_lease_store import (
    LEASE_PREFIX,
    resource_prefix,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.models.lease import EnumLeaseFenceStatus


async def check_fence(
    adapter: HandlerPlacementEtcd,
    *,
    kind: str,
    resource: str,
    fence: int,
    prefix: str = LEASE_PREFIX,
) -> EnumLeaseFenceStatus:
    try:
        entries = await adapter.get_prefix_entries(
            resource_prefix(kind, resource, prefix=prefix)
        )
    except InfraConnectionError:
        return EnumLeaseFenceStatus.UNKNOWN
    live = {create_revision for _, _, create_revision, _ in entries}
    if fence in live:
        return EnumLeaseFenceStatus.OK
    if live and fence < max(live):
        return EnumLeaseFenceStatus.STALE
    return EnumLeaseFenceStatus.EXPIRED


class HandlerLeaseFence:
    """The ``check_fence`` port bound to one etcd adapter."""

    def __init__(
        self, adapter: HandlerPlacementEtcd, *, prefix: str = LEASE_PREFIX
    ) -> None:
        self._adapter = adapter
        self._prefix = prefix

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def check(
        self, *, kind: str, resource: str, fence: int
    ) -> EnumLeaseFenceStatus:
        return await check_fence(
            self._adapter,
            kind=kind,
            resource=resource,
            fence=fence,
            prefix=self._prefix,
        )


__all__ = ["HandlerLeaseFence", "check_fence"]
