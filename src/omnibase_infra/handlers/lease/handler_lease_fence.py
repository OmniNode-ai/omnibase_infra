# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``check_fence`` port of ``HandlerLeaseFence`` (the ticket's runtime/placement/fence.py; the canonical-file-shape ratchet keeps new code under handlers/): does a fence still name its resource's live grant (OMN-21424)?

An effect target calls it before it acts on a fenced effect. It answers ``ok``
only when a live slot of the resource carries exactly that fence; never ``ok``
for a lower fence (``stale``), an absent key or a fence above every live grant
(``expired``: revisions only rise, so such a grant was issued and is gone) or
an unreachable etcd (``unknown``). ``unknown`` means only that etcd was not read.
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


class HandlerLeaseFence:
    """Answers whether a fence still names its resource's live grant."""

    def __init__(
        self, adapter: HandlerPlacementEtcd, *, prefix: str = LEASE_PREFIX
    ) -> None:
        self._adapter = adapter
        self._prefix = prefix

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def check_fence(
        self, *, kind: str, resource: str, fence: int
    ) -> EnumLeaseFenceStatus:
        try:
            entries = await self._adapter.get_prefix_entries(
                resource_prefix(kind, resource, prefix=self._prefix)
            )
        except InfraConnectionError:
            return EnumLeaseFenceStatus.UNKNOWN
        live = {create_revision for _, _, create_revision, _ in entries}
        if not live:
            return EnumLeaseFenceStatus.EXPIRED
        if fence in live:
            return EnumLeaseFenceStatus.OK
        if fence < max(live):
            return EnumLeaseFenceStatus.STALE
        return EnumLeaseFenceStatus.EXPIRED


__all__ = ["HandlerLeaseFence"]
