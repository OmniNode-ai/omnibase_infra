# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease handlers: acquire, renew, release and reconcile over etcd (OMN-21424).

Each returns the reply the orchestrator publishes; no handler touches the bus.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseProjectionRow,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)


class HandlerLeaseOrchestratorBase:
    """Shared construction and classification of the lease handlers."""

    def __init__(self, store: HandlerLeaseStore) -> None:
        self._store = store

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT


class HandlerLeaseAcquire(HandlerLeaseOrchestratorBase):
    """One etcd transaction: ``lease-granted`` with a fresh fence, or ``lease-refused``."""

    async def handle(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        return await self._store.acquire(request)


class HandlerLeaseRenew(HandlerLeaseOrchestratorBase):
    """``lease-renewed`` at the same fence, or ``lease-expired``."""

    async def handle(self, request: ModelLeaseRenewRequest) -> ModelLeaseReply:
        return await self._store.renew(request.kind, request.resource, request.lease_id)


class HandlerLeaseRelease(HandlerLeaseOrchestratorBase):
    """``lease-released``, or ``lease-expired`` when the grant was already gone."""

    async def handle(self, request: ModelLeaseReleaseRequest) -> ModelLeaseReply:
        return await self._store.release(
            request.kind, request.resource, request.lease_id
        )


class HandlerLeaseReconcile(HandlerLeaseOrchestratorBase):
    """Read back a held projection row and publish its observed grant or expiry."""

    async def handle(self, request: ModelLeaseProjectionRow) -> ModelLeaseReply:
        return await self._store.reconcile(request)


__all__ = [
    "HandlerLeaseAcquire",
    "HandlerLeaseReconcile",
    "HandlerLeaseRelease",
    "HandlerLeaseRenew",
]
