# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane runner's side of a ``lane-work`` lease (OMN-21428).

A lane runner acquires one ``lane-work`` lease on ``lane:<lane>`` in place of
writing a CLAIM row, renews it every ``ttl_s/3``, and releases it at its
TERMINAL with its outcome. It writes no ledger row: ``node_projection_lease``
writes the CLAIM from the grant and the TERMINAL from the release or expiry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.handler_lease_ledger import LANE_WORK_KIND
from omnibase_infra.handlers.lease.handler_lease_orchestrator import (
    HandlerLeaseRelease,
)
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.models.lease import (
    ModelLaneWorkHolder,
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)


def renew_interval_s(ttl_s: int) -> float:
    """A renewal every third of the TTL: two may be lost before the grant expires."""
    return ttl_s / 3


class LeaseCommands(ABC):
    """The lease orchestrator's three commands, answered with its reply."""

    @abstractmethod
    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply: ...

    @abstractmethod
    async def renew(self, request: ModelLeaseRenewRequest) -> ModelLeaseReply: ...

    @abstractmethod
    async def release(self, request: ModelLeaseReleaseRequest) -> ModelLeaseReply: ...


class StoreLeaseCommands(LeaseCommands):
    """The commands answered in process by the etcd lease store."""

    def __init__(self, store: HandlerLeaseStore) -> None:
        self._store = store
        self._release = HandlerLeaseRelease(store)

    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        return await self._store.acquire(request)

    async def renew(self, request: ModelLeaseRenewRequest) -> ModelLeaseReply:
        return await self._store.renew(request.kind, request.resource, request.lease_id)

    async def release(self, request: ModelLeaseReleaseRequest) -> ModelLeaseReply:
        return await self._release.handle(request)


class HandlerLaneWorkLease:
    """Acquire, renew and release one lane's ``lane-work`` lease."""

    def __init__(
        self,
        commands: LeaseCommands,
        holder: ModelLaneWorkHolder,
        *,
        ttl_s: int,
    ) -> None:
        self._commands = commands
        self._holder = holder
        self._ttl_s = ttl_s
        self._lease_id: UUID | None = None
        # One request_id per acquisition: a retry after a lost reply reads back
        # the grant the store committed instead of being refused by it.
        self._request_id = uuid4()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @property
    def renew_interval_s(self) -> float:
        return renew_interval_s(self._ttl_s)

    async def acquire(self) -> ModelLeaseReply:
        reply = await self._commands.acquire(
            ModelLeaseAcquireRequest(
                kind=LANE_WORK_KIND,
                resource=self._holder.resource,
                holder=self._holder.holder(),
                request_id=self._request_id,
                ttl_s=self._ttl_s,
            )
        )
        if reply.outcome == "lease-granted":
            self._lease_id = reply.lease_id
        return reply

    def _granted(self) -> UUID:
        if self._lease_id is None:
            raise RuntimeError(f"lane {self._holder.lane} holds no lane-work lease")
        return self._lease_id

    async def renew(self) -> ModelLeaseReply:
        return await self._commands.renew(
            ModelLeaseRenewRequest(
                kind=LANE_WORK_KIND,
                resource=self._holder.resource,
                lease_id=self._granted(),
            )
        )

    async def release(self, outcome: str) -> ModelLeaseReply:
        reply = await self._commands.release(
            ModelLeaseReleaseRequest(
                kind=LANE_WORK_KIND,
                resource=self._holder.resource,
                lease_id=self._granted(),
                outcome=outcome,
            )
        )
        self._lease_id = None
        self._request_id = uuid4()
        return reply


__all__ = [
    "HandlerLaneWorkLease",
    "LeaseCommands",
    "StoreLeaseCommands",
    "renew_interval_s",
]
