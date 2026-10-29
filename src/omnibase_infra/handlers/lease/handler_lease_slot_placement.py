# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Places a unit on a host by a slot lease (OMN-21427).

Keeps no lock file or marker; a lane with no slot lease is not placing.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseReply


class HandlerLeaseSlotPlacement:
    """Place on the first host with an available slot, replaying existing grants."""

    def __init__(self, store: HandlerLeaseStore, *, kind: str = "slot") -> None:
        self._store = store
        self._kind = kind

    async def place(
        self, hosts: Sequence[str], holder: str, request_id: UUID
    ) -> ModelLeaseReply:
        """Replay across all hosts before attempting a grant in host order.

        A refusal names every host it tried, so it is never attributed to one.
        """
        if not hosts:
            raise ValueError("place() needs at least one host")
        for host in hosts:
            replay = await self._store.granted(
                ModelLeaseAcquireRequest(
                    kind=self._kind,
                    resource=f"slot:{host}",
                    holder=holder,
                    request_id=request_id,
                )
            )
            if replay is not None:
                return replay
        for host in hosts:
            reply = await self._store.acquire(
                ModelLeaseAcquireRequest(
                    kind=self._kind,
                    resource=f"slot:{host}",
                    holder=holder,
                    request_id=request_id,
                )
            )
            if reply.outcome == "lease-granted":
                return reply
        return ModelLeaseReply(
            outcome="lease-refused",
            kind=self._kind,
            resource=f"slot:{','.join(hosts)}",
            holder=holder,
            request_id=request_id,
            reason=f"no free slot on {','.join(hosts)}",
        )

    async def renew(self, host: str, lease_id: UUID) -> ModelLeaseReply:
        """Renew the host's slot grant."""
        return await self._store.renew(self._kind, f"slot:{host}", lease_id)

    async def release(self, host: str, lease_id: UUID) -> ModelLeaseReply:
        """Release the host's slot grant."""
        return await self._store.release(self._kind, f"slot:{host}", lease_id)

    async def free_slots(self, host: str) -> int:
        """Return the host's available slot count."""
        return await self._store.free(self._kind, f"slot:{host}")

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT


__all__ = ["HandlerLeaseSlotPlacement"]
