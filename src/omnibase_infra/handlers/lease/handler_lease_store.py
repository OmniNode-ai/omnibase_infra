# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic lease store over etcd (OMN-21424).

Implements the generic lease model of OMN-21418
(``formal/generic_lease/GenericLease.tla``). A resource of a kind with
capacity ``n`` has ``n`` slot keys. A grant is one etcd transaction that
creates a free slot key and the request key together, both bound to a fresh
etcd lease of the kind's TTL; the fence is that transaction's revision, taken
from the store's one revision counter, so every grant of a resource has a fence
above every earlier one (``FenceMonotonicPerResource``), also when another
runtime answers. A renewal is a keepalive of the same etcd lease and keeps the
fence. An expired lease's keys are deleted by etcd, which frees the slot
(``ExpiredLeaseFreesResource``). The request key makes an acquire idempotent:
the same ``request_id`` reads back the grant it made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
    b64,
    unb64,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReply,
)

LEASE_PREFIX = "/onex/lease/"
_REQUEST_PREFIX = "/onex/lease-request/"


def resource_prefix(kind: str, resource: str, *, prefix: str = LEASE_PREFIX) -> str:
    """The key prefix holding every slot of one resource."""
    return f"{prefix}{kind}/{resource}/"


class HandlerLeaseStore:
    """Acquire, renew, release and show leases; every grant is one etcd transaction."""

    def __init__(
        self,
        adapter: HandlerPlacementEtcd,
        kinds: Sequence[ModelLeaseKindPolicy],
        *,
        prefix: str = LEASE_PREFIX,
    ) -> None:
        self._adapter = adapter
        self._kinds = {policy.kind: policy for policy in kinds}
        self._prefix = prefix

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _policy(self, kind: str) -> ModelLeaseKindPolicy:
        policy = self._kinds.get(kind)
        if policy is None:
            raise ValueError(f"lease kind {kind!r} is not declared")
        return policy

    def _request_key(self, request: ModelLeaseAcquireRequest) -> str:
        return (
            f"{_REQUEST_PREFIX}{request.kind}/{request.resource}/{request.request_id}"
        )

    async def granted(
        self, request: ModelLeaseAcquireRequest
    ) -> ModelLeaseReply | None:
        """Replay the request's live grant, if it has one."""
        replay = await self._adapter.get(self._request_key(request))
        if replay is not None:
            return self._replayed(request, replay[0], replay[1])
        return None

    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        policy = self._policy(request.kind)
        request_key = self._request_key(request)
        replay = await self.granted(request)
        if replay is not None:
            return replay
        etcd_lease, ttl = await self._adapter.lease_grant(request.ttl_s or policy.ttl_s)
        lease_id = uuid4()
        value = json.dumps(
            {
                "lease_id": str(lease_id),
                "holder": request.holder,
                "request_id": str(request.request_id),
            },
            sort_keys=True,
        )
        slots = resource_prefix(request.kind, request.resource, prefix=self._prefix)
        for slot in range(policy.capacity_of(request.resource)):
            slot_key = f"{slots}{slot}"
            succeeded, revision, responses = await self._adapter.txn(
                compare=[
                    _absent(slot_key),
                    _absent(request_key),
                ],
                success=[
                    _put(slot_key, value, etcd_lease),
                    _put(request_key, value, etcd_lease),
                ],
                failure=[{"requestRange": {"key": b64(request_key)}}],
            )
            if succeeded:
                return ModelLeaseReply(
                    outcome="lease-granted",
                    kind=request.kind,
                    resource=request.resource,
                    holder=request.holder,
                    lease_id=lease_id,
                    fence=revision,
                    request_id=request.request_id,
                    ttl_s=ttl,
                )
            existing = _range_entry(responses)
            if existing is not None:
                await self._adapter.lease_revoke(etcd_lease)
                return self._replayed(request, existing[0], existing[1])
        await self._adapter.lease_revoke(etcd_lease)
        return ModelLeaseReply(
            outcome="lease-refused",
            kind=request.kind,
            resource=request.resource,
            holder=request.holder,
            request_id=request.request_id,
            reason=f"all {policy.capacity_of(request.resource)} slot(s) held",
        )

    async def free(self, kind: str, resource: str) -> int:
        """The resource's capacity minus its live grants."""
        policy = self._policy(kind)
        entries = await self._adapter.get_prefix_entries(
            resource_prefix(kind, resource, prefix=self._prefix)
        )
        return policy.capacity_of(resource) - len(entries)

    def _replayed(
        self, request: ModelLeaseAcquireRequest, value: str, fence: int
    ) -> ModelLeaseReply:
        held = json.loads(value)
        return ModelLeaseReply(
            outcome="lease-granted",
            kind=request.kind,
            resource=request.resource,
            holder=str(held["holder"]),
            lease_id=UUID(str(held["lease_id"])),
            fence=fence,
            request_id=request.request_id,
            reason="replayed request_id",
        )

    async def _slot_of(
        self, kind: str, resource: str, lease_id: UUID
    ) -> tuple[str, int, int] | None:
        """The slot key, fence and etcd lease of ``lease_id`` on the resource."""
        entries = await self._adapter.get_prefix_entries(
            resource_prefix(kind, resource, prefix=self._prefix)
        )
        for key, value, create_revision, etcd_lease in entries:
            if json.loads(value).get("lease_id") == str(lease_id):
                return key, create_revision, etcd_lease
        return None

    async def renew(self, kind: str, resource: str, lease_id: UUID) -> ModelLeaseReply:
        """A keepalive of the grant's own etcd lease; the fence does not change."""
        self._policy(kind)
        slot = await self._slot_of(kind, resource, lease_id)
        if slot is None:
            return ModelLeaseReply(
                outcome="lease-expired", kind=kind, resource=resource, lease_id=lease_id
            )
        ttl = await self._adapter.lease_keepalive(slot[2])
        if ttl <= 0:
            return ModelLeaseReply(
                outcome="lease-expired", kind=kind, resource=resource, lease_id=lease_id
            )
        return ModelLeaseReply(
            outcome="lease-renewed",
            kind=kind,
            resource=resource,
            lease_id=lease_id,
            fence=slot[1],
            ttl_s=ttl,
        )

    async def release(
        self, kind: str, resource: str, lease_id: UUID
    ) -> ModelLeaseReply:
        """Revoke the grant's etcd lease; a release of a lost grant changes nothing."""
        self._policy(kind)
        slot = await self._slot_of(kind, resource, lease_id)
        if slot is None:
            return ModelLeaseReply(
                outcome="lease-expired", kind=kind, resource=resource, lease_id=lease_id
            )
        await self._adapter.lease_revoke(slot[2])
        return ModelLeaseReply(
            outcome="lease-released",
            kind=kind,
            resource=resource,
            lease_id=lease_id,
            fence=slot[1],
        )

    async def show(self, kind: str, resource: str) -> list[ModelLeaseReply]:
        """Every live grant of the resource, lowest fence first."""
        self._policy(kind)
        entries = await self._adapter.get_prefix_entries(
            resource_prefix(kind, resource, prefix=self._prefix)
        )
        held = [
            ModelLeaseReply(
                outcome="lease-granted",
                kind=kind,
                resource=resource,
                holder=str(json.loads(value)["holder"]),
                lease_id=UUID(str(json.loads(value)["lease_id"])),
                fence=create_revision,
            )
            for _, value, create_revision, _ in entries
        ]
        return sorted(held, key=lambda reply: reply.fence)


def _absent(key: str) -> dict[str, object]:
    return {
        "key": b64(key),
        "target": "CREATE",
        "result": "EQUAL",
        "createRevision": "0",
    }


def _put(key: str, value: str, etcd_lease: int) -> dict[str, object]:
    return {
        "requestPut": {"key": b64(key), "value": b64(value), "lease": str(etcd_lease)}
    }


def _range_entry(responses: list[dict[str, object]]) -> tuple[str, int] | None:
    """The ``(value, create_revision)`` of a failure-branch range, if it found a key."""
    for response in responses:
        ranged = response.get("response_range")
        if not isinstance(ranged, dict):
            continue
        kvs = ranged.get("kvs")
        if isinstance(kvs, list) and kvs and isinstance(kvs[0], dict):
            return unb64(kvs[0].get("value")), int(
                str(kvs[0].get("create_revision", "0"))
            )
    return None


__all__ = ["LEASE_PREFIX", "HandlerLeaseStore", "resource_prefix"]
