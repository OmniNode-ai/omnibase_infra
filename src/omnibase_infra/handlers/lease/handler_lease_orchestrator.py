# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease orchestrator handlers: acquire, renew, release over the etcd store (OMN-21424).

Each takes its typed request and returns the reply as its outcome's own class; the
runtime publishes it on the topic the contract's ``published_events`` names for that
class. No handler touches the bus.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import UUID

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.lease_store import (
    LeaseStore,
    lease_store_from_config,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseExpired,
    ModelLeaseGranted,
    ModelLeaseRefused,
    ModelLeaseReleased,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewed,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "nodes"
    / "node_lease_orchestrator"
    / "contract.yaml"
)


@lru_cache(maxsize=1)
def _contract() -> dict[str, object]:
    """The orchestrator contract: the source of its store configuration."""
    loaded = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Contract {_CONTRACT} must contain a mapping")
    return loaded


async def _lease_store_from_contract() -> LeaseStore:
    """Build the store from the contract's overlay-resolved descriptor values."""
    descriptor = _contract().get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(f"Contract {_CONTRACT} must declare a descriptor")
    resolved = {
        key: expand_contract_env_refs(str(descriptor.get(key, "")))
        for key in ("lease_etcd_endpoint", "lease_kinds")
    }
    for key, value in resolved.items():
        if not value.strip():
            raise ValueError(
                f"Contract {_CONTRACT} descriptor.{key} resolved empty; "
                "set the environment variable it references"
            )
    return await lease_store_from_config(
        resolved["lease_etcd_endpoint"], resolved["lease_kinds"]
    )


_OUTCOME_MODELS: dict[str, type[ModelLeaseReply]] = {
    "lease-granted": ModelLeaseGranted,
    "lease-refused": ModelLeaseRefused,
    "lease-renewed": ModelLeaseRenewed,
    "lease-released": ModelLeaseReleased,
    "lease-expired": ModelLeaseExpired,
}


def lease_reply_event(reply: ModelLeaseReply) -> ModelLeaseReply:
    """The reply as its outcome's own class, the one the contract routes to a topic."""
    return _OUTCOME_MODELS[reply.outcome].model_validate(reply, from_attributes=True)


def _refused_for_lease(
    kind: str, resource: str, lease_id: UUID, error: ValueError
) -> ModelLeaseReply:
    """A renew or release the store cannot take (an undeclared kind) still gets its answer.

    The caller waits on the reply topics for this ``lease_id``; an unanswered
    command would leave it waiting out its whole budget.
    """
    return ModelLeaseReply(
        outcome="lease-refused",
        kind=kind,
        resource=resource,
        lease_id=lease_id,
        reason=str(error),
    )


class HandlerLeaseOrchestratorBase:
    """Shared construction and classification of the three lease handlers."""

    def __init__(self, store: LeaseStore | None = None) -> None:
        self._store = store

    async def _lease_store(self) -> LeaseStore:
        if self._store is None:
            self._store = await _lease_store_from_contract()
        return self._store

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT


class HandlerLeaseAcquire(HandlerLeaseOrchestratorBase):
    """One etcd transaction: ``lease-granted`` with a fresh fence, or ``lease-refused``."""

    async def handle(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        store = await self._lease_store()
        try:
            reply = await store.acquire(request)
        except ValueError as exc:
            reply = ModelLeaseReply(
                outcome="lease-refused",
                kind=request.kind,
                resource=request.resource,
                holder=request.holder,
                request_id=request.request_id,
                reason=str(exc),
            )
        return lease_reply_event(reply)


class HandlerLeaseRenew(HandlerLeaseOrchestratorBase):
    """``lease-renewed`` at the same fence, ``lease-expired``, or ``lease-refused``."""

    async def handle(self, request: ModelLeaseRenewRequest) -> ModelLeaseReply:
        store = await self._lease_store()
        try:
            reply = await store.renew(request.kind, request.resource, request.lease_id)
        except ValueError as exc:
            reply = _refused_for_lease(
                request.kind, request.resource, request.lease_id, exc
            )
        return lease_reply_event(reply)


class HandlerLeaseRelease(HandlerLeaseOrchestratorBase):
    """``lease-released``, ``lease-expired`` when the grant was already gone, or ``lease-refused``."""

    async def handle(self, request: ModelLeaseReleaseRequest) -> ModelLeaseReply:
        store = await self._lease_store()
        try:
            reply = await store.release(
                request.kind, request.resource, request.lease_id
            )
        except ValueError as exc:
            reply = _refused_for_lease(
                request.kind, request.resource, request.lease_id, exc
            )
        return lease_reply_event(reply)


__all__ = ["HandlerLeaseAcquire", "HandlerLeaseRelease", "HandlerLeaseRenew"]
