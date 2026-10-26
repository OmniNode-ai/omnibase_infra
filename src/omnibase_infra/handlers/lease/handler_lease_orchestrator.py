# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease orchestrator handlers: acquire, renew, release over the etcd store (OMN-21424).

Each returns the reply the orchestrator publishes; no handler touches the bus.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import yaml

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.handler_lease_store import (
    HandlerLeaseStore,
    lease_store_from_config,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
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
    """The orchestrator contract: the source of its topics and store configuration."""
    loaded = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Contract {_CONTRACT} must contain a mapping")
    return loaded


async def _lease_store_from_contract() -> HandlerLeaseStore:
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


@lru_cache(maxsize=5)
def _reply_topic(outcome: str) -> str:
    """Resolve one outcome's publish topic from the orchestrator contract."""
    contract = _CONTRACT
    loaded = _contract()
    event_bus = loaded.get("event_bus")
    if not isinstance(event_bus, dict):
        raise ValueError(f"Contract {contract} must declare event_bus")
    topics = event_bus.get("publish_topics")
    if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
        raise ValueError(f"Contract {contract} must declare publish_topics")
    matches = [topic for topic in topics if topic.endswith(f"{outcome}.v1")]
    if len(matches) != 1:
        raise ValueError(
            f"Contract {contract} must publish exactly one topic for {outcome!r}"
        )
    return str(matches[0])


class HandlerLeaseOrchestratorBase:
    """Shared construction and classification of the three lease handlers."""

    def __init__(self, store: HandlerLeaseStore | None = None) -> None:
        self._store = store

    async def _lease_store(self) -> HandlerLeaseStore:
        if self._store is None:
            self._store = await _lease_store_from_contract()
        return self._store

    def _output(
        self, envelope: ModelEventEnvelope[object], reply: ModelLeaseReply
    ) -> ModelHandlerOutput[None]:
        correlation_id = envelope.correlation_id or uuid4()
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=type(self).__name__,
            events=(
                ModelEventEnvelope(
                    # The result applier unwraps model payloads before publishing.
                    payload=reply,
                    correlation_id=correlation_id,
                    event_type=_reply_topic(reply.outcome),
                    tenant_id=envelope.tenant_id,
                ),
            ),
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT


class HandlerLeaseAcquire(HandlerLeaseOrchestratorBase):
    """One etcd transaction: ``lease-granted`` with a fresh fence, or ``lease-refused``."""

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[None]:
        request = ModelLeaseAcquireRequest.model_validate(envelope.payload)
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
        return self._output(envelope, reply)


class HandlerLeaseRenew(HandlerLeaseOrchestratorBase):
    """``lease-renewed`` at the same fence, or ``lease-expired``."""

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[None]:
        request = ModelLeaseRenewRequest.model_validate(envelope.payload)
        reply = await (await self._lease_store()).renew(
            request.kind, request.resource, request.lease_id
        )
        return self._output(envelope, reply)


class HandlerLeaseRelease(HandlerLeaseOrchestratorBase):
    """``lease-released``, or ``lease-expired`` when the grant was already gone."""

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[None]:
        request = ModelLeaseReleaseRequest.model_validate(envelope.payload)
        reply = await (await self._lease_store()).release(
            request.kind, request.resource, request.lease_id
        )
        return self._output(envelope, reply)


__all__ = ["HandlerLeaseAcquire", "HandlerLeaseRelease", "HandlerLeaseRenew"]
