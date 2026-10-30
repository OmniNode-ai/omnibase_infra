# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The caller's side of the lease orchestrator: publish a command, read its answer (OMN-21424).

``onex lease acquire|renew|release`` is a client of ``node_lease_orchestrator``.
It publishes the command on that node's own command topic and reads the outcome
from the orchestrator's event topics; only the orchestrator's effect touches
etcd. The reply topics are read before the command is published, so a reply
that arrives at once is not missed (``subscribe_ungrouped`` returns only after
every partition has a position).

A reply is matched to its command by what the command carries. An acquire has a
unique ``request_id`` that the reply echoes. A renew or release names its
``lease_id`` and the reply echoes that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic as Topic,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)
from omnibase_infra.protocols.protocol_lease_bus import ProtocolLeaseBus

REPLY_TOPICS: tuple[str, ...] = (
    Topic.EVT_LEASE_GRANTED_V1.value,
    Topic.EVT_LEASE_REFUSED_V1.value,
    Topic.EVT_LEASE_RENEWED_V1.value,
    Topic.EVT_LEASE_RELEASED_V1.value,
    Topic.EVT_LEASE_EXPIRED_V1.value,
)


class LeaseReplyTimeoutError(TimeoutError):
    """The orchestrator did not answer the command within the caller's budget."""


def _addressing(request: BaseModel) -> tuple[str, str, str, UUID]:
    """The command topic, the kind and resource, and the id the reply echoes."""
    if isinstance(request, ModelLeaseAcquireRequest):
        topic, echoed = Topic.CMD_LEASE_ACQUIRE_REQUESTED_V1, request.request_id
    elif isinstance(request, ModelLeaseRenewRequest):
        topic, echoed = Topic.CMD_LEASE_RENEW_REQUESTED_V1, request.lease_id
    elif isinstance(request, ModelLeaseReleaseRequest):
        topic, echoed = Topic.CMD_LEASE_RELEASE_REQUESTED_V1, request.lease_id
    else:
        raise TypeError(f"{type(request).__name__} is not a lease command")
    return topic.value, request.kind, request.resource, echoed


def _reply_of(message: ModelEventMessage) -> ModelLeaseReply | None:
    """The lease reply a message carries, or ``None`` for anything else."""
    try:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if not isinstance(envelope.payload, dict):
            return None
        return ModelLeaseReply.model_validate(envelope.payload)
    except (ValidationError, ValueError):
        return None


async def request_lease(
    bus: ProtocolLeaseBus, request: BaseModel, *, timeout_s: float
) -> ModelLeaseReply:
    """Publish ``request`` to the orchestrator's command topic and return its answer.

    Raises:
        LeaseReplyTimeoutError: no matching reply arrived within ``timeout_s``.
    """
    topic, kind, resource, echoed = _addressing(request)
    answered: asyncio.Future[ModelLeaseReply] = (
        asyncio.get_running_loop().create_future()
    )

    async def on_message(message: ModelEventMessage) -> None:
        reply = _reply_of(message)
        if (
            reply is not None
            and (reply.kind, reply.resource) == (kind, resource)
            and echoed in (reply.request_id, reply.lease_id)
            and not answered.done()
        ):
            answered.set_result(reply)

    unsubscribers: list[Callable[[], Awaitable[None]]] = []
    try:
        for reply_topic in REPLY_TOPICS:
            unsubscribers.append(await bus.subscribe_ungrouped(reply_topic, on_message))
        envelope = ModelEventEnvelope[object](
            payload=request.model_dump(mode="json"),
            correlation_id=uuid4(),
            event_type=topic,
            tenant_id=None,
        )
        await bus.publish(
            topic, f"{kind}/{resource}".encode(), envelope.model_dump_json().encode()
        )
        try:
            return await asyncio.wait_for(answered, timeout_s)
        except TimeoutError as exc:
            raise LeaseReplyTimeoutError(
                f"no reply to {topic} within {timeout_s:g}s; is node_lease_orchestrator "
                "running on this bus?"
            ) from exc
    finally:
        for unsubscribe in unsubscribers:
            await unsubscribe()


__all__ = ["REPLY_TOPICS", "LeaseReplyTimeoutError", "request_lease"]
