# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tenant-ingress effect service (OMN-14349, OMN-14208 Path A).

Subscribes to a tenant-prefixed variant of a canonical topic for every
configured tenant, stamps the config-scoped verified tenant_id into the
payload -- overwriting any client-supplied value -- and republishes onto
the bare canonical topic every other consumer already subscribes to.

Distinct from node_bus_forwarder_effect's ServiceGatewayForwarder: that
service bridges TWO separate brokers (a customer-owned local runtime bus
and the hosted cloud bus) for the opposite traffic direction (cloud
orchestrator delegating work OUT to a customer's own local compute,
OMN-12908). This service has exactly one broker -- there is no bridge --
it strips a tenant wire prefix and republishes on the same broker.

The verified-identity guarantee this service provides is only as strong as
the Kafka broker ACL that restricts a tenant's credential to publishing on
its own tenant-<slug>.* prefix (OMN-12911 / OMN-14110). That ACL
provisioning is Daniyal's AWS/MSK domain, tracked separately -- this
service is the correct, ready CONSUMER side of the mechanism regardless of
when the ACL side lands, but is not itself a security boundary until the
ACL is live and verified with a real cross-prefix-publish-rejected
readback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_tenant_ingress_effect.handlers.handler_stamp_tenant_id import (
    HandlerStampTenantId,
)
from omnibase_infra.nodes.node_tenant_ingress_effect.models.model_tenant_ingress_config import (
    ModelTenantIngressConfig,
)


class ProtocolIngressBus(Protocol):
    """Structural subset shared by EventBusKafka, EventBusInmemory, and tests."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        """Publish bytes to a topic."""

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[object], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to a topic and return an async unsubscribe callback."""


class ServiceTenantIngress:
    """Own tenant-prefixed subscriptions; stamp verified tenant_id; republish."""

    def __init__(
        self,
        *,
        config: ModelTenantIngressConfig,
        bus: ProtocolIngressBus,
    ) -> None:
        self._config = config
        self._bus = bus
        self._stamp_handler = HandlerStampTenantId()
        self._unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._started = False

    async def start(self) -> None:
        """Subscribe to every configured tenant's wire-prefixed topic."""
        if self._started:
            return

        for slug in self._config.tenants:
            wire_topic = f"tenant-{slug}.{self._config.canonical_topic}"
            unsubscribe = await self._bus.subscribe(
                topic=wire_topic,
                group_id=self._group_id(slug),
                on_message=self._make_handler(slug),
            )
            self._unsubscribe_callbacks.append(unsubscribe)

        self._started = True

    async def stop(self) -> None:
        """Stop all active subscriptions."""
        callbacks = list(reversed(self._unsubscribe_callbacks))
        self._unsubscribe_callbacks.clear()
        self._started = False
        for unsubscribe in callbacks:
            await unsubscribe()

    def _make_handler(self, slug: str) -> Callable[[object], Awaitable[None]]:
        # Closes over ``slug`` at subscribe time -- the subscription itself IS
        # the tenant identity, matching the OMN-12908/12911 topic-per-tenant
        # ACL boundary. No per-message topic lookup, no shared mutable state.
        async def _on_message(message: object) -> None:
            value = _message_bytes(message)
            envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(value)
            # Config-bound identity always wins -- never a self-asserted
            # fallback. HandlerStampTenantId.stamp() overwrites, never
            # merges-if-absent.
            stamped_envelope = self._stamp_handler.stamp(envelope, slug)
            await self._bus.publish(
                self._config.canonical_topic,
                getattr(message, "key", None),
                stamped_envelope.model_dump_json().encode("utf-8"),
                getattr(message, "headers", None),
            )

        return _on_message

    def _group_id(self, slug: str) -> str:
        return f"tenant-ingress-{slug}"


def _message_bytes(message: object) -> bytes:
    value = getattr(message, "value", message)
    if isinstance(value, str):
        return value.encode("utf-8")
    if not isinstance(value, bytes):
        raise TypeError("tenant-ingress bus message value must be bytes or string")
    return value


__all__ = ["ProtocolIngressBus", "ServiceTenantIngress"]
