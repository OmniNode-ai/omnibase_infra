# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Cross-boundary seam tests for ServiceTenantIngress (OMN-14349, OMN-14208 Path A).

These tests drive the real omnibase_core.ModelEventEnvelope the whole ONEX
runtime dispatch pipeline actually uses (the same class OMN-14345/#2252's
fix targets) -- not a bespoke envelope model. That is the point of this
service: unlike node_bus_forwarder_effect's ModelGatewayEnvelope bridge
(built for a different, opposite-direction topology), this service's
input and output wire shape is identical to every other producer/consumer
on the canonical topic, so a message it republishes is directly consumable
downstream with no shape translation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_tenant_ingress_effect.models.model_tenant_ingress_config import (
    ModelTenantIngressConfig,
)
from omnibase_infra.nodes.node_tenant_ingress_effect.services.service_tenant_ingress import (
    ServiceTenantIngress,
)

pytestmark = pytest.mark.asyncio

CANONICAL_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


@dataclass(frozen=True)
class _Message:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None = None


class _MockBus:
    """Structural stand-in for the real event bus (subscribe/publish only)."""

    def __init__(self) -> None:
        self.subscriptions: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.subscription_groups: dict[str, str] = {}
        self.published: list[_Message] = []

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[Any], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **_kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        assert on_message is not None
        assert group_id is not None
        self.subscriptions[topic] = on_message
        self.subscription_groups[topic] = group_id

        async def _unsubscribe() -> None:
            self.subscriptions.pop(topic, None)
            self.subscription_groups.pop(topic, None)

        return _unsubscribe

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(_Message(topic, key, value, headers))

    async def emit(self, topic: str, envelope: ModelEventEnvelope[dict]) -> None:
        await self.subscriptions[topic](
            _Message(
                topic=topic,
                key=b"key-1",
                value=envelope.model_dump_json().encode("utf-8"),
                headers={"trace": "preserved"},
            )
        )


def _config(**overrides: object) -> ModelTenantIngressConfig:
    values: dict[str, object] = {
        "tenants": ("acme", "beta"),
        "canonical_topic": CANONICAL_TOPIC,
    }
    values.update(overrides)
    return ModelTenantIngressConfig(**values)  # type: ignore[arg-type]


def _inbound_envelope(**overrides: object) -> ModelEventEnvelope[dict]:
    values: dict[str, object] = {
        "payload": {"prompt": "hi", "task_type": "review", "source": "claude-code"},
        "correlation_id": uuid4(),
        "event_type": "omnimarket.delegate-skill",
    }
    values.update(overrides)
    return ModelEventEnvelope[dict](**values)  # type: ignore[arg-type]


async def test_start_subscribes_one_topic_per_configured_tenant() -> None:
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)

    await service.start()

    assert set(bus.subscriptions) == {
        f"tenant-acme.{CANONICAL_TOPIC}",
        f"tenant-beta.{CANONICAL_TOPIC}",
    }
    assert bus.subscription_groups[f"tenant-acme.{CANONICAL_TOPIC}"].endswith("acme")
    assert bus.subscription_groups[f"tenant-beta.{CANONICAL_TOPIC}"].endswith("beta")


async def test_inbound_message_gets_verified_tenant_id_not_forged_or_missing() -> None:
    """The stamp must come from the SUBSCRIBED topic, never the payload.

    A payload-supplied tenant_id (forged or absent) must never survive --
    the topic this message structurally arrived on (which only a
    tenant-scoped credential could publish to, once the broker ACL from
    OMN-12911/OMN-14110 is live) is the sole source of truth.
    """
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)
    await service.start()

    forged_payload = {"prompt": "steal data", "tenant_id": "evil-forged-tenant"}
    await bus.emit(
        f"tenant-acme.{CANONICAL_TOPIC}",
        _inbound_envelope(payload=forged_payload),
    )

    assert len(bus.published) == 1
    published = bus.published[0]
    assert published.topic == CANONICAL_TOPIC
    # The REAL consumer-side parse -- the exact model every downstream node uses.
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published.value)
    assert consumer_view.payload["tenant_id"] == "acme"
    assert consumer_view.payload["tenant_id"] != "evil-forged-tenant"
    assert consumer_view.payload["prompt"] == "steal data"


async def test_inbound_message_missing_tenant_id_gets_one_stamped() -> None:
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)
    await service.start()

    await bus.emit(
        f"tenant-beta.{CANONICAL_TOPIC}",
        _inbound_envelope(payload={"prompt": "hi"}),
    )

    published = bus.published[0]
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published.value)
    assert consumer_view.payload["tenant_id"] == "beta"


async def test_republished_envelope_preserves_correlation_and_key() -> None:
    """The stamp must not disturb anything else on the envelope or transport."""
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)
    await service.start()

    cid = uuid4()
    await bus.emit(
        f"tenant-acme.{CANONICAL_TOPIC}",
        _inbound_envelope(correlation_id=cid, payload={"prompt": "hi"}),
    )

    published = bus.published[0]
    assert published.key == b"key-1"
    assert published.headers == {"trace": "preserved"}
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published.value)
    assert consumer_view.correlation_id == cid


async def test_message_on_unconfigured_tenant_topic_is_never_received() -> None:
    """Fail-closed BY CONSTRUCTION: only configured tenants' topics are subscribed.

    There is no runtime branch to get wrong here -- an unconfigured tenant's
    topic was never subscribed to, so a message on it structurally cannot
    reach this service at all.
    """
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)
    await service.start()

    assert (
        "tenant-widgets.onex.cmd.omnimarket.delegate-skill.v1" not in bus.subscriptions
    )


async def test_stop_unsubscribes_all_configured_tenants() -> None:
    bus = _MockBus()
    service = ServiceTenantIngress(config=_config(), bus=bus)
    await service.start()

    await service.stop()

    assert bus.subscriptions == {}
