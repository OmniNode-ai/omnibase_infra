# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for post-freeze dynamic contract registration (OMN-11247)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

import omnibase_infra.runtime.service_runtime_host_process as runtime_host_module
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

pytestmark = [pytest.mark.integration]


VALID_CONTRACT_YAML = """\
name: node_dynamic_test
handler_id: proto.dynamic_test
contract_version:
  major: 1
  minor: 0
  patch: 0
description: Dynamically registered test handler
input_model:
  name: JsonDict
  module: omnibase_infra.models.types
output_model:
  name: ModelHandlerOutput
  module: omnibase_core.models.dispatch.model_handler_output
descriptor:
  node_archetype: EFFECT_GENERIC
metadata:
  handler_class: omnibase_infra.handlers.handler_http.HandlerHttp
event_bus:
  subscribe_topics:
    - onex.evt.test.dynamic-reg.v1
  publish_topics: []
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerHttp
        module: omnibase_infra.handlers.handler_http
"""


@dataclass
class _Subscription:
    topic: str
    on_message: Any
    node_identity: Any
    purpose: Any


class _InMemoryEventBus:
    def __init__(self) -> None:
        self.subscriptions: list[_Subscription] = []
        self.unsubscribe_count = 0

    async def subscribe(self, **kwargs: Any) -> Any:
        self.subscriptions.append(_Subscription(**kwargs))

        async def unsubscribe() -> None:
            self.unsubscribe_count += 1

        return unsubscribe

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        message = MagicMock()
        message.value = json.dumps(payload).encode("utf-8")
        message.headers = MagicMock()
        message.headers.correlation_id = payload.get("correlation_id")

        matching = [
            subscription
            for subscription in self.subscriptions
            if subscription.topic == topic
        ]
        assert len(matching) == 1
        await matching[0].on_message(message)


class _CachingContractSource:
    environment = "test"

    def __init__(self) -> None:
        self.cached: dict[str, object] = {}
        self.registered: list[tuple[str, str, UUID]] = []
        self.deregistered: list[tuple[str, UUID]] = []

    def on_contract_registered(
        self,
        *,
        node_name: str,
        contract_yaml: str,
        correlation_id: UUID,
    ) -> bool:
        descriptor = object()
        self.cached[node_name] = descriptor
        self.registered.append((node_name, contract_yaml, correlation_id))
        return True

    def on_contract_deregistered(
        self,
        *,
        node_name: str,
        correlation_id: UUID,
    ) -> None:
        self.deregistered.append((node_name, correlation_id))
        self.cached.pop(node_name, None)

    def get_cached_descriptor(self, node_name: str) -> object | None:
        return self.cached.get(node_name)


class _RuntimeCreatedContractSource(_CachingContractSource):
    def __init__(self, *, environment: str, graceful_mode: bool) -> None:
        super().__init__()
        self.environment = environment
        self.graceful_mode = graceful_mode
        self.correlation_id = uuid4()


def _make_process(
    event_bus: _InMemoryEventBus,
    contract_source: _CachingContractSource | None,
) -> RuntimeHostProcess:
    process = RuntimeHostProcess.__new__(RuntimeHostProcess)
    process._event_bus = event_bus
    process._kafka_contract_source = contract_source
    process._node_identity = MagicMock()
    process._node_identity.service = "test-service"
    process._node_identity.node_name = "test-node"
    process._node_identity.version = "v1"
    process._dynamic_contract_unsubscribe = None
    process.materialized: list[tuple[str, object, UUID]] = []
    process._get_environment_from_config = MagicMock(return_value="test")

    async def materialize_handler_live(
        *,
        node_name: str,
        descriptor: object,
        correlation_id: UUID,
    ) -> bool:
        process.materialized.append((node_name, descriptor, correlation_id))
        return True

    process._materialize_handler_live = materialize_handler_live
    return process


@pytest.mark.asyncio
async def test_dynamic_registration_event_crosses_bus_to_materialize_handler() -> None:
    """Registration events travel through the bus subscription into materialization."""
    event_bus = _InMemoryEventBus()
    contract_source = _CachingContractSource()
    process = _make_process(event_bus, contract_source)
    correlation_id = uuid4()

    await process._start_dynamic_contract_listener()
    await event_bus.publish(
        SUFFIX_NODE_REGISTRATION,
        {
            "node_name": "node_dynamic_test",
            "contract_yaml": VALID_CONTRACT_YAML,
            "event_type": "registered",
            "correlation_id": str(correlation_id),
        },
    )

    assert [subscription.topic for subscription in event_bus.subscriptions] == [
        SUFFIX_NODE_REGISTRATION
    ]
    assert contract_source.registered == [
        ("node_dynamic_test", VALID_CONTRACT_YAML, correlation_id)
    ]
    descriptor = contract_source.get_cached_descriptor("node_dynamic_test")
    assert process.materialized == [("node_dynamic_test", descriptor, correlation_id)]


@pytest.mark.asyncio
async def test_dynamic_listener_initializes_contract_source_in_hybrid_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid runtime startup still wires the post-freeze dynamic listener."""
    event_bus = _InMemoryEventBus()
    process = _make_process(event_bus, None)

    monkeypatch.setattr(
        runtime_host_module,
        "KafkaContractSource",
        _RuntimeCreatedContractSource,
    )

    await process._start_dynamic_contract_listener()

    assert isinstance(process._kafka_contract_source, _RuntimeCreatedContractSource)
    assert process._kafka_contract_source.environment == "test"
    assert [subscription.topic for subscription in event_bus.subscriptions] == [
        SUFFIX_NODE_REGISTRATION
    ]


@pytest.mark.asyncio
async def test_dynamic_deregistration_event_crosses_bus_without_materialization() -> (
    None
):
    """Deregistration events clear the cache and do not materialize a handler."""
    event_bus = _InMemoryEventBus()
    contract_source = _CachingContractSource()
    process = _make_process(event_bus, contract_source)
    correlation_id = uuid4()

    contract_source.cached["node_dynamic_test"] = object()
    await process._start_dynamic_contract_listener()
    await event_bus.publish(
        SUFFIX_NODE_REGISTRATION,
        {
            "node_name": "node_dynamic_test",
            "event_type": "deregistered",
            "correlation_id": str(correlation_id),
        },
    )

    assert contract_source.deregistered == [("node_dynamic_test", correlation_id)]
    assert contract_source.get_cached_descriptor("node_dynamic_test") is None
    assert process.materialized == []
