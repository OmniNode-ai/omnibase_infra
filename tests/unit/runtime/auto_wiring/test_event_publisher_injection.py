# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for auto-wired handler event_publisher injection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)


class RecordingEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        self.published.append((topic, key, value))


class HandlerGenerationConsumerShape:
    """Fake with the constructor shape that regressed in OMN-12851."""

    def __init__(
        self,
        event_publisher: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self._event_publisher = event_publisher

    async def handle(self, envelope: object) -> None:
        if self._event_publisher is None:
            raise AssertionError("event_publisher was not injected")
        self._event_publisher("onex.cmd.omnimarket.node-deploy.v1", b'{"ok":true}')
        self._event_publisher("onex.evt.platform.node-registration.v1", b'{"ok":true}')


def _make_generation_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_generation_consumer",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_generation_consumer",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.node-generation-requested.v1",),
            publish_topics=(
                "onex.evt.omnimarket.node-generation-completed.v1",
                "onex.cmd.omnimarket.node-deploy.v1",
                "onex.evt.platform.node-registration.v1",
            ),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerGenerationConsumer",
                        module="omnimarket.nodes.node_generation_consumer.handlers.handler_generation_consumer",
                    ),
                ),
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_wired_optional_event_publisher_emits_side_topics() -> None:
    contract = _make_generation_contract()
    event_bus = RecordingEventBus()
    resolver = ServiceHandlerResolver()
    ownership_query = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerGenerationConsumerShape,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=contract.handler_routing.handlers[0],  # type: ignore[union-attr]
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership_query,
            event_bus=event_bus,
            container=None,
        )

    assert (
        prepared.resolution_outcome
        is EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY
    )

    await prepared.dispatcher(
        ModelEventEnvelope[dict[str, str]](
            payload={"prompt": "generate a node"},
            event_type="omnimarket.node-generation-requested",
        )
    )
    await asyncio.sleep(0)

    assert event_bus.published == [
        ("onex.cmd.omnimarket.node-deploy.v1", None, b'{"ok":true}'),
        ("onex.evt.platform.node-registration.v1", None, b'{"ok":true}'),
    ]


@pytest.mark.unit
def test_event_publisher_handler_fails_fast_without_runtime_event_bus() -> None:
    contract = _make_generation_contract()
    resolver = ServiceHandlerResolver()
    ownership_query = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerGenerationConsumerShape,
    ):
        with pytest.raises(ModelOnexError, match="event_publisher"):
            _prepare_handler_wiring(
                contract=contract,
                entry=contract.handler_routing.handlers[0],  # type: ignore[union-attr]
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership_query,
                event_bus=None,
                container=None,
            )
