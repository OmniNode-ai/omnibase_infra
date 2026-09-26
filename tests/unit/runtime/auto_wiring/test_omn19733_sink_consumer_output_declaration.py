# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19733: subscription flow rows carry contract output capability."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)

_TOPIC = "onex.cmd.omnimarket.sink-test-start.v1"
_OUTPUT_TOPIC = "onex.evt.omnimarket.sink-test-completed.v1"
_T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


class _Handler:
    async def handle(self, envelope: object) -> None:
        return None


def _contract(*, publish_topics: tuple[str, ...]) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_sink_test",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/node_sink_test/contract.yaml"),
        entry_point_name="node_sink_test",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_TOPIC,),
            publish_topics=publish_topics,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerSinkTest",
                        module="fake.module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _clean_flow_counters() -> Iterator[None]:
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("publish_topics", "expected"),
    [
        pytest.param((), False, id="sink"),
        pytest.param((_OUTPUT_TOPIC,), True, id="publisher"),
    ],
)
async def test_subscription_delta_declares_contract_bus_output(
    publish_topics: tuple[str, ...], expected: bool
) -> None:
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=_T0)

    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(publish_topics=publish_topics),)
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_Handler,
    ):
        await wire_from_manifest(
            manifest,
            MessageDispatchEngine(),
            event_bus=event_bus,
            environment="local",
        )

    window = counters.drain(node_id=carrier, now=_T0 + timedelta(seconds=30))
    assert window is not None
    assert len(window.consumer_deltas) == 1
    assert window.consumer_deltas[0].declares_output is expected
