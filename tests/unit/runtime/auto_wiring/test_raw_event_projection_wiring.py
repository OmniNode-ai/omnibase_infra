# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Raw Kafka event projection wiring tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.nodes.node_build_loop_projection_compute.handlers.handler_build_loop_projection import (
    HandlerBuildLoopProjection,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)

TERMINAL_TOPIC = "onex.evt.omnimarket.build-loop-orchestrator-completed.v1"
CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_build_loop_projection_compute/contract.yaml"
)


class CapturingApplier:
    """Capture dispatch results emitted by the raw projection callback."""

    def __init__(self) -> None:
        self.results = []
        self.correlation_ids: list[UUID | None] = []

    async def apply(self, result: object, correlation_id: UUID | None = None) -> None:
        self.results.append(result)
        self.correlation_ids.append(correlation_id)


def _synthetic_contract(
    name: str,
    topic: str,
    *,
    consumer_purpose: str | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
            consumer_purpose=consumer_purpose,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="FakeHandler",
                        module="fake.module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, message: object) -> None:
            return None

    return FakeHandler


@pytest.mark.asyncio
async def test_raw_event_projection_requires_explicit_result_applier() -> None:
    manifest = discover_contracts_from_paths([CONTRACT])
    engine = MessageDispatchEngine()

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,
        environment="test",
        subscribe_immediately=False,
    )

    assert report.total_failed == 0
    assert report.total_skipped == 1
    assert "requires dedicated raw event projection wiring" in report.results[0].reason


@pytest.mark.asyncio
async def test_raw_event_projection_dispatches_model_event_message_to_applier() -> None:
    manifest = discover_contracts_from_paths([CONTRACT])
    engine = MessageDispatchEngine()
    event_bus = EventBusInmemory()
    await event_bus.start()
    applier = CapturingApplier()
    run_id = f"unit-build-loop-{uuid4().hex[:8]}"
    correlation_id = uuid4()

    mock_container = MagicMock()
    handler_instance = HandlerBuildLoopProjection(container=mock_container)
    mock_container.get_service_async = AsyncMock(return_value=handler_instance)

    try:
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=engine,
            event_bus=event_bus,
            environment="test",
            container=mock_container,
            subscribe_immediately=False,
            result_appliers_by_contract={
                "node_build_loop_projection_compute": applier,
            },
        )
        assert report.total_failed == 0
        assert report.total_wired == 1

        engine.freeze()
        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=event_bus,
            environment="test",
            result_appliers_by_contract={
                "node_build_loop_projection_compute": applier,
            },
        )
        assert subscriptions == {
            "node_build_loop_projection_compute": (TERMINAL_TOPIC,)
        }

        body = {
            "run_id": run_id,
            "workflow_name": "build_loop",
            "event_type": "build-loop-orchestrator-completed",
            "terminal_event_at": datetime.now(UTC).isoformat(),
            "correlation_id": str(correlation_id),
        }
        await event_bus.publish(
            TERMINAL_TOPIC,
            key=None,
            value=json.dumps(body).encode("utf-8"),
            headers=ModelEventHeaders(
                source="unit-test",
                event_type="build-loop-orchestrator-completed",
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC),
            ),
        )

        assert len(applier.results) == 1
        result = applier.results[0]
        assert len(result.output_intents) == 1
        intent = result.output_intents[0]
        assert intent.intent_type == "build_loop.append"
        assert intent.payload.run_id == run_id
        assert applier.correlation_ids == [correlation_id]
    finally:
        await event_bus.close()


@pytest.mark.asyncio
async def test_explicit_result_applier_contract_subscribes_before_generic_backlog() -> (
    None
):
    generic_contract = _synthetic_contract(
        "aaa_generic_backlog",
        "onex.cmd.omnimarket.generic-backlog.v1",
    )
    projection_contract = _synthetic_contract(
        "node_build_loop_projection_compute",
        TERMINAL_TOPIC,
        consumer_purpose="projection",
    )
    manifest = ModelAutoWiringManifest(
        contracts=(generic_contract, projection_contract),
    )
    engine = MessageDispatchEngine()
    applier = CapturingApplier()
    unsubscribe = AsyncMock()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=unsubscribe)

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_fake_handler_cls(),
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=engine,
            event_bus=event_bus,
            environment="test",
            subscribe_immediately=False,
            result_appliers_by_contract={
                "node_build_loop_projection_compute": applier,
            },
        )

    assert report.total_failed == 0
    assert report.total_wired == 2

    engine.freeze()
    subscriptions = await subscribe_wired_contract_topics(
        manifest=manifest,
        report=report,
        dispatch_engine=engine,
        event_bus=event_bus,
        environment="test",
        result_appliers_by_contract={
            "node_build_loop_projection_compute": applier,
        },
    )

    assert tuple(subscriptions) == (
        "node_build_loop_projection_compute",
        "aaa_generic_backlog",
    )
    subscribed_topics = [
        call.kwargs["topic"] for call in event_bus.subscribe.call_args_list
    ]
    assert subscribed_topics == [
        TERMINAL_TOPIC,
        "onex.cmd.omnimarket.generic-backlog.v1",
    ]
