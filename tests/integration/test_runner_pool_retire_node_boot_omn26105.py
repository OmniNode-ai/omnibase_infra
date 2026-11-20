# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot node_runner_pool_retire from its real contract and round-trip a refused command (OMN-26105)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models.model_runner_pool_retire_request import (
    ModelRunnerPoolRetireRequest,
)
from omnibase_infra.models.model_runner_pool_retire_result import (
    ModelRunnerPoolRetireResult,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
FLEET = ROOT / "config/runner_fleet.yaml"
CONTRACT = ROOT / "src/omnibase_infra/nodes/node_runner_pool_retire/contract.yaml"


@pytest.mark.asyncio
async def test_real_contract_boots_and_refuses_a_non_ci_pool() -> None:
    manifest = discover_contracts_from_paths([CONTRACT])
    assert manifest.errors == ()
    declaration = manifest.contracts[0]
    assert declaration.event_bus is not None
    (command_topic,) = declaration.event_bus.subscribe_topics
    (terminal_topic,) = declaration.event_bus.publish_topics
    correlation = uuid4()
    result_future: asyncio.Future[ModelRunnerPoolRetireResult] = (
        asyncio.get_running_loop().create_future()
    )

    async def collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation and not result_future.done():
            result_future.set_result(
                ModelRunnerPoolRetireResult.model_validate(envelope.payload)
            )

    bus = EventBusInmemory(environment="dev", group=str(uuid4()))
    await bus.start()
    try:
        await bus.subscribe(terminal_topic, group_id=str(uuid4()), on_message=collect)
        engine = MessageDispatchEngine()
        await wire_from_manifest(manifest, engine, event_bus=bus, environment="dev")
        engine.freeze()
        request = ModelRunnerPoolRetireRequest(
            host="h053",
            prefix="not-a-declared-pool",
            fleet_config=FLEET,
            dry_run=True,
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=correlation,
            event_type=request.operation,
            payload=request.model_dump(mode="json"),
        )
        await bus.publish(
            command_topic, None, envelope.model_dump_json().encode(), None
        )
        result = await asyncio.wait_for(result_future, timeout=30)
    finally:
        await bus.close()
    assert result.success is False
    assert result.planned == ()
    assert result.error_message is not None
