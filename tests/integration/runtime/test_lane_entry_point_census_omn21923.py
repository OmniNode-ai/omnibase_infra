# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Census commands traverse the real runtime wiring under both bus overlays."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_lane_entry_point_census import load_policy
from omnibase_infra.models.lane_entry_point_census import (
    ModelLaneEntryPointCensusRequest,
    ModelLaneEntryPointCensusResult,
)
from omnibase_infra.nodes.node_lane_entry_point_census_effect import (
    NodeLaneEntryPointCensusEffect,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO / "src/omnibase_infra/nodes/node_lane_entry_point_census_effect/contract.yaml"
)
FIXTURES = REPO / "tests/unit/nodes/node_lane_entry_point_census_effect/fixtures"
POLICY = REPO / "deploy/lane-census/runtime-entry-points.yaml"
PROFILES = yaml.safe_load((FIXTURES / "runtime-profiles.yaml").read_text())


async def _run(profile: str, policy: Path) -> ModelLaneEntryPointCensusResult:
    overlay = PROFILES[profile]
    node = NodeLaneEntryPointCensusEffect(ModelONEXContainer())
    assert isinstance(node, NodeLaneEntryPointCensusEffect)
    manifest = discover_contracts_from_paths([CONTRACT])
    assert manifest.errors == ()
    assert len(manifest.contracts) == 1
    declaration = manifest.contracts[0]
    assert declaration.runtime_profiles == ("main",)
    assert declaration.event_bus is not None
    (command_topic,) = declaration.event_bus.subscribe_topics
    (result_topic,) = declaration.event_bus.publish_topics
    bus: EventBusInmemory | EventBusKafka
    if overlay["backend"] == "inmemory":
        bus = EventBusInmemory(environment=overlay["environment"], group=str(uuid4()))
    else:
        bootstrap = os.environ[overlay["bootstrap_env"]]
        bus = EventBusKafka(
            config=ModelKafkaEventBusConfig(
                bootstrap_servers=bootstrap,
                environment=overlay["environment"],
                auto_offset_reset="earliest",
                security_protocol=overlay["security_protocol"],
                instance_id=str(uuid4()),
            )
        )
    correlation_id = uuid4()
    received: asyncio.Future[ModelLaneEntryPointCensusResult] = (
        asyncio.get_running_loop().create_future()
    )

    async def collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation_id and not received.done():
            received.set_result(
                ModelLaneEntryPointCensusResult.model_validate(envelope.payload)
            )

    await bus.start()
    try:
        await bus.subscribe(
            result_topic, group_id=f"census-proof-{uuid4()}", on_message=collect
        )
        engine = MessageDispatchEngine()
        await wire_from_manifest(
            manifest, engine, event_bus=bus, environment=overlay["environment"]
        )
        engine.freeze()
        request = ModelLaneEntryPointCensusRequest(
            policy=load_policy(policy) if policy.exists() else None,
            policy_path="" if policy.exists() else str(policy),
            inspection=tuple(
                json.loads((FIXTURES / "inspect_planted.json").read_text())
            ),
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=correlation_id,
            event_type=request.operation,
            payload=request.model_dump(mode="json"),
        )
        await bus.publish(
            command_topic, None, envelope.model_dump_json().encode(), None
        )
        return await asyncio.wait_for(received, timeout=overlay["timeout_seconds"])
    finally:
        await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile", ["inmemory", pytest.param("redpanda", marks=pytest.mark.kafka)]
)
async def test_bus_reports_the_planted_python_dash_m_container(profile: str) -> None:
    result = await _run(profile, POLICY)
    assert result.success, result.error_message
    lane = next(c for c in result.censuses if c.lane == "proj-a")
    assert "proj-a-ctx-consumer" in lane.remove
    assert all(
        entry.command for census in result.censuses for entry in census.containers
    )
    assert "proj-a-ctx-consumer" in result.table_markdown
    assert "entry point to remove" in result.table_markdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile", ["inmemory", pytest.param("redpanda", marks=pytest.mark.kafka)]
)
async def test_bus_reports_an_unreadable_policy_as_a_refusal(
    profile: str, tmp_path: Path
) -> None:
    result = await _run(profile, tmp_path / "absent.yaml")
    assert result.success is False
    assert result.censuses == ()
    assert result.error_message
