# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof that runtime-error triage is a non-emitting effect.

The real triage contract consumes ``runtime-error.v1`` and deliberately
declares no publish topic. This exercises its actual handler through
``wire_from_manifest`` and an in-memory bus, proving a successful database
effect is not misclassified as an undeliverable event by the consume boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
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

pytestmark = pytest.mark.integration

_TOPIC = "onex.evt.omnibase-infra.runtime-error.v1"
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/nodes/node_runtime_error_triage_effect/contract.yaml"
)


def _triage_contract() -> ModelDiscoveredContract:
    """Build the auto-wiring manifest entry from the checked-in triage contract."""
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert "output_model" not in contract
    assert contract["event_bus"]["subscribe_topics"] == [_TOPIC]
    assert contract["event_bus"]["publish_topics"] == []

    return ModelDiscoveredContract(
        name=contract["name"],
        node_type=contract["node_type"],
        contract_version=ModelContractVersion(**contract["contract_version"]),
        contract_path=_CONTRACT_PATH,
        entry_point_name=contract["name"],
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(_TOPIC,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerRuntimeErrorTriage",
                        module=(
                            "omnibase_infra.nodes.node_runtime_error_triage_effect."
                            "handlers.handler_runtime_error_triage"
                        ),
                    ),
                    operation="triage_runtime_error",
                    message_category="event",
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_runtime_error_triage_does_not_dlq_a_successful_effect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC1: one runtime error runs triage with neither undeliverable output nor DLQ."""
    bus = EventBusInmemory(environment="test", group="omn19801-runtime-error")
    engine = MessageDispatchEngine()
    await bus.start()
    try:
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(_triage_contract(),)),
            engine,
            event_bus=bus,
            environment="local",
        )
        engine.freeze()

        event = ModelRuntimeErrorEvent.create(
            logger_family="aiokafka.consumer",
            log_level="ERROR",
            message_template="Heartbeat failed",
            raw_message="Heartbeat failed",
            error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            severity=EnumRuntimeErrorSeverity.ERROR,
        )
        envelope = ModelEventEnvelope[object](
            payload=event.model_dump(mode="json"),
            correlation_id=event.correlation_id,
            event_type="omnibase-infra.runtime-error",
        )

        with (
            caplog.at_level(
                logging.ERROR,
                logger="omnibase_infra.runtime.auto_wiring.handler_wiring",
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_route_apply_publish_failure",
                new_callable=AsyncMock,
            ) as route_failure,
        ):
            await bus.publish(_TOPIC, None, envelope.model_dump_json().encode("utf-8"))

        route_failure.assert_not_awaited()
        assert "UndeliverableDispatchOutputError" not in caplog.text
        assert "no result applier for output-producing dispatch" not in caplog.text
    finally:
        await bus.close()
