# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
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


class ModelTerminalResult(BaseModel):
    correlation_id: UUID
    status: str


class HandlerTerminalResult:
    async def handle(self, envelope: object) -> ModelTerminalResult:
        if isinstance(envelope, dict):
            payload = envelope["payload"]
            debug_trace = envelope["__debug_trace"]
            assert isinstance(debug_trace, dict)
            correlation_id = debug_trace["correlation_id"]
        else:
            payload = envelope.payload
            correlation_id = envelope.correlation_id
        return ModelTerminalResult(
            correlation_id=UUID(str(correlation_id)),
            status=str(payload["status"]),
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_publish_contract_emits_handler_result_to_terminal_event() -> None:
    command_topic = "onex.cmd.omnimarket.session-bootstrap-start.v2"
    terminal_topic = "onex.evt.omnimarket.session-bootstrap-completed.v2"
    aux_topic = "onex.evt.omnimarket.session-cron-health-violation.v1"
    correlation_id = UUID("11111111-1111-4111-8111-111111111111")

    contract = ModelDiscoveredContract(
        name="session_bootstrap",
        node_type="orchestrator",
        contract_version=ModelContractVersion(major=2, minor=0, patch=0),
        contract_path=Path("/tmp/session_bootstrap/contract.yaml"),  # noqa: S108
        entry_point_name="session_bootstrap",
        package_name="omnimarket",
        terminal_event=terminal_topic,
        event_bus=ModelEventBusWiring(
            subscribe_topics=(command_topic,),
            publish_topics=(terminal_topic, aux_topic),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerTerminalResult",
                        module=__name__,
                    ),
                    event_model=None,
                ),
            ),
        ),
    )
    bus = EventBusInmemory(environment="test", group="pattern-b-terminal")
    await bus.start()
    try:
        terminal_results: asyncio.Queue[ModelTerminalResult] = asyncio.Queue()

        async def collect_terminal(message: ModelEventMessage) -> None:
            envelope = ModelEventEnvelope[ModelTerminalResult].model_validate_json(
                message.value
            )
            if envelope.correlation_id == correlation_id:
                await terminal_results.put(envelope.payload)

        await bus.subscribe(
            terminal_topic,
            group_id="terminal-collector",
            on_message=collect_terminal,
        )

        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerTerminalResult,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload={"status": "ready"},
            correlation_id=correlation_id,
            event_type="omnimarket.session-bootstrap-start",
        )
        await bus.publish(
            command_topic,
            None,
            command.model_dump_json().encode("utf-8"),
            None,
        )

        result = await asyncio.wait_for(terminal_results.get(), timeout=2)

        assert result.status == "ready"
        assert result.correlation_id == correlation_id
    finally:
        await bus.close()
