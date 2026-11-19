# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seam tests: runtime -> heartbeat topic -> registry, runtime -> runtime-error topic -> triage (OMN-24559).

Two seams carry what a runtime says about its own heartbeat:

1. The heartbeat: a real runtime (``heartbeat_required=True``) publishes
   ``ModelNodeHeartbeatEvent`` on ``SUFFIX_NODE_HEARTBEAT``; the registry side
   reads that topic through the registration orchestrator's
   ``DispatcherNodeHeartbeat`` (which validates it as the same
   ``ModelNodeHeartbeatEvent``) and the orchestrator contract subscribes to the
   same topic string. The test fails when the producer's bytes stop parsing as
   the consumer's model, or either side changes the topic. (The separate
   contract-registry router parses this topic with a different core model;
   that mismatch predates this change and is reported, not covered here.)
2. The refusal: a heartbeat refusal is published on ``SUFFIX_RUNTIME_ERROR`` as
   ``ModelRuntimeErrorEvent``; the runtime-error triage node's contract
   declares that topic and the input fields it reads. The test fails when the
   refusal event stops carrying a field the triage node reads, or either side
   changes the topic.

Owner of the other side: the registry side is the registration orchestrator and
contract registry reducer in this repository (the registry lanes registry-build
and the siblings omnibase_infra#5989 and #5995 touch the introspection event,
not the heartbeat event or these topics; read at their heads, not guessed).

Related:
    - OMN-24559: a runtime that cannot heartbeat is shown not healthy
    - OMN-24496: mandatory heartbeat
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.enums import EnumRuntimeHeartbeatRefusalReason
from omnibase_infra.event_bus import ModelEventMessage
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    dispatcher_node_heartbeat,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.service_node_introspection import ServiceNodeIntrospection
from omnibase_infra.topics import SUFFIX_NODE_HEARTBEAT, SUFFIX_RUNTIME_ERROR
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.integration

NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"
ORCHESTRATOR_CONTRACT = NODES_DIR / "node_registration_orchestrator" / "contract.yaml"
TRIAGE_CONTRACT = NODES_DIR / "node_runtime_error_triage_effect" / "contract.yaml"


def _payload(message: ModelEventMessage) -> dict[str, object]:
    """Decode one history message, unwrapping the event envelope."""
    decoded = json.loads(message.value)
    assert isinstance(decoded, dict)
    inner = decoded.get("payload")
    return inner if isinstance(inner, dict) else decoded


def _contract_topics(path: Path) -> set[str]:
    """Every string in a contract that looks like a topic."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node.startswith("onex."):
            found.add(node)

    walk(yaml.safe_load(path.read_text()))
    return found


class TestHeartbeatSeamRuntimeToRegistry:
    """The runtime's heartbeat bytes are what the registry side reads."""

    @pytest.mark.asyncio
    async def test_runtime_heartbeat_parses_as_the_registry_model(self) -> None:
        """A started required-heartbeat runtime's beat parses as the dispatcher's model."""
        HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
        bus = EventBusInmemory()
        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=NODES_DIR / "node_registration_orchestrator",
            event_bus=bus,
            node_name="heartbeat-seam-runtime",
            environment="local",
        )
        process = RuntimeHostProcess(
            event_bus=bus,
            config=make_runtime_config(),
            introspection_service=service,
            introspection_config=ModelRuntimeIntrospectionConfig(
                enabled=True, jitter_max_ms=0, throttle_min_interval_s=10
            ),
            heartbeat_required=True,
        )
        seed_mock_handlers(process)
        try:
            await process.start()
            history = await bus.get_event_history(topic=SUFFIX_NODE_HEARTBEAT)
            assert history, "a required-heartbeat runtime published no heartbeat"
            # The registry consumer validates with the class the producer emits.
            assert dispatcher_node_heartbeat.ModelNodeHeartbeatEvent is (
                ModelNodeHeartbeatEvent
            )
            for message in history:
                parsed = (
                    dispatcher_node_heartbeat.ModelNodeHeartbeatEvent.model_validate(
                        _payload(message)
                    )
                )
                assert parsed.node_id == service._introspection_node_id
        finally:
            await process.stop()

    def test_registry_contract_subscribes_to_the_heartbeat_topic(self) -> None:
        """The orchestrator contract subscribes the topic the runtime publishes."""
        assert SUFFIX_NODE_HEARTBEAT in _contract_topics(ORCHESTRATOR_CONTRACT)

    def test_control_a_changed_topic_is_not_subscribed(self) -> None:
        """Failing control: a renamed heartbeat topic is not in the contract."""
        renamed = SUFFIX_NODE_HEARTBEAT.replace("node-heartbeat", "node-heartbeat-x")
        assert renamed not in _contract_topics(ORCHESTRATOR_CONTRACT)


class TestRefusalSeamRuntimeToTriage:
    """The refusal event carries what the triage node reads."""

    @pytest.mark.asyncio
    async def test_refusal_event_carries_every_field_the_triage_node_reads(
        self,
    ) -> None:
        """A published refusal parses as the triage input model with its fields."""
        HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
        bus = EventBusInmemory()
        await bus.start()
        handler = HandlerRuntimeIntrospectionStartup()
        err = handler.refusal_error(
            reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILING,
            runtime_name="seam-runtime",
            detail="boom",
            correlation_id=uuid4(),
        )
        await handler.publish_refusal_event(err.refusal, bus)

        history = await bus.get_event_history(topic=SUFFIX_RUNTIME_ERROR)
        assert len(history) == 1
        payload = _payload(history[0])
        event = ModelRuntimeErrorEvent.model_validate(payload)
        assert "heartbeat_publish_failing" in event.raw_message
        contract = yaml.safe_load(TRIAGE_CONTRACT.read_text())
        input_fields = next(
            op["input_fields"]
            for op in contract["io_operations"]
            if op["operation"] == "triage_runtime_error"
        )
        for field in input_fields:
            assert field in payload, f"triage reads {field}; the refusal lacks it"

    def test_triage_contract_consumes_the_runtime_error_topic(self) -> None:
        """The triage node consumes the topic the refusal is published on."""
        assert SUFFIX_RUNTIME_ERROR in _contract_topics(TRIAGE_CONTRACT)

    def test_control_a_changed_topic_is_not_consumed(self) -> None:
        """Failing control: a renamed runtime-error topic is not in the contract."""
        renamed = SUFFIX_RUNTIME_ERROR.replace("runtime-error", "runtime-error-x")
        assert renamed not in _contract_topics(TRIAGE_CONTRACT)
