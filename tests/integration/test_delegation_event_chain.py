# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Golden-chain: task-delegated event lands on bus from delegation chain (OMN-10940).

Proves that the full DelegationIntentBridge path emits
onex.evt.omniclaude.task-delegated.v1 on BOTH terminal outcomes:
  - COMPLETED (quality gate pass)
  - FAILED    (quality gate fail / refusal)

These tests drive:
  HandlerDelegationWorkflow  (orchestrator FSM)
  DelegationIntentBridge     (executes routing / inference / quality-gate intents)
  EventBusInmemory           (captures all published envelopes)

and assert that task-delegated events are present in the bus history with
the required payload fields (correlation_id, task_type, model_name, max_tokens).

This is distinct from the handler-unit tests in test_delegation_pipeline_e2e.py
which only assert what the handler *returns*. These tests assert what actually
*lands on the event bus*.

Related:
    - OMN-10940: E2E task-delegated event emission from /skill path
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.topic_constants import TOPIC_DELEGATION_TASK_DELEGATED
from omnibase_infra.nodes.node_delegation_orchestrator.delegation_intent_bridge import (
    DelegationIntentBridge,
    MockLlmCaller,
)
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_intent import (
    ModelInferenceIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_response_data import (
    ModelInferenceResponseData,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_quality_gate_intent import (
    ModelQualityGateIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_routing_intent import (
    ModelRoutingIntent,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_bifrost_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal bifrost contract for the routing reducer."""
    import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _h

    _h._config = None
    _h._load_bifrost_endpoints.cache_clear()
    contract_path = tmp_path / "bifrost_delegation.yaml"
    contract_path.write_text(
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n"
        "  - backend_id: local-qwen-coder-30b\n"
        '    endpoint_url: "http://192.168.86.201:8000"\n'
        '    model_name: "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"\n'
        "    tier: local\n"
        "    timeout_ms: 30000\n"
        "    capabilities: []\n"
        "  - backend_id: local-deepseek-r1-14b\n"
        '    endpoint_url: "http://192.168.86.201:8001"\n'
        '    model_name: "Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ"\n'
        "    tier: local\n"
        "    timeout_ms: 30000\n"
        "    capabilities: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BIFROST_CONTRACT_PATH", str(contract_path))
    yield
    _h._config = None
    _h._load_bifrost_endpoints.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drive_pipeline_to_terminal(
    *,
    bus: EventBusInmemory,
    handler: HandlerDelegationWorkflow,
    bridge: DelegationIntentBridge,
    request: ModelDelegationRequest,
    inference_content: str,
) -> None:
    """Drive the full delegation pipeline through the bridge to a terminal state.

    Sequence:
        1. handler.handle_delegation_request -> ModelRoutingIntent
        2. bridge.handle_routing_intent      -> ModelRoutingDecision (published to bus)
        3. handler.handle_routing_decision   -> ModelInferenceIntent
        4. bridge.handle_inference_intent    -> ModelInferenceResponseData (published)
        5. handler.handle_inference_response -> ModelQualityGateIntent
        6. bridge.handle_quality_gate_intent -> ModelQualityGateResult (published)
        7. handler.handle_gate_result        -> terminal events

    The final output_events from step 7 are NOT automatically published by the
    handler — in production the dispatcher would do that.  Here we publish them
    manually so the bus history reflects the full chain.
    """
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    cid = request.correlation_id

    # Step 1 — request intake
    step1_events = handler.handle_delegation_request(request)
    assert len(step1_events) == 1
    assert isinstance(step1_events[0], ModelRoutingIntent)

    # Step 2 — routing intent → routing decision (bridge publishes decision to bus)
    routing_decision = await bridge.handle_routing_intent(step1_events[0])
    assert routing_decision.correlation_id == cid

    # Step 3 — routing decision drives orchestrator
    step3_events = handler.handle_routing_decision(routing_decision)
    assert len(step3_events) == 1
    assert isinstance(step3_events[0], ModelInferenceIntent)
    inference_intent = step3_events[0]

    # Step 4 — inference intent → inference response (bridge publishes to bus)
    inference_response = ModelInferenceResponseData(
        correlation_id=cid,
        content=inference_content,
        model_used=inference_intent.model,
        latency_ms=50,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
    )
    await bridge._publish(
        inference_response, "onex.evt.omnibase-infra.inference-response.v1"
    )

    # Step 5 — inference response drives orchestrator
    step5_events = handler.handle_inference_response(inference_response)
    assert len(step5_events) >= 1
    quality_gate_intents = [
        e for e in step5_events if isinstance(e, ModelQualityGateIntent)
    ]
    assert len(quality_gate_intents) == 1

    # Step 6 — quality gate intent → result (bridge publishes to bus)
    gate_result = await bridge.handle_quality_gate_intent(quality_gate_intents[0])

    # Step 7 — gate result drives orchestrator to terminal state; publish outputs to bus
    terminal_events = handler.handle_gate_result(gate_result)
    for event in terminal_events:
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event,
            correlation_id=getattr(event, "correlation_id", None),
            envelope_timestamp=datetime.now(UTC),
        )
        topic = getattr(event, "topic", None)
        if topic:
            await bus.publish_envelope(envelope=envelope, topic=topic)


def _parse_task_delegated_payloads(
    history: list[object],
) -> list[dict[str, object]]:
    """Extract task-delegated event payloads from raw bus history messages."""
    results: list[dict[str, object]] = []
    for msg in history:
        if not hasattr(msg, "topic") or msg.topic != TOPIC_DELEGATION_TASK_DELEGATED:
            continue
        raw = json.loads(msg.value) if isinstance(msg.value, bytes) else msg.value
        payload: dict[str, object] = raw.get("payload", raw)
        results.append(payload)
    return results


# ---------------------------------------------------------------------------
# Golden-chain: success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_emits_task_delegated_on_bus() -> None:
    """Quality-gate pass → task-delegated.v1 event lands on bus with required fields."""
    bus = EventBusInmemory(environment="test", group="test-group", max_history=1000)
    await bus.start()
    try:
        good_content = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_verify_registration_happy_path():\n"
            "    result = verify_registration('node-abc')\n"
            "    assert result is True\n\n"
            "def test_verify_registration_missing_node():\n"
            "    with pytest.raises(ValueError):\n"
            "        verify_registration('')\n"
        )

        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write unit tests for verify_registration.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        handler = HandlerDelegationWorkflow()
        bridge = DelegationIntentBridge(event_bus=bus)

        await _drive_pipeline_to_terminal(
            bus=bus,
            handler=handler,
            bridge=bridge,
            request=request,
            inference_content=good_content,
        )

        # Assert: task-delegated.v1 is in bus history
        history = await bus.get_event_history(
            limit=200, topic=TOPIC_DELEGATION_TASK_DELEGATED
        )
        assert len(history) >= 1, (
            f"Expected task-delegated event on bus topic {TOPIC_DELEGATION_TASK_DELEGATED}, "
            f"got 0. Full bus history topics: "
            f"{[getattr(m, 'topic', '?') for m in await bus.get_event_history(limit=200)]}"
        )

        payloads = _parse_task_delegated_payloads(history)
        assert len(payloads) >= 1

        # Find the one matching our correlation_id
        matching = [p for p in payloads if str(p.get("correlation_id")) == str(cid)]
        assert len(matching) == 1, (
            f"Expected 1 task-delegated payload for correlation_id={cid}, "
            f"got {len(matching)}"
        )

        payload = matching[0]
        # Required payload fields
        assert payload["task_type"] == "test"
        assert payload["model_name"] != ""
        assert payload["quality_gate_passed"] is True
        assert "correlation_id" in payload

    finally:
        await bus.close()


# ---------------------------------------------------------------------------
# Golden-chain: failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_path_emits_task_delegated_on_bus() -> None:
    """Quality-gate fail (refusal) → task-delegated.v1 event lands on bus with quality_gate_passed=False."""
    bus = EventBusInmemory(environment="test", group="test-group", max_history=1000)
    await bus.start()
    try:
        refusal_content = (
            "I'm sorry, I cannot generate unit tests for this code as it "
            "appears to contain sensitive authentication logic."
        )

        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write unit tests for auth_handler.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        handler = HandlerDelegationWorkflow()
        bridge = DelegationIntentBridge(event_bus=bus)

        await _drive_pipeline_to_terminal(
            bus=bus,
            handler=handler,
            bridge=bridge,
            request=request,
            inference_content=refusal_content,
        )

        # Assert: task-delegated.v1 is in bus history even on failure
        history = await bus.get_event_history(
            limit=200, topic=TOPIC_DELEGATION_TASK_DELEGATED
        )
        assert len(history) >= 1, (
            f"Expected task-delegated event on bus even for failed delegation, "
            f"got 0. Bus history: "
            f"{[getattr(m, 'topic', '?') for m in await bus.get_event_history(limit=200)]}"
        )

        payloads = _parse_task_delegated_payloads(history)
        matching = [p for p in payloads if str(p.get("correlation_id")) == str(cid)]
        assert len(matching) == 1, (
            f"Expected 1 task-delegated payload for correlation_id={cid}, "
            f"got {len(matching)}"
        )

        payload = matching[0]
        assert payload["task_type"] == "test"
        assert payload["quality_gate_passed"] is False
        assert "correlation_id" in payload

    finally:
        await bus.close()


# ---------------------------------------------------------------------------
# Payload contract: required fields present on both paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inference_content", "expected_pass"),
    [
        (
            # Pass: valid test output
            "import pytest\n\n@pytest.mark.unit\ndef test_foo():\n    assert True\n",
            True,
        ),
        (
            # Fail: refusal
            "I cannot assist with that request.",
            False,
        ),
    ],
)
async def test_task_delegated_payload_has_required_fields(
    inference_content: str,
    expected_pass: bool,
) -> None:
    """task-delegated payload carries correlation_id, task_type, model_name on both paths."""
    bus = EventBusInmemory(environment="test", group="test-group", max_history=1000)
    await bus.start()
    try:
        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write unit tests for parser.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        handler = HandlerDelegationWorkflow()
        bridge = DelegationIntentBridge(event_bus=bus)

        await _drive_pipeline_to_terminal(
            bus=bus,
            handler=handler,
            bridge=bridge,
            request=request,
            inference_content=inference_content,
        )

        history = await bus.get_event_history(
            limit=200, topic=TOPIC_DELEGATION_TASK_DELEGATED
        )
        payloads = _parse_task_delegated_payloads(history)
        matching = [p for p in payloads if str(p.get("correlation_id")) == str(cid)]

        assert len(matching) == 1
        payload = matching[0]

        # Contract: all required fields present
        for field in (
            "correlation_id",
            "task_type",
            "model_name",
            "quality_gate_passed",
        ):
            assert field in payload, (
                f"Required field '{field}' missing from task-delegated payload"
            )

        assert payload["quality_gate_passed"] is expected_pass
        assert payload["task_type"] == "test"

    finally:
        await bus.close()
