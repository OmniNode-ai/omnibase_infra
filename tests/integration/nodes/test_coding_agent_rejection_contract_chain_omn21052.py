# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent rejection chain against the real contracts on disk (OMN-21052).

The workspace COMPUTE node publishes its verdict on the topic its contract
declares; the orchestrator must subscribe to that same topic and, fed the verdict
exactly as it crosses the wire (JSON), end the run in one REJECTED result on the
contract-declared failed terminal topic. Every topic here is read from the node
contracts, so the test breaks if the contracts and the handlers drift apart.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumAgentStatus,
    EnumCodingAgent,
    ModelCodingAgentInvokeCommand,
    ModelCodingAgentResult,
    ModelWorkspaceValidateCommand,
    ModelWorkspaceValidateResult,
)
from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
    HandlerCodingAgentOrchestrator,
)
from omnibase_infra.nodes.node_coding_agent_workspace_compute.handlers.handler_workspace_validate import (
    HandlerWorkspaceValidate,
)

pytestmark = pytest.mark.integration

_NODES = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"


def _event_bus(node: str) -> dict[str, list[str]]:
    contract = yaml.safe_load((_NODES / node / "contract.yaml").read_text())
    event_bus: dict[str, list[str]] = contract["event_bus"]
    return event_bus


def _one(topics: list[str], suffix: str) -> str:
    matches = [t for t in topics if t.endswith(suffix)]
    assert len(matches) == 1, f"expected one topic ending {suffix!r}, got {matches}"
    return matches[0]


@pytest.mark.asyncio
async def test_compute_verdict_from_disk_contract_reaches_one_rejected_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(allowed_root))

    compute_bus = _event_bus("node_coding_agent_workspace_compute")
    orchestrator_bus = _event_bus("node_coding_agent_orchestrator")
    validate_topic = _one(compute_bus["subscribe_topics"], "workspace-validate.v1")
    validated_topic = _one(compute_bus["publish_topics"], "workspace-validated.v1")
    invoke_topic = _one(orchestrator_bus["subscribe_topics"], "coding-agent-invoke.v1")
    failed_topic = _one(orchestrator_bus["publish_topics"], "coding-agent-failed.v1")
    fsm_topic = _one(orchestrator_bus["publish_topics"], "fsm-advance.v1")
    assert validate_topic in orchestrator_bus["publish_topics"]
    assert validated_topic in orchestrator_bus["subscribe_topics"]

    command = ModelCodingAgentInvokeCommand(
        correlation_id=uuid4(),
        agent=EnumCodingAgent.CLAUDE,
        prompt="read-only proof: say hello",
        workspace_path=str(tmp_path / "outside-allowed-root"),
        sandbox=EnumAgentSandbox.READ_ONLY,
        timeout_ms=60000,
    )
    orchestrator = HandlerCodingAgentOrchestrator()

    entry = await orchestrator.handle(
        ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type=derive_event_type_alias_for_topic(invoke_topic),
        )
    )
    (validate_envelope,) = entry.events
    assert validate_envelope.event_type == validate_topic

    validate_command = ModelWorkspaceValidateCommand.model_validate_json(
        validate_envelope.payload.model_dump_json()
    )
    verdict = await HandlerWorkspaceValidate().handle(validate_command)
    assert verdict.valid is False
    assert verdict.invoke_command == command

    wire_verdict = ModelWorkspaceValidateResult.model_validate_json(
        verdict.model_dump_json()
    ).model_dump(mode="json")
    rejected = await orchestrator.handle(
        ModelEventEnvelope(
            payload=wire_verdict,
            correlation_id=command.correlation_id,
            event_type=derive_event_type_alias_for_topic(validated_topic),
        )
    )

    emitted_topics = [e.event_type for e in rejected.events]
    assert sorted(emitted_topics) == sorted([fsm_topic, failed_topic])
    terminal = next(e.payload for e in rejected.events if e.event_type == failed_topic)
    assert isinstance(terminal, ModelCodingAgentResult)
    assert terminal.correlation_id == command.correlation_id
    assert terminal.status == EnumAgentStatus.REJECTED
    assert terminal.exit_code is None
