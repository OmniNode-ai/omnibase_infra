# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rejection path of the coding-agent workflow, end to end over real dispatch (OMN-21052).

An invoke command naming a workspace outside the allowed roots must reach exactly
one terminal event on the failed terminal topic, with status REJECTED, no
completed event and no subprocess.

The live dev-lane run that motivated this test stopped at the orchestrator: the
COMPUTE node published its verdict (``valid: false``) and the orchestrator
refused it, because the orchestrator wanted a ``{verdict, command}`` payload the
COMPUTE node never publishes. Every earlier suite fed the orchestrator a
hand-built payload of that shape, so none of them could see it. This test feeds
each hop the payload the PREVIOUS hop actually published through the real
dispatch engine and the real dispatch result applier, so the chain breaks here if
any hop's output is not the next hop's input.

Only the CLI subprocess seam is mocked, and it is never reached.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumAgentStatus,
    EnumCodingAgent,
    EnumCodingAgentFsmState,
    ModelCodingAgentInvokeCommand,
    ModelCodingAgentResult,
    ModelWorkspaceValidateCommand,
    ModelWorkspaceValidateResult,
)
from omnibase_infra.nodes.node_coding_agent_fsm_reducer.handlers.handler_coding_agent_fsm import (
    HandlerCodingAgentFsm,
)
from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
    HandlerCodingAgentOrchestrator,
)
from omnibase_infra.nodes.node_coding_agent_workspace_compute.handlers.handler_workspace_validate import (
    HandlerWorkspaceValidate,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.unit.nodes.node_coding_agent.test_real_dispatch_multitopic_routing import (
    _entry_event_model,
    _make_engine,
    _published_topics,
    _real_applier_for,
)

pytestmark = pytest.mark.unit

_INVOKE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-invoke.v1"  # onex-topic-allow: the chain entry topic this dispatch test drives
_VALIDATE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-workspace-validate.v1"  # onex-topic-allow: chain step topic, asserted against the topic the bus receives
_VALIDATED_TOPIC = "onex.evt.omnibase-infra.coding-agent-workspace-validated.v1"  # onex-topic-allow: chain step topic, asserted against the topic the bus receives
_EFFECT_INVOKE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-effect-invoke.v1"  # onex-topic-allow: asserted never published on the rejection path
_FSM_ADVANCE_TOPIC = "onex.evt.omnibase-infra.coding-agent-fsm-advance.v1"  # onex-topic-allow: chain step topic, asserted against the topic the bus receives
_COMPLETED_TOPIC = "onex.evt.omnibase-infra.coding-agent-completed.v1"  # onex-topic-allow: asserted never published on the rejection path
_FAILED_TOPIC = "onex.evt.omnibase-infra.coding-agent-failed.v1"  # onex-topic-allow: the terminal topic a rejection must reach


def _orchestrator_engine(
    *, topic: str, category: EnumMessageCategory, route_id: str, pattern: str
) -> MessageDispatchEngine:
    return _make_engine(
        handler=HandlerCodingAgentOrchestrator(),
        dispatcher_id=f"coding-agent-orchestrator-{route_id}",
        category=category,
        message_types={derive_event_type_alias_for_topic(topic)},
        route_id=f"route.coding-agent-orchestrator.{route_id}",
        topic_pattern=pattern,
        event_model=_entry_event_model("node_coding_agent_orchestrator"),
        # The validated event is not the entrypoint model; the orchestrator is a
        # single handler matched by topic, as in production.
        use_payload_matcher=topic == _INVOKE_TOPIC,
    )


async def _dispatch_and_publish(
    engine: MessageDispatchEngine,
    *,
    node_dir: str,
    topic: str,
    envelope: ModelEventEnvelope[object],
) -> AsyncMock:
    result = await engine.dispatch(topic=topic, envelope=envelope)
    assert result.status == EnumDispatchStatus.SUCCESS, result.error_message
    mock_bus = AsyncMock(spec=ProtocolEventBusLike)
    await _real_applier_for(node_dir, mock_bus).apply(result)
    return mock_bus


def _wire_envelope(
    mock_bus: AsyncMock, *, topic: str, tenant: object = None
) -> ModelEventEnvelope[object]:
    """Re-present the one envelope the bus received as the next hop's input.

    The next hop receives the published payload as the consume boundary hands it
    over: serialised JSON with the topic's event-type alias stamped.
    """
    published = mock_bus.publish_envelope.call_args.kwargs["envelope"]
    return ModelEventEnvelope(
        payload=published.payload.model_dump(mode="json"),
        correlation_id=published.correlation_id,
        event_type=derive_event_type_alias_for_topic(topic),
        tenant_id=tenant,
    )


class TestRejectedWorkspaceReachesOneFailedTerminal:
    @pytest.mark.asyncio
    async def test_out_of_root_workspace_ends_in_one_rejected_terminal_no_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(allowed_root))
        outside = tmp_path / "outside-allowed-root"

        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CLAUDE,
            prompt="read-only proof: say hello",
            workspace_path=str(outside),
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=60000,
        )
        published_topics: list[str] = []

        # Hop 1: the orchestrator turns the invoke command into a validate command.
        entry_bus = await _dispatch_and_publish(
            _orchestrator_engine(
                topic=_INVOKE_TOPIC,
                category=EnumMessageCategory.COMMAND,
                route_id="invoke",
                pattern="*.cmd.omnibase-infra.coding-agent-invoke.*",
            ),
            node_dir="node_coding_agent_orchestrator",
            topic=_INVOKE_TOPIC,
            envelope=ModelEventEnvelope(
                payload=command,
                correlation_id=command.correlation_id,
                event_type=derive_event_type_alias_for_topic(_INVOKE_TOPIC),
            ),
        )
        published_topics += _published_topics(entry_bus)
        assert _published_topics(entry_bus) == [_VALIDATE_TOPIC]

        # Hop 2: the COMPUTE node rejects the workspace and publishes its verdict.
        compute_bus = await _dispatch_and_publish(
            _make_engine(
                handler=HandlerWorkspaceValidate(),
                dispatcher_id="coding-agent-workspace-compute",
                category=EnumMessageCategory.COMMAND,
                message_types={
                    derive_event_type_alias_for_topic(_VALIDATE_TOPIC),
                    "ModelWorkspaceValidateCommand",
                },
                route_id="route.coding-agent-workspace-compute",
                topic_pattern="*.cmd.omnibase-infra.coding-agent-workspace-validate.*",
                event_model=_entry_event_model("node_coding_agent_workspace_compute"),
            ),
            node_dir="node_coding_agent_workspace_compute",
            topic=_VALIDATE_TOPIC,
            envelope=_wire_envelope(entry_bus, topic=_VALIDATE_TOPIC),
        )
        published_topics += _published_topics(compute_bus)
        verdict = compute_bus.publish_envelope.call_args.kwargs["envelope"].payload
        assert isinstance(verdict, ModelWorkspaceValidateResult)
        assert verdict.valid is False

        # Hop 3: the orchestrator consumes THAT verdict, exactly as published.
        validated_bus = await _dispatch_and_publish(
            _orchestrator_engine(
                topic=_VALIDATED_TOPIC,
                category=EnumMessageCategory.EVENT,
                route_id="validated",
                pattern="*.evt.omnibase-infra.coding-agent-workspace-validated.*",
            ),
            node_dir="node_coding_agent_orchestrator",
            topic=_VALIDATED_TOPIC,
            envelope=_wire_envelope(compute_bus, topic=_VALIDATED_TOPIC),
        )
        validated_topics = _published_topics(validated_bus)
        published_topics += validated_topics

        # Rejection: the FSM advance and exactly one failed terminal, nothing else.
        assert sorted(validated_topics) == sorted([_FSM_ADVANCE_TOPIC, _FAILED_TOPIC])
        assert _EFFECT_INVOKE_TOPIC not in published_topics
        assert _COMPLETED_TOPIC not in published_topics
        assert published_topics.count(_FAILED_TOPIC) == 1

        by_topic = {
            c.kwargs["topic"]: c.kwargs["envelope"]
            for c in validated_bus.publish_envelope.call_args_list
        }
        terminal = by_topic[_FAILED_TOPIC].payload
        assert isinstance(terminal, ModelCodingAgentResult)
        assert terminal.correlation_id == command.correlation_id
        assert terminal.status == EnumAgentStatus.REJECTED
        assert terminal.exit_code is None, "a rejected run never starts a subprocess"

        # The FSM advance must be a payload the reducer can fold: the reducer
        # accepts it and lands in REJECTED.
        advance = by_topic[_FSM_ADVANCE_TOPIC].payload
        reducer_output = await HandlerCodingAgentFsm().handle(
            ModelEventEnvelope(
                payload=advance.model_dump(mode="json"),
                correlation_id=command.correlation_id,
            )
        )
        folded_state = reducer_output.projections[0]
        assert folded_state.current_state == EnumCodingAgentFsmState.REJECTED

    @pytest.mark.asyncio
    async def test_validate_command_carries_the_originating_invoke_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(tmp_path))
        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CODEX,
            prompt="read-only proof",
            workspace_path=str(tmp_path),
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=1234,
        )
        bus = await _dispatch_and_publish(
            _orchestrator_engine(
                topic=_INVOKE_TOPIC,
                category=EnumMessageCategory.COMMAND,
                route_id="invoke",
                pattern="*.cmd.omnibase-infra.coding-agent-invoke.*",
            ),
            node_dir="node_coding_agent_orchestrator",
            topic=_INVOKE_TOPIC,
            envelope=ModelEventEnvelope(
                payload=command,
                correlation_id=command.correlation_id,
                event_type=derive_event_type_alias_for_topic(_INVOKE_TOPIC),
            ),
        )
        validate = bus.publish_envelope.call_args.kwargs["envelope"].payload
        assert isinstance(validate, ModelWorkspaceValidateCommand)
        assert validate.invoke_command == command
