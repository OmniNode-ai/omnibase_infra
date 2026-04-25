# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_agent_task_lifecycle_event import (
    ModelAgentTaskLifecycleEvent,
)
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_infra.event_bus.topic_constants import (
    TOPIC_DELEGATION_COMPLETED,
    TOPIC_DELEGATION_FAILED,
)
from omnibase_infra.nodes.node_delegation_orchestrator.enums import (
    EnumDelegationState,
)
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_event import (
    ModelDelegationEvent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)


def _request() -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt="triage these findings",
        task_type="research",
        correlation_id=uuid4(),
        emitted_at=datetime.now(UTC),
    )


def _command(correlation_id) -> ModelInvocationCommand:
    return ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=correlation_id,
        invocation_kind=EnumInvocationKind.AGENT,
        agent_protocol=EnumAgentProtocol.A2A,
        target_ref="adk-type-debt-scout",
        payload={
            "repo_path": ModelSchemaValue.from_value(
                "/Users/jonah/Code/omni_worktrees/OMN-9620/OMN-9632/omnibase_infra"
            )
        },
    )


@pytest.mark.unit
def test_agent_lifecycle_completion_emits_terminal_events() -> None:
    handler = HandlerDelegationWorkflow()
    request = _request()
    handler.handle_delegation_request(request)

    commands = handler.handle_invocation_command(_command(request.correlation_id))
    assert len(commands) == 1
    assert handler.workflows[request.correlation_id].state == EnumDelegationState.ROUTED

    submitted = ModelAgentTaskLifecycleEvent(
        task_id=commands[0].task_id,
        correlation_id=request.correlation_id,
        lifecycle_type=EnumAgentTaskLifecycleType.SUBMITTED,
        remote_task_handle="remote-123",
        occurred_at=datetime.now(UTC),
    )
    assert handler.handle_agent_task_lifecycle(submitted) == []
    assert (
        handler.workflows[request.correlation_id].state == EnumDelegationState.EXECUTING
    )

    completed = ModelAgentTaskLifecycleEvent(
        task_id=commands[0].task_id,
        correlation_id=request.correlation_id,
        lifecycle_type=EnumAgentTaskLifecycleType.COMPLETED,
        remote_task_handle="remote-123",
        artifact={"summary": ModelSchemaValue.from_value({"count": 3})},
        occurred_at=datetime.now(UTC),
    )
    events = handler.handle_agent_task_lifecycle(completed)
    assert len(events) == 2
    assert isinstance(events[0], ModelDelegationEvent)
    assert events[0].topic == TOPIC_DELEGATION_COMPLETED
    assert isinstance(events[1], ModelTaskDelegatedEvent)
    assert (
        handler.workflows[request.correlation_id].state == EnumDelegationState.COMPLETED
    )


@pytest.mark.unit
def test_agent_lifecycle_failure_emits_failed_event() -> None:
    handler = HandlerDelegationWorkflow()
    request = _request()
    handler.handle_delegation_request(request)
    command = _command(request.correlation_id)
    handler.handle_invocation_command(command)

    failed = ModelAgentTaskLifecycleEvent(
        task_id=command.task_id,
        correlation_id=request.correlation_id,
        lifecycle_type=EnumAgentTaskLifecycleType.FAILED,
        remote_task_handle="remote-456",
        error="remote failure",
        occurred_at=datetime.now(UTC),
    )
    events = handler.handle_agent_task_lifecycle(failed)
    assert len(events) == 2
    assert isinstance(events[0], ModelDelegationEvent)
    assert events[0].topic == TOPIC_DELEGATION_FAILED
    assert handler.workflows[request.correlation_id].state == EnumDelegationState.FAILED
