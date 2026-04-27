# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_delegation_orchestrator.contract_topics import (
    TOPIC_ID_AGENT_TASK_LIFECYCLE,
    TOPIC_ID_INVOCATION_COMMAND,
)
from omnibase_infra.nodes.node_delegation_orchestrator.dispatchers.dispatcher_agent_task_lifecycle import (
    DispatcherAgentTaskLifecycle,
)
from omnibase_infra.nodes.node_delegation_orchestrator.dispatchers.dispatcher_invocation_command import (
    DispatcherInvocationCommand,
)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_invocation_dispatcher_rejects_unsupported_envelope_with_correlation_id() -> (
    None
):
    dispatcher = DispatcherInvocationCommand(handler=object())

    result = await dispatcher.handle(object())

    assert result.status is EnumDispatchStatus.INVALID_MESSAGE
    assert result.topic == TOPIC_ID_INVOCATION_COMMAND
    assert result.correlation_id is not None
    assert result.error_message == "Unsupported envelope type: object"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_lifecycle_dispatcher_rejects_unsupported_envelope_with_correlation_id() -> (
    None
):
    dispatcher = DispatcherAgentTaskLifecycle(handler=object())

    result = await dispatcher.handle(object())

    assert result.status is EnumDispatchStatus.INVALID_MESSAGE
    assert result.topic == TOPIC_ID_AGENT_TASK_LIFECYCLE
    assert result.correlation_id is not None
    assert result.error_message == "Unsupported envelope type: object"
