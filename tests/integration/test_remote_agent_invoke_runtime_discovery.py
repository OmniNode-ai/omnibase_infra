# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for remote-agent runtime discovery behavior."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_infra.nodes.node_remote_agent_invoke_effect.handlers.handler_a2a_task import (
    HandlerA2ATask,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_handler_runtime_discovery_fails_closed_without_dependencies() -> (
    None
):
    """Runtime-discovered A2A handler rejects unregistered targets before side effects."""
    handler = HandlerA2ATask()
    command = ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=uuid4(),
        invocation_kind=EnumInvocationKind.AGENT,
        agent_protocol=EnumAgentProtocol.A2A,
        target_ref="agent:unregistered-runtime-target",
        payload={"prompt": ModelSchemaValue.from_value("runtime discovery smoke")},
    )

    with pytest.raises(ValueError, match="Unknown target_ref"):
        await handler.submit(command)
