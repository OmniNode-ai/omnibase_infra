# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for node_remote_agent_invoke_effect contract and models (OMN-9632).

Validates that the scaffold models are importable Pydantic BaseModel subclasses
and that the contract declares the correct topics and module references.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import BaseModel

from omnibase_infra.nodes.node_remote_agent_invoke_effect.models import (
    EnumAgentTaskStatus,
    ModelAgentTaskLifecycleEvent,
    ModelInvocationCommand,
)

CONTRACT_PATH = (
    Path("src")
    / "omnibase_infra"
    / "nodes"
    / "node_remote_agent_invoke_effect"
    / "contract.yaml"
)

EXPECTED_MODULE = "omnibase_infra.nodes.node_remote_agent_invoke_effect.models"


@pytest.mark.integration
class TestRemoteAgentInvokeEffectContract:
    def test_invocation_command_is_pydantic_model(self) -> None:
        assert issubclass(ModelInvocationCommand, BaseModel)

    def test_agent_task_lifecycle_event_is_pydantic_model(self) -> None:
        assert issubclass(ModelAgentTaskLifecycleEvent, BaseModel)

    def test_invocation_command_instantiates(self) -> None:
        agent_id = uuid4()
        cmd = ModelInvocationCommand(
            correlation_id=uuid4(),
            agent_id=agent_id,
            payload={"task": "ping"},
        )
        assert cmd.agent_id == agent_id

    def test_agent_task_lifecycle_event_instantiates(self) -> None:
        agent_id = uuid4()
        evt = ModelAgentTaskLifecycleEvent(
            correlation_id=uuid4(),
            status=EnumAgentTaskStatus.SUBMITTED,
            agent_id=agent_id,
        )
        assert evt.status == EnumAgentTaskStatus.SUBMITTED

    def test_contract_model_modules_reference_local_package(self) -> None:
        with CONTRACT_PATH.open() as f:
            data = yaml.safe_load(f)
        assert data["input_model"]["module"] == EXPECTED_MODULE
        assert data["output_model"]["module"] == EXPECTED_MODULE
