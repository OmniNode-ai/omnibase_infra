# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scaffold tests for node_remote_agent_invoke_effect."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.nodes.node_remote_agent_invoke_effect import (
    NodeRemoteAgentInvokeEffect,
)
from omnibase_infra.utils import compute_consumer_group_id

CONTRACT_PATH = (
    Path("src")
    / "omnibase_infra"
    / "nodes"
    / "node_remote_agent_invoke_effect"
    / "contract.yaml"
)

REMOTE_AGENT_INVOKE_TOPIC = "onex.cmd.omnibase-infra.remote-agent-invoke.v1"
AGENT_TASK_LIFECYCLE_TOPIC = "onex.evt.omnibase-infra.agent-task-lifecycle.v1"


def _load_contract() -> dict:
    with CONTRACT_PATH.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


@pytest.mark.unit
class TestNodeRemoteAgentInvokeEffectScaffold:
    """Validate the remote-agent invoke effect scaffold."""

    def test_node_exports_declarative_effect_shell(self) -> None:
        assert NodeRemoteAgentInvokeEffect.__name__ == "NodeRemoteAgentInvokeEffect"

    def test_contract_file_exists(self) -> None:
        assert CONTRACT_PATH.exists(), f"Missing contract: {CONTRACT_PATH}"

    def test_contract_identity(self) -> None:
        data = _load_contract()
        assert data["name"] == "node_remote_agent_invoke_effect"
        assert data["contract_name"] == "node_remote_agent_invoke_effect"
        assert data["node_name"] == "node_remote_agent_invoke_effect"
        assert data["node_type"] == "EFFECT_GENERIC"

    def test_input_topic_is_remote_agent_invoke(self) -> None:
        data = _load_contract()
        assert data["event_bus"]["subscribe_topics"] == [REMOTE_AGENT_INVOKE_TOPIC]
        consumed_topics = {event["topic"] for event in data["consumed_events"]}
        assert consumed_topics == {REMOTE_AGENT_INVOKE_TOPIC}

    def test_output_topic_is_agent_task_lifecycle(self) -> None:
        data = _load_contract()
        assert data["event_bus"]["publish_topics"] == [AGENT_TASK_LIFECYCLE_TOPIC]
        published_topics = {event["topic"] for event in data["published_events"]}
        assert published_topics == {AGENT_TASK_LIFECYCLE_TOPIC}

    def test_handler_routing_declares_a2a_handler(self) -> None:
        data = _load_contract()
        handlers = data["handler_routing"]["handlers"]
        assert len(handlers) == 1
        assert handlers[0]["operation"] == "agent.a2a_task"
        assert handlers[0]["handler"]["name"] == "HandlerA2ATask"

    def test_input_output_models_use_canonical_core_module(self) -> None:
        # OMN-14489: the contract must declare the CANONICAL omnibase_core classes
        # the producer (delegation orchestrator) and HandlerA2ATask actually use —
        # not the removed local stub models (which required agent_id / forbade the
        # rich fields, so the runtime input_model validation dropped the real
        # payload: remote-agent-invoke.v1 HW=19 IN / agent-task-lifecycle.v1 HW=0 OUT).
        data = _load_contract()
        assert (
            data["input_model"]["module"]
            == "omnibase_core.models.delegation.model_invocation_command"
        )
        assert (
            data["output_model"]["module"]
            == "omnibase_core.models.delegation.model_agent_task_lifecycle_event"
        )

    def test_consumer_group_id_matches_expected_shape(self) -> None:
        identity = ModelNodeIdentity(
            env="dev",
            service="omnibase-infra",
            node_name="node_remote_agent_invoke_effect",
            version="v0.1.0",
        )

        assert (
            compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME)
            == "dev.omnibase-infra.node_remote_agent_invoke_effect.consume.v0.1.0"
        )
