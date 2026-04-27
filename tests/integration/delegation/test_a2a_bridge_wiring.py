# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration smoke coverage for reducer -> orchestrator A2A bridge wiring."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums.enum_agent_capability import EnumAgentCapability
from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.delegation.model_routing_rule import ModelRoutingRule
from omnibase_core.models.delegation.model_target_agent import ModelTargetAgent
from omnibase_core.topics import TopicBase
from omnibase_infra.nodes.node_delegation_orchestrator.dispatch_resolver import (
    resolve_effect_topic,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    resolve_invocation_command,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CONTRACT_PATH = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_delegation_routing_reducer"
    / "contract.yaml"
)


def _load_contract() -> dict[str, object]:
    with CONTRACT_PATH.open(encoding="utf-8") as contract_file:
        loaded = yaml.safe_load(contract_file)
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.integration
def test_agent_routing_end_to_topic_boundary() -> None:
    """Reducer routing rules and orchestrator dispatch resolver stay aligned."""
    contract = _load_contract()
    rules = tuple(
        ModelRoutingRule.model_validate(item)
        for item in contract.get("routing_rules", [])
    )
    agents = tuple(
        ModelTargetAgent.model_validate(item)
        for item in contract.get("target_agents", [])
    )

    command = resolve_invocation_command(
        rules=rules,
        capability=EnumAgentCapability.TECH_DEBT_TRIAGE,
        payload={"findings": [], "limit": 10},
        task_id=uuid4(),
        correlation_id=uuid4(),
    )

    assert command.invocation_kind is EnumInvocationKind.AGENT
    assert command.agent_protocol is EnumAgentProtocol.A2A
    assert (
        resolve_effect_topic(command.invocation_kind) is TopicBase.REMOTE_AGENT_INVOKE
    )
    assert any(agent.target_ref == command.target_ref for agent in agents)


@pytest.mark.integration
def test_routing_contract_yaml_single_target_agents_block() -> None:
    """Guard against append-style duplication of the target_agents block."""
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert text.count("target_agents:") == 1


@pytest.mark.integration
def test_contract_yaml_target_agents_parse_as_models() -> None:
    """Production contract loader parity for target_agents declarations."""
    contract = _load_contract()
    parsed = [
        ModelTargetAgent.model_validate(item)
        for item in contract.get("target_agents", [])
    ]
    assert parsed == [
        ModelTargetAgent(
            target_ref="adk-type-debt-scout",
            protocol=EnumAgentProtocol.A2A,
            base_url="${DEBT_SCOUT_BASE_URL}",
            protocol_version="0.3",
        )
    ]


@pytest.mark.integration
def test_debt_scout_target_url_is_config_dependency() -> None:
    contract = _load_contract()
    deps = {item["key"] for item in contract.get("config_dependencies", [])}
    assert "DEBT_SCOUT_BASE_URL" in deps
    target = contract["target_agents"][0]
    assert target["base_url"] == "${DEBT_SCOUT_BASE_URL}"
