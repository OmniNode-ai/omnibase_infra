# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.enums.enum_agent_capability import EnumAgentCapability
from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_routing_rule import ModelRoutingRule
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    resolve_invocation_command,
)

_REDUCER_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_delegation_routing_reducer"
    / "contract.yaml"
)


@pytest.mark.unit
def test_agent_rule_emits_invocation_command() -> None:
    rules = (
        ModelRoutingRule(
            capability=EnumAgentCapability.TECH_DEBT_TRIAGE,
            invocation_kind=EnumInvocationKind.AGENT,
            agent_protocol=EnumAgentProtocol.A2A,
            model_backend=None,
            target_ref="adk-type-debt-scout",
            fallbacks=(),
        ),
    )
    cmd = resolve_invocation_command(
        rules=rules,
        capability=EnumAgentCapability.TECH_DEBT_TRIAGE,
        payload={"findings": []},
        task_id=uuid.uuid4(),
        correlation_id=uuid.uuid4(),
    )
    assert isinstance(cmd, ModelInvocationCommand)
    assert cmd.invocation_kind is EnumInvocationKind.AGENT
    assert cmd.agent_protocol is EnumAgentProtocol.A2A
    assert cmd.target_ref == "adk-type-debt-scout"


@pytest.mark.unit
def test_missing_rule_raises() -> None:
    with pytest.raises(LookupError):
        resolve_invocation_command(
            rules=(),
            capability=EnumAgentCapability.TECH_DEBT_TRIAGE,
            payload={},
            task_id=uuid.uuid4(),
            correlation_id=uuid.uuid4(),
        )


@pytest.mark.unit
def test_contract_yaml_routing_rules_parse_as_models() -> None:
    contract = yaml.safe_load(_REDUCER_CONTRACT.read_text())
    rules_raw = contract.get("routing_rules", [])
    parsed = [ModelRoutingRule.model_validate(rule) for rule in rules_raw]
    assert all(isinstance(rule, ModelRoutingRule) for rule in parsed)
    assert any(
        rule.capability is EnumAgentCapability.TECH_DEBT_TRIAGE for rule in parsed
    )


@pytest.mark.unit
def test_invalid_routing_rule_yaml_rejected_at_load() -> None:
    bogus = {
        "capability": "tech_debt_triage",
        "invocation_kind": "agent",
        "target_ref": "adk-type-debt-scout",
        "fallbacks": [],
    }
    with pytest.raises(ValidationError):
        ModelRoutingRule.model_validate(bogus)


@pytest.mark.unit
def test_contract_yaml_has_single_routing_rules_block() -> None:
    text = _REDUCER_CONTRACT.read_text()
    assert text.count("routing_rules:") == 1
