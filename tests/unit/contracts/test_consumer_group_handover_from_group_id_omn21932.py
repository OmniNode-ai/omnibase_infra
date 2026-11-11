# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract declaring ``from_group_id`` loads and reaches the preflight (OMN-21932).

Broker-free: the contract loader reads the YAML and the preflight hands over on
the in-memory bus, so the retired group it names is observable in the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    HandlerContractSubscribePreflight,
    load_consumer_group_handover,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_request import (
    ModelContractSubscribePreflightRequest,
)
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    load_consume_concurrency,
)

pytestmark = pytest.mark.unit

ENVIRONMENT = "local"
TOPIC = "onex.evt.platform.node-heartbeat.v1"
LITERAL_GROUP = "agent-observability-postgres"
CONSUMER_GROUP = "local.omnimarket.agent_observability.consume.v1"

CONTRACT = f"""
name: agent_observability
node_type: REDUCER_GENERIC
contract_version: 1.0.0
description: OMN-21932 literal-group handover fixture
event_bus:
  subscribe_topics:
    - {TOPIC}
consumer_group_handover:
  from_service: omnimarket-projections
  from_node: agent-observability
  from_version: v1
  from_auto_offset_reset: earliest
  from_group_id: {LITERAL_GROUP}
"""


def _contract(tmp_path: Path, text: str = CONTRACT) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_contract_loader_accepts_a_declared_from_group_id(tmp_path: Path) -> None:
    handover = load_consumer_group_handover(_contract(tmp_path))

    assert handover is not None
    assert handover.from_group_id == LITERAL_GROUP
    assert handover.source_group(ENVIRONMENT) == LITERAL_GROUP


@pytest.mark.parametrize("declared", ["''", "'  '", "' padded'", "7"])
def test_the_contract_loader_refuses_an_invalid_from_group_id(
    tmp_path: Path, declared: str
) -> None:
    text = CONTRACT.replace(
        f"from_group_id: {LITERAL_GROUP}", f"from_group_id: {declared}"
    )

    with pytest.raises(ValueError, match="from_group_id"):
        load_consumer_group_handover(_contract(tmp_path, text))


@pytest.mark.asyncio
async def test_the_preflight_hands_over_from_the_literal_group(tmp_path: Path) -> None:
    bus = EventBusInmemory(environment=ENVIRONMENT)
    await bus.start()
    contract_path = _contract(tmp_path)
    preflight = HandlerContractSubscribePreflight(
        contract_name="agent_observability",
        concurrency=load_consume_concurrency(contract_path),
        handover=load_consumer_group_handover(contract_path),
        event_bus=bus,
        environment=ENVIRONMENT,
        consumer_group=CONSUMER_GROUP,
    )

    result = await preflight.handle(ModelContractSubscribePreflightRequest(topic=TOPIC))

    assert result.handover is not None
    assert result.handover.from_group == LITERAL_GROUP
    assert result.handover.to_group == CONSUMER_GROUP
