# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: drain-proof gate over the real adapter/observer stack (OMN-18917).

Wires ServiceDrainProofGate -> ServiceConsumerLagObserver -> AdapterKafkaAdminLag
with in-memory admin/consumer doubles (no broker) and the topic namespace taken
from the environment, as the runtime does.
"""

from __future__ import annotations

import pytest

from omnibase_infra.migration.adapter_kafka_admin_lag import AdapterKafkaAdminLag
from omnibase_infra.migration.service_consumer_lag_observer import (
    ServiceConsumerLagObserver,
)
from omnibase_infra.migration.service_drain_proof_gate import ServiceDrainProofGate
from tests.unit.migration.test_consumer_group_lag import (
    _FakeAdmin,
    _FakeConsumer,
    _FakeTopicPartition,
    _migration_contract,
)

pytestmark = pytest.mark.integration

_PHYSICAL = "prepr1.onex.evt.orders.order-placed.v1"


@pytest.mark.asyncio
async def test_gate_resolves_physical_topic_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    tp = _FakeTopicPartition(_PHYSICAL, 0)
    observer = ServiceConsumerLagObserver(
        AdapterKafkaAdminLag(_FakeAdmin({tp: 5}), _FakeConsumer({tp: 5}))
    )
    contract = _migration_contract()

    decision = await ServiceDrainProofGate(observer).evaluate(contract)

    assert decision.old_topic == contract.old_binding.topic
    assert decision.retirement_allowed is True
    assert decision.residual_lag == 0


@pytest.mark.asyncio
async def test_gate_blocks_on_namespaced_residual_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    tp = _FakeTopicPartition(_PHYSICAL, 0)
    observer = ServiceConsumerLagObserver(
        AdapterKafkaAdminLag(_FakeAdmin({tp: 2}), _FakeConsumer({tp: 5}))
    )

    decision = await ServiceDrainProofGate(observer).evaluate(_migration_contract())

    assert decision.retirement_allowed is False
    assert decision.residual_lag == 3
