# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the topic-migration executor handler (OMN-12623).

TDD: asserts the executor provisions the new topic on DUAL_WRITE, mints the new
group, blocks CUTOVER while old lag remains, and rejects backward transitions.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.enums.enum_cutover_criterion import EnumCutoverCriterion
from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)
from omnibase_core.models.contracts.model_topic_schema_binding import (
    ModelTopicSchemaBinding,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.migration.models.model_consumer_group_lag import (
    ModelConsumerGroupLag,
)
from omnibase_infra.migration.service_drain_proof_gate import (
    ModelDrainProofDecision,
    ServiceDrainProofGate,
)
from omnibase_infra.nodes.node_topic_migration_executor_effect.handlers.handler_topic_migration_executor import (
    HandlerTopicMigrationExecutor,
)
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_command import (
    ModelTopicMigrationCommand,
)

pytestmark = pytest.mark.unit


class _RecordingProvisioner:
    def __init__(self) -> None:
        self.provisioned: list[str] = []

    async def ensure_topic_exists(self, topic_name: str) -> bool:
        self.provisioned.append(topic_name)
        return True


class _FakeGate(ServiceDrainProofGate):
    def __init__(self, *, allowed: bool, residual: int) -> None:
        self._allowed = allowed
        self._residual = residual

    async def evaluate(
        self, contract: ModelTopicMigrationContract
    ) -> ModelDrainProofDecision:
        return ModelDrainProofDecision(
            migration_ticket=contract.ticket,
            old_consumer_group=contract.old_consumer_group,
            old_topic=contract.old_binding.topic,
            retirement_allowed=self._allowed,
            residual_lag=self._residual,
            reason="fake gate decision",
        )


def _binding(topic: str, major: int) -> ModelTopicSchemaBinding:
    return ModelTopicSchemaBinding(
        topic=topic,
        event_name="ORDER_PLACED",
        schema_version=ModelSemVer(major=major, minor=0, patch=0),
    )


def _contract(phase: EnumMigrationPhase) -> ModelTopicMigrationContract:
    return ModelTopicMigrationContract(
        contract_version=ModelSemVer(major=1, minor=0, patch=0),
        ticket="OMN-12623",
        old_binding=_binding("onex.evt.orders.order-placed.v1", 1),
        new_binding=_binding("onex.evt.orders.order-placed.v2", 2),
        old_consumer_group="dev.orders.order-placed.consume.v1",
        new_consumer_group="dev.orders.order-placed.consume.v2",
        compatibility_window_hours=24,
        cutover_criteria=(EnumCutoverCriterion.OLD_TOPIC_DRAINED,),
        drain_proof_required=True,
        phase=phase,
    )


def _command(
    contract: ModelTopicMigrationContract, target: EnumMigrationPhase
) -> ModelTopicMigrationCommand:
    return ModelTopicMigrationCommand(
        correlation_id=uuid4(),
        contract=contract,
        target_phase=target,
    )


@pytest.mark.asyncio
async def test_dual_write_provisions_new_topic() -> None:
    prov = _RecordingProvisioner()
    handler = HandlerTopicMigrationExecutor(prov, _FakeGate(allowed=True, residual=0))
    event = await handler.execute(
        _command(_contract(EnumMigrationPhase.PLANNED), EnumMigrationPhase.DUAL_WRITE)
    )
    assert prov.provisioned == ["onex.evt.orders.order-placed.v2"]
    assert event.new_topic_provisioned is True
    assert event.phase is EnumMigrationPhase.DUAL_WRITE
    assert "minted new group" in event.detail


@pytest.mark.asyncio
async def test_cutover_blocks_while_old_lag_remains() -> None:
    handler = HandlerTopicMigrationExecutor(
        _RecordingProvisioner(), _FakeGate(allowed=False, residual=4)
    )
    with pytest.raises(ValueError, match="cutover blocked"):
        await handler.execute(
            _command(
                _contract(EnumMigrationPhase.DUAL_READ), EnumMigrationPhase.CUTOVER
            )
        )


@pytest.mark.asyncio
async def test_cutover_allowed_after_drain_proof() -> None:
    handler = HandlerTopicMigrationExecutor(
        _RecordingProvisioner(), _FakeGate(allowed=True, residual=0)
    )
    event = await handler.execute(
        _command(_contract(EnumMigrationPhase.DUAL_READ), EnumMigrationPhase.CUTOVER)
    )
    assert event.phase is EnumMigrationPhase.CUTOVER
    assert event.retirement_allowed is True
    assert event.residual_lag == 0


@pytest.mark.asyncio
async def test_backward_transition_rejected() -> None:
    handler = HandlerTopicMigrationExecutor(
        _RecordingProvisioner(), _FakeGate(allowed=True, residual=0)
    )
    with pytest.raises(ValueError, match="not strictly forward"):
        await handler.execute(
            _command(
                _contract(EnumMigrationPhase.CUTOVER),
                EnumMigrationPhase.DUAL_WRITE,
            )
        )
