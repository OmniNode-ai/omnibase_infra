# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: ``/ready`` reports the contracts the lazy-start front door woke.

Measured on the dev runtime (omni-201, container 5f81d5119e91, started
2026-11-23T16:01:28Z): the front door logged 202 ``started on its first request``
lines for 26 dormant contracts by 16:38Z, while ``/ready`` read
``contract_attach`` 25 of 25 and ``/health`` read runtime wiring 68 of 68 the
whole time. Nothing outside ``lazy_start_front_door.py`` reads the front door's
``serving`` set or ``wake_failures``, so no reader outside the process could see
a hot load happen or a wake fail (the hot-loading plan's serving-set read slice).

RED at the merge base: ``StoreLazyStartDormant.record_front_door`` does not
exist and the ``contract_attach`` probe detail carries no ``lazy_start`` block.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
)

pytestmark = pytest.mark.unit

WOKEN = "node_serving_read_woken"
FAILING = "node_serving_read_failing"
IDLE = "node_serving_read_idle"
INGRESS = "node_serving_read_ingress"


def _topic(name: str) -> str:
    return f"onex.cmd.platform.{name.replace('_', '-')}.v1"


def _contract(tmp_path: Path, name: str) -> ModelDiscoveredContract:
    path = tmp_path / name / "contract.yaml"
    path.parent.mkdir()
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name=name,
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_topic(name),), publish_topics=()
        ),
    )


class _Transaction:
    """Builds every contract but ``FAILING``, whose build always raises."""

    def __init__(self) -> None:
        self.delivered: list[str] = []

    async def build(self, contract_name: str) -> None:
        if contract_name == FAILING:
            raise RuntimeError(f"contract '{FAILING}' did not wire: fixture")

    async def attach(self, contract_name: str) -> None:
        return None

    async def rollback(self, contract_name: str) -> None:
        return None

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        self.delivered.append(contract_name)
        return {"handled_by": contract_name}


def _request(name: str) -> ModelLazyStartRequest:
    return ModelLazyStartRequest(
        target_contract=name,
        correlation_id=uuid4(),
        topic=_topic(name),
        envelope={"payload": {}},
    )


@pytest.fixture
def front_door(tmp_path: Path) -> Iterator[LazyStartFrontDoor]:
    StoreLazyStartDormant.clear()
    dormant = {n: _contract(tmp_path, n) for n in (WOKEN, FAILING, IDLE)}
    policy = ModelLazyStartPolicy(
        ingress_contract=INGRESS, boot_contracts=(), budget_seconds=5, max_retries=0
    )
    door = LazyStartFrontDoor(_Transaction(), dormant, policy)
    StoreLazyStartDormant.record(policy, tuple(dormant.values()))
    StoreLazyStartDormant.record_front_door(door)
    yield door
    StoreLazyStartDormant.clear()


def _lazy_start_block() -> dict[str, object]:
    gate = ContractAttachReadinessGate(())
    gate.record(())
    ready, detail = gate.probe()
    assert ready is True, "the serving read must never change readiness"
    block = detail["lazy_start"]
    assert isinstance(block, dict)
    return block


@pytest.mark.asyncio
async def test_ready_reports_a_woken_contract_and_its_count_rises(
    front_door: LazyStartFrontDoor,
) -> None:
    before = _lazy_start_block()
    assert before["serving_count"] == 0
    assert before["serving_contracts"] == []
    assert before["dormant_contract_count"] == 3

    result = await front_door.handle(_request(WOKEN))
    assert result.outcome is EnumLazyStartOutcome.DELIVERED
    assert result.answer == {"handled_by": WOKEN}

    after = _lazy_start_block()
    assert after["serving_contracts"] == [WOKEN]
    assert after["serving_count"] == before["serving_count"] + 1


@pytest.mark.asyncio
async def test_ready_reports_a_failed_wake_by_contract_and_reason(
    front_door: LazyStartFrontDoor,
) -> None:
    await front_door.wake_for_topic(_topic(FAILING))

    block = _lazy_start_block()
    failures = block["wake_failures"]
    assert isinstance(failures, dict)
    assert list(failures) == [FAILING]
    assert "did not wire" in str(failures[FAILING])
    assert block["serving_contracts"] == []


def test_ready_carries_no_lazy_start_block_without_a_front_door() -> None:
    StoreLazyStartDormant.clear()
    gate = ContractAttachReadinessGate(())
    gate.record(())
    ready, detail = gate.probe()
    assert ready is True
    assert "lazy_start" not in detail
