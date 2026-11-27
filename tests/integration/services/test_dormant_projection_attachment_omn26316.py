# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: the health monitor, with a real lazy-start front door, excuses a
dormant projection and still judges a woken or eager unattached one."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.integration

_MONITOR = "omnibase_infra.services.service_runtime_health_monitor"
LIVE = "projection_live_attached"
WAITING = "projection_dormant_waiting"
WOKEN = "projection_dormant_woken_unattached"


def _topic(name: str) -> str:
    return f"onex.evt.omnimarket.{name.replace('_', '-')}.v1"


def _projection(name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(__file__),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_topic(name),), publish_topics=()
        ),
        db_io=ModelDbOwnershipSubcontract(
            db_tables=[
                ModelDbTableDeclaration(
                    name=f"{name}_rows",
                    database_ref="application",
                    schema="omninode_internal",
                    migration="0001_init.sql",
                    access="read_write",
                    role="projection_target",
                )
            ]
        ),
    )


class _Transaction:
    async def build(self, contract_name: str) -> None:
        return None

    async def attach(self, contract_name: str) -> None:
        return None

    async def rollback(self, contract_name: str) -> None:
        return None

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        return {"handled_by": contract_name}


@pytest.fixture(autouse=True)
def _clean_store() -> Iterator[None]:
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


def _lane_with_dormant(*names: str) -> LazyStartFrontDoor:
    dormant = {name: _projection(name) for name in names}
    policy = ModelLazyStartPolicy(
        ingress_contract="node_dormant_ingress",
        boot_contracts=(),
        budget_seconds=5,
        max_retries=0,
    )
    door = LazyStartFrontDoor(_Transaction(), dormant, policy)
    StoreLazyStartDormant.record(policy, tuple(dormant.values()))
    StoreLazyStartDormant.record_front_door(door)
    return door


async def _attachment(*contracts: ModelDiscoveredContract) -> tuple[int, str, str]:
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts), errors=())
    live_group = f"runtime.live.consume.v1.__t.{_topic(LIVE)}"
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.get_consumer_groups.return_value = {
        (_topic(LIVE), "runtime.live.consume.v1"): live_group
    }
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus, bootstrap_servers="localhost:9092"
    )
    with (
        patch(f"{_MONITOR}._discover_contracts", return_value=manifest),
        patch(
            f"{_MONITOR}._filter_manifest_for_runtime_profile", return_value=manifest
        ),
        patch(
            f"{_MONITOR}._list_consumer_group_snapshots",
            return_value=[ConsumerGroupSnapshot(group_id=live_group, state="STABLE")],
        ),
    ):
        event = await monitor.run_once()
    dim = next(d for d in event.dimensions if d.name == "projection_attachment")
    return event.unattached_projection_count, dim.status, dim.detail


@pytest.mark.asyncio
async def test_dormant_projection_is_listed_and_does_not_degrade() -> None:
    _lane_with_dormant(WAITING)
    count, status, detail = await _attachment(_projection(LIVE), _projection(WAITING))
    assert (count, status) == (0, "HEALTHY")
    assert WAITING in detail


@pytest.mark.asyncio
async def test_woken_projection_without_consumer_degrades() -> None:
    door = _lane_with_dormant(WOKEN)
    await door.handle(
        ModelLazyStartRequest(
            target_contract=WOKEN,
            correlation_id=uuid4(),
            topic=_topic(WOKEN),
            envelope={"payload": {}},
        )
    )
    count, status, detail = await _attachment(_projection(LIVE), _projection(WOKEN))
    assert (count, status) == (1, "DEGRADED")
    assert WOKEN in detail
