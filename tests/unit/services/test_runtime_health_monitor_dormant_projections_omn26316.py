# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: a projection left dormant by lazy start is not an unattached one.

Measured on the dev runtime (omni-201, container started 2026-11-24T23:02:25Z):
``/health`` read ``projection_attachment`` DEGRADED, "39/95 declared projection(s)
have no attached consumer", and the semantic healthcheck failed on it, so docker
read the container unhealthy. The lane runs lazy start (OMN-21592): 111 contracts
wired at boot and 1179 left dormant until their first request. Read from the
container, 37 of the 38 projections without a consumer group were dormant ones:
none was in the lane's boot set, none had a retained record on its topic, and the
front door had neither woken nor failed to wake any of them. Being unattached is
their documented state until a first message or request reaches them.

The monitor read the manifest of every contract the profile owns, dormant ones
included, and judged each projection against the live bus registry alone. It named
the first 8 (``MAX_NAMED_PROJECTIONS``) and the rest read ``+31 more``, so the other
projections were never named anywhere the operator reads.

RED at the merge base: a dormant projection reads unattached, and a dimension that
lists more than 8 projections names only 8 of them anywhere.
"""

from __future__ import annotations

import logging
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
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
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

pytestmark = pytest.mark.unit

_MONITOR = "omnibase_infra.services.service_runtime_health_monitor"
LIVE = "projection_live_attached"
EAGER = "projection_eager_unattached"
WAITING = "projection_dormant_waiting"
WOKEN = "projection_dormant_woken_unattached"
FAILING = "projection_dormant_wake_failed"
INGRESS = "node_dormant_ingress"


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
    """Builds every contract but ``FAILING``, whose build always raises."""

    async def build(self, contract_name: str) -> None:
        if contract_name == FAILING:
            raise RuntimeError(f"contract '{FAILING}' did not wire: fixture")

    async def attach(self, contract_name: str) -> None:
        return None

    async def rollback(self, contract_name: str) -> None:
        return None

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        return {"handled_by": contract_name}


def _lane_with_dormant(*names: str) -> LazyStartFrontDoor:
    """The lazy-start lane's boot: ``names`` left dormant, the front door recorded."""
    StoreLazyStartDormant.clear()
    dormant = {name: _projection(name) for name in names}
    policy = ModelLazyStartPolicy(
        ingress_contract=INGRESS, boot_contracts=(), budget_seconds=5, max_retries=0
    )
    door = LazyStartFrontDoor(_Transaction(), dormant, policy)
    StoreLazyStartDormant.record(policy, tuple(dormant.values()))
    StoreLazyStartDormant.record_front_door(door)
    return door


@pytest.fixture(autouse=True)
def _clean_store() -> Iterator[None]:
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


async def _health(*contracts: ModelDiscoveredContract) -> ModelRuntimeHealthCheckEvent:
    """One health check; the live bus registry holds only ``LIVE``'s consumer."""
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
        return await monitor.run_once()


def _attachment(event: ModelRuntimeHealthCheckEvent) -> tuple[str, str]:
    dim = next(d for d in event.dimensions if d.name == "projection_attachment")
    return dim.status, dim.detail


@pytest.mark.asyncio
async def test_dormant_projection_awaiting_its_first_request_is_not_unattached() -> (
    None
):
    _lane_with_dormant(WAITING)

    event = await _health(_projection(LIVE), _projection(WAITING))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 0
    assert status == "HEALTHY"
    assert WAITING in detail
    assert "dormant" in detail


@pytest.mark.asyncio
async def test_a_dormant_projection_that_has_a_consumer_is_attached_not_dormant() -> (
    None
):
    """Dormant is a reading of the unattached ones only: a consumer decides first."""
    _lane_with_dormant(LIVE)

    event = await _health(_projection(LIVE))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 0
    assert status == "HEALTHY"
    assert "dormant" not in detail
    assert detail.startswith("All 1 ")


@pytest.mark.asyncio
async def test_without_lazy_start_the_same_projection_is_unattached() -> None:
    """Control: the exemption is the lazy-start state, not a blanket one."""
    event = await _health(_projection(LIVE), _projection(WAITING))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 1
    assert status == "DEGRADED"
    assert WAITING in detail


@pytest.mark.asyncio
async def test_an_eager_unattached_projection_stays_reported_beside_a_dormant_one() -> (
    None
):
    _lane_with_dormant(WAITING)

    event = await _health(_projection(LIVE), _projection(WAITING), _projection(EAGER))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 1
    assert status == "DEGRADED"
    assert detail.startswith("1/")
    assert EAGER in detail


@pytest.mark.asyncio
async def test_a_woken_projection_is_judged_by_the_live_registry_again() -> None:
    """The wake completed but registered no consumer: the monitor must report it."""
    door = _lane_with_dormant(WOKEN)
    await door.handle(
        ModelLazyStartRequest(
            target_contract=WOKEN,
            correlation_id=uuid4(),
            topic=_topic(WOKEN),
            envelope={"payload": {}},
        )
    )
    assert WOKEN in door.serving

    event = await _health(_projection(LIVE), _projection(WOKEN))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 1
    assert status == "DEGRADED"
    assert WOKEN in detail


@pytest.mark.asyncio
async def test_a_projection_whose_wake_failed_stays_reported() -> None:
    door = _lane_with_dormant(FAILING)
    await door.wake_for_topic(_topic(FAILING))
    assert FAILING in door.wake_failures

    event = await _health(_projection(LIVE), _projection(FAILING))

    status, detail = _attachment(event)
    assert event.unattached_projection_count == 1
    assert status == "DEGRADED"
    assert FAILING in detail


@pytest.mark.asyncio
async def test_every_unattached_projection_is_named_in_the_log_past_the_detail_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The detail names 8 and counts the rest; the log must name all of them."""
    names = [f"projection_eager_{index:02d}" for index in range(12)]

    with caplog.at_level(logging.INFO):
        event = await _health(_projection(LIVE), *(_projection(n) for n in names))

    _, detail = _attachment(event)
    assert "+4 more" in detail
    logged = " ".join(
        r.getMessage() for r in caplog.records if "unattached" in r.getMessage()
    )
    for name in names:
        assert name in logged, f"{name} is named nowhere the operator reads"


@pytest.mark.asyncio
async def test_every_dormant_projection_is_named_in_the_log_past_the_detail_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    names = [f"projection_dormant_{index:02d}" for index in range(12)]
    _lane_with_dormant(*names)

    with caplog.at_level(logging.INFO):
        event = await _health(_projection(LIVE), *(_projection(n) for n in names))

    status, detail = _attachment(event)
    assert status == "HEALTHY"
    assert "12 dormant" in detail
    logged = " ".join(
        r.getMessage() for r in caplog.records if "dormant" in r.getMessage()
    )
    for name in names:
        assert name in logged, f"{name} is named nowhere the operator reads"
