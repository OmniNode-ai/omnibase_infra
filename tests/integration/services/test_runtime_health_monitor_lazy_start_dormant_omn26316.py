# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The health monitor does not expect a lazy-start lane's dormant contracts attached (OMN-26316).

On a lane whose hot-load row is ``on_demand`` with a ``lazy_start`` block, the
kernel wires only the boot set and leaves every other contract dormant: its
topics are never subscribed until a first request wakes it
(``lazy_start_boot_split``). The health monitor discovers the FULL manifest and
selected every ``db_io`` projection in it as one the runtime "must have
attached", so each dormant projection read as unattached. Measured on the dev
lane on 2026-11-23: ``projection_attachment`` DEGRADED, 73/98 projections, from
17:08:41Z, with 202 first-request starts of 26 dormant contracts running
normally.

The scope is built from the manifest alone, so it needs the boot split's own
record of what it held dormant. A dormant contract that a first request has
woken has its topics on the live bus registry and stays in scope.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
    dormant_unattached_topics,
)
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.observability import reset_consumer_flow_counters
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
)

INGRESS = "node_lazy_start_ingress"
INGRESS_TOPIC = "onex.cmd.platform.lazy-start-ingress.v1"
PROJECTIONS = {
    "node_projection_alpha": "onex.evt.omnimarket.alpha-recorded.v1",
    "node_projection_beta": "onex.evt.omnimarket.beta-recorded.v1",
}


def _contract(
    name: str, topic: str, *, projection: bool = True
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER" if projection else "ORCHESTRATOR",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        db_io=(
            ModelDbOwnershipSubcontract(
                db_tables=[
                    ModelDbTableDeclaration(
                        name=f"{name}_table",
                        database_ref="application",
                        schema="omninode_internal",
                        migration="0001_init.sql",
                        access="read_write",
                        role="projection_target",
                    )
                ]
            )
            if projection
            else None
        ),
    )


def _full_manifest() -> ModelAutoWiringManifest:
    """The ingress (boot set) plus two projections the boot split leaves dormant."""
    return ModelAutoWiringManifest(
        contracts=(
            _contract(INGRESS, INGRESS_TOPIC, projection=False),
            *(_contract(name, topic) for name, topic in PROJECTIONS.items()),
        ),
        errors=(),
    )


@pytest.fixture
def lazy_start_lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """The lane's hot-load row: on_demand, boot set = the ingress only."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n"
        "    lazy_start:\n"
        f"      ingress_contract: {INGRESS}\n"
        f"      boot_contracts: [{INGRESS}]\n"
        "      budget_seconds: 30\n      max_retries: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


async def _run_monitor(
    manifest: ModelAutoWiringManifest, live_topics: dict[str, str]
) -> ModelRuntimeHealthCheckEvent:
    """One check over *manifest*, with *live_topics* (topic -> group) subscribed."""
    live = {
        (topic, group): f"{group}.__t.{topic}" for topic, group in live_topics.items()
    }
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.get_consumer_groups.return_value = live
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus, bootstrap_servers="localhost:9092"
    )
    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
            return_value=[
                ConsumerGroupSnapshot(group_id=group_id, state="STABLE")
                for group_id in live.values()
            ],
        ),
    ):
        return await monitor.run_once()


def _attachment(event: ModelRuntimeHealthCheckEvent) -> ModelRuntimeHealthDimension:
    return next(d for d in event.dimensions if d.name == "projection_attachment")


@pytest.fixture(autouse=True)
def _counters() -> Iterator[None]:
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


@pytest.mark.asyncio
async def test_a_dormant_projection_does_not_degrade_projection_attachment(
    lazy_start_lane: None,
) -> None:
    """The dev lane's reading: dormant projections are not unattached, they are dormant."""
    full = _full_manifest()
    boot = boot_manifest(full)
    assert [c.name for c in boot.contracts] == [INGRESS]
    assert set(StoreLazyStartDormant.contracts()) == set(PROJECTIONS)

    event = await _run_monitor(full, {INGRESS_TOPIC: "runtime.ingress.consume.v1"})

    attachment = _attachment(event)
    assert attachment.status == "HEALTHY", attachment.detail
    assert event.unattached_projection_count == 0


@pytest.mark.asyncio
async def test_without_a_lazy_start_boot_the_same_projections_are_unattached(
    lazy_start_lane: None,
) -> None:
    """Positive control: the same manifest and registry, no dormant record, is DEGRADED."""
    event = await _run_monitor(
        _full_manifest(), {INGRESS_TOPIC: "runtime.ingress.consume.v1"}
    )

    attachment = _attachment(event)
    assert attachment.status == "DEGRADED"
    assert event.unattached_projection_count == len(PROJECTIONS)


@pytest.mark.asyncio
async def test_a_woken_projection_with_a_dropped_subscription_is_still_reported(
    lazy_start_lane: None,
) -> None:
    """A dormant contract a first request woke is back in scope: its registry decides.

    ``beta`` has two topics; one is on the live registry (the wake attached it)
    and one is not. That is a real finding, not dormancy.
    """
    second_topic = "onex.evt.omnimarket.beta-second.v1"
    beta = _contract(
        "node_projection_beta", PROJECTIONS["node_projection_beta"]
    ).model_copy(
        update={
            "event_bus": ModelEventBusWiring(
                subscribe_topics=(PROJECTIONS["node_projection_beta"], second_topic),
                publish_topics=(),
            )
        }
    )
    full = ModelAutoWiringManifest(
        contracts=(
            _contract(INGRESS, INGRESS_TOPIC, projection=False),
            _contract("node_projection_alpha", PROJECTIONS["node_projection_alpha"]),
            beta,
        ),
        errors=(),
    )
    boot_manifest(full)

    event = await _run_monitor(
        full,
        {
            INGRESS_TOPIC: "runtime.ingress.consume.v1",
            PROJECTIONS["node_projection_beta"]: "runtime.beta.consume.v1",
        },
    )

    assert event.unattached_projection_count == 1
    assert "node_projection_beta" in _attachment(event).detail
    assert "node_projection_alpha" not in _attachment(event).detail


def test_dormant_topics_are_the_dormant_contracts_topics_less_the_attached_ones(
    lazy_start_lane: None,
) -> None:
    assert dormant_unattached_topics(frozenset()) == frozenset()  # nothing recorded yet

    boot_manifest(_full_manifest())
    every_projection_topic = frozenset(PROJECTIONS.values())

    assert dormant_unattached_topics(frozenset()) == every_projection_topic
    # The boot set's own topic is never dormant, and an attached one is left out.
    assert dormant_unattached_topics(frozenset({INGRESS_TOPIC})) == (
        every_projection_topic
    )
    woken = PROJECTIONS["node_projection_alpha"]
    assert dormant_unattached_topics(frozenset({woken})) == (
        every_projection_topic - {woken}
    )


def test_a_lane_with_no_lazy_start_block_has_no_dormant_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONEX_RUNTIME_LANE", raising=False)
    StoreLazyStartDormant.clear()

    assert boot_manifest(_full_manifest()) == _full_manifest()
    assert dormant_unattached_topics(frozenset()) == frozenset()
