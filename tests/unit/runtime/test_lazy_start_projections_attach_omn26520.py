# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26520: a lazy-start lane boots its declared projections, and a wake wires db_io.

The dev runtime of 2026-11-23 (infra b550268717, boot at 15:08:28Z) read
``unhealthy`` in Docker with ``projection_attachment: 75/98 declared
projection(s) have no attached consumer``. The recorded boot line was
``Lazy start: 104 contract(s) wired at boot, 1173 dormant until their first
request`` and 77 of the 103 declared projections were among the dormant.

Two causes, both pinned here:

1. A projection's job is to consume its topic continuously. The wake-on-publish
   gate only fires for a message this process publishes, so a projection fed by
   another process (``projection_done_items``: consumer group with 0 members, lag
   224 growing to 243 over 20 minutes) is never woken. A lazy-start lane keeps
   every declared projection in the boot manifest.
2. A projection that is woken wired through ``wire_from_manifest`` with no
   ``topology``, so every ``db_io`` contract refused with ``declares db_io but
   wire_from_manifest received no checked-in ModelDeploymentTopology`` (12 of
   the 13 contracts whose wake failed on that boot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers import lazy_start_runtime_transaction as transaction_module
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    split_boot_manifest,
)
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.topology import load_topology_profile

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omnimarket.work-ledger-terminal.v1"
POLICY = ModelLazyStartPolicy(ingress_contract="node_lazy_start_ingress")


def _contract(
    name: str,
    *,
    db_io: bool,
    topic: str | None = TOPIC,
    plugin_managed: bool = False,
    requires_cloud_gateway: bool = False,
) -> ModelDiscoveredContract:
    tables = (
        ModelDbOwnershipSubcontract(
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
        )
        if db_io
        else None
    )
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(__file__),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,) if topic else (),
            publish_topics=(),
            plugin_managed=plugin_managed,
        ),
        db_io=tables,
        requires_cloud_gateway=requires_cloud_gateway,
    )


def _split(
    *contracts: ModelDiscoveredContract,
) -> tuple[set[str], set[str]]:
    boot, dormant = split_boot_manifest(
        ModelAutoWiringManifest(contracts=contracts), POLICY
    )
    return {c.name for c in boot.contracts}, {c.name for c in dormant}


def test_a_declared_projection_is_booted_not_left_dormant() -> None:
    """Fails at origin/dev: the projection lands in the dormant set."""
    boot, dormant = _split(
        _contract("node_lazy_start_ingress", db_io=False, topic=None),
        _contract("projection_done_items", db_io=True),
    )
    assert "projection_done_items" in boot
    assert "projection_done_items" not in dormant


def test_a_node_that_is_not_a_projection_stays_dormant() -> None:
    """Positive control: the lazy split still defers what it was built to defer."""
    boot, dormant = _split(
        _contract("node_lazy_start_ingress", db_io=False, topic=None),
        _contract("node_request_driven_effect", db_io=False),
    )
    assert "node_request_driven_effect" in dormant
    assert "node_request_driven_effect" not in boot


def test_projections_the_health_surface_excludes_stay_dormant() -> None:
    """A plugin-owned or cloud-gateway projection is out of scope of the health
    dimension, so keeping it dormant cannot hide an unattached consumer."""
    boot, dormant = _split(
        _contract("node_lazy_start_ingress", db_io=False, topic=None),
        _contract("projection_plugin_owned", db_io=True, plugin_managed=True),
        _contract("projection_cloud_only", db_io=True, requires_cloud_gateway=True),
    )
    assert {"projection_plugin_owned", "projection_cloud_only"} <= dormant
    assert not {"projection_plugin_owned", "projection_cloud_only"} & boot


class _Bus:
    environment = "local"

    async def publish(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


def _transaction(
    contract: ModelDiscoveredContract, topology: ModelDeploymentTopology | None
) -> LazyStartRuntimeTransaction:
    return LazyStartRuntimeTransaction(
        {contract.name: contract},
        cast("ProtocolLazyStartBus", _Bus()),
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
        topology,
    )


async def _wiring_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    contract: ModelDiscoveredContract,
    topology: ModelDeploymentTopology | None,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def _wire(*args: Any, **kwargs: Any) -> object:
        seen.update(kwargs)
        raise RuntimeError("stop after the wiring call")

    monkeypatch.setattr(transaction_module, "wire_from_manifest", _wire)
    with pytest.raises(RuntimeError, match="stop after the wiring call"):
        await _transaction(contract, topology).build(contract.name)
    return seen


@pytest.mark.asyncio
async def test_a_wake_wires_a_db_io_contract_with_the_database_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails at origin/dev: build() passes no topology, so a woken db_io contract
    refuses with 'received no checked-in ModelDeploymentTopology'."""
    topology = load_topology_profile("local")
    contract = _contract("projection_lab_lane_health", db_io=True)
    seen = await _wiring_kwargs(monkeypatch, contract, topology)
    assert seen["topology"] == topology


@pytest.mark.asyncio
async def test_a_runtime_with_no_topology_wires_with_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a boot with no topology wakes with none, as the kernel wired."""
    contract = _contract("node_request_driven_effect", db_io=False)
    seen = await _wiring_kwargs(monkeypatch, contract, None)
    assert seen["topology"] is None


def test_the_ingress_front_door_wakes_with_the_boot_database_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails at origin/dev: the ingress builds its transaction with no topology."""
    dormant = _contract("projection_lab_lane_health", db_io=True)
    StoreLazyStartDormant.record(POLICY, (dormant,), None, ())
    StoreLazyStartDormant.record_topology(load_topology_profile("local"))
    container = cast(
        "ModelONEXContainer",
        type("_Container", (), {"get_service": lambda self, key: _Bus()})(),
    )
    front_door = HandlerLazyStartIngress(container)._front_door_of_this_boot()
    assert front_door is not None
    transaction = vars(front_door)["_transaction"]
    assert transaction._topology == load_topology_profile("local")
