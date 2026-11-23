# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: a lazy start of a ``db_io`` contract wires with the boot's database topology.

The dev runtime of 2026-11-23T14:15:39Z booted with ``projection_metering_summary`` dormant. Its first
request (``onex.cmd.omnimarket.metering-summary-refresh-requested.v1``) started it through
``LazyStartRuntimeTransaction.build``, which called ``wire_from_manifest`` without the topology the kernel
loads at boot, and ``handler_wiring`` refused the contract: ``contract 'projection_metering_summary' declares
db_io but wire_from_manifest received no checked-in ModelDeploymentTopology``. The same refusal read 70 times
for ``projection_metering_summary`` and for eight more ``db_io`` contracts in the first twelve minutes of that
boot (``Lazy start: contract ... did not start (attempt 4 of 4)``).

``boot_manifest`` loads the topology the kernel loads (the ``ONEX_DATABASE_TOPOLOGY_PROFILE`` profile); the dormant
store keeps it beside the dormant contracts; the ingress hands it to the transaction; the transaction hands it to
``wire_from_manifest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers import handler_lazy_start_ingress as ingress_module
from omnibase_infra.handlers import lazy_start_runtime_transaction as transaction_module
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
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
)

pytestmark = pytest.mark.unit

DORMANT = "projection_fixture_db_io"
POLICY = ModelLazyStartPolicy(ingress_contract="node_ingress_fixture")


def _contract(tmp_path: Path, name: str) -> ModelDiscoveredContract:
    path = tmp_path / name / "contract.yaml"
    path.parent.mkdir()
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name=name,
        package_name="omnimarket",
    )


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


class _Report:
    total_wired = 0
    results: tuple[object, ...] = ()


@pytest.mark.asyncio
async def test_build_hands_the_boot_topology_to_wire_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def _wire(*_args: Any, **kwargs: Any) -> _Report:
        seen.update(kwargs)
        return _Report()

    monkeypatch.setattr(transaction_module, "wire_from_manifest", _wire)
    topology = MagicMock(name="deployment_topology")
    transaction = LazyStartRuntimeTransaction(
        {DORMANT: _contract(tmp_path, DORMANT)},
        cast("ProtocolLazyStartBus", MagicMock()),
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
        topology=topology,
    )

    with pytest.raises(RuntimeError, match="did not wire"):
        await transaction.build(
            DORMANT
        )  # the stub wires nothing; the call is the point

    assert seen.get("topology") is topology


def test_boot_manifest_keeps_the_boot_topology_with_the_dormant_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "omnibase_infra.handlers.lazy_start_boot_split.lazy_start_policy_of_this_lane",
        lambda _environ=None: POLICY,
    )
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract(tmp_path, "node_ingress_fixture"),
            _contract(tmp_path, DORMANT),
        )
    )
    topology = MagicMock(name="deployment_topology")

    boot = boot_manifest(manifest, None, topology=topology)

    assert [c.name for c in boot.contracts] == ["node_ingress_fixture"]
    assert DORMANT in StoreLazyStartDormant.contracts()
    assert StoreLazyStartDormant.topology() is topology


def test_boot_manifest_loads_the_topology_the_kernel_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "omnibase_infra.handlers.lazy_start_boot_split.lazy_start_policy_of_this_lane",
        lambda _environ=None: POLICY,
    )
    loaded = MagicMock(name="loaded_topology")
    profiles: list[str] = []

    def _load(profile: str) -> Any:
        profiles.append(profile)
        return loaded

    monkeypatch.setattr("omnibase_infra.topology.load_topology_profile", _load)
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract(tmp_path, "node_ingress_fixture"),
            _contract(tmp_path, DORMANT),
        )
    )

    boot_manifest(manifest, None, {"ONEX_DATABASE_TOPOLOGY_PROFILE": "dev-fixture"})

    assert profiles == ["dev-fixture"]
    assert StoreLazyStartDormant.topology() is loaded

    StoreLazyStartDormant.clear()
    boot_manifest(manifest, None, {})  # no profile: no topology, as the kernel has none
    assert StoreLazyStartDormant.topology() is None


@pytest.mark.asyncio
async def test_the_ingress_builds_its_transaction_with_the_recorded_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = MagicMock(name="deployment_topology")
    StoreLazyStartDormant.record(
        POLICY,
        (_contract(tmp_path, DORMANT),),
        None,
        (),
        topology,
    )
    built: dict[str, Any] = {}

    class _Capturing:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built["args"] = args
            built.update(kwargs)

    monkeypatch.setattr(ingress_module, "LazyStartRuntimeTransaction", _Capturing)
    monkeypatch.setattr(ingress_module, "install_publish_gate", lambda *_a, **_k: True)
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, MagicMock())

    HandlerLazyStartIngress(container)

    assert built.get("topology") is topology
