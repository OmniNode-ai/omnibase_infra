# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The graph effect activates only with complete, contract-bound resources."""

from __future__ import annotations

import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import omnibase_infra
from omnibase_core.container import ModelONEXContainer
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.runtime import execution_graph_runtime_composition as composition
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.execution_graph_runtime_composition import (
    GRAPH_READ_CONTRACT_NAME,
    GRAPH_READ_WORKFLOW_TYPE,
    open_execution_graph_runtime_executor,
    select_execution_graph_contract,
)
from omnibase_infra.runtime.execution_graph_terminal_publisher import (
    ExecutionGraphTerminalPublisher,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig
from omnibase_infra.runtime.runtime_host_process import (
    _discover_package_node_contracts,
    _wire_package_node_subscriptions,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
_TERMINAL_TOPIC = "onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _contract(
    name: str, command_topic: str = _COMMAND_TOPIC
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("contract.yaml"),
        entry_point_name=name,
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(command_topic,), publish_topics=(_TERMINAL_TOPIC,)
        ),
    )


def _config(tmp_path: Path) -> ModelRuntimeConfig:
    public_key = tmp_path / "gateway.pem"
    private_key = tmp_path / "terminal.pem"
    public_key.touch()
    private_key.touch()
    return ModelRuntimeConfig.model_validate(
        {
            "execution_graph_read_gateway": {
                "command_topic": _COMMAND_TOPIC,
                "runtime_id": "gateway",
                "realm": "lab",
                "bus_id": "lab-bus",
                "public_key_path": str(public_key),
            },
            "execution_graph_read": {
                "databases": {
                    "analytics_dsn": "postgresql://user:password@localhost/analytics",
                    "ledger_dsn": "postgresql://user:password@localhost/ledger",
                },
                "topology_version": {
                    "contract_version": {"major": 1, "minor": 3, "patch": 0},
                    "topology_sha256": "0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
                },
                "terminal_publisher": {
                    "terminal_topic": _TERMINAL_TOPIC,
                    "runtime_id": "graph-runtime",
                    "realm": "lab",
                    "bus_id": "lab-bus",
                    "workflow_type": GRAPH_READ_WORKFLOW_TYPE,
                },
                "private_key_path": str(private_key),
            },
        }
    )


@pytest.mark.unit
def test_disabled_graph_contract_is_removed_before_wiring() -> None:
    other = _contract("unrelated_effect")
    graph = _contract(GRAPH_READ_CONTRACT_NAME)
    manifest = ModelAutoWiringManifest(contracts=(other, graph))

    selected, graph_contract = select_execution_graph_contract(manifest, enabled=False)

    assert selected.contracts == (other,)
    assert graph_contract is None
    assert manifest.contracts == (other, graph)
    assert select_execution_graph_contract(manifest, enabled=True) == (manifest, graph)
    with pytest.raises(ValueError, match="not uniquely discovered"):
        select_execution_graph_contract(
            ModelAutoWiringManifest(contracts=(other,)), enabled=True
        )


@pytest.mark.unit
def test_graph_node_entry_point_and_contract_are_discoverable() -> None:
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module_name = project["project"]["entry-points"]["onex.nodes"][
        GRAPH_READ_CONTRACT_NAME
    ]
    module = import_module(module_name)
    assert module.__file__ is not None
    contract_path = Path(module.__file__).parent / "contract.yaml"

    manifest = discover_contracts_from_paths([contract_path])

    assert manifest.errors == ()
    assert len(manifest.contracts) == 1
    contract = manifest.contracts[0]
    assert contract.name == GRAPH_READ_CONTRACT_NAME
    assert contract.event_bus is not None
    assert contract.event_bus.subscribe_topics == (_COMMAND_TOPIC,)
    assert contract.event_bus.publish_topics == (_TERMINAL_TOPIC,)


@pytest.mark.unit
def test_graph_opt_in_requires_gateway_and_distinct_databases(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.execution_graph_read is not None
    assert config.execution_graph_read_gateway is not None
    raw = config.model_dump()
    raw["execution_graph_read_gateway"] = None
    with pytest.raises(ValidationError, match="requires execution_graph_read_gateway"):
        ModelRuntimeConfig.model_validate(raw)


@pytest.mark.unit
def test_handler_dependency_is_supplied_only_with_executor() -> None:
    assert _build_runtime_handler_dependencies(None) is None
    executor = object()
    container = ModelONEXContainer()
    dependencies = _build_runtime_handler_dependencies(
        None,
        execution_graph_read_executor=executor,
        execution_graph_read_container=container,
    )
    assert dependencies == {
        "HandlerExecutionGraphRead": {"container": container, "executor": executor}
    }
    with pytest.raises(ValueError, match="requires a runtime container"):
        _build_runtime_handler_dependencies(
            None, execution_graph_read_executor=executor
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_package_scan_does_not_subscribe_disabled_graph() -> None:
    package_root = Path(omnibase_infra.__file__).parent
    graph_contracts = [
        contract
        for contract in _discover_package_node_contracts(package_root)
        if contract["name"] == GRAPH_READ_CONTRACT_NAME
    ]
    assert len(graph_contracts) == 1
    wiring = MagicMock(spec=EventBusSubcontractWiring)
    wiring.wire_subscriptions = AsyncMock(
        spec=EventBusSubcontractWiring.wire_subscriptions
    )

    wired, _, skipped = await _wire_package_node_subscriptions(
        graph_contracts,
        wiring,
        set(),
        runtime_profile="effects",
    )
    assert wired == 0
    assert skipped == 1
    wiring.wire_subscriptions.assert_not_called()

    # Even an enabled graph uses kernel auto-wiring, never this second path.
    assert wiring.wire_subscriptions.await_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boot_composition_refuses_contract_topic_mismatch_before_pool_open(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert config.execution_graph_read is not None
    assert config.execution_graph_read_gateway is not None
    wrong_contract = _contract(GRAPH_READ_CONTRACT_NAME, "wrong.command")
    with pytest.raises(ValueError, match="gateway command differs"):
        async with open_execution_graph_runtime_executor(
            config=config.execution_graph_read,
            gateway=config.execution_graph_read_gateway,
            contract=wrong_contract,
            event_bus=AsyncMock(spec=ProtocolEventBusLike),
        ):
            pytest.fail("mismatched graph contract was accepted")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boot_composition_supplies_two_pools_fold_and_signed_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    assert config.execution_graph_read is not None
    assert config.execution_graph_read_gateway is not None
    analytics_pool, ledger_pool = object(), object()
    closed: list[bool] = []

    @asynccontextmanager
    async def pools(_: object) -> AsyncIterator[tuple[object, object]]:
        try:
            yield analytics_pool, ledger_pool
        finally:
            closed.append(True)

    topology = object()
    resolver = MagicMock(spec=PackagedExecutionGraphTopologyContract)
    resolver.resolve.return_value = topology
    compose = MagicMock(
        spec=composition.compose_execution_graph_read_executor,
        return_value=object(),
    )
    publisher = MagicMock(spec=ExecutionGraphTerminalPublisher)
    publisher.publish = AsyncMock(spec=ExecutionGraphTerminalPublisher.publish)
    publisher_factory = MagicMock(
        spec=ExecutionGraphTerminalPublisher,
        return_value=publisher,
    )
    monkeypatch.setattr(composition, "open_execution_graph_read_pools", pools)
    monkeypatch.setattr(
        composition, "PackagedExecutionGraphTopologyContract", lambda: resolver
    )
    monkeypatch.setattr(
        composition,
        "load_private_key_from_pem",
        MagicMock(spec=composition.load_private_key_from_pem),
    )
    monkeypatch.setattr(
        composition, "ExecutionGraphTerminalPublisher", publisher_factory
    )
    monkeypatch.setattr(composition, "compose_execution_graph_read_executor", compose)

    async with open_execution_graph_runtime_executor(
        config=config.execution_graph_read,
        gateway=config.execution_graph_read_gateway,
        contract=_contract(GRAPH_READ_CONTRACT_NAME),
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
    ) as executor:
        assert executor is compose.return_value
        kwargs = compose.call_args.kwargs
        assert kwargs["analytics_pool"] is analytics_pool
        assert kwargs["ledger_pool"] is ledger_pool
        assert kwargs["topology"] is topology
        assert kwargs["workflow_type"] == GRAPH_READ_WORKFLOW_TYPE
        assert kwargs["publish_terminal"] is publisher.publish
    assert closed == [True]
