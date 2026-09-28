# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed disposable graph/ledger selection, without runtime startup."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.runtime.auto_wiring.graph_ledger_node_selection import (
    select_graph_ledger_manifest,
    validate_graph_ledger_boot,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.models.model_graph_ledger_node_allowlist import (
    GRAPH_LEDGER_NODES,
    ModelGraphLedgerNodeAllowlist,
)
from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

pytestmark = pytest.mark.unit


def selection() -> ModelGraphLedgerNodeAllowlist:
    return ModelGraphLedgerNodeAllowlist.model_validate(
        {"runtime_lane": "sim-202", "nodes": list(GRAPH_LEDGER_NODES)}
    )


def contract(name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract.model_validate(
        {
            "name": name,
            "node_type": "EFFECT_GENERIC",
            "contract_version": {"major": 1, "minor": 0, "patch": 0},
            "contract_path": Path("contract.yaml"),
            "entry_point_name": name,
            "package_name": "omnibase_infra",
        }
    )


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [GRAPH_LEDGER_NODES[0]],
        [*GRAPH_LEDGER_NODES, "node_delegation_orchestrator"],
        [*GRAPH_LEDGER_NODES, GRAPH_LEDGER_NODES[0]],
    ],
)
def test_selection_rejects_missing_extra_and_duplicate_nodes(nodes: list[str]) -> None:
    with pytest.raises(ValidationError):
        ModelGraphLedgerNodeAllowlist.model_validate(
            {"runtime_lane": "sim-202", "nodes": nodes}
        )


def test_absent_selection_preserves_normal_config() -> None:
    assert ModelRuntimeConfig().graph_ledger_node_allowlist is None


def test_selection_precedes_registration_and_requires_every_node() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=tuple(
            contract(n) for n in (*GRAPH_LEDGER_NODES, "node_delegation_orchestrator")
        )
    )
    selected = select_graph_ledger_manifest(manifest, selection())
    assert {c.name for c in selected.contracts} == set(GRAPH_LEDGER_NODES)
    with pytest.raises(ValueError, match="exactly once"):
        select_graph_ledger_manifest(
            ModelAutoWiringManifest(contracts=manifest.contracts[1:]), selection()
        )


@pytest.mark.parametrize(
    ("lane", "profile"), [("compose-dev", "main"), ("", "main"), ("sim-202", "workers")]
)
def test_boot_refuses_wrong_lane_or_profile(lane: str, profile: str) -> None:
    with pytest.raises(ValueError):
        validate_graph_ledger_boot(selection(), profile, {"ONEX_RUNTIME_LANE": lane})


def test_boot_accepts_only_explicit_disposable_profiles() -> None:
    for profile in ("main", "effects"):
        validate_graph_ledger_boot(
            selection(), profile, {"ONEX_RUNTIME_LANE": "sim-202"}
        )


@pytest.mark.asyncio
async def test_excluded_descriptor_never_imports_or_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text("name: node_delegation_orchestrator\n")
    descriptor = ModelHandlerDescriptor.model_validate(
        {
            "handler_id": "forbidden.handler",
            "name": "Forbidden handler",
            "version": "1.0.0",
            "handler_kind": "effect",
            "input_model": "test.models.Input",
            "output_model": "test.models.Output",
            "handler_class": "forbidden.side_effect.Handler",
            "contract_path": str(path),
        }
    )
    host = object.__new__(RuntimeHostProcess)
    host._graph_ledger_node_allowlist = selection()
    host._handler_descriptors = {}
    registry = MagicMock()
    monkeypatch.setattr(
        host, "_resolve_handler_descriptors", AsyncMock(return_value=[descriptor])
    )
    monkeypatch.setattr(host, "_get_handler_registry", AsyncMock(return_value=registry))
    await host._discover_or_wire_handlers()
    registry.register.assert_not_called()
    assert host._handler_descriptors == {}


@pytest.mark.asyncio
async def test_all_legacy_subscription_paths_are_closed() -> None:
    host = object.__new__(RuntimeHostProcess)
    host._graph_ledger_node_allowlist = selection()
    await host._wire_event_bus_subscriptions()
    await host._wire_package_node_subscriptions()
    await host._wire_baseline_subscriptions()
    await host._start_dynamic_contract_listener()


def test_real_contract_profile_ownership_keeps_exact_two_per_runtime() -> None:
    from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
    from omnibase_infra.runtime.auto_wiring.profile_ownership import (
        filter_manifest_for_runtime_profile,
    )

    root = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"
    names = (*GRAPH_LEDGER_NODES, "node_runtime_error_triage_effect")
    manifest = ModelAutoWiringManifest(
        contracts=tuple(
            _parse_contract(
                contract_path=root / n / "contract.yaml",
                entry_point_name=n,
                package_name="omnibase_infra",
                package_version="0.38.60",
            )
            for n in names
        )
    )
    selected = select_graph_ledger_manifest(manifest, selection())
    for profile, expected in (
        ("main", set(GRAPH_LEDGER_NODES[:2])),
        ("effects", set(GRAPH_LEDGER_NODES[2:])),
    ):
        owned = filter_manifest_for_runtime_profile(
            selected, profile, environ={"ONEX_RUNTIME_LANE": "sim-202"}
        )
        assert {c.name for c in owned.manifest.contracts} == expected
        assert not owned.manifest.errors
