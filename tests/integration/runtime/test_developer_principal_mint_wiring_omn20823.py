# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot the real mint handler's wiring seam without contacting the broker."""

from __future__ import annotations

from pathlib import Path

import pytest

import omnibase_infra
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_MINT = "node_developer_principal_mint_orchestrator"


def _mint_manifest() -> ModelAutoWiringManifest:
    """Select the real installed entry point, as kernel discovery does at boot."""
    discover_contracts_cache_clear()
    discovered = discover_contracts()
    errors = [e for e in discovered.errors if e.entry_point_name == _MINT]
    assert not errors, f"Mint discovery failed: {errors}"
    contracts = tuple(c for c in discovered.contracts if c.name == _MINT)
    assert len(contracts) == 1, "The mint entry point and contract must be installed"
    assert omnibase_infra.__file__ is not None
    expected = Path(omnibase_infra.__file__).parent / "nodes" / _MINT / "contract.yaml"
    assert Path(contracts[0].contract_path).resolve() == expected.resolve()
    assert expected.is_file(), "Wiring must load the real on-disk mint contract"
    return ModelAutoWiringManifest(contracts=contracts, errors=())


@pytest.mark.asyncio
async def test_mint_real_manifest_wires_without_failures() -> None:
    """The actual handler must wire; zero failures with a skipped node is insufficient."""
    manifest = filter_manifest_for_runtime_profile(
        manifest=_mint_manifest(), runtime_profile="local-dev", environ={}
    ).manifest
    assert len(manifest.contracts) == 1
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        environment="local",
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract={},
        materialized_explicit_dependencies={},
        topology=None,
    )
    assert report.total_failed == 0, report.model_dump()
    assert report.total_wired == 1, report.model_dump()
    assert report.total_skipped == 0, report.model_dump()
    assert {r.contract_name for r in report.results} == {_MINT}


@pytest.mark.parametrize("profile", ["main", "effects"])
def test_deployed_profiles_do_not_own_the_mint_handler(profile: str) -> None:
    """Losing the local-dev ownership declaration must not expose broker minting."""
    ownership = filter_manifest_for_runtime_profile(
        manifest=_mint_manifest(), runtime_profile=profile, environ={}
    )
    assert not ownership.manifest.errors
    assert ownership.manifest.contracts == ()
