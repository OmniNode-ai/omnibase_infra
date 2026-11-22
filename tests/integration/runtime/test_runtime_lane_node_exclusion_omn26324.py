# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: dev wires only the contracts it runs, as a deployed runtime reads it (OMN-26324).

A deployed runtime does not hand the ownership filter an environment mapping: it
reads ``ONEX_RUNTIME_LANE`` and ``ONEX_CONTRACTS_DIR`` from its own process
environment. This drives the auto-wiring ownership filter that way, against the
repository's real ``contracts/`` directory, and asserts that the dev lane leaves
each declared contract unattached while it attaches an unrelated one, and that
every other registered lane attaches all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_infra.models.runtime.model_runtime_lane_node_exclusion import (
    ENV_CONTRACTS_DIR,
    ENV_RUNTIME_LANE,
    LANE_PROFILE_PREFIX,
    ModelRuntimeLaneNodeExclusion,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "contracts"
DEV_LANE = f"{LANE_PROFILE_PREFIX}dev"
UNRELATED = "node_unrelated_to_the_dev_profile_omn26324"


def _declared_exclusion() -> tuple[str, ...]:
    declaration = ModelRuntimeLaneNodeExclusion.load(
        ModelRuntimeLaneNodeExclusion.declaration_path(CONTRACTS_DIR)
    )
    return declaration.entry_for(DEV_LANE).excluded_contracts


def _manifest(names: tuple[str, ...]) -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(
        contracts=tuple(
            ModelDiscoveredContract(
                name=name,
                node_type="COMPUTE_GENERIC",
                contract_version=ModelContractVersion(major=1, minor=0, patch=0),
                contract_path=REPO_ROOT / "src" / name / "contract.yaml",
                entry_point_name=name,
                package_name="omnibase_infra",
            )
            for name in names
        )
    )


def _wired(names: tuple[str, ...]) -> set[str]:
    # No environ argument: the filter reads the process environment, as deployed.
    result = filter_manifest_for_runtime_profile(_manifest(names), "main")
    return {contract.name for contract in result.manifest.contracts}


def test_dev_lane_process_environment_leaves_declared_contracts_unattached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    excluded = _declared_exclusion()
    assert excluded, "the dev lane declares no excluded contract"
    kept = (UNRELATED,)
    monkeypatch.setenv(ENV_RUNTIME_LANE, DEV_LANE)
    monkeypatch.setenv(ENV_CONTRACTS_DIR, str(CONTRACTS_DIR))

    wired = _wired((*excluded, *kept))

    assert wired.isdisjoint(excluded)
    assert wired == set(kept)


@pytest.mark.parametrize("lane", sorted(REGISTERED_RUNTIME_LANES - {DEV_LANE}))
def test_other_lane_process_environment_attaches_every_declared_contract(
    monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    excluded = _declared_exclusion()
    monkeypatch.setenv(ENV_RUNTIME_LANE, lane)
    monkeypatch.setenv(ENV_CONTRACTS_DIR, str(CONTRACTS_DIR))

    assert set(excluded) <= _wired(excluded)
