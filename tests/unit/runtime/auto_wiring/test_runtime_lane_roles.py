# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane-scoped contract attaches only where the lane's overlay admits it.

OMN-19747, plan task LO2. A contract names the lane roles it needs
(``runtime_lane_roles``), or, until OMN-19753 refuses it, the lane ids it may
attach on (``runtime_lanes``). The lane and its roles come from the
deployment's ``runtime.lane`` overlay; no lane is compiled into a package.
An excluded contract is listed in ``lane_excluded_contracts`` and is never a
discovery error.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_core.models.config_overlay import ModelRuntimeLaneDeclaration
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_role_requirement import (
    ModelRuntimeLaneRoleRequirement,
)
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    filter_manifest_for_runtime_profile,
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.health import runtime_lane_identity

pytestmark = pytest.mark.unit


def _lane(lane_id: str, *roles: str) -> ModelRuntimeLaneDeclaration:
    return ModelRuntimeLaneDeclaration.model_validate(
        {
            "schema_version": "runtime_lane.v1",
            "lane_id": lane_id,
            "roles": list(roles),
            "description": "test lane",
        }
    )


def _contract(
    name: str,
    *,
    roles: tuple[str, ...] | None = None,
    lanes: tuple[str, ...] | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_lane_roles=(
            None
            if roles is None
            else ModelRuntimeLaneRoleRequirement.model_validate({"roles": list(roles)})
        ),
        runtime_lanes=(
            None
            if lanes is None
            else ModelRuntimeLaneScope.model_validate({"lanes": list(lanes)})
        ),
    )


@pytest.fixture(autouse=True)
def _no_established_lane() -> Iterator[None]:
    runtime_lane_identity.clear_established_runtime_lane()
    yield
    runtime_lane_identity.clear_established_runtime_lane()


def test_a_lab_role_contract_attaches_on_a_lab_lane_and_is_excluded_elsewhere() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("lab_only", roles=("lab",)), _contract("anywhere"))
    )

    on_lab = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("new-lab", "lab")
    )
    on_dev = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("new-dev")
    )

    assert [c.name for c in on_lab.manifest.contracts] == ["lab_only", "anywhere"]
    assert on_lab.lane_excluded_contracts == ()
    assert [c.name for c in on_dev.manifest.contracts] == ["anywhere"]
    assert on_dev.lane_excluded_contracts == ("lab_only",)
    assert on_dev.manifest.errors == ()
    assert on_dev.runtime_lane == "new-dev"


def test_every_listed_role_is_required() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("lab_faults", roles=("lab", "fault_injection")),)
    )
    only_lab = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("x", "lab")
    )
    assert only_lab.lane_excluded_contracts == ("lab_faults",)


def test_the_transitional_lane_list_admits_by_lane_id() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("legacy", lanes=("compose-dev",)),)
    )
    kept = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("compose-dev", "lab")
    )
    excluded = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("another-lane")
    )
    assert [c.name for c in kept.manifest.contracts] == ["legacy"]
    assert excluded.lane_excluded_contracts == ("legacy",)
    assert excluded.manifest.errors == ()


def test_the_established_lane_is_the_default() -> None:
    runtime_lane_identity._established[:] = []
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("lab_only", roles=("lab",)),)
    )
    from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
    from omnibase_core.models.config_overlay import ModelConfigOverlayScope
    from omnibase_infra.config_overlay.models import ModelRuntimeLaneResolution

    runtime_lane_identity.establish_runtime_lane(
        ModelRuntimeLaneResolution(
            declaration=_lane("new-lab", "lab"),
            scope=ModelConfigOverlayScope(environment="local", lane="new-lab"),
            source=EnumConfigOverlaySource.LOCAL_HOME,
            location="/x/runtime.lane.json",
            sha256="0" * 64,
        )
    )
    result = filter_manifest_for_runtime_profile(manifest, "main")
    assert [c.name for c in result.manifest.contracts] == ["lab_only"]


def test_a_lane_scoped_contract_with_no_lane_known_is_refused_not_guessed() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("lab_only", roles=("lab",)),)
    )
    with pytest.raises(ProtocolConfigurationError, match="lab_only"):
        filter_manifest_for_runtime_profile(manifest, "main")


def test_unscoped_contracts_need_no_lane() -> None:
    manifest = ModelAutoWiringManifest(contracts=(_contract("anywhere"),))
    result = filter_manifest_for_runtime_profile(manifest, "main")
    assert [c.name for c in result.manifest.contracts] == ["anywhere"]
    assert result.runtime_lane is None


def test_the_raw_contract_path_applies_the_same_rule() -> None:
    raw = {"name": "lab_only", "runtime_lane_roles": ["lab"]}
    assert runtime_profile_owns_contract(raw, "main", lane=_lane("a", "lab")) is True
    assert runtime_profile_owns_contract(raw, "main", lane=_lane("a")) is False
    bad = {"name": "typo", "runtime_lane_roles": ["labb"]}
    assert runtime_profile_owns_contract(bad, "main", lane=_lane("a", "lab")) is False
    with pytest.raises(ProtocolConfigurationError):
        runtime_profile_owns_contract(raw, "main")


def test_discovery_parses_runtime_lane_roles(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text(
        "name: lab_only\n"
        "node_type: reducer\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "runtime_lane_roles: [lab]\n"
    )
    contract = _parse_contract(
        contract_path=path,
        entry_point_name="lab_only",
        package_name="pkg",
        package_version="0.0.0",
    )
    assert contract.runtime_lane_roles is not None
    assert [r.value for r in contract.runtime_lane_roles.roles] == ["lab"]


def test_discovery_refuses_an_unknown_role(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text(
        "name: typo\n"
        "node_type: reducer\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "runtime_lane_roles: [labb]\n"
    )
    with pytest.raises(ValueError, match="labb"):
        _parse_contract(
            contract_path=path,
            entry_point_name="typo",
            package_name="pkg",
            package_version="0.0.0",
        )


# --- the transitional runtime_lanes list, as OMN-19408 shipped it ------------

_LAB_PROJECTION_CONTRACT = """
name: projection_lab_lane_health
contract_version: {major: 1, minor: 0, patch: 0}
node_type: reducer
runtime_lanes:
  - compose-dev
  - onex-lab
  - onex-lab-k3s
""".strip()


def _discover(tmp_path: Path, body: str) -> ModelAutoWiringManifest:
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )

    contract_dir = tmp_path / "node"
    contract_dir.mkdir()
    path = contract_dir / "contract.yaml"
    path.write_text(body, encoding="utf-8")
    return discover_contracts_from_paths([path])


def test_the_lab_projection_stays_off_a_lane_not_in_its_list(
    tmp_path: Path,
) -> None:
    manifest = _discover(tmp_path, _LAB_PROJECTION_CONTRACT)
    assert manifest.total_errors == 0

    stability = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("no-role-lane")
    )
    dev = filter_manifest_for_runtime_profile(
        manifest, "main", lane=_lane("compose-dev", "lab")
    )

    assert stability.manifest.contracts == ()
    assert stability.lane_excluded_contracts == ("projection_lab_lane_health",)
    assert stability.manifest.total_errors == 0
    assert [c.name for c in dev.manifest.contracts] == ["projection_lab_lane_health"]


def test_a_profile_that_does_not_own_the_node_needs_no_lane(tmp_path: Path) -> None:
    manifest = _discover(tmp_path, _LAB_PROJECTION_CONTRACT)
    result = filter_manifest_for_runtime_profile(manifest, "effects")
    assert result.manifest.contracts == ()
    assert result.skipped_contracts == ("projection_lab_lane_health",)


def test_discovery_refuses_a_misspelt_lane_list_entry(tmp_path: Path) -> None:
    manifest = _discover(
        tmp_path, _LAB_PROJECTION_CONTRACT.replace("onex-lab-k3s", "onex-lab-k3z")
    )
    assert manifest.contracts == ()
    assert "onex-lab-k3z" in manifest.errors[0].error
