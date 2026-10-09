# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19812: fails before the fix because effects has no lane and records an error."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_runtime_lane_role import EnumRuntimeLaneRole
from omnibase_core.models.config_overlay import ModelRuntimeLaneDeclaration
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    filter_manifest_for_runtime_profile,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_NAME = "branch_claim_check_effect"
LAB_SCOPE = ("compose-dev", "onex-lab", "onex-lab-k3s")

# The shipped omnimarket#3547 header, parsed by the kernel's real discovery path.
CONTRACT = """\
name: branch_claim_check_effect
node_type: effect
contract_version: {major: 1, minor: 0, patch: 0}
node_version: {major: 1, minor: 0, patch: 0}
runtime_lanes: [compose-dev, onex-lab, onex-lab-k3s]
descriptor:
  runtime_profiles: [effects]
"""


def _lane_declaration(lane: str) -> ModelRuntimeLaneDeclaration:
    """The runtime.lane overlay document the deployment supplies for ``lane``."""
    return ModelRuntimeLaneDeclaration(
        schema_version="runtime_lane.v1",
        lane_id=lane,
        roles=(EnumRuntimeLaneRole.LAB,) if lane in LAB_SCOPE else (),
        description=f"{lane} runtime lane",
    )


def _render_environments(files: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Resolve YAML anchors and merge environment keys in deployment order."""
    environments: dict[str, dict[str, str]] = {}
    for filename in files:
        text = (REPO_ROOT / "docker" / filename).read_text(encoding="utf-8")
        # SafeLoader resolves << / !!merge; !override is a compose-only tag.
        document = yaml.safe_load(text.replace("!override", ""))
        for name in ("runtime-effects", "omninode-runtime"):
            environment = document["services"].get(name, {}).get("environment", {})
            assert isinstance(environment, dict)
            environments.setdefault(name, {}).update(
                {str(key): str(value) for key, value in environment.items()}
            )
    for environment in environments.values():
        if "ONEX_RUNTIME_LANE" in environment:
            environment["ONEX_RUNTIME_LANE"] = environment["ONEX_RUNTIME_LANE"].replace(
                "${ONEX_PREPR_SLOT}", "1"
            )
    return environments


@pytest.mark.parametrize(
    ("overlay", "lane", "attaches"),
    [(None, "compose-dev", True), ("docker-compose.prepr.yml", "prepr-1", False)],
    ids=["dev", "prepr-1"],
)
def test_lane_scoped_effects_contract_uses_rendered_placement(
    tmp_path: Path, overlay: str | None, lane: str, attaches: bool
) -> None:
    contract_dir = tmp_path / "node_branch_claim_check_effect"
    contract_dir.mkdir()
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text(CONTRACT, encoding="utf-8")
    parsed = discover_contracts_from_paths([contract_path])
    assert not parsed.errors
    assert len(parsed.contracts) == 1
    contract = parsed.contracts[0]
    assert contract.name == CONTRACT_NAME
    assert contract.runtime_profiles == ("effects",)
    assert contract.runtime_lanes is not None
    assert contract.runtime_lanes.lanes == LAB_SCOPE
    assert contract.contract_version.model_dump() == {
        "major": 1,
        "minor": 0,
        "patch": 0,
    }
    assert contract.node_version == "1.0.0"

    repo_contracts = sorted(
        (REPO_ROOT / "src/omnibase_infra/nodes").glob("*/contract.yaml")
    )
    assert repo_contracts, "positive control: the real repository manifest is present"
    manifest = discover_contracts_from_paths([*repo_contracts, contract_path])
    assert len(manifest.contracts) > 1
    files: tuple[str, ...] = ("docker-compose.infra.yml", "docker-compose.dev-lane.yml")
    if overlay is not None:
        files += (overlay,)
    environments = _render_environments(files)

    for service, expected_profile in (
        ("runtime-effects", "effects"),
        ("omninode-runtime", "main"),
    ):
        environment = environments[service]
        profile = environment["RUNTIME_PROFILE"]
        assert profile == expected_profile
        ownership = filter_manifest_for_runtime_profile(
            manifest,
            profile,
            lane=_lane_declaration(environment["ONEX_RUNTIME_LANE"]),
        )
        lane_errors = [
            error
            for error in ownership.manifest.errors
            if CONTRACT_NAME in error.error or "OMN-19408" in error.error
        ]
        assert not lane_errors, lane_errors
        assert environment["ONEX_RUNTIME_LANE"] == lane
        owned = {contract.name for contract in ownership.manifest.contracts}
        if profile == "effects":
            assert ownership.runtime_lane == lane
            assert (CONTRACT_NAME in owned) is attaches
            assert (CONTRACT_NAME in ownership.lane_excluded_contracts) is not attaches
            assert CONTRACT_NAME not in ownership.skipped_contracts
        else:
            assert CONTRACT_NAME not in owned
            assert CONTRACT_NAME in ownership.skipped_contracts
            assert CONTRACT_NAME not in ownership.lane_excluded_contracts
