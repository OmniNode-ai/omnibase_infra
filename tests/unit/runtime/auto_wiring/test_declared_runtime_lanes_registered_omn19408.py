# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard compose runtime-lane declarations against registry drift (OMN-19408).

Per-host dev compose files declared lanes absent from the core lane registry.
The ownership filter consequently treated those runtimes as declaring no lane.
Every lane-scoped contract then failed closed during discovery, leaving runtime
health permanently DEGRADED. This test scans all compose YAML, including
extension blocks and anchors, and proves each declared lane is registered.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.unit

_LAB_RUNTIME_LANES = ("compose-dev", "onex-lab", "onex-lab-k3s")
_RUNTIME_LANE_VARIABLE = "ONEX_RUNTIME_LANE"

LaneDeclaration = tuple[str, str]


def _walk_runtime_lane_declarations(document: object) -> set[str]:
    """Return runtime lanes declared anywhere in a loaded compose document."""
    lanes: set[str] = set()
    visited: set[tuple[int, bool]] = set()

    def visit(value: object, *, environment_list: bool = False) -> None:
        if isinstance(value, Mapping):
            visit_key = (id(value), environment_list)
            if visit_key in visited:
                return
            visited.add(visit_key)

            mapping = cast("Mapping[object, object]", value)
            for key, nested_value in mapping.items():
                if key == _RUNTIME_LANE_VARIABLE:
                    lanes.add(
                        nested_value
                        if isinstance(nested_value, str)
                        else str(nested_value)
                    )
                visit(
                    nested_value,
                    environment_list=(
                        key == "environment" and isinstance(nested_value, list)
                    ),
                )
            return

        if isinstance(value, list):
            visit_key = (id(value), environment_list)
            if visit_key in visited:
                return
            visited.add(visit_key)

            prefix = f"{_RUNTIME_LANE_VARIABLE}="
            for item in cast("list[object]", value):
                if (
                    environment_list
                    and isinstance(item, str)
                    and item.startswith(prefix)
                ):
                    lanes.add(item.removeprefix(prefix))
                visit(item, environment_list=environment_list)

    visit(document)
    return lanes


def _load_declared_runtime_lanes() -> tuple[LaneDeclaration, ...]:
    repo_root = Path(__file__).resolve().parents[4]
    declarations: set[LaneDeclaration] = set()

    for compose_path in sorted((repo_root / "docker").glob("docker-compose*.yml")):
        compose_yaml = compose_path.read_text(encoding="utf-8").replace("!override", "")
        document: object = yaml.safe_load(compose_yaml)
        relative_path = compose_path.relative_to(repo_root).as_posix()
        declarations.update(
            (relative_path, lane) for lane in _walk_runtime_lane_declarations(document)
        )

    return tuple(sorted(declarations))


DECLARED_RUNTIME_LANES = _load_declared_runtime_lanes()
DECLARATION_IDS = [
    f"{compose_path}:{lane}" for compose_path, lane in DECLARED_RUNTIME_LANES
]


def _lane_scoped_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="synthetic_lab_lane_contract",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("synthetic/synthetic_lab_lane_contract/contract.yaml"),
        entry_point_name="synthetic_lab_lane_contract",
        package_name="test-package",
        runtime_lanes=ModelRuntimeLaneScope(lanes=_LAB_RUNTIME_LANES),
    )


def test_walker_finds_mapping_list_and_nested_anchor_declarations() -> None:
    document: object = yaml.safe_load(
        """
x-runtime:
  nested-environment: &runtime-environment
    ONEX_RUNTIME_LANE: compose-dev
services:
  mapping-form:
    environment:
      <<: *runtime-environment
      NOT_ONEX_RUNTIME_LANE: ignored-mapping-key
  list-form:
    environment:
      - ONEX_RUNTIME_LANE=stability-test
      - NOT_ONEX_RUNTIME_LANE=ignored-list-item
    command: ONEX_RUNTIME_LANE=ignored-outside-environment
"""
    )

    assert _walk_runtime_lane_declarations(document) == {
        "compose-dev",
        "stability-test",
    }


def test_compose_scan_has_positive_controls() -> None:
    assert len(DECLARED_RUNTIME_LANES) >= 5, (
        "the compose scan must find declarations before absence proves anything"
    )
    assert ("docker/docker-compose.dev-lane.yml", "compose-dev") in (
        DECLARED_RUNTIME_LANES
    )
    assert ("docker/docker-compose.stability-test.yml", "stability-test") in (
        DECLARED_RUNTIME_LANES
    )


@pytest.mark.parametrize(
    ("compose_path", "lane"),
    DECLARED_RUNTIME_LANES,
    ids=DECLARATION_IDS,
)
def test_declared_runtime_lane_is_registered(compose_path: str, lane: str) -> None:
    assert "${" not in lane, (
        f"{compose_path} declares {_RUNTIME_LANE_VARIABLE}={lane!r}; "
        "a lane must be a literal so it can be checked"
    )
    assert lane in REGISTERED_RUNTIME_LANES, (
        f"{compose_path} declares {_RUNTIME_LANE_VARIABLE}={lane!r}, but the sorted "
        f"registered set is {sorted(REGISTERED_RUNTIME_LANES)}. Register the lane in "
        "omnibase_core constants_runtime_lanes.REGISTERED_RUNTIME_LANES (and bump "
        "this repo's omnibase-core pin) or declare a registered lane; an unregistered "
        "lane makes every lane-scoped contract fail closed and the runtime read "
        "DEGRADED (OMN-19408)."
    )


@pytest.mark.parametrize(
    ("compose_path", "lane"),
    DECLARED_RUNTIME_LANES,
    ids=DECLARATION_IDS,
)
def test_declared_runtime_lane_passes_lane_scope_filter(
    compose_path: str,
    lane: str,
) -> None:
    manifest = ModelAutoWiringManifest(contracts=(_lane_scoped_contract(),))

    result = filter_manifest_for_runtime_profile(
        manifest,
        "main",
        environ={_RUNTIME_LANE_VARIABLE: lane},
    )

    assert not result.manifest.errors, (
        f"{compose_path} declares {_RUNTIME_LANE_VARIABLE}={lane!r}, which the lane "
        "scope filter rejected"
    )
    assert result.runtime_lane == lane
