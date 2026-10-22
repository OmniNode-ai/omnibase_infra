# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deployment must actually own the board-probe projection (AC4).

The instance's lane class and the class contracts it takes reach the runtime
from the lane's private env fragment through the base passthrough (OMN-20867).
The public instance overlay must not override either: a hardcoded contract list
there replaces the fragment's whole selection, so the landing nodes the
fragment names would drop off the instance.
"""

from pathlib import Path

import pytest

from omnibase_core.constants.constants_runtime_lanes import LAB_RUNTIME_LANES
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from tests.ci.test_dev_202_lane_boundary_omn19505 import OVERLAY_PATH
from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.ci
WRITER = "projection_board_probe_results"
LANDING = ("node_pr_landing_orchestrator", "node_pr_landing_github_effect")
CLASS_KEYS = ("ONEX_RUNTIME_LANE_CLASS", "ONEX_RUNTIME_LANE_CLASS_CONTRACTS")
INFRA_PATH = OVERLAY_PATH.parent / "docker-compose.infra.yml"


def _overlay_env() -> dict[str, str]:
    overlay = load_compose(OVERLAY_PATH)
    return dict(overlay["services"]["omninode-runtime"]["environment"])


def _deployment_env() -> dict[str, str]:
    """The overlay environment plus what the lane's private fragment sets."""
    env = _overlay_env()
    classes = [
        lane
        for lane in LAB_RUNTIME_LANES
        if env["ONEX_RUNTIME_LANE"].startswith(f"{lane}-")
    ]
    assert len(classes) == 1
    env["ONEX_RUNTIME_LANE_CLASS"] = classes[0]
    env["ONEX_RUNTIME_LANE_CLASS_CONTRACTS"] = ",".join((*LANDING, WRITER))
    return env


def _contract(name: str) -> ModelDiscoveredContract:
    # The published omnimarket contracts are scoped to the lab lanes.
    return ModelDiscoveredContract(
        name=name,
        node_type="reducer",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/contracts/{name}/contract.yaml"),
        entry_point_name=f"node_{name}",
        package_name="omnimarket",
        runtime_lanes=ModelRuntimeLaneScope(lanes=tuple(sorted(LAB_RUNTIME_LANES))),
    )


def test_dev_202_overlay_leaves_the_class_selection_to_the_fragment() -> None:
    """The base passes both names through; the overlay sets neither."""
    passthrough = load_compose(INFRA_PATH)["x-runtime-passthrough-env"]
    for key in CLASS_KEYS:
        assert key in passthrough
        assert passthrough[key] is None
        assert key not in _overlay_env()


def test_dev_202_owns_board_probe_writer_without_admitting_other_class_nodes() -> None:
    """Run the real ownership filter on the deployment's environment."""
    env = _deployment_env()
    names = (*LANDING, WRITER, "unselected_class_node")
    manifest = ModelAutoWiringManifest(contracts=tuple(map(_contract, names)))
    result = filter_manifest_for_runtime_profile(manifest, "main", environ=env)
    assert not result.manifest.errors
    assert {c.name for c in result.manifest.contracts} == {*LANDING, WRITER}
    assert result.runtime_lane == env["ONEX_RUNTIME_LANE"]
    assert set(result.lane_class_admitted_contracts) == {*LANDING, WRITER}
    assert result.lane_excluded_contracts == ("unselected_class_node",)


def test_absent_class_selection_reproduces_the_live_missing_consumer() -> None:
    """A registered instance name alone does not inherit class contracts."""
    env = _deployment_env()
    result = filter_manifest_for_runtime_profile(
        ModelAutoWiringManifest(contracts=(_contract(WRITER),)),
        "main",
        environ={"ONEX_RUNTIME_LANE": env["ONEX_RUNTIME_LANE"]},
    )
    assert not result.manifest.contracts
    assert result.lane_excluded_contracts == (WRITER,)
