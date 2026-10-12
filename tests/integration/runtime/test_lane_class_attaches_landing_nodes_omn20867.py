# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20867: a dev-lane instance attaches the landing nodes its overlay selects.

The acting landing lane is a per-host instance of the dev lane: its runtime
declares its own registered lane, ``<dev lane>-<instance>``, and both landing
contracts are scoped to the dev lane only, so the ownership filter dropped them
there and the instance emitted zero landing commands. The shipped contracts may
not name a host's lane, so the deployment says it: the instance's private
overlay fragment declares the runtime an instance of the dev-lane class and
names the class contracts it takes. The dev lane itself sets neither and is
unchanged.

The environments are rendered from the real compose files in deployment order,
with each value-less passthrough name taking the host value the deploy agent
overlays from the lane's fragment. Every per-host dev overlay in ``docker/`` is
found, not named. The contract headers are the shipped omnimarket ones (lane
lists filled from the rendered dev lane), parsed by the kernel's real discovery
path beside this repository's own contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.ticket.model_ticket_contract import ModelTicketContract
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    filter_manifest_for_runtime_profile,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER = REPO_ROOT / "docker"
BASE_FILES = ("docker-compose.infra.yml", "docker-compose.dev-lane.yml")
ORCHESTRATOR = "node_pr_landing_orchestrator"
GITHUB_EFFECT = "node_pr_landing_github_effect"
LANDING = (ORCHESTRATOR, GITHUB_EFFECT)
# A dev-lane-scoped contract the overlay does NOT select: it stays off an
# instance (the lab lane-health fold keys rows for the lab lanes only).
LAB_HEALTH = "node_projection_lab_lane_health"


def test_ticket_contract_retains_lane_class_and_mirror_source_evidence() -> None:
    # These changes share a ticket and collided in dod_evidence. Both test
    # commands and both deployment probes must survive a base merge.
    contract = ModelTicketContract.from_yaml(
        (REPO_ROOT / "contracts/OMN-20867.yaml").read_text(encoding="utf-8")
    )
    evidence = {item.id: item for item in contract.dod_evidence}
    assert len(evidence) == len(contract.dod_evidence)
    for prefix, criterion, test_path, source_path in (
        (
            "lane-class-scope",
            "AC7",
            "tests/integration/runtime/test_lane_class_attaches_landing_nodes_omn20867.py",
            "src/omnibase_infra/runtime/auto_wiring/profile_ownership.py",
        ),
        (
            "mirror-source-consume-only",
            "AC4",
            "tests/integration/gateway/test_mirror_source_consume_only_omn20867.py",
            "src/omnibase_infra/event_bus/kafka_transport.py",
        ),
    ):
        test_id = f"dod-omn20867-{prefix}"
        if prefix == "lane-class-scope":
            test_id += "-tests"
        tests = evidence[test_id]
        assert criterion in tests.binds_ac
        assert any(test_path in check.check_value for check in tests.checks)
        probe_prefix = prefix.replace("mirror-source-", "mirror-")
        probe = evidence[f"dod-omn20867-{probe_prefix}-deploy-probe"]
        assert any(source_path in check.check_value for check in probe.checks)


def _render_runtime_environment(
    files: tuple[str, ...], host_env: Mapping[str, str]
) -> dict[str, str]:
    """The omninode-runtime environment compose would hand the container."""
    environment: dict[str, object] = {}
    for filename in files:
        text = (DOCKER / filename).read_text(encoding="utf-8")
        # SafeLoader resolves << / !!merge; !override is a compose-only tag.
        document = yaml.safe_load(text.replace("!override", ""))
        service = document["services"].get("omninode-runtime", {})
        declared = service.get("environment", {})
        assert isinstance(declared, dict)
        environment.update(declared)
    rendered: dict[str, str] = {}
    for key, value in environment.items():
        if value is None:
            # A value-less name passes the host value through, or stays unset.
            if key in host_env:
                rendered[str(key)] = host_env[key]
            continue
        rendered[str(key)] = str(value)
    return rendered


DEV_LANE = _render_runtime_environment(BASE_FILES, {})["ONEX_RUNTIME_LANE"]


def _instance_overlays() -> list[str]:
    """Every compose overlay whose runtime declares an instance of the dev lane."""
    found: list[str] = []
    for path in sorted(DOCKER.glob("docker-compose.*.yml")):
        document = yaml.safe_load(
            path.read_text(encoding="utf-8").replace("!override", "")
        )
        services = document.get("services") if isinstance(document, dict) else None
        runtime = (
            services.get("omninode-runtime") if isinstance(services, dict) else None
        )
        env = runtime.get("environment") if isinstance(runtime, dict) else None
        lane = env.get("ONEX_RUNTIME_LANE") if isinstance(env, dict) else None
        if isinstance(lane, str) and lane.startswith(f"{DEV_LANE}-"):
            found.append(path.name)
    return found


INSTANCE_OVERLAYS = _instance_overlays()
NON_ACTING = {
    "ONEX_PR_LANDING_GITHUB_MODE": "dry_run",
    "ONEX_PR_LANDING_ARM_ACTION_MODE": "report_only",
}
# What an acting instance's overlay fragment sets.
ACTING_INSTANCE = {
    "ONEX_PR_LANDING_GITHUB_MODE": "enforce",
    "ONEX_PR_LANDING_ARM_ACTION_MODE": "enforce",
    "ONEX_RUNTIME_LANE_CLASS": DEV_LANE,
    "ONEX_RUNTIME_LANE_CLASS_CONTRACTS": f"{ORCHESTRATOR},{GITHUB_EFFECT}",
}


def _manifest(tmp_path: Path) -> ModelAutoWiringManifest:
    # The shipped omnimarket origin/dev headers, cut to what discovery reads.
    headers = {
        ORCHESTRATOR: (
            "name: node_pr_landing_orchestrator\n"
            "contract_version: {major: 2, minor: 0, patch: 0}\n"
            "node_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            "runtime_profiles: [main]\n"
            f"runtime_lanes: [{DEV_LANE}]\n"
        ),
        GITHUB_EFFECT: (
            "name: node_pr_landing_github_effect\n"
            "contract_version: {major: 1, minor: 3, patch: 0}\n"
            "node_type: EFFECT_GENERIC\n"
            "node_version: {major: 1, minor: 0, patch: 0}\n"
            "runtime_profiles: [main]\n"
            f"runtime_lanes: [{DEV_LANE}]\n"
        ),
        LAB_HEALTH: (
            "name: node_projection_lab_lane_health\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: reducer\n"
            f"runtime_lanes: [{DEV_LANE}]\n"
        ),
    }
    paths: list[Path] = []
    for name, header in headers.items():
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "contract.yaml"
        path.write_text(header, encoding="utf-8")
        paths.append(path)
    repo_contracts = sorted(
        (REPO_ROOT / "src/omnibase_infra/nodes").glob("*/contract.yaml")
    )
    assert repo_contracts, "positive control: the real repository manifest is present"
    manifest = discover_contracts_from_paths([*repo_contracts, *paths])
    names = {contract.name for contract in manifest.contracts}
    assert set(headers) <= names, "positive control: the headers parsed"
    return manifest


def test_the_repository_ships_an_instance_overlay() -> None:
    # Positive control for the parametrization below: an empty list would pass
    # the instance case vacuously.
    assert INSTANCE_OVERLAYS


def test_the_dev_lane_is_unchanged(tmp_path: Path) -> None:
    environment = _render_runtime_environment(BASE_FILES, NON_ACTING)
    assert environment["RUNTIME_PROFILE"] == "main"
    assert environment["ONEX_PR_LANDING_GITHUB_MODE"] == "dry_run"
    ownership = filter_manifest_for_runtime_profile(
        _manifest(tmp_path), "main", environ=environment
    )
    owned = {contract.name for contract in ownership.manifest.contracts}
    assert {*LANDING, LAB_HEALTH} <= owned
    assert ownership.runtime_lane == DEV_LANE
    assert ownership.runtime_lane_class is None
    assert ownership.lane_class_admitted_contracts == ()


@pytest.mark.parametrize("overlay", INSTANCE_OVERLAYS)
def test_an_instance_attaches_the_landing_nodes_its_overlay_selects(
    tmp_path: Path, overlay: str
) -> None:
    files = (*BASE_FILES, overlay)
    manifest = _manifest(tmp_path)

    # Before its overlay selects them, the instance excludes both nodes.
    bare = _render_runtime_environment(files, NON_ACTING)
    before = filter_manifest_for_runtime_profile(manifest, "main", environ=bare)
    for name in LANDING:
        assert name in before.lane_excluded_contracts

    environment = _render_runtime_environment(files, ACTING_INSTANCE)
    assert environment["RUNTIME_PROFILE"] == "main"
    assert environment["ONEX_RUNTIME_LANE"].startswith(f"{DEV_LANE}-")
    # The fragment reached the container (passthrough rendering took it).
    assert environment["ONEX_PR_LANDING_GITHUB_MODE"] == "enforce"

    ownership = filter_manifest_for_runtime_profile(
        manifest, "main", environ=environment
    )

    assert not [e for e in ownership.manifest.errors if "OMN-20867" in e.error]
    owned = {contract.name for contract in ownership.manifest.contracts}
    assert set(LANDING) <= owned
    assert ownership.runtime_lane_class == DEV_LANE
    assert ownership.lane_class_admitted_contracts == LANDING
    # Only the selected class contracts attach on the instance.
    assert LAB_HEALTH not in owned
    assert LAB_HEALTH in ownership.lane_excluded_contracts
