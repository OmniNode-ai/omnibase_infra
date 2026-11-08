# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288 (one-runtime plan T17): the dev projection API is a node inside main.

``node_projection_query_api_effect`` answers every route the ``projection-api``
service answered, and the runtime serves its ASGI app on its own HTTP port under the
``http_mount.path_prefix`` its contract declares. On primary dev the policy folds the
``projection-api`` role into main, so the separate container is retired through the same
contract mapping as the folded writers, and the lab pass reads the API from the mount.

Failure modes each test is written against:

* (F1) the container is retired in the policy but still rendered into the dev stack;
* (F2) every other dev instance loses its projection-api container with the primary's;
* (F3) main does not carry the database binding the mounted node reads through;
* (F4) the lab pass keeps probing the retired container's port, so it fails on the
  first deploy after the retirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    DEV_COMPOSE_FILES,
    dev_stack_services,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ROUTING = ROOT / "config/deploy_lane_routing.yaml"
ONEX_API_DELIVERY = ROOT / ".github/workflows/onex-api-lab-delivery-reusable.yml"
SERVICE = "projection-api"
PROFILE = "projection-api"
# The prefix node_projection_query_api_effect's contract declares under http_mount in
# omnimarket (path_prefix). Spelled here because this repo must not read the other clone.
MOUNT_PREFIX = "/projection-api"
DSN = "OMNIDASH_ANALYTICS_DB_URL"


def _routing_verify(instance: str) -> dict[str, str]:
    table = yaml.safe_load(ROUTING.read_text(encoding="utf-8"))
    verify = table["instances"][instance]["verify"]
    assert isinstance(verify, dict)
    return verify


def test_policy_folds_the_projection_api_into_dev_and_stability_main_only() -> None:
    policy = load_contract()
    main = policy.profiles["dev"].processes["main"]
    assert PROFILE in main.folded_profiles
    assert main.foldable_services[PROFILE] == SERVICE
    # An HTTP query service holds no consumer group, so there is nothing to hand over.
    assert PROFILE not in main.fold_handovers
    project = render_env(policy)["STABILITY_TEST_COMPOSE_PROJECT"]
    folding = {
        name
        for name, profile in policy.profiles.items()
        if profile.compose_project == project
    }
    (stability_lane,) = folding
    stability = policy.profiles[stability_lane].processes["main"]
    assert PROFILE in stability.folded_profiles
    assert stability.foldable_services[PROFILE] == SERVICE
    for name, profile in policy.profiles.items():
        if name not in ("dev", stability_lane):
            assert PROFILE not in profile.processes["main"].folded_profiles, name
    rendered = render_env(policy)
    assert PROFILE in rendered["DEV_RUNTIME_MAIN_FOLDED_PROFILES"].split(",")
    assert rendered["DEV_RUNTIME_MAIN_PROJECTION_API_RUNTIME_PROFILE"] == (
        "folded-projection-api-runtime"
    )
    assert rendered["DEV_RUNTIME_MAIN_PROJECTION_API_FULL_PROFILE"] == (
        "folded-projection-api-full"
    )
    assert (ROOT / "docker/runtime-policy.env").read_text() == format_dotenv(rendered)


def test_dev_stack_runs_no_projection_api_container() -> None:
    stack = dev_stack_services()
    assert "omninode-runtime" in stack  # positive control: the stack really rendered
    assert SERVICE not in stack


def test_other_dev_instances_keep_their_projection_api_container() -> None:
    definition = load_compose(DEV_COMPOSE_FILES[1])["services"][SERVICE]
    assert definition["profiles"] == [
        "${DEV_RUNTIME_MAIN_PROJECTION_API_RUNTIME_PROFILE:-runtime}",
        "${DEV_RUNTIME_MAIN_PROJECTION_API_FULL_PROFILE:-full}",
    ]
    # The primary-dev fold overlay is the only file that folds; the shared overlay does not.
    assert SERVICE not in (load_compose(DEV_COMPOSE_FILES[2]).get("services") or {})


def test_main_carries_the_database_binding_the_mounted_node_reads() -> None:
    definition = load_compose(DEV_COMPOSE_FILES[1])["services"]
    main = dev_stack_services()["omninode-runtime"]["environment"]
    assert DSN in main
    assert main[DSN] == definition[SERVICE]["environment"][DSN]


def test_lab_pass_reads_the_dev_projection_api_from_the_mount() -> None:
    dev = _routing_verify("dev-201")
    assert dev["projection_url"] == f"{dev['main_url']}{MOUNT_PREFIX}"
    # A dev instance that keeps its container keeps reading its own, not the mount.
    table = yaml.safe_load(ROUTING.read_text(encoding="utf-8"))
    others = [
        instance["verify"]
        for name, instance in table["instances"].items()
        if name != "dev-201" and "verify" in instance
    ]
    assert others  # positive control: another instance with a verify block exists
    for other in others:
        assert not other["projection_url"].endswith(MOUNT_PREFIX)


def test_the_onex_api_delivery_probe_reads_the_same_projection_url_as_the_table() -> (
    None
):
    workflow = yaml.safe_load(ONEX_API_DELIVERY.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["verify-onex-api-delivered"]["steps"]
    probe = next(step for step in steps if step.get("id") == "probe")
    assert (
        probe["env"]["DEV_LANE_PROJECTION_URL"]
        == (_routing_verify("dev-201")["projection_url"])
    )
