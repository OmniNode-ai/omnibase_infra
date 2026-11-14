# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24476: the dev lane has no separate onex-api container.

The operator ruled the separate HTTP API a mistake for the local runtime
(2026-11-10) and consented to deleting the container (2026-11-14). No compose
file of the dev lane declares an ``onex-api``, ``cloud-migration-files`` or
``cloud-migration`` service any more -- the collaborator's lane and the
per-slot verification lane are not the dev lane and keep their own onex-api -- and the dev
policy retires the previous revision's containers by name before main is
recreated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_dev_boot_guard import folded_dev_services
from scripts.render_runtime_policy_env import load_contract, render_env

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
REMOVED = ("onex-api", "cloud-migration-files", "cloud-migration")
#: The compose files of the dev lane: the shared base overlay, its tenant fold and the
#: per-host instance overlays. Every other compose file is another lane's.
DEV_LANE_FILES = "docker-compose.dev-*.yml"


def _services(path: Path) -> dict[str, object]:
    # compose's ``!override`` and merge tags only change how overlays merge.
    text = re.sub(r"!override[ \t]*|!!merge[ \t]*", "", path.read_text("utf-8"))
    services = yaml.safe_load(text).get("services") or {}
    assert isinstance(services, dict)
    return services


def test_no_dev_lane_compose_file_declares_a_removed_service() -> None:
    files = sorted(DOCKER.glob(DEV_LANE_FILES))
    assert len(files) >= 3
    holders = [
        (f.name, removed)
        for f in files
        for removed in REMOVED
        if removed in _services(f)
    ]
    assert holders == []


def test_another_lane_keeps_its_own_onex_api() -> None:
    # Positive control, and the boundary of the removal: a compose file outside the
    # dev lane still declares onex-api, so only the dev lane dropped it.
    dev = set(DOCKER.glob(DEV_LANE_FILES))
    others = [
        f
        for f in sorted(DOCKER.glob("docker-compose*.yml"))
        if f not in dev and "onex-api" in _services(f)
    ]
    assert others, "no other lane declares onex-api: the removal reached too far"


def test_the_dev_policy_retires_each_removed_service_by_name() -> None:
    process = load_contract().profiles["dev"].processes["main"]
    rendered = render_env(load_contract())["DEV_RUNTIME_MAIN_RETIRED_SERVICES"]
    for removed in REMOVED:
        assert removed in process.retired_services
        assert removed not in process.foldable_services.values()
        assert removed in rendered.split(",")
    env_lines = (DOCKER / "runtime-policy.env").read_text("utf-8").splitlines()
    committed = [
        line.split("=", 1)[1]
        for line in env_lines
        if line.startswith("DEV_RUNTIME_MAIN_RETIRED_SERVICES=")
    ]
    assert committed == [rendered]


def test_the_deploy_agent_lane_filter_drops_the_removed_services() -> None:
    # The deploy agent reads this committed table -- render_env(load_contract())
    # renders exactly it -- and, on the dev lane whose main folds the effects
    # profile, retires folded_dev_services(policy, folded); that returned set is
    # what lane_config_for merges into disabled_services.
    policy = render_env(load_contract())
    assert "effects" in policy["DEV_RUNTIME_MAIN_FOLDED_PROFILES"].split(",")
    mapping = json.loads(policy["DEV_RUNTIME_MAIN_FOLDABLE_SERVICES"])
    folded = "effects" if "effects" in mapping else next(iter(mapping))
    retired = folded_dev_services(policy, frozenset([folded]))
    assert set(REMOVED) <= retired
    # The set is exactly the folded service plus everything the contract
    # retires -- the same set the deploy agent lane filter retires.
    assert retired == {
        mapping[folded],
        *load_contract().profiles["dev"].processes["main"].retired_services,
    }
