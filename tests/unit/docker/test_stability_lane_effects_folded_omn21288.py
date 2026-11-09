# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288 step 1: one lab lane's main runtime carries the effects role.

The lane is found through its rendered policy prefix (``STABILITY_TEST_``) and
its compose project, never by spelling its id here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FOLDED_KEY = "STABILITY_TEST_RUNTIME_MAIN_FOLDED_PROFILES"
ARCHIVE_DIRS = {
    "ONEX_DEAD_LETTER_ARCHIVE_DIR": "/app/data/dead-letter-archive",
    "ONEX_CONSUMER_FLOW_ARCHIVE_DIR": "/app/data/consumer-flow-archive",
}


def _lane_project() -> str:
    return render_env(load_contract())["STABILITY_TEST_COMPOSE_PROJECT"]


def _lane_overlay() -> Path:
    project = _lane_project()
    for path in sorted((ROOT / "docker").glob("docker-compose.*.yml")):
        match = re.search(r"(?m)^name:\s*(\S+)\s*$", path.read_text(encoding="utf-8"))
        if match and match.group(1) == project:
            return path
    raise AssertionError(f"no compose overlay names project {project!r}")


def test_only_the_stability_lane_and_primary_dev_fold_effects() -> None:
    """The stability lane (OMN-21288) and primary dev (OMN-22930); no other lane."""
    contract = load_contract()
    folding = {
        name: profile
        for name, profile in contract.profiles.items()
        if "effects" in profile.processes["main"].folded_profiles
    }
    # The lane is found through its compose project, never by spelling its id.
    lane = [n for n, p in folding.items() if p.compose_project == _lane_project()]
    assert sorted(folding) == sorted([*lane, "dev"]) and len(lane) == 1, sorted(folding)
    stability = folding[lane[0]]
    # OMN-22032 (T19): effects is one role of the list the lane folds.
    assert "effects" in stability.processes["main"].folded_profiles
    # effects stays a declared process on both: it is a required process name.
    assert all("effects" in profile.processes for profile in folding.values())


def test_rendered_env_carries_the_fold_for_those_lanes_only() -> None:
    rendered = render_env(load_contract())
    folds = {k: v for k, v in rendered.items() if k.endswith("_FOLDED_PROFILES")}
    # Primary dev folds its own roles (OMN-21288 tenant and writer folds) and,
    # since OMN-22930, effects.
    dev_folds = folds.pop("DEV_RUNTIME_MAIN_FOLDED_PROFILES").split(",")
    assert "effects" in dev_folds
    assert "effects" in folds.pop(FOLDED_KEY).split(",")
    assert folds == {}
    committed = (ROOT / "docker" / "runtime-policy.env").read_text(encoding="utf-8")
    assert committed == format_dotenv(rendered)


def test_main_reads_the_fold_and_preserves_the_effects_archive_volume() -> None:
    services = load_compose(_lane_overlay())["services"]
    main_env = services["omninode-runtime"]["environment"]
    effects_env = services["runtime-effects"]["environment"]
    assert main_env["ONEX_RUNTIME_FOLDED_PROFILES"] == (
        f"${{{FOLDED_KEY}:?runtime policy contract must set {FOLDED_KEY}}}"
    )
    assert "runtime_data:/app/data" in services["omninode-runtime"]["volumes"]
    assert "effects_data:/app/effects-data" in services["omninode-runtime"]["volumes"]
    assert (
        "effects_logs:/app/effects-logs:ro" in services["omninode-runtime"]["volumes"]
    )
    assert "effects_data:/app/data" in services["runtime-effects"]["volumes"]
    for key, value in ARCHIVE_DIRS.items():
        assert effects_env[key] == value
        # Both runtime paths address the same directory in the old volume.
        assert main_env[key] == value.replace("/app/data/", "/app/effects-data/")


def test_effects_service_is_opt_in_only() -> None:
    # The tag matters as well as the value: a plain list merges with the base
    # service's runtime/full profiles and would keep the effects container up.
    document = yaml.compose(_lane_overlay().read_text(encoding="utf-8"))
    assert isinstance(document, yaml.MappingNode)
    services = next(v for k, v in document.value if k.value == "services")
    assert isinstance(services, yaml.MappingNode)
    effects = next(v for k, v in services.value if k.value == "runtime-effects")
    assert isinstance(effects, yaml.MappingNode)
    profiles = next(v for k, v in effects.value if k.value == "profiles")
    assert isinstance(profiles, yaml.SequenceNode)
    assert profiles.tag == "!override"
    assert [item.value for item in profiles.value] == ["folded-effects"]
