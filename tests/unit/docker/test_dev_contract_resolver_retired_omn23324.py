# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23324 (plan T16): the dev stack runs no contract resolver container.

The runtime mounts the resolver (``node_contract_resolver_bridge`` declares an
``http_mount``), so the ``omninode-contract-resolver`` container of the
previous revision is inert on the dev stack: its stanza sits under a profile no
bring-up requests, the policy retires it by name so the deploy handler stops
it and leaves it out of the restart set, and the fold overlay adds no
declaration. Other dev instances keep their own resolver container.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts.render_runtime_policy_env import load_contract, render_env
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
FOLD_OVERLAY = DOCKER / "docker-compose.dev-tenant-fold.yml"
CONSUMER = "omninode-contract-resolver"
# The per-instance dev overlays: every dev overlay except the primary-dev pair.
PRIMARY_DEV_OVERLAYS = {
    "docker-compose.dev-lane.yml",
    "docker-compose.dev-tenant-fold.yml",
}
OTHER_DEV_INSTANCES = tuple(
    sorted(
        path.name
        for path in DOCKER.glob("docker-compose.dev-*.yml")
        if path.name not in PRIMARY_DEV_OVERLAYS
    )
)


def _overlay_services(path: Path) -> dict[str, dict[str, object]]:
    # compose's ``!override`` tag only changes how overlays merge; the value it wraps is plain YAML.
    parsed = yaml.safe_load(re.sub(r"!override[ \t]*", "", path.read_text("utf-8")))
    services = parsed["services"]
    assert isinstance(services, dict)
    return services


def test_contract_resolver_retired_by_name_in_policy() -> None:
    process = load_contract().profiles["dev"].processes["main"]
    assert CONSUMER in process.retired_services
    assert CONSUMER not in process.foldable_services.values()
    rendered = render_env(load_contract())["DEV_RUNTIME_MAIN_RETIRED_SERVICES"]
    assert CONSUMER in rendered.split(",")


def test_contract_resolver_retired_in_checked_in_policy_env() -> None:
    env_text = (DOCKER / "runtime-policy.env").read_text("utf-8")
    line = next(
        item
        for item in env_text.splitlines()
        if item.startswith("DEV_RUNTIME_MAIN_RETIRED_SERVICES=")
    )
    assert CONSUMER in line.split("=", 1)[1].split(",")


def test_contract_resolver_retired_by_folded_profile_gate() -> None:
    services = _overlay_services(DOCKER / "docker-compose.dev-lane.yml")
    assert services[CONSUMER]["profiles"] == ["folded-contract-resolver"]


def test_contract_resolver_retired_fold_overlay_adds_no_declaration() -> None:
    # The stack-shape baseline counts compose files that declare the service and
    # only shrinks, so the gate lives in the stanza dev-lane already holds.
    assert CONSUMER not in _overlay_services(FOLD_OVERLAY)


def test_contract_resolver_retired_dev_stack_renders_none() -> None:
    stack = dev_stack_services()
    assert "omninode-runtime" in stack  # positive control: the stack really rendered
    assert CONSUMER not in stack


def test_contract_resolver_retired_only_on_primary_dev() -> None:
    keepers = [
        overlay
        for overlay in OTHER_DEV_INSTANCES
        if CONSUMER in _overlay_services(DOCKER / overlay)
    ]
    # Positive control: instance overlays exist and at least one declares the
    # resolver; a primary-dev gate that leaked into them would empty this list.
    assert OTHER_DEV_INSTANCES
    assert keepers
    for overlay in keepers:
        profiles = _overlay_services(DOCKER / overlay)[CONSUMER].get("profiles")
        assert profiles != ["folded-contract-resolver"], overlay
