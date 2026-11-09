# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22391 (plan T15): the runtime serves the intelligence API; its container is gone.

omniintelligence declares the API as an ``http_mount`` on ``/api/v1`` in the contract of
``node_pattern_storage_effect``, so the runtime's own HTTP port answers it. No compose
file of this repository declares an ``intelligence-api`` service any more, the catalog and
bundles do not name it, and the dev policy retires the previous revision's container by
name before main is recreated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts.render_runtime_policy_env import load_contract, render_env

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
SERVICE = "intelligence-api"
CONTAINER = "omnibase-intelligence-api"


def _services(path: Path) -> dict[str, object]:
    # compose's ``!override`` and merge tags only change how overlays merge.
    text = re.sub(r"!override[ \t]*|!!merge[ \t]*", "", path.read_text("utf-8"))
    services = yaml.safe_load(text).get("services") or {}
    assert isinstance(services, dict)
    return services


def test_no_compose_file_declares_the_service() -> None:
    files = sorted(DOCKER.glob("docker-compose*.yml"))
    assert len(files) > 5
    holders = [f.name for f in files if SERVICE in _services(f)]
    assert holders == []


def test_catalog_and_bundles_do_not_name_the_service() -> None:
    assert not (DOCKER / "catalog" / "services" / f"{SERVICE}.yaml").exists()
    bundles = yaml.safe_load((DOCKER / "catalog" / "bundles.yaml").read_text("utf-8"))
    assert bundles, "bundles.yaml parsed empty"
    named = [
        name
        for name, bundle in bundles.items()
        if isinstance(bundle, dict) and SERVICE in (bundle.get("services") or [])
    ]
    assert named == []


def test_the_dev_policy_retires_the_previous_container_by_name() -> None:
    process = load_contract().profiles["dev"].processes["main"]
    assert SERVICE in process.retired_services
    assert SERVICE not in process.foldable_services.values()
    rendered = render_env(load_contract())["DEV_RUNTIME_MAIN_RETIRED_SERVICES"]
    assert SERVICE in rendered.split(",")
    env_lines = (DOCKER / "runtime-policy.env").read_text("utf-8").splitlines()
    committed = [
        line.split("=", 1)[1]
        for line in env_lines
        if line.startswith("DEV_RUNTIME_MAIN_RETIRED_SERVICES=")
    ]
    assert committed == [rendered]


def test_the_deploy_surfaces_name_no_runtime_service_to_build_or_start() -> None:
    handler = (
        ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"
    ).read_text("utf-8")
    services = handler.split("readonly RUNTIME_SERVICES=(", 1)[1].split(")", 1)[0]
    assert SERVICE not in services.split()
    events = (ROOT / "scripts/deploy-agent/deploy_agent/events.py").read_text("utf-8")
    assert f'"{SERVICE}"' not in events


def test_the_mount_is_not_a_standalone_service_in_the_stack_shape_data() -> None:
    data = (ROOT / "config/validation/stack_shape.yaml").read_text("utf-8")
    assert f"service: {SERVICE}" not in data


def test_the_container_name_is_the_one_the_retirement_targets() -> None:
    # Positive control: the name the retirement census filters on is the compose
    # service key, the same string the removed stanza used.
    assert CONTAINER.endswith(SERVICE)
