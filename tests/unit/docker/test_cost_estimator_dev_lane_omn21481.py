# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21481: the cost estimator page and its API are declared services of the
dev lane, and every dev-lane redeploy brings them back.

AC1 -- the dev lane's compose declaration carries the estimator page as its own
service, published at the fixed lane port 3080, and the estimator API stays
declared with its /health route on 8090.

AC2 -- an upload on the lane page reaches the lane's own API: the page is wired
to it in-network with the estimator feature on, serves the lab host's own name
(a production build otherwise redirects it to its public origin), and is not
held behind the cloud migration gate. The upload itself is walked by the front
end's estimator_lane_upload Playwright test against the lane URL.

AC3 -- the page is in the restart set of every dev-lane deploy path, as a
tag-referenced service (recreated, never built by the lane). Its own compose
profile keeps it off the other dev instances, which bring up only the runtime
profile, while every dev-lane deploy names it and so starts it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DEV_LANE = _REPO / "docker" / "docker-compose.dev-lane.yml"
_REFRESH = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_HANDLER = _REPO / "src" / "omnibase_infra" / "handlers" / "handler_runtime_deploy.sh"
_EVENTS = _REPO / "scripts" / "deploy-agent" / "deploy_agent" / "events.py"

_SERVICE = "cost-estimator-web"
_API = "onex-api"
_LANE_PORT = "3080"
_HOST_OPT_IN = "OMNI" + "WEB_SERVE_ANY_HOST"

# Compose's merge tags (`!override`, `!reset`) are not YAML tags safe_load
# knows. This module reads values, never merge semantics, so the tag is dropped
# and the value kept.
_COMPOSE_TAG = re.compile(r"!(?:override|reset)(?=\s)")


def _services(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(_COMPOSE_TAG.sub("", path.read_text(encoding="utf-8")))
    services = doc["services"]
    assert isinstance(services, dict)
    return services


def _bash_array(path: Path, name: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^(?:readonly )?{name}=\(\n(.*?)^\)", text, re.S | re.M)
    assert match is not None, f"{name} not found in {path.name}"
    entries: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            entries.append(stripped)
    assert entries, f"{name} parsed empty in {path.name}"
    return entries


def _python_assignment(path: Path, name: str) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
    raise AssertionError(f"{name} not assigned in {path.name}")


def _string_constants(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


# AC1: the lane's compose declaration carries the page and its API


def test_ac1_dev_lane_declares_the_estimator_page_service() -> None:
    services = _services(_DEV_LANE)
    assert _SERVICE in services, "the dev lane declares no estimator service"
    svc = services[_SERVICE]
    assert svc["restart"] == "unless-stopped"
    assert "omnibase-infra-network" in svc["networks"]
    assert f"{_LANE_PORT}:3000" in svc["ports"]
    labels = svc["labels"]
    assert "com.omninode.lane=dev" in labels
    assert "com.omninode.ticket=OMN-21481" in labels
    assert "/cost-estimator" in " ".join(svc["healthcheck"]["test"])


def test_ac1_the_estimator_api_stays_declared_with_its_health_route() -> None:
    api = _services(_DEV_LANE)[_API]
    assert "8090:8000" in api["ports"]
    assert "http://localhost:8000/health" in " ".join(api["healthcheck"]["test"])


def test_ac1_the_page_image_is_tag_referenced_and_never_pulled() -> None:
    svc = _services(_DEV_LANE)[_SERVICE]
    assert "build" not in svc
    assert "cost-estimator-web:dev-lane" in svc["image"]
    assert svc["pull_policy"] == "never"


# AC2: an upload on the lane page reaches the lane's own API


def test_ac2_the_page_reaches_the_lanes_own_api_with_the_estimator_on() -> None:
    svc = _services(_DEV_LANE)[_SERVICE]
    env = svc["environment"]
    assert env["FEATURE_ESTIMATOR_ENABLED"] == "true"
    assert env["DELEGATION_API_BASE_URL"] == f"http://{_API}:8000"
    # Without it a production build redirects the lab host to its public origin.
    assert env[_HOST_OPT_IN] == "true"
    # A loopback AUTH_URL makes the front end answer 500 on every page.
    assert not re.search(r"//(localhost|127\.0\.0\.1)", env["AUTH_URL"])
    # Not behind the cloud migration gate that holds the API.
    assert "depends_on" not in svc


# AC3: every dev-lane redeploy recreates it; no other instance starts it


def test_ac3_the_governed_refresh_recreates_the_page() -> None:
    assert _SERVICE in _bash_array(_REFRESH, "REFRESH_BUILD_SERVICES")


def test_ac3_onex_runtime_deploy_recreates_the_page_on_the_dev_lane() -> None:
    assert _SERVICE in _bash_array(_HANDLER, "DEV_LANE_ONLY_RUNTIME_SERVICES")


def test_ac3_the_deploy_agent_recreates_the_page_and_never_builds_it() -> None:
    runtime = _string_constants(
        _python_assignment(_EVENTS, "DEV_LANE_ONLY_RUNTIME_SERVICES")
    )
    tagged = _string_constants(
        _python_assignment(_EVENTS, "DEV_LANE_ONLY_TAG_REFERENCED_SERVICES")
    )
    assert _SERVICE in runtime
    assert _SERVICE in tagged


def test_ac3_the_page_sits_outside_the_runtime_profile() -> None:
    """Other dev instances layer this file and bring up `--profile runtime`.

    A runtime-profile page would start on each of them. Every deploy path here
    names the service, and compose enables the profile of a service named on
    the command line, so it is recreated on this lane and nowhere else.
    """
    profiles = _services(_DEV_LANE)[_SERVICE]["profiles"]
    assert profiles == ["cost-estimator"]


def test_ac3_rt6_certifies_the_page_by_its_source_repo_label() -> None:
    """A recreated tag-referenced container is read back by its source repo.

    The page's image is stamped with the source-repo and revision labels at
    build; the readback must keep reading them rather than comparing the image
    with this repository's ref, or every refresh that recreates it would fail.
    """
    handler = _HANDLER.read_text(encoding="utf-8")
    assert '"ai.omninode.image.source-repo"' in handler
    assert "readback_foreign_sourced_service" in handler
    compose = _DEV_LANE.read_text(encoding="utf-8")
    assert "--label ai.omninode.image.source-repo=" in compose
    assert "--label org.opencontainers.image.revision=" in compose
