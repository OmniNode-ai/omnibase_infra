# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render check: no service the deploy agent names for a lane keeps the dev lane's name.

The deploy agent hands its runtime service list to ``docker compose up -d --no-deps``
by name for every lane it drives, and an explicit name bypasses a profile-disable. A
service the overlay disables by profile and leaves on the base file's fixed
``container_name`` is then created under the dev lane's name, and the follow job fails
at its runtime phase with a name conflict.

This renders the dev composition (base file) and the overlay lane with every profile
enabled, takes the lane's up list from the agent's own ``_requested_services_for_up``,
and asserts that no service in it renders the same fixed name in both. Needs
``docker compose`` 2.24.4 or newer for the ``!override`` tag.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, cast

import pytest
from deploy_agent.events import EnumRuntimeLane, Scope
from deploy_agent.executor import _requested_services_for_up

from tests.integration.infra.test_stability_test_runtime_compose_render import (
    COMPOSE_FILES,
    REPO_ROOT,
    _compose_render_env,
    lane_overlay_path,
)

AGENT_NAMED_SERVICES = ("autoheal", "context-audit-consumer")


def _render(project: str, *compose_files: str) -> dict[str, Any]:
    command = ["docker", "compose", "--env-file", "docker/runtime-policy.env"]
    for compose_file in compose_files:
        command += ["-f", compose_file]
    command += [
        "--project-name",
        project,
        "--profile",
        "*",
        "config",
        "--format",
        "json",
    ]
    # A headless lane's compose isolation wrapper needs its own identity to find the
    # real binary; the project comes from --project-name, never from the environment.
    wrapper_env = {
        key: os.environ[key]
        for key in ("LANDING_REAL_DOCKER", "LANDING_COMPOSE_STATE")
        if key in os.environ
    }
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env={**_compose_render_env(), **wrapper_env},
        text=True,
    )
    rendered = json.loads(result.stdout)
    assert isinstance(rendered, dict)
    return cast("dict[str, Any]", rendered)


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's runtime policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


@pytest.mark.integration
def test_overlay_lane_up_list_names_no_service_on_the_dev_lane_name() -> None:
    up_list = _requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.STABILITY_TEST
    )
    dev = _render("render-base", COMPOSE_FILES[0])["services"]
    overlay_lane = _render("render-overlay", *COMPOSE_FILES)["services"]

    # Positive controls: the render sees the two services at fixed dev names, and the
    # lane still starts its runtime by name.
    for service in AGENT_NAMED_SERVICES:
        assert dev[service].get("container_name"), f"{service} has no fixed dev name"
        assert service in overlay_lane
    assert "omninode-runtime" in up_list

    shared = {
        service: dev[service]["container_name"]
        for service in up_list
        if service in dev
        and dev[service].get("container_name")
        and overlay_lane.get(service, {}).get("container_name")
        == dev[service]["container_name"]
    }
    assert not shared, (
        f"the agent starts {sorted(shared)} by name in the overlay lane, but they "
        f"render the dev lane's container names {shared}"
    )
    assert set(AGENT_NAMED_SERVICES).isdisjoint(up_list)
