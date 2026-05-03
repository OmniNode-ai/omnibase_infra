# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for CI resilience fixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKER_BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-build.yml"
ENV_PARITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "env-parity.yml"
SETUP_PYTHON_UV_ACTION = (
    REPO_ROOT / ".github" / "actions" / "setup-python-uv" / "action.yml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_migration_integration_resolves_reachable_postgres_host() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-integration"]

    ports = job["services"]["postgres"]["ports"]
    assert ports == ["5432/tcp"]

    steps = job["steps"]
    resolve_step = next(
        step for step in steps if step.get("name") == "Resolve Postgres service host"
    )
    assert resolve_step["id"] == "postgres_host"
    assert (
        resolve_step["env"]["POSTGRES_PORT"]
        == "${{ job.services.postgres.ports['5432'] }}"
    )
    assert "socket.create_connection" in resolve_step["run"]
    assert "/proc/net/route" in resolve_step["run"]

    apply_step = next(
        step for step in steps if step.get("name") == "Apply all migrations"
    )
    assert (
        apply_step["env"]["OMNIBASE_INFRA_DB_URL"]
        == "postgresql://postgres:test_password@${{ steps.postgres_host.outputs.host }}:${{ job.services.postgres.ports['5432'] }}/omnibase_infra"
    )

    assert_step = next(
        step for step in steps if step.get("name") == "Assert manifest tables exist"
    )
    assert (
        assert_step["env"]["POSTGRES_HOST"] == "${{ steps.postgres_host.outputs.host }}"
    )
    assert (
        assert_step["env"]["POSTGRES_PORT"]
        == "${{ job.services.postgres.ports['5432'] }}"
    )
    assert 'host=os.environ["POSTGRES_HOST"]' in assert_step["run"]
    assert 'port=int(os.environ["POSTGRES_PORT"])' in assert_step["run"]

    client_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Verify Python Postgres client"
    )
    assert_step_index = steps.index(assert_step)
    assert client_step_index < assert_step_index
    client_step = steps[client_step_index]
    assert "import asyncpg" in client_step["run"]
    assert "apt-get" not in client_step["run"]
    assert "sudo" not in client_step["run"]

    assert "import asyncpg" in assert_step["run"]
    assert "asyncpg.connect" in assert_step["run"]
    assert "EXPECTED_TABLES" in assert_step["run"]
    assert "psql" not in assert_step["run"]


def test_migration_conflict_action_startup_failure_is_warn_only() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-conflict-check"]

    validate_step = next(
        step
        for step in job["steps"]
        if step.get("uses")
        == "OmniNode-ai/onex_change_control/.github/actions/validate-boundaries@main"
    )
    assert validate_step["continue-on-error"] is True
    assert validate_step["with"]["warn-only"] == "true"

    report_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Report non-blocking boundary validator startup failure"
    )
    assert report_step["if"] == "steps.migration_conflicts.outcome == 'failure'"


def test_docker_integration_build_timeout_matches_workflow_budget() -> None:
    workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    job = workflow["jobs"]["docker-integration-tests"]
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Run Docker integration tests"
    )

    assert step["env"]["OMNI_DOCKER_BUILD_TIMEOUT_SECONDS"] == "1200"
    assert '--timeout="${OMNI_DOCKER_BUILD_TIMEOUT_SECONDS}"' in step["run"]


def test_docker_integration_installs_compose_plugin_before_tests() -> None:
    workflow = _load_yaml(DOCKER_BUILD_WORKFLOW)
    assert workflow["env"]["DOCKER_COMPOSE_VERSION"] == "v2.40.3"

    job = workflow["jobs"]["docker-integration-tests"]
    steps = job["steps"]
    compose_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install Docker Compose plugin"
    )
    test_step_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run Docker integration tests"
    )

    compose_step = steps[compose_step_index]
    assert compose_step_index < test_step_index
    assert "docker compose version" in compose_step["run"]
    assert "DOCKER_COMPOSE_VERSION" in compose_step["run"]
    assert "docker-compose-linux-x86_64" in compose_step["run"]


def test_short_gates_can_disable_uv_cache_cleanup() -> None:
    action = _load_yaml(SETUP_PYTHON_UV_ACTION)
    assert action["inputs"]["cache-enabled"]["default"] == "true"
    cache_step = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Load cached uv"
    )
    assert cache_step["if"] == "inputs.cache-enabled != 'false'"

    ci_workflow = _load_yaml(CI_WORKFLOW)
    for job_name, job in ci_workflow["jobs"].items():
        setup_steps = [
            step
            for step in job.get("steps", [])
            if step.get("uses") == "./.github/actions/setup-python-uv"
        ]
        if not setup_steps:
            continue

        assert len(setup_steps) == 1
        setup_step = setup_steps[0]
        assert setup_step["with"]["cache-enabled"] == "false"

    env_parity_workflow = _load_yaml(ENV_PARITY_WORKFLOW)
    setup_step = next(
        step
        for step in env_parity_workflow["jobs"]["env-parity"]["steps"]
        if step.get("uses") == "./omnibase_infra/.github/actions/setup-python-uv"
    )
    assert setup_step["with"]["cache-enabled"] == "false"
