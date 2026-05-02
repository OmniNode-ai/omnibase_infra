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


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_migration_integration_uses_dynamic_postgres_port() -> None:
    workflow = _load_yaml(CI_WORKFLOW)
    job = workflow["jobs"]["migration-integration"]

    ports = job["services"]["postgres"]["ports"]
    assert ports == ["5432/tcp"]

    steps = job["steps"]
    apply_step = next(
        step for step in steps if step.get("name") == "Apply all migrations"
    )
    assert (
        apply_step["env"]["OMNIBASE_INFRA_DB_URL"]
        == "postgresql://postgres:test_password@localhost:${{ job.services.postgres.ports[5432] }}/omnibase_infra"
    )

    assert_step = next(
        step for step in steps if step.get("name") == "Assert manifest tables exist"
    )
    assert (
        assert_step["env"]["POSTGRES_PORT"]
        == "${{ job.services.postgres.ports[5432] }}"
    )
    assert '-p "$POSTGRES_PORT"' in assert_step["run"]


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
