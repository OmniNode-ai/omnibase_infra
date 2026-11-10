# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pin compose guard node handler against real git clones (OMN-23368).

The handler runs with no mocks: a real bare "origin" and a real clone, so the staleness
reading comes from ``git rev-list`` and the refusal from the declared-service set.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_dev_lane_pin_compose_guard import (
    ComposeServiceMissingError,
    HandlerDevLanePinComposeGuard,
)
from omnibase_infra.models.model_dev_lane_pin_guard_request import (
    ModelDevLanePinGuardRequest,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
    )


@pytest.fixture
def behind_clone(tmp_path: Path) -> Path:
    """A clone whose origin/dev holds two commits the clone's HEAD lacks."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "dev")
    (origin / "f").write_text("1")
    _git(origin, "add", "f")
    _git(origin, "commit", "-m", "one")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    for n in ("2", "3"):
        (origin / "f").write_text(n)
        _git(origin, "commit", "-am", n)
    _git(clone, "fetch", "origin")
    return clone


def _request(clone: Path, targets: tuple[str, ...]) -> ModelDevLanePinGuardRequest:
    return ModelDevLanePinGuardRequest(
        targets=targets,
        compose_files=("docker-compose.yml",),
        declared_services=frozenset({"alpha", "beta"}),
        env={"DEPLOY_AGENT_TRACKING_REF": "dev"},
        agent_dir=str(clone),
        git_timeout_seconds=30.0,
        now=datetime.now(UTC),
    )


@pytest.mark.integration
def test_declared_services_report_pin_two_commits_behind(behind_clone: Path) -> None:
    result = HandlerDevLanePinComposeGuard().handle(
        _request(behind_clone, ("alpha", "beta"))
    )
    assert "behind origin/dev by 2 commits" in result.report


@pytest.mark.integration
def test_undeclared_service_is_refused_naming_it(behind_clone: Path) -> None:
    with pytest.raises(ComposeServiceMissingError) as exc:
        HandlerDevLanePinComposeGuard().handle(
            _request(behind_clone, ("alpha", "gone"))
        )
    assert exc.value.missing == ("gone",)
    assert "2 commits behind origin/dev" in str(exc.value)
