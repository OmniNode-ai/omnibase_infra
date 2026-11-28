# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent reading over a real git history, routed by the real contract (OMN-26724)."""

from __future__ import annotations

import importlib
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.model_deploy_agent_reading_request import (
    ModelDeployAgentReadingRequest,
)

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_agent_reading_compute/contract.yaml"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _routed_handler() -> Any:
    routed = yaml.safe_load(CONTRACT.read_text())["handler_routing"]["handlers"][0]
    assert routed["operation"] == "deploy_agent_reading.judge"
    spec = routed["handler"]
    return getattr(importlib.import_module(spec["module"]), spec["name"])()


def test_restart_onto_the_merge_is_read_from_real_history(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    before = _commit(tmp_path, "before")
    merge = _commit(tmp_path, "merge")
    after = _commit(tmp_path, "after")
    # The merge was committed just now; the reading is taken an hour later.
    merge_age = (
        int(time.time())
        - int(_git(tmp_path, "show", "-s", "--format=%ct", merge))
        + 3600
    )

    def has_merge(sha: str) -> bool:
        return (
            subprocess.run(
                ["git", "-C", str(tmp_path), "merge-base", "--is-ancestor", merge, sha],
                check=False,
                env=scrub_git_location_env(),
            ).returncode
            == 0
        )

    def read(sha: str) -> Any:
        return _routed_handler().handle(
            ModelDeployAgentReadingRequest(
                dev_loaded_code_sha=sha,
                dev_uptime_seconds=600,
                dev_loaded_sha_has_merge=has_merge(sha),
                merge_age_seconds=merge_age,
                dev_last_status="success",
                dev_last_runtime_lane="dev",
                dev_last_phase_result_count=7,
                stability_uptime_seconds=merge_age + 100,
                prod_uptime_seconds=merge_age + 100,
            )
        )

    assert read(after).ac1_met is True
    assert read(before).ac1_met is False
