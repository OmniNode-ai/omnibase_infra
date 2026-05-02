# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration checks for runner fleet config path propagation."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    default_runner_fleet_config_path,
    load_runner_fleet_config,
)

REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.integration
def test_runner_fleet_config_default_is_repo_anchored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RUNNER_FLEET_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    config_path = default_runner_fleet_config_path()

    assert config_path == REPO_ROOT / "config" / "runner_fleet.yaml"
    assert load_runner_fleet_config(config_path).runner_name_prefix == "omninode-runner"


@pytest.mark.integration
def test_deploy_runner_config_path_is_propagated_to_sync_and_cron() -> None:
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'RUNNER_FLEET_CONFIG="${RUNNER_FLEET_CONFIG_PATH:-${RUNNER_FLEET_CONFIG:-'
        '${REPO_ROOT}/config/runner_fleet.yaml}}"'
    ) in deploy_script
    assert '"${RUNNER_FLEET_CONFIG}" \\' in deploy_script
    assert "RUNNER_FLEET_CONFIG_PATH=${RUNNER_FLEET_CONFIG}" in deploy_script
    assert "RUNNER_FLEET_CONFIG_PATH=${repo_root}/config/runner_fleet.yaml" not in (
        deploy_script
    )
