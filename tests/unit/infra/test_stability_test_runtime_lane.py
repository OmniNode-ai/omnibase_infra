# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for the OMN-10281 stability-test runtime lane."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_FILE = REPO_ROOT / "docker" / "docker-compose.stability-test.yml"
RUNBOOK_FILE = REPO_ROOT / "docs" / "runbooks" / "stability-test-runtime-lane.md"

RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
PRODUCTION_CONTAINER_NAMES = {
    "omninode-runtime",
    "omninode-runtime-effects",
    "omnibase-infra-postgres",
    "omnibase-infra-redpanda",
    "omnibase-infra-valkey",
}
PRODUCTION_GROUP_IDS = {
    "onex-runtime-main",
    "onex-runtime-effects",
    "onex-runtime-workers",
}
PRODUCTION_VOLUME_NAMES = {
    "omninode-runtime-data",
    "omninode-runtime-logs",
    "omninode-effects-data",
    "omninode-effects-logs",
    "omninode-worker-logs",
    "omnibase-infra-postgres-data",
    "omnibase-infra-redpanda-data",
    "omnibase-infra-valkey-data",
}


def _load_overlay() -> dict:
    with OVERLAY_FILE.open(encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle)
    assert isinstance(overlay, dict)
    return overlay


@pytest.mark.unit
def test_stability_lane_has_distinct_compose_project_name() -> None:
    overlay = _load_overlay()

    assert overlay["name"] == "omnibase-infra-stability-test"
    assert overlay["name"] != "omnibase-infra"


@pytest.mark.unit
def test_stability_lane_runtime_services_do_not_reuse_production_names() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        service = services[service_name]
        container_name = service["container_name"]
        assert "stability-test" in container_name
        assert container_name not in PRODUCTION_CONTAINER_NAMES
        assert service["image"] == "runtime:stability-test-workspace"
        assert service["restart"] == "no"


@pytest.mark.unit
def test_stability_lane_uses_workspace_selector_and_isolated_groups() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        assert environment["BUILD_SOURCE"] == "workspace"
        assert environment["ONEX_ENVIRONMENT"] == "stability-test"
        assert environment["ONEX_STATE_ROOT"] == "/app/data/.onex_state_stability_test"
        assert environment["KAFKA_INSTANCE_ID"].startswith("stability-test-")

        group_id = environment["ONEX_GROUP_ID"]
        assert group_id.startswith("onex-stability-test-")
        assert group_id not in PRODUCTION_GROUP_IDS


@pytest.mark.unit
def test_stability_lane_core_infra_names_are_distinct() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in ("postgres", "redpanda", "valkey", "migration-gate"):
        container_name = services[service_name]["container_name"]
        assert "stability-test" in container_name
        assert container_name not in PRODUCTION_CONTAINER_NAMES


@pytest.mark.unit
def test_stability_lane_volumes_do_not_reuse_production_names() -> None:
    overlay = _load_overlay()
    volumes = overlay["volumes"]
    concrete_names = [volume_config["name"] for volume_config in volumes.values()]

    for volume_name, volume_config in volumes.items():
        concrete_name = volume_config["name"]
        assert concrete_name not in PRODUCTION_VOLUME_NAMES, volume_name
        assert "stability-test" in concrete_name, volume_name

    assert len(concrete_names) == len(set(concrete_names))


@pytest.mark.unit
def test_stability_runbook_is_validation_only() -> None:
    runbook = RUNBOOK_FILE.read_text(encoding="utf-8")

    assert "install-infra-watchdog" not in runbook
    assert "systemctl" not in runbook
    assert "does not deploy, restart, or change `.201`" in runbook
    assert "It does not run `docker compose up`." in runbook
    assert "config" in runbook
    assert "BUILD_SOURCE=workspace" in runbook
