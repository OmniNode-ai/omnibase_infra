# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No lab address ships in the cloud runtime path; cloud mode refuses one (OMN-26709).

AC1: the files the runtime image copies carry no lab address.
AC2: in cloud mode (a topology profile whose instance is not local) the boot
preflight refuses to start, with a named error, when a setting holds a lab address.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_cloud_lab_guard import (
    CloudModeLabAddressError,
    HandlerCloudLabGuard,
    holds_lab_address,
)
from omnibase_infra.models.runtime.model_cloud_mode_lab_address_request import (
    ModelCloudModeLabAddressRequest,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.runtime import entrypoint_preflight
from omnibase_infra.topology.application_database import (
    TOPOLOGY_PROFILE_INSTANCE_MAP,
)

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_FLEET = _ROOT / "config" / "image" / "runner_fleet.yaml"
_LAB = ".".join(("192", "168", "7", "7"))
_PROFILE = "ONEX_DATABASE_TOPOLOGY_PROFILE"
_CLOUD = next(p for p, i in TOPOLOGY_PROFILE_INSTANCE_MAP.items() if i != "local")


def _guard(env: dict[str, str]) -> None:
    HandlerCloudLabGuard().handle(
        ModelCloudModeLabAddressRequest(environment=tuple(sorted(env.items())))
    )


def test_shipped_files_carry_no_lab_address() -> None:
    shipped = [
        _ROOT / "contracts" / "llm_endpoints.yaml",
        _IMAGE_FLEET,
        _ROOT / "config" / "infisical_projects.yaml",
    ]
    for path in shipped:
        hits = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if holds_lab_address(line)
        ]
        assert hits == [], f"{path.name} carries a lab address"


def test_dockerfile_bakes_the_image_fleet_not_the_lab_fleet() -> None:
    dockerfile = (_ROOT / "docker" / "Dockerfile.runtime").read_text(encoding="utf-8")
    assert "config/image/runner_fleet.yaml /app/config/runner_fleet.yaml" in dockerfile
    assert "COPY --chown=omniinfra:omniinfra config/runner_fleet.yaml" not in dockerfile
    assert load_runner_fleet_config(_IMAGE_FLEET).hosts


def test_cloud_mode_with_lab_llm_url_is_refused() -> None:
    env = {
        _PROFILE: _CLOUD,
        "LLM_CODER_URL": f"http://{_LAB}:8000",
    }
    with pytest.raises(CloudModeLabAddressError) as excinfo:
        _guard(env)
    assert "LLM_CODER_URL" in str(excinfo.value)
    assert _LAB not in str(excinfo.value)


def test_cloud_mode_with_lab_infisical_addr_is_refused() -> None:
    env = {
        _PROFILE: _CLOUD,
        "INFISICAL_ADDR": f"http://{_LAB}:8880",
    }
    with pytest.raises(CloudModeLabAddressError, match="INFISICAL_ADDR"):
        _guard(env)


def test_cloud_mode_with_lab_runner_fleet_file_is_refused(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text("runner_host: h777\n", encoding="utf-8")
    env = {
        _PROFILE: _CLOUD,
        "RUNNER_FLEET_CONFIG_PATH": str(fleet),
    }
    with pytest.raises(CloudModeLabAddressError, match="RUNNER_FLEET_CONFIG_PATH"):
        _guard(env)


def test_cloud_mode_with_no_lab_values_starts() -> None:
    _guard(
        {
            _PROFILE: _CLOUD,
            "LLM_CODER_URL": "https://models.example.com",
            "INFISICAL_ADDR": "https://secrets.example.com",
            "RUNNER_FLEET_CONFIG_PATH": str(_IMAGE_FLEET),
        }
    )


def test_lab_mode_keeps_lab_addresses() -> None:
    _guard(
        {
            _PROFILE: "local",
            "LLM_CODER_URL": f"http://{_LAB}:8000",
        }
    )
    _guard({"LLM_CODER_URL": f"http://{_LAB}:8000"})


def test_preflight_aborts_boot_in_cloud_mode_with_lab_address() -> None:
    env = {
        _PROFILE: _CLOUD,
        "INFISICAL_ADDR": f"http://{_LAB}:8880",
    }
    assert entrypoint_preflight.run_preflight(env) != 0


def test_detector_has_a_positive_and_a_negative_control() -> None:
    assert holds_lab_address(f"http://{_LAB}:8000")
    assert holds_lab_address("runner_host: h777")
    assert not holds_lab_address("https://models.example.com")
    assert not holds_lab_address("http://8.8.8.8:8000")
    assert not holds_lab_address(".".join(("10", "20", "30", "40")))
    assert not holds_lab_address("http://127.0.0.1:8104")
