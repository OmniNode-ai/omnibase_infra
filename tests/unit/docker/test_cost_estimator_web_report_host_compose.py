# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The report host's cost estimator page binds one address and takes the gate from the host env (OMN-22162)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

COMPOSE = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "docker-compose.cost-estimator-web.yml"
)
ALL_INTERFACES = ".".join(("0",) * 4)
REQUIRED = (
    "COST_ESTIMATOR_WEB_IMAGE",
    "COST_ESTIMATOR_BIND_ADDR",
    "COST_ESTIMATOR_PORT",
    "COST_ESTIMATOR_API_BASE_URL",
    "COST_ESTIMATOR_AUTH_URL",
)
GATE = ("FEATURE_ESTIMATOR_GATE_ENABLED", "ESTIMATOR_GATE_PASSWORD")
RENDER_ENV = {
    "COST_ESTIMATOR_WEB_IMAGE": "example/cost-estimator-web:report-host",
    "COST_ESTIMATOR_BIND_ADDR": "192.0.2.1",
    "COST_ESTIMATOR_PORT": "3080",
    "COST_ESTIMATOR_API_BASE_URL": "http://192.0.2.2:8090",
    "COST_ESTIMATOR_AUTH_URL": "https://report.example.invalid",
}


def _service() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert list(document["services"]) == ["cost-estimator-web"]
    service = document["services"]["cost-estimator-web"]
    assert isinstance(service, dict)
    return service


def _render(env_values: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # Only the deployment and gate values are replaced, so one set in the caller's shell can never
    # satisfy a refusal test; --env-file os.devnull stops Compose reading a stray .env beside the file.
    env = {k: v for k, v in os.environ.items() if k not in REQUIRED + GATE}
    env.update(env_values)
    return subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(COMPOSE),
            "config",
            "--format",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _rendered(extra: dict[str, str] | None = None) -> dict[str, Any]:
    done = _render({**RENDER_ENV, **(extra or {})})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["services"]["cost-estimator-web"]


def test_page_is_published_on_one_address_only() -> None:
    (port,) = _service()["ports"]
    assert port.startswith("${COST_ESTIMATOR_BIND_ADDR:?"), port
    assert ALL_INTERFACES not in COMPOSE.read_text(encoding="utf-8")
    published = _rendered()["ports"]
    assert [(p["host_ip"], p["published"], p["target"]) for p in published] == [
        ("192.0.2.1", "3080", 3000)
    ]


@pytest.mark.parametrize("name", REQUIRED)
def test_render_refuses_without_each_required_value(name: str) -> None:
    """Positive control: all five render; removing any one makes compose refuse and name it."""
    assert _render(RENDER_ENV).returncode == 0
    refused = _render({k: v for k, v in RENDER_ENV.items() if k != name})
    assert refused.returncode != 0
    assert name in refused.stderr


def test_gate_flag_defaults_on_and_the_password_to_empty() -> None:
    """Unset, the gate reads on (the page's own default, so it fails closed) and no password is named."""
    environment = _rendered()["environment"]
    assert environment["FEATURE_ESTIMATOR_GATE_ENABLED"] == "true"
    assert environment["ESTIMATOR_GATE_PASSWORD"] == ""


def test_gate_values_pass_through_from_the_host_env() -> None:
    environment = _rendered(
        {
            "FEATURE_ESTIMATOR_GATE_ENABLED": "false",
            "ESTIMATOR_GATE_PASSWORD": "from-the-host-env-file",
        }
    )["environment"]
    assert environment["FEATURE_ESTIMATOR_GATE_ENABLED"] == "false"
    assert environment["ESTIMATOR_GATE_PASSWORD"] == "from-the-host-env-file"


def test_the_file_names_no_gate_password() -> None:
    declared = _service()["environment"]
    assert declared["ESTIMATOR_GATE_PASSWORD"] == "${ESTIMATOR_GATE_PASSWORD:-}"
    assert declared["FEATURE_ESTIMATOR_GATE_ENABLED"] == (
        "${FEATURE_ESTIMATOR_GATE_ENABLED:-true}"
    )


def test_page_calls_the_api_named_by_the_host_env_and_holds_no_state() -> None:
    environment = _rendered()["environment"]
    assert environment["ONEX_API_BASE_URL"] == "http://192.0.2.2:8090"
    assert environment["DELEGATION_API_BASE_URL"] == "http://192.0.2.2:8090"
    service = _service()
    assert "volumes" not in service
    assert service["pull_policy"] == "never"
    assert service["cap_drop"] == ["ALL"]
