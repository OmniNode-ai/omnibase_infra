# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the OMN-12924 judge lane."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
COMPOSE_FILE = "docker/docker-compose.judge.yml"

EXPECTED_RENDERED_SERVICES = {
    "postgres",
    "redpanda",
    "redpanda-partition-cap",
    "valkey",
    "forward-migration",
    "migration-gate",
    "intelligence-migration",
    "omninode-runtime",
    "runtime-effects",
    "projection-api",
}
OUT_OF_SCOPE_SERVICES = {
    "keycloak",
    "infisical",
    "runtime-worker",
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "intelligence-api",
    "omninode-contract-resolver",
    "phoenix",
    "autoheal",
}
EXPECTED_PUBLISHED_PORTS = {
    "postgres": {"35436"},
    "redpanda": {"49092", "49644"},
    "redpanda-partition-cap": set(),
    "valkey": {"56379"},
    "forward-migration": set(),
    "migration-gate": set(),
    "intelligence-migration": set(),
    "omninode-runtime": {"48085"},
    "runtime-effects": {"48086"},
    "projection-api": {"43002"},
}
DEV_OR_PROD_PUBLISHED_PORTS = {
    "5436",
    "15436",
    "19092",
    "39092",
    "9644",
    "29644",
    "16379",
    "26379",
    "8085",
    "18085",
    "28085",
    "8086",
    "18086",
    "28086",
    "3002",
    "13002",
    "23002",
    "28080",
    "38080",
    "8880",
}


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _compose_render_env() -> dict[str, str]:
    python_path = os.pathsep.join(
        path
        for path in (
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        )
        if path
    )
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": python_path,
        "USER": os.environ.get("USER", ""),
    }


def _docker_compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        "docker/runtime-policy.env",
        "--env-file",
        "docker/judge.env.example",
        "-f",
        COMPOSE_FILE,
        "--profile",
        "judge",
        *args,
    ]


def _run_compose_config(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _docker_compose_command("config", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env=_compose_render_env(),
        text=True,
    )


def _compose_config_json() -> dict[str, Any]:
    result = _run_compose_config("--format", "json")
    rendered_config = json.loads(result.stdout)

    assert isinstance(rendered_config, dict)
    return cast("dict[str, Any]", rendered_config)


def _published_ports(service_config: dict[str, Any]) -> set[str]:
    ports = cast("list[dict[str, Any]]", service_config.get("ports", []))
    return {str(port["published"]) for port in ports}


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_judge_lane_services_render_with_judge_profile() -> None:
    result = _run_compose_config("--services")

    rendered_services = set(result.stdout.splitlines())

    assert rendered_services == EXPECTED_RENDERED_SERVICES
    assert rendered_services.isdisjoint(OUT_OF_SCOPE_SERVICES)


@pytest.mark.integration
def test_judge_lane_render_contains_isolated_runtime_identity() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    assert services["omninode-runtime"]["container_name"] == "omninode-judge-runtime"
    assert services["runtime-effects"]["container_name"] == (
        "omninode-judge-runtime-effects"
    )
    assert services["projection-api"]["container_name"] == (
        "omnimarket-judge-projection-api"
    )

    for service_name in ("omninode-runtime", "runtime-effects"):
        environment = services[service_name]["environment"]
        assert environment["ONEX_ENVIRONMENT"] == "judge"
        assert environment["KAFKA_ENVIRONMENT"] == "judge"
        assert environment["ONEX_BOX_ID"] == "judge-local"
        assert environment["ONEX_STATE_ROOT"] == "/app/data/.onex_state_judge"
        assert environment["ONEX_STATE_DIR"] == "/app/data/.onex_state_judge"
        assert environment["KAFKA_INSTANCE_ID"].startswith("judge-")
        assert environment["ONEX_RUNTIME_ADDRESS"].startswith(
            "runtime://judge-local/judge/"
        )
        assert environment["ONEX_RUNTIME_ID"].startswith("judge-")
        assert environment["ONEX_SECRET_RESOLVER_CONFIG_PATH"] == (
            "/app/data/delegation/secret_resolver.yaml"
        )
        resolver_config = json.loads(environment["ONEX_SECRET_RESOLVER_CONFIG_JSON"])
        assert resolver_config["enable_convention_fallback"] is False

    assert services["omninode-runtime"]["environment"]["ONEX_GROUP_ID"] == (
        "onex-judge-runtime-main"
    )
    assert services["runtime-effects"]["environment"]["ONEX_GROUP_ID"] == (
        "onex-judge-runtime-effects"
    )


@pytest.mark.integration
def test_judge_lane_render_omits_keycloak_and_infisical() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    assert set(services) == EXPECTED_RENDERED_SERVICES
    assert set(services).isdisjoint(OUT_OF_SCOPE_SERVICES)

    for service in services.values():
        depends_on = service.get("depends_on", {})
        assert "keycloak" not in depends_on
        assert "infisical" not in depends_on
        environment = service.get("environment", {})
        assert environment.get("INFISICAL_ADDR", "") == ""
        assert environment.get("KEYCLOAK_ISSUER", "") == ""


@pytest.mark.integration
def test_judge_lane_render_does_not_expose_dev_stability_or_prod_ports() -> None:
    rendered_config = _compose_config_json()

    for service_name, service_config in rendered_config["services"].items():
        published_ports = _published_ports(service_config)
        assert published_ports == EXPECTED_PUBLISHED_PORTS[service_name]
        assert published_ports.isdisjoint(DEV_OR_PROD_PUBLISHED_PORTS)


@pytest.mark.integration
def test_judge_lane_render_raises_redpanda_fd_and_partition_capacity() -> None:
    rendered_config = _compose_config_json()
    redpanda = rendered_config["services"]["redpanda"]
    command = redpanda["command"]

    assert redpanda["ulimits"]["nofile"] == {"soft": 65535, "hard": 65535}
    assert "--overprovisioned" in command
    assert command[command.index("--memory") + 1] == "8G"
    assert "--reserve-memory" in command
    assert command[command.index("--reserve-memory") + 1] == "0M"
    assert "--check=false" in command
    assert "topic_partitions_per_shard=7000" in command


@pytest.mark.integration
def test_judge_secret_refs_are_rendered_from_runtime_policy() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    for service_name in ("omninode-runtime", "runtime-effects"):
        environment = services[service_name]["environment"]
        resolver_config = json.loads(environment["ONEX_SECRET_RESOLVER_CONFIG_JSON"])
        mappings = {
            mapping["logical_name"]: mapping["source"]
            for mapping in resolver_config["mappings"]
        }

        assert mappings["llm.gemini.api_key"] == {
            "source_type": "env",
            "source_path": "GEMINI_API_KEY",
        }
        assert mappings["llm.openrouter.api_key"] == {
            "source_type": "env",
            "source_path": "OPEN_ROUTER_API_KEY",
        }
        assert mappings["llm.glm.api_key"] == {
            "source_type": "env",
            "source_path": "LLM_GLM_API_KEY",
        }
