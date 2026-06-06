# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the OMN-10281 stability-test lane."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (
    "docker/docker-compose.infra.yml",
    "docker/docker-compose.stability-test.yml",
)
REQUIRED_RUNTIME_SERVICES = {
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
}
EXPECTED_RENDERED_SERVICES = {
    "postgres",
    "redpanda",
    "redpanda-partition-cap",
    "valkey",
    "forward-migration",
    "intelligence-migration",
    "migration-gate",
    "keycloak",
    "projection-api",
    *REQUIRED_RUNTIME_SERVICES,
}
OUT_OF_LANE_SERVICES = {
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "intelligence-api",
    "omninode-contract-resolver",
    "phoenix",
    "autoheal",
    "infisical",
}
EXPECTED_PUBLISHED_PORTS = {
    "postgres": {"15436"},
    "redpanda": {"39092", "29644"},
    "valkey": {"26379"},
    "omninode-runtime": {"18085"},
    "runtime-effects": {"18086"},
    "runtime-worker": set(),
    "projection-api": {"13002"},
    "forward-migration": set(),
    "intelligence-migration": set(),
    "migration-gate": set(),
    "redpanda-partition-cap": set(),
    "keycloak": {"38080"},
}
PRODUCTION_PUBLISHED_PORTS = {
    "5436",
    "19092",
    "18082",
    "18081",
    "9644",
    "16379",
    "8880",
    "8085",
    "8086",
    "8087",
    "8053",
    "8091",
    "8092",
    "8093",
    "6006",
    "28080",
}
PRODUCTION_CONTAINER_NAMES = {
    "omninode-runtime",
    "omninode-runtime-effects",
    "omnibase-infra-postgres",
    "omnibase-infra-redpanda",
    "omnibase-infra-valkey",
    "omnibase-forward-migration",
    "omnibase-infra-infisical",
    "omninode-agent-actions-consumer",
    "omninode-skill-lifecycle-consumer",
    "omninode-context-audit-consumer",
    "omnibase-intelligence-api",
    "omnibase-intelligence-migration",
    "omninode-contract-resolver",
    "omnibase-infra-phoenix",
    "omnibase-infra-autoheal",
    "omnibase-infra-keycloak",
    "omnimarket-projection-api",
}


def _http_url(authority: str) -> str:
    return "http" + "://" + authority


def _cidr(prefix: str, suffix: str) -> str:
    return prefix + "." + suffix


COMPOSE_RENDER_ENV = {
    "INFISICAL_AUTH_SECRET": "render-only-infisical-auth-secret",
    "INFISICAL_DB_CONNECTION_URI": "postgresql://postgres:postgres@postgres:5432/infisical",
    "INFISICAL_ENCRYPTION_KEY": "render-only-infisical-encryption-key-32",
    "INFISICAL_REDIS_URL": "redis://:render-only-valkey-password@valkey:6379",
    "GITHUB_TOKEN": "render-only-github-token",
    "LINEAR_API_KEY": "render-only-linear-api-key",
    "LLM_CODER_FAST_URL": _http_url("llm-coder-fast.invalid"),
    "LLM_CODER_URL": _http_url("llm-coder.invalid"),
    "LLM_DEEPSEEK_R1_URL": _http_url("llm-deepseek.invalid"),
    "LLM_EMBEDDING_URL": _http_url("llm-embedding.invalid"),
    "LLM_ENDPOINT_CIDR_ALLOWLIST": _cidr("192.168.86", "0/24"),
    "LOCAL_LLM_SHARED_SECRET": "render-only-local-llm-secret",
    "OMNI_HOME": "/data/omninode/omni_home",
    "ONEX_REGISTRATION_AUTO_ACK": "false",
    "ONEX_INFRA_HOST": "192.168.86.201",
    "ONEX_INFRA_USER": "jonah",
    "ONEX_SERVICE_CLIENT_SECRET": "render-only-client-secret",
    "POSTGRES_PASSWORD": "postgres",
    "REDPANDA_ADVERTISE_HOST": "192.168.86.201",
    "STABILITY_TEST_REDPANDA_ADVERTISE_HOST": "100.109.203.94",
    "STABILITY_TEST_POSTGRES_EXTERNAL_PORT": "15436",
    "STABILITY_TEST_VALKEY_EXTERNAL_PORT": "26379",
    "STABILITY_TEST_REDPANDA_ADMIN_PORT": "29644",
    "STABILITY_TEST_REDPANDA_EXTERNAL_PORT": "39092",
    "STABILITY_TEST_REDPANDA_PANDAPROXY_PORT": "28082",
    "STABILITY_TEST_KEYCLOAK_EXTERNAL_PORT": "38080",
}


def _docker_compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        "docker/runtime-policy.env",
        "-f",
        COMPOSE_FILES[0],
        "-f",
        COMPOSE_FILES[1],
        "--profile",
        "runtime",
        *args,
    ]


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
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "USER": os.environ.get("USER", ""),
        **COMPOSE_RENDER_ENV,
    }


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


def _label_value(service_config: dict[str, Any], key: str) -> str | None:
    labels = service_config.get("labels", {})
    if isinstance(labels, dict):
        value = labels.get(key)
        return str(value) if value is not None else None
    if isinstance(labels, list):
        prefix = f"{key}="
        for label in labels:
            if isinstance(label, str) and label.startswith(prefix):
                return label.removeprefix(prefix)
    return None


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_stability_lane_runtime_services_render_with_runtime_profile() -> None:
    result = _run_compose_config("--services")

    rendered_services = set(result.stdout.splitlines())

    assert rendered_services == EXPECTED_RENDERED_SERVICES
    assert rendered_services.isdisjoint(OUT_OF_LANE_SERVICES)


@pytest.mark.integration
def test_stability_lane_render_contains_isolated_runtime_identity() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    assert services["omninode-runtime"]["container_name"] == (
        "omninode-stability-test-runtime"
    )
    assert services["runtime-effects"]["container_name"] == (
        "omninode-stability-test-runtime-effects"
    )
    assert services["runtime-worker"]["container_name"] == (
        "omninode-stability-test-runtime-worker"
    )

    for service_name in REQUIRED_RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        assert "BUILD_SOURCE" not in environment
        assert "EXPECTED_BUILD_SOURCE" not in environment
        assert environment["OMNIMEMORY_ENABLED"] == "false"
        assert environment["OMNIMEMORY_MEMGRAPH_HOST"] == ""
        assert environment["ONEX_ENVIRONMENT"] == "stability-test"
        assert environment["KAFKA_ENVIRONMENT"] == "stability-test"
        assert environment["ONEX_INFRA_HOST"] == "192.168.86.201"
        assert environment["ONEX_INFRA_USER"] == "jonah"
        assert environment["ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
        assert environment["KAFKA_INSTANCE_ID"].startswith("stability-test-")
        assert environment["KAFKA_MAX_POLL_INTERVAL_MS"] == "1800000"
        assert "image" not in services[service_name]
        assert services[service_name]["restart"] == "unless-stopped"
        assert environment["ONEX_RUNTIME_ADDRESS"].startswith(
            "runtime://omninode-pc/stability-test/"
        )
        assert environment["ONEX_RUNTIME_ID"].startswith("stability-test-")
        assert environment["ONEX_STATE_DIR"] == environment["ONEX_STATE_ROOT"]
        assert (
            _label_value(
                services[service_name],
                "com.omninode.runtime.address",
            )
            == environment["ONEX_RUNTIME_ADDRESS"]
        )
        assert (
            _label_value(
                services[service_name],
                "com.omninode.runtime.id",
            )
            == environment["ONEX_RUNTIME_ID"]
        )

    assert services["omninode-runtime"]["environment"]["ONEX_GROUP_ID"] == (
        "onex-stability-test-runtime-main"
    )
    assert services["forward-migration"]["container_name"] == (
        "omnibase-infra-stability-test-forward-migration"
    )
    assert services["intelligence-migration"]["container_name"] == (
        "omnibase-infra-stability-test-intelligence-migration"
    )
    assert services["redpanda-partition-cap"]["container_name"] == (
        "omnibase-infra-stability-test-redpanda-partition-cap"
    )
    assert (
        services["omninode-runtime"]["depends_on"]["intelligence-migration"][
            "condition"
        ]
        == "service_completed_successfully"
    )
    assert (
        services["omninode-runtime"]["depends_on"]["redpanda-partition-cap"][
            "condition"
        ]
        == "service_completed_successfully"
    )
    for service_name in REQUIRED_RUNTIME_SERVICES:
        assert (
            services[service_name]["depends_on"]["migration-gate"]["condition"]
            == "service_healthy"
        )
    assert services["redpanda-partition-cap"]["depends_on"]["redpanda"][
        "condition"
    ] == ("service_healthy")


@pytest.mark.integration
def test_stability_lane_render_inherits_release_build_source() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    for service_name in REQUIRED_RUNTIME_SERVICES:
        build = services[service_name]["build"]
        assert build["context"] == str(REPO_ROOT)
        assert build["dockerfile"] == "docker/Dockerfile.runtime"
        assert build["args"] == {
            "BUILD_SOURCE": "release",
            "EXPECTED_BUILD_SOURCE": "release",
        }


@pytest.mark.integration
def test_stability_lane_runtime_socket_uses_owned_tmpfs() -> None:
    rendered_config = _compose_config_json()
    runtime_service = rendered_config["services"]["omninode-runtime"]

    assert runtime_service["tmpfs"] == ["/run/onex-runtime:uid=1000,gid=1000,mode=0770"]
    volume_targets = {
        volume["target"]
        for volume in runtime_service.get("volumes", [])
        if isinstance(volume, dict) and "target" in volume
    }
    assert "/run/onex-runtime" not in volume_targets


@pytest.mark.integration
def test_stability_projection_api_has_separate_infra_and_analytics_dsns() -> None:
    rendered_config = _compose_config_json()
    environment = rendered_config["services"]["projection-api"]["environment"]

    assert environment["ONEX_ENVIRONMENT"] == "stability-test"
    assert environment["KAFKA_ENVIRONMENT"] == "stability-test"
    assert environment["OMNIBASE_INFRA_DB_URL"].endswith("/omnibase_infra")
    assert environment["OMNIDASH_ANALYTICS_DB_URL"].endswith("/omnidash_analytics")


@pytest.mark.integration
def test_stability_lane_render_inherits_failing_runtime_healthcheck() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    for service_name in REQUIRED_RUNTIME_SERVICES:
        assert services[service_name]["healthcheck"]["test"] == [
            "CMD",
            "curl",
            "-sf",
            "http://localhost:8085/health",
        ]


@pytest.mark.integration
def test_stability_lane_render_does_not_expose_production_ports_or_services() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    assert set(services) == EXPECTED_RENDERED_SERVICES
    assert set(services).isdisjoint(OUT_OF_LANE_SERVICES)
    assert rendered_config["networks"]["omnibase-infra-network"]["name"] == (
        "omnibase-infra-stability-test-network"
    )
    assert rendered_config["networks"]["omnibase-infra-network"]["driver"] == "bridge"
    assert rendered_config["networks"]["omnimemory-network"]["name"] == (
        "omnibase-infra-stability-test-omnimemory-network"
    )
    assert rendered_config["networks"]["omnimemory-network"]["driver"] == "bridge"
    assert (
        rendered_config["networks"]["omnimemory-network"].get("external", False)
        is False
    )

    rendered_container_names = {
        service_config["container_name"]
        for service_config in services.values()
        if "container_name" in service_config
    }
    assert rendered_container_names.isdisjoint(PRODUCTION_CONTAINER_NAMES)

    for service_name, service_config in services.items():
        published_ports = _published_ports(service_config)
        assert published_ports == EXPECTED_PUBLISHED_PORTS[service_name]
        assert published_ports.isdisjoint(PRODUCTION_PUBLISHED_PORTS)

    redpanda_command = " ".join(services["redpanda"]["command"])
    assert "100.109.203.94:39092" in redpanda_command
    assert "100.109.203.94:28082" in redpanda_command
    assert "192.168.86.201:39092" not in redpanda_command
    assert "localhost:19092" not in redpanda_command

    partition_cap_command = "\n".join(services["redpanda-partition-cap"]["command"])
    assert "/usr/bin/rpk -X brokers=redpanda:9092" in partition_cap_command
    assert "admin.hosts=redpanda:9644" in partition_cap_command
    assert "topic_partitions_per_shard" in partition_cap_command
    assert "7000" in partition_cap_command
    assert "topic_memory_per_partition" in partition_cap_command
    assert "1048576" in partition_cap_command

    for service_name in REQUIRED_RUNTIME_SERVICES:
        volume_targets = {
            volume["target"] for volume in services[service_name].get("volumes", [])
        }
        assert "/app/contracts" not in volume_targets, service_name
        assert "/app/skills" in volume_targets, service_name
        assert "/data/omninode/omni_home" in volume_targets, service_name


@pytest.mark.integration
def test_stability_lane_render_exposes_session_health_contract() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    for service_name in REQUIRED_RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        assert environment["ONEX_INFRA_HOST"] == "192.168.86.201"
        assert environment["ONEX_INFRA_USER"] == "jonah"
        assert environment["OMNI_HOME"] == "/data/omninode/omni_home"
        assert environment["SSH_STRICT_HOST_KEY_CHECKING"] == "accept-new"
