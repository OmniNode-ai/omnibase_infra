# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for the OMN-12924 judge compose profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.judge.yml"
JUDGE_ENV_EXAMPLE = REPO_ROOT / "docker" / "judge.env.example"
RUNTIME_POLICY_ENV_FILE = REPO_ROOT / "docker" / "runtime-policy.env"

EXPECTED_SERVICES = {
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
RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects")
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
PRODUCTION_CONTAINER_NAMES = {
    "omninode-runtime",
    "omninode-runtime-effects",
    "omnimarket-projection-api",
    "omnibase-infra-postgres",
    "omnibase-infra-redpanda",
    "omnibase-infra-valkey",
    "omnibase-infra-keycloak",
    "omnibase-infra-infisical",
}


def _load_compose() -> dict[str, Any]:
    compose = yaml.safe_load(JUDGE_COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(compose, dict)
    return compose


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        env[key] = value
    return env


def _labels_by_key(labels: list[str]) -> dict[str, str]:
    return dict(label.split("=", maxsplit=1) for label in labels)


@pytest.mark.unit
def test_judge_compose_is_standalone_minimal_stack() -> None:
    compose = _load_compose()
    services = compose["services"]

    assert compose["name"] == "omnibase-infra-judge"
    assert set(services) == EXPECTED_SERVICES
    assert set(services).isdisjoint(OUT_OF_SCOPE_SERVICES)

    for service in services.values():
        assert service["profiles"] == ["judge"]


@pytest.mark.unit
def test_judge_profile_omits_keycloak_and_infisical_dependencies() -> None:
    compose = _load_compose()
    services = compose["services"]
    compose_text = JUDGE_COMPOSE_FILE.read_text(encoding="utf-8")

    assert "keycloak:" not in compose_text
    assert "infisical:" not in compose_text
    assert 'KEYCLOAK_ISSUER: ""' in compose_text
    assert 'INFISICAL_ADDR: ""' in compose_text

    for service in services.values():
        depends_on = service.get("depends_on", {})
        assert "keycloak" not in depends_on
        assert "infisical" not in depends_on


@pytest.mark.unit
def test_judge_runtime_identity_and_secret_refs_are_contract_owned() -> None:
    compose = _load_compose()
    policy_env = _load_env(RUNTIME_POLICY_ENV_FILE)
    services = compose["services"]

    assert policy_env["JUDGE_RUNTIME_MAIN_PORT"] == "48085"
    assert policy_env["JUDGE_RUNTIME_EFFECTS_PORT"] == "48086"
    assert policy_env["JUDGE_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
    assert (
        "llm.gemini.api_key"
        in policy_env["JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        "OPENROUTER_API_KEY"
        not in policy_env["JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )

    for service_name in RUNTIME_SERVICES:
        service = services[service_name]
        environment = service["environment"]
        labels = _labels_by_key(service["labels"])

        assert service["container_name"] not in PRODUCTION_CONTAINER_NAMES
        assert environment["ONEX_ENVIRONMENT"] == "judge"
        assert environment["KAFKA_ENVIRONMENT"] == "judge"
        assert environment["ONEX_BOX_ID"] == "judge-local"
        assert environment["ONEX_RUNTIME_ADDRESS"].startswith(
            "runtime://judge-local/judge/"
        )
        assert (
            environment["ONEX_RUNTIME_ADDRESS"]
            == labels["com.omninode.runtime.address"]
        )
        assert environment["ONEX_RUNTIME_ID"] == labels["com.omninode.runtime.id"]
        assert "JUDGE_RUNTIME" in environment["ONEX_RUNTIME_CAPABILITIES"]
        assert "JUDGE_RUNTIME" in environment["ONEX_SECRET_RESOLVER_CONFIG_JSON"]
        assert environment["ONEX_SECRET_RESOLVER_CONFIG_PATH"].startswith(
            "${JUDGE_RUNTIME_"
        )


@pytest.mark.unit
def test_judge_runtime_build_defaults_are_valid_dockerfile_selectors() -> None:
    compose = _load_compose()
    build_args = compose["x-runtime-base"]["build"]["args"]

    assert build_args["BUILD_SOURCE"] == "${BUILD_SOURCE:-release}"
    assert build_args["EXPECTED_BUILD_SOURCE"] == "${EXPECTED_BUILD_SOURCE:-release}"


@pytest.mark.unit
def test_judge_bifrost_defaults_are_chat_completion_urls() -> None:
    compose = _load_compose()
    environment = compose["x-judge-runtime-env"]

    for key in (
        "BIFROST_LOCAL_CODER_ENDPOINT_URL",
        "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
        "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
        "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
    ):
        assert "/chat/completions" in environment[key]
    assert (
        "/openai/embeddings" not in environment["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"]
    )


@pytest.mark.unit
def test_judge_redpanda_can_host_full_contract_topic_catalog() -> None:
    compose = _load_compose()
    redpanda = compose["services"]["redpanda"]
    command = redpanda["command"]

    assert redpanda["ulimits"]["nofile"] == {"soft": 65535, "hard": 65535}
    assert "--overprovisioned" in command
    assert command[command.index("--memory") + 1] == "${REDPANDA_MEMORY:-8G}"
    assert "--reserve-memory" in command
    assert command[command.index("--reserve-memory") + 1] == "0M"
    assert "--check=false" in command
    assert "topic_partitions_per_shard=7000" in command


@pytest.mark.unit
def test_judge_example_env_contains_only_operator_inputs() -> None:
    example_env = _load_env(JUDGE_ENV_EXAMPLE)

    assert example_env["GEMINI_API_KEY"] == "replace-with-gemini-api-key"
    assert "POSTGRES_PASSWORD" in example_env
    assert "VALKEY_PASSWORD" in example_env
    assert "KEYCLOAK_ISSUER" not in example_env
    assert "INFISICAL_ADDR" not in example_env
    assert "JUDGE_RUNTIME_MAIN_PORT" not in example_env
    assert "ONEX_ACTIVE_RUNTIME_PACKAGES" not in example_env


@pytest.mark.unit
def test_judge_profile_names_and_ports_do_not_reuse_dev_or_prod() -> None:
    compose = _load_compose()
    services = compose["services"]

    assert services["postgres"]["ports"] == [
        "${JUDGE_POSTGRES_EXTERNAL_PORT:-35436}:5432"
    ]
    assert services["redpanda"]["ports"] == ["49092:19092", "49644:9644"]
    assert services["valkey"]["ports"] == ["${JUDGE_VALKEY_EXTERNAL_PORT:-56379}:6379"]
    assert services["omninode-runtime"]["ports"] == [
        "${JUDGE_RUNTIME_MAIN_PORT:?runtime policy contract must set JUDGE_RUNTIME_MAIN_PORT}:8085"
    ]
    assert services["runtime-effects"]["ports"] == [
        "${JUDGE_RUNTIME_EFFECTS_PORT:?runtime policy contract must set JUDGE_RUNTIME_EFFECTS_PORT}:8085"
    ]
    assert services["projection-api"]["ports"] == [
        "${JUDGE_PROJECTION_API_PORT:-43002}:3002"
    ]

    for service_name, service in services.items():
        if "container_name" in service:
            assert "judge" in service["container_name"], service_name
            assert service["container_name"] not in PRODUCTION_CONTAINER_NAMES


@pytest.mark.unit
def test_judge_runtime_services_mount_contracts_from_clone() -> None:
    compose = _load_compose()

    for service_name in (*RUNTIME_SERVICES, "projection-api"):
        volumes = compose["services"][service_name]["volumes"]
        assert "../contracts:/app/contracts:ro" in volumes
        assert any(volume.endswith(":/app/skills:ro") for volume in volumes)
