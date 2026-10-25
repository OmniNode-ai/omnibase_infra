# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21279: lab hosts share dev's GitHub command consumer groups."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.dev-bus-member.yml"
GITHUB_NODES = (
    "node_github_repo_gateway_effect",
    "node_github_repo_write_effect",
)
GROUP_IDENTITY = {
    "KAFKA_INSTANCE_ID": "runtime-effects",
    "ONEX_GROUP_ID": "onex-runtime-effects",
    "ONEX_ENVIRONMENT": "local",
    "KAFKA_ENVIRONMENT": "local",
    "RUNTIME_PROFILE": "effects",
    "ONEX_RUNTIME_NODE_SELECTION": ",".join(GITHUB_NODES),
}


@pytest.fixture
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


def test_standalone_compose_topology(compose: dict[str, Any]) -> None:
    assert compose["name"] == "omnibase-infra-dev-bus-member"
    assert set(compose["services"]) == {"runtime-effects"}
    service = compose["services"]["runtime-effects"]
    assert service["container_name"] == "omninode-dev-bus-member-runtime-effects"
    assert service["restart"] == "unless-stopped"
    assert service["labels"] == {
        "com.omninode.lane": "dev",
        "com.omninode.ticket": "OMN-21279",
    }
    assert "extends" not in service
    assert "include" not in compose
    assert "profiles" not in service
    assert "depends_on" not in service
    assert service["ports"] == ["127.0.0.1:${DEV_BUS_MEMBER_HEALTH_PORT:-43086}:8085"]
    assert service["volumes"] == [
        "../contracts:/app/contracts:ro",
        "effects_data:/app/data",
    ]
    assert set(compose["volumes"]) == {"effects_data"}


@pytest.mark.parametrize(("key", "expected"), GROUP_IDENTITY.items())
def test_group_identity_is_literal_and_shared(
    compose: dict[str, Any], key: str, expected: str
) -> None:
    value = compose["services"]["runtime-effects"]["environment"][key]
    assert "${" not in value
    assert value == expected


def test_group_identity_matches_dev_runtime_effects(compose: dict[str, Any]) -> None:
    infra = yaml.safe_load(
        (DOCKER_DIR / "docker-compose.infra.yml").read_text(encoding="utf-8")
    )
    dev_env = infra["services"]["runtime-effects"]["environment"]
    member_env = compose["services"]["runtime-effects"]["environment"]
    assert member_env["KAFKA_INSTANCE_ID"] == dev_env["KAFKA_INSTANCE_ID"]
    default = re.fullmatch(
        r"\$\{ONEX_ENVIRONMENT:-([^}]+)\}", dev_env["ONEX_ENVIRONMENT"]
    )
    assert default is not None, "expected dev's explicit ONEX_ENVIRONMENT default"
    assert member_env["ONEX_ENVIRONMENT"] == default.group(1)


def test_runtime_and_broker_settings(compose: dict[str, Any]) -> None:
    env = compose["services"]["runtime-effects"]["environment"]
    lane = env["ONEX_RUNTIME_LANE"]
    assert lane.startswith("${DEV_BUS_MEMBER_RUNTIME_LANE:?")
    assert lane.endswith("}")
    for phrase in (
        "deploy path",
        "private lane overlay",
        "lanes.dev.bus_member.runtime_lane",
        "never an env file",
    ):
        assert phrase in lane
    assert env["ONEX_RUNTIME_LANE_HEALTH_SPEAKER"] == "false"
    assert env["ONEX_ACTIVE_RUNTIME_PACKAGES"] == "omnibase_infra,omnimarket"
    assert env["KAFKA_SECURITY_PROTOCOL"] == "SASL_PLAINTEXT"
    assert env["KAFKA_SASL_MECHANISM"] == "SCRAM-SHA-256"
    assert env["ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
    assert env["ONEX_RUNTIME_NODE_SELECTION"].split(",") == list(GITHUB_NODES)


def test_deploy_inputs_are_required_and_document_their_sources(
    compose: dict[str, Any],
) -> None:
    service = compose["services"]["runtime-effects"]
    image = service["image"]
    assert image.startswith("${DEV_BUS_MEMBER_RUNTIME_IMAGE:?")
    assert image.endswith("}")
    assert "@sha256" not in image
    assert "tag" in image
    assert "release the dev lane runs" in image
    env = service["environment"]
    broker = env["KAFKA_BOOTSTRAP_SERVERS"]
    assert broker.startswith("${DEV_BUS_MEMBER_BOOTSTRAP_SERVERS:?")
    assert broker.endswith("}")
    for phrase in (
        "dev lane's external listener",
        "deploy path",
        "private lane overlay",
        "config/lab/ci_bus_lanes.yaml lanes.dev.broker",
        "never an env file",
    ):
        assert phrase in broker
    for field in ("USERNAME", "PASSWORD"):
        value = env[f"KAFKA_SASL_{field}"]
        assert value.startswith(f"${{DEV_BUS_MEMBER_SASL_{field}:?")
        assert value.endswith("}")
        for phrase in (
            "host's own lane principal",
            "host's lane credential store",
            "at deploy time",
            "never an env file",
        ):
            assert phrase in value


def test_github_secret_uses_only_a_file_resolver(compose: dict[str, Any]) -> None:
    assert "env_file" not in compose
    for service in compose["services"].values():
        assert "env_file" not in service
        assert not {"GITHUB_TOKEN", "GH_TOKEN"} & service["environment"].keys()
    service = compose["services"]["runtime-effects"]
    assert service["secrets"] == ["github_token"]
    assert set(compose["secrets"]) == {"github_token"}
    secret = compose["secrets"]["github_token"]
    assert set(secret) == {"file"}
    assert secret["file"].startswith("${DEV_BUS_MEMBER_GITHUB_TOKEN_FILE:?")
    assert secret["file"].endswith("}")
    assert "host secret store's materialized file" in secret["file"]
    assert "mode 0600" in secret["file"]
    resolver = json.loads(service["environment"]["ONEX_SECRET_RESOLVER_CONFIG_JSON"])
    assert resolver == {
        "enable_convention_fallback": False,
        "mappings": [
            {
                "logical_name": "GITHUB_TOKEN",
                "source": {
                    "source_path": "/run/secrets/github_token",
                    "source_type": "file",
                },
            }
        ],
    }
