# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for runtime-lane Kafka environment isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
REPO_ROOT_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"


def _construct_compose_value(
    loader: yaml.SafeLoader,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


class _TestSafeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_TestSafeLoader.add_constructor("!override", _construct_compose_value)


def _load_compose_yaml(path: Path) -> dict:
    compose = yaml.load(path.read_text(encoding="utf-8"), Loader=_TestSafeLoader)  # noqa: S506
    assert isinstance(compose, dict)
    return compose


@pytest.mark.parametrize(
    ("overlay_name", "expected_environment"),
    [
        ("docker-compose.prod.yml", "prod"),
        ("docker-compose.stability-test.yml", "stability-test"),
    ],
)
def test_non_local_runtime_lanes_set_kafka_environment(
    overlay_name: str,
    expected_environment: str,
) -> None:
    overlay_path = REPO_ROOT / "docker" / overlay_name
    overlay = _load_compose_yaml(overlay_path)

    for service_name in RUNTIME_SERVICES:
        environment = overlay["services"][service_name]["environment"]
        assert environment["ONEX_ENVIRONMENT"] == expected_environment
        assert environment["KAFKA_ENVIRONMENT"] == expected_environment


def test_runtime_lanes_use_lane_specific_redpanda_advertise_hosts() -> None:
    dev = _load_compose_yaml(REPO_ROOT_COMPOSE)
    stability = _load_compose_yaml(
        REPO_ROOT / "docker" / "docker-compose.stability-test.yml"
    )
    prod = _load_compose_yaml(REPO_ROOT / "docker" / "docker-compose.prod.yml")

    assert "DEV_REDPANDA_ADVERTISE_HOST" in " ".join(
        dev["services"]["redpanda"]["command"]
    )
    assert "${REDPANDA_ADVERTISE_HOST" not in " ".join(
        dev["services"]["redpanda"]["command"]
    )

    stability_redpanda_command = " ".join(stability["services"]["redpanda"]["command"])
    assert "100.109.203.94:39092" in stability_redpanda_command
    assert "STABILITY_TEST_REDPANDA_ADVERTISE_HOST" not in stability_redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in stability_redpanda_command
    assert "192.168.86.201:39092" not in stability_redpanda_command

    prod_redpanda_command = " ".join(prod["services"]["redpanda"]["command"])
    assert "PROD_REDPANDA_ADVERTISE_HOST" in prod_redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in prod_redpanda_command
