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

    dev_redpanda_command = " ".join(dev["services"]["redpanda"]["command"])
    assert "DEV_REDPANDA_ADVERTISE_HOST" in dev_redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in dev_redpanda_command

    stability_redpanda_command = " ".join(stability["services"]["redpanda"]["command"])
    assert "100.109.203.94:39092" in stability_redpanda_command
    assert "STABILITY_TEST_REDPANDA_ADVERTISE_HOST" not in stability_redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in stability_redpanda_command
    assert "192.168.86.201:39092" not in stability_redpanda_command

    prod_redpanda_command = " ".join(prod["services"]["redpanda"]["command"])
    assert "PROD_REDPANDA_ADVERTISE_HOST" in prod_redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in prod_redpanda_command


def test_dev_redpanda_advertise_host_fails_fast_when_unset() -> None:
    """OMN-15173: unset DEV_REDPANDA_ADVERTISE_HOST must never silently render
    a localhost advertise address.

    Before the fix, an unset var silently defaulted to `localhost` via
    `${DEV_REDPANDA_ADVERTISE_HOST:-localhost}` — a latent off-host regression
    (CI runners / other machines get an advertised address they cannot reach).
    The dev lane must use the Compose fail-fast form (`${VAR:?message}`),
    matching the existing PROD_REDPANDA_ADVERTISE_HOST precedent in
    docker-compose.prod.yml, so an unset var breaks the compose render loudly
    instead of the client failing silently later.
    """
    dev = _load_compose_yaml(REPO_ROOT_COMPOSE)
    dev_redpanda_command = " ".join(dev["services"]["redpanda"]["command"])

    # The old silent-default form must be gone.
    assert "${DEV_REDPANDA_ADVERTISE_HOST:-localhost}" not in dev_redpanda_command
    assert "${DEV_REDPANDA_ADVERTISE_HOST:-}" not in dev_redpanda_command

    # The Compose fail-fast form (colon-question-mark) must be present for
    # both the Kafka and Pandaproxy advertise addresses.
    assert dev_redpanda_command.count("${DEV_REDPANDA_ADVERTISE_HOST:?") == 2
