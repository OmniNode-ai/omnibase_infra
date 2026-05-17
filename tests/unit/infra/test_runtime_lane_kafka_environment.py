# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for runtime-lane Kafka environment isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")


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
