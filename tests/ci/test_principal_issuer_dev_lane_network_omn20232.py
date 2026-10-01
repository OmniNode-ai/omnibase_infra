# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20232: prevent issuer network drift that breaks redpanda DNS resolution.

Joining omnibase-infra_default isolates the principal issuer from the dev lane's
broker, making redpanda:9092 and redpanda:9644 unreachable.
"""

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relative_path: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8")),
    )


def _dev_broker_network_name() -> str:
    infra = _load_yaml("docker/docker-compose.infra.yml")
    (network_key,) = infra["services"]["redpanda"]["networks"]
    name = infra["networks"][network_key]["name"]
    assert isinstance(name, str)
    return name


@pytest.mark.unit
def test_issuer_external_network_matches_dev_broker() -> None:
    issuer = _load_yaml("docker/docker-compose.principal-issuer.yml")
    assert issuer["networks"]["dev-lane"]["name"] == _dev_broker_network_name()


@pytest.mark.unit
def test_manifest_issuer_and_dev_networks_match_dev_broker() -> None:
    manifest = _load_yaml("deploy/lane-census/lane-manifest.yaml")
    lanes = manifest["lanes"]
    assert (
        lanes["principal-issuer"]["network"]
        == lanes["dev"]["network"]
        == _dev_broker_network_name()
    )
