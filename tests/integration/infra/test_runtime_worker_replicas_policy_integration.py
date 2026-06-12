# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime worker replica policy rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from scripts.render_runtime_policy_env import format_dotenv, render_env

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = REPO_ROOT / "docker" / "runtime-policy.env"
PROD_COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.prod.yml"
STABILITY_COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.stability-test.yml"

PROFILE_ENV_PREFIXES = {
    "dev": "DEV",
    "stability-test": "STABILITY_TEST",
    "judge": "JUDGE",
    "prod": "PROD",
}

pytestmark = pytest.mark.integration


def _construct_compose_value(
    loader: yaml.SafeLoader,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)


def _load_contract() -> ModelRuntimePolicyContract:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return ModelRuntimePolicyContract.model_validate(raw)


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        env[key] = value
    return env


def _load_compose(path: Path) -> dict[str, Any]:
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(raw, dict)
    return raw


def test_worker_replica_policy_integrates_contract_env_and_lane_overrides() -> None:
    """OMN-12990: lane recreates must fail-fast instead of dropping workers."""
    contract = _load_contract()

    for profile_name, profile in contract.profiles.items():
        worker = profile.processes["worker"]
        assert worker.replicas >= 1, profile_name

    expected_env = format_dotenv(render_env(contract))
    observed_env = POLICY_ENV_PATH.read_text(encoding="utf-8")
    assert observed_env == expected_env

    policy_env = _load_dotenv(POLICY_ENV_PATH)
    for profile_name, env_prefix in PROFILE_ENV_PREFIXES.items():
        expected_replicas = str(
            contract.profiles[profile_name].processes["worker"].replicas
        )
        assert policy_env[f"{env_prefix}_WORKER_REPLICAS"] == expected_replicas
        assert int(expected_replicas) >= 1

    stability_worker = _load_compose(STABILITY_COMPOSE_PATH)["services"][
        "runtime-worker"
    ]
    prod_worker = _load_compose(PROD_COMPOSE_PATH)["services"]["runtime-worker"]

    assert stability_worker["deploy"]["replicas"].startswith(
        "${STABILITY_TEST_WORKER_REPLICAS:?"
    )
    assert ":-" not in stability_worker["deploy"]["replicas"]
    assert prod_worker["deploy"]["replicas"].startswith("${PROD_WORKER_REPLICAS:?")
    assert ":-" not in prod_worker["deploy"]["replicas"]
