# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime policy contract and compose boundary checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = ROOT / "docker" / "runtime-policy.env"
COMPOSE_PATH = ROOT / "docker" / "docker-compose.infra.yml"
STABILITY_COMPOSE_PATH = ROOT / "docker" / "docker-compose.stability-test.yml"
PROD_COMPOSE_PATH = ROOT / "docker" / "docker-compose.prod.yml"

pytestmark = pytest.mark.unit


def _load_contract() -> ModelRuntimePolicyContract:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return ModelRuntimePolicyContract.model_validate(raw)


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def test_runtime_policy_contract_declares_three_runtime_lanes() -> None:
    contract = _load_contract()

    assert set(contract.profiles) == {"dev", "stability-test", "prod"}
    assert contract.profiles["dev"].main_port == 8085
    assert contract.profiles["dev"].effects_port == 8086
    assert contract.profiles["stability-test"].main_port == 18085
    assert contract.profiles["stability-test"].effects_port == 18086
    assert contract.profiles["stability-test"].topic_provisioner_max_partitions == 1
    assert contract.profiles["prod"].main_port == 28085
    assert contract.profiles["prod"].effects_port == 28086


def test_runtime_policy_env_matches_contract_renderer() -> None:
    from scripts.render_runtime_policy_env import format_dotenv, render_env

    contract = _load_contract()
    expected = format_dotenv(render_env(contract))
    observed = POLICY_ENV_PATH.read_text(encoding="utf-8")

    assert observed == expected


def test_runtime_policy_env_has_expected_lane_values() -> None:
    env = _load_dotenv(POLICY_ENV_PATH)

    assert env["AUXILIARY_SERVICES_OMNIMEMORY_ENABLED"] == "false"
    assert env["ONEX_ACTIVE_RUNTIME_PACKAGES"] == "omnibase_infra,omnimarket"
    assert env["DEV_RUNTIME_MAIN_PORT"] == "8085"
    assert env["STABILITY_TEST_RUNTIME_MAIN_PORT"] == "18085"
    assert env["STABILITY_TEST_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
    assert env["PROD_RUNTIME_MAIN_PORT"] == "28085"
    assert (
        env["STABILITY_TEST_RUNTIME_MAIN_CAPABILITIES"]
        == "market.skill-proof,workflow.orchestration,runtime.main"
    )


def test_compose_consumes_policy_env_instead_of_hardcoded_policy_values() -> None:
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    stability_text = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")
    prod_text = PROD_COMPOSE_PATH.read_text(encoding="utf-8")

    assert (
        "ONEX_ACTIVE_RUNTIME_PACKAGES: ${ONEX_ACTIVE_RUNTIME_PACKAGES:-"
        not in compose_text
    )
    assert "OMNIMEMORY_ENABLED: ${OMNIMEMORY_ENABLED:-" not in compose_text
    assert 'OMNIINTELLIGENCE_PUBLISH_INTROSPECTION: "true"' not in compose_text

    for text in (stability_text, prod_text):
        assert 'OMNIMEMORY_ENABLED: ""' not in text
        assert 'OMNIMEMORY_MEMGRAPH_HOST: ""' not in text
        assert 'BIFROST_VERIFY_ENDPOINTS: "0"' not in text
        assert "ONEX_RUNTIME_CAPABILITIES: market.skill-proof" not in text
        assert "ONEX_RUNTIME_CAPABILITIES: effects.consumer" not in text
        assert "ONEX_RUNTIME_CAPABILITIES: workflow.dispatch" not in text
