# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for contract-owned runtime policy."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = ROOT / "docker" / "runtime-policy.env"


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def test_runtime_policy_contract_controls_all_runtime_lane_ports() -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract = ModelRuntimePolicyContract.model_validate(raw)
    rendered_env = _load_dotenv(POLICY_ENV_PATH)

    assert contract.llm_cloud_endpoint_host_allowlist == (
        "generativelanguage.googleapis.com",
        "api.z.ai",
    )
    assert (
        rendered_env["LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST"]
        == "generativelanguage.googleapis.com,api.z.ai"
    )

    expected_ports = {
        "DEV_RUNTIME_MAIN_PORT": contract.profiles["dev"].main_port,
        "DEV_RUNTIME_EFFECTS_PORT": contract.profiles["dev"].effects_port,
        "STABILITY_TEST_RUNTIME_MAIN_PORT": contract.profiles[
            "stability-test"
        ].main_port,
        "STABILITY_TEST_RUNTIME_EFFECTS_PORT": contract.profiles[
            "stability-test"
        ].effects_port,
        "JUDGE_RUNTIME_MAIN_PORT": contract.profiles["judge"].main_port,
        "JUDGE_RUNTIME_EFFECTS_PORT": contract.profiles["judge"].effects_port,
        "PROD_RUNTIME_MAIN_PORT": contract.profiles["prod"].main_port,
        "PROD_RUNTIME_EFFECTS_PORT": contract.profiles["prod"].effects_port,
    }

    assert expected_ports == {key: int(rendered_env[key]) for key in expected_ports}


def test_runtime_policy_contract_controls_logical_secret_resolver_refs() -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract = ModelRuntimePolicyContract.model_validate(raw)
    rendered_env = _load_dotenv(POLICY_ENV_PATH)

    stability_profile = contract.profiles["stability-test"]
    judge_profile = contract.profiles["judge"]
    logical_names = {
        mapping.logical_name for mapping in stability_profile.secret_resolver_mappings
    }
    judge_logical_names = {
        mapping.logical_name for mapping in judge_profile.secret_resolver_mappings
    }

    assert logical_names == {
        "llm.openrouter.api_key",
        "llm.glm.api_key",
        "llm.gemini.api_key",
    }
    assert judge_logical_names == logical_names
    assert (
        rendered_env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH"]
        == stability_profile.secret_resolver_config_path
    )
    assert (
        "OPENROUTER_API_KEY"
        not in rendered_env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        rendered_env["JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH"]
        == judge_profile.secret_resolver_config_path
    )
    assert (
        "OPENROUTER_API_KEY"
        not in rendered_env["JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
